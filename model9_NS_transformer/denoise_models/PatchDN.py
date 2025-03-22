

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp
import torch.nn.functional as F
import argparse
from einops import repeat
from .dm_layers.embedders import Time_series_PatchEmbed
from .dm_layers.PatchTST_layers import positional_encoding
# from model9_NS_transformer.ns_layers.RevIN import RevIN
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.embedding_table = nn.Linear(d_model, hidden_size , bias=True)

    def forward(self, y):
        embeddings=self.embedding_table(y)

        return embeddings


#################################################################################
#                                 Core PatchDN Model                                #
#################################################################################

class PatchDNBlock(nn.Module):
    """
    A PatchDN block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of PatchDN.
    """
    def __init__(self, hidden_size, patch_num, context_window):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(hidden_size*patch_num, context_window , bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class PatchDN(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        MTS_args,
        depth=2,
        mlp_ratio=4.0,
        learn_sigma=False,
        pe='zeros',
        learn_pe=True,

    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels =MTS_args.enc_in
        self.out_channels =self.in_channels * 2 if learn_sigma else self.in_channels
        self.num_heads = MTS_args.n_heads_d
        self.context_window= MTS_args.pred_len
        self.condition_hidden_size= MTS_args.d_model_c
        self.hidden_size= MTS_args.d_model_d
        self.use_pretraining_condition = MTS_args.use_pretraining_condition

        ###patch 参数
        self.patch_size = MTS_args.patch_size
        self.stride = MTS_args.stride
        self.padding_patch = MTS_args.padding_patch

        self.x_embedder = Time_series_PatchEmbed( self.context_window,self.patch_size,self.stride,self.padding_patch,self.hidden_size)
        patch_num= self.x_embedder.patch_num

        self.t_embedder = TimestepEmbedder(self.hidden_size)
        if self.use_pretraining_condition:
          self.y_embedder = LabelEmbedder(self.context_window, self.hidden_size)
        else:
          self.y_embedder = LabelEmbedder(12*self.condition_hidden_size, self.hidden_size) 
        
        


        # self.pos_embed = nn.Parameter(torch.zeros(1, patch_num, self.hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            PatchDNBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(self.hidden_size, patch_num, self.context_window)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.patch_num ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.normal_(self.x_embedder.value_embedding.weight, std=0.02)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self,x,t,y):
        """
        Forward pass of DiT.
        x: bs x pred_len x nvars       y_t
        t: bs                           时间步
        y: bs x pred_len x nvars       条件            

        """
        x=x.permute(0,2,1)                                                        #x: [bs x nvars x pred_len]
        
        bs, nvars,pred_len=x.shape

        # do patching
        x = self.x_embedder(x)                                     #u: [bs * nvars x patch_num x d_model]

       
        t = repeat(t, "b -> b d", d=nvars).reshape(-1)                             #t:  [bs * nvars ]
        t = self.t_embedder(t)                                                      #t: (bs * nvars, D)
       
            
        if self.use_pretraining_condition:
            #y:[ bsz x pred_len  x n_vars ]
            bsz, n_vars,pred_len=y.permute(0,2,1).shape 
            y=y.reshape(bsz * n_vars,pred_len)         
            y = self.y_embedder(y)  
        else:

            #y:[ bsz x n_vars x patch_num  x d_model ]
            bsz, n_vars,patch_num,d_model=y.shape 
            y = y.reshape(bsz * n_vars, patch_num*d_model)                                
            y = self.y_embedder(y)                                                      #y: (bs * nvars, D)
        c = t + y                                                                   #c:  (bs * nvars, D)
         
        for block in self.blocks:
            x = block(x, c)                                                         #  u: [bs * nvars x patch_num x d_model]

        x = self.final_layer(x, c)                                                  #  x: [bs * nvars x context_window]

        x=x.reshape(bs, nvars, pred_len).permute(0,2,1) 
        return x






    