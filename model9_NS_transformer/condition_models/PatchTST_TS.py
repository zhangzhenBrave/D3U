import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding, DataEmbedding_inverted
from layers.Decompose import series_decomp, FourierLayer
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class encoder(nn.Module):
    def __init__(self, configs):
        super(encoder, self).__init__()
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor_c, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), 
                        configs.d_model_c, configs.n_heads_c),
                    configs.d_model_c,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers_c)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model_c), Transpose(1,2))
        )

    def forward(self, x_enc, attn_mask=None, tau=None, delta=None):
        """
        x_enc : [bs * nvars x patch_num x d_model]
        """
        enc_out, attns = self.encoder(x_enc, attn_mask, tau, delta)
        return enc_out, attns


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        self.decomposition=configs.decomposition
     
        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model_c, patch_len, stride, padding, configs.padding_patch,configs.dropout)

        # Series decomposition
        self.decomp = series_decomp(kernel_size=configs.kernel_size) if self.decomposition else None
        self.seasonal = FourierLayer(d_model=configs.d_model_c, factor=configs.fourier_factor) if self.decomposition else None

        # Encoder
        self.encoder = encoder(configs)

        # Prediction Head
        self.head_nf = configs.d_model_c * \
                       int((configs.seq_len - patch_len) / stride + 2)

        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        condition_out=enc_out 


        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        condition_out=condition_out.permute(0,1,3,2)                   #condition_out:[ bsz, n_vars ,patch_num,d_model ]
        return dec_out,condition_out

   

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.decomposition:
            res, trend = self.decomp(x_enc) # trend/res : [bsz, seq_len, n_vars]
            seasonal = self.seasonal(res)
            x_enc = trend + seasonal
        x_mark_enc=None
        x_dec=None
        x_mark_dec=None
        dec_out,enc_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out, torch.tensor([0.0]) ,enc_out   # [B, L, D]
       