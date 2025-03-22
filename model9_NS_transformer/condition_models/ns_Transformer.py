import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F
from layers.Decompose import series_decomp, FourierLayer

class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y

class encoder(nn.Module):
    def __init__(self, configs):
        super(encoder, self).__init__()
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor_c, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), 
                                    configs.d_model_c, configs.n_heads_c),
                    configs.d_model_c,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers_c)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model_c)
        )
    def forward(self, x_enc, attn_mask=None, tau=None, delta=None):
        enc_out, attns = self.encoder(x_enc, attn_mask, tau, delta)
        return enc_out, attns

class decoder(nn.Module):
    def __init__(self, configs):
        super(decoder, self).__init__()
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(True, configs.factor_c, attention_dropout=configs.dropout,
                                    output_attention=False),
                                configs.d_model_c, configs.n_heads_c),
                    AttentionLayer(
                        DSAttention(False, configs.factor_c, attention_dropout=configs.dropout,
                                    output_attention=False),
                            configs.d_model_c, configs.n_heads_c),
                    configs.d_model_c,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers_c)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model_c),
            projection=nn.Linear(configs.d_model_c, configs.c_out, bias=True)
        )

    def forward(self, dec_out, enc_out, x_mask=None, cross_mask=None, tau=None, delta=None):
        dec_out = self.decoder(dec_out, enc_out, x_mask, cross_mask, tau, delta)
        return dec_out

class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        self.decomposition = configs.decomposition

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model_c, configs.embed, configs.freq,
                                           configs.dropout)

        # Series decomposition
        self.decomp = series_decomp(kernel_size=configs.kernel_size) if self.decomposition else None
        self.seasonal = FourierLayer(d_model=configs.d_model_c, factor=configs.fourier_factor) if self.decomposition else None


        # Encoder
        self.encoder = encoder(configs)
        # Decoder
        
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model_c, configs.embed, configs.freq,
                                               configs.dropout)
        self.decoder = decoder(configs)

        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)

        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out,enc_out

   
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
            if self.decomposition:
                res, trend = self.decomp(x_enc) # trend/res : [bsz, seq_len, n_vars]
                seasonal = self.seasonal(res)
                x_enc = trend + seasonal
            dec_out,enc_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out, torch.tensor([0.0]) ,enc_out # [B, L, D]
