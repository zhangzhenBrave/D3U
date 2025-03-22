from typing import Optional
from torch import nn
from torch import Tensor
from layers.SVQ.SVQ_backbone import SVQ_backbone
from layers.Decompose import series_decomp, FourierLayer

class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        self.enc_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        n_layers = configs.e_layers_c
        n_heads = configs.n_heads_c
        d_model = configs.d_model_c
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        individual = configs.individual
        sout = configs.sout
    
        patch_len = configs.patch_size
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        

        kernel_size = configs.kernel_size
        
        num_codebook = configs.num_codebook
        codebook_size = configs.codebook_size
        wFFN = configs.wFFN
        svq = configs.svq,
        length = configs.length

        self.model = SVQ_backbone(codebook_size, length, num_codebook=num_codebook, svq=svq, wFFN=wFFN, c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, sout =sout, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)

        # Series decomposition
        self.decomposition = configs.decomposition
        self.decomp = series_decomp(kernel_size=configs.kernel_size) if self.decomposition else None
        self.seasonal = FourierLayer(d_model=configs.d_model_c, factor=configs.fourier_factor) if self.decomposition else None


    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark, vq_details=True):
        batch_x_mark=None
        dec_inp=None
        batch_y_mark=None
        # Series Decompose
        if self.decomposition:
            res, trend = self.decomp(x) # trend/res : [bsz, seq_len, n_vars]
            seasonal = self.seasonal(res)
            x = trend + seasonal

        x = x.permute(0,2,1)
        x, loss, enc_out, vq_details_lst = self.model(x, vq_details=vq_details)     # enc_out : [bsz x nvars x d_model x patch_num]
    
        x = x.permute(0,2,1)                                                        # x: [bsz x pred len x nvars]
        enc_out = enc_out.permute(0, 1, 3, 2)                                       # enc_out : [bsz x nvars x patch_num x d_model]

        '''print('x.shape = ', x.shape)
        print('enc_out.shape = ', enc_out.shape)
        if vq_details:
            print('index.shape = ', vq_details_lst[0].shape)
            print('distance.shape = ', vq_details_lst[1].shape)'''

        # return x, loss, vq_details_lst  # for Case_study.ipynb
        return x, loss,enc_out 
