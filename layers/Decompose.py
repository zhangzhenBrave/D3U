import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
import math

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x:torch.Tensor):
        if len(x.shape)==3:
            # padding on the both ends of time series
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=1)
            x = self.avg(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        elif len(x.shape)==2:
            front = x[0:1, :].repeat((self.kernel_size - 1) // 2, 1)
            end = x[-1:, :].repeat((self.kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=0)
            x = self.avg(x.permute(1, 0))
            x = x.permute(1, 0)
        else:
            raise Exception(
                'Unsupported data shape: {}, [seq_len, bsz, patch_len] or [series_len, n_vars]'.format(x.shape))
        return x

class raw_series_decomp(nn.Module):
    """
    Series decomposition block used in data loading stage
    """
    def __init__(self, kernel_size, stride=1):
        super(raw_series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=stride)

    def forward(self, x:torch.Tensor):
        # x : [series_len, n_vars]
        moving_mean = self.moving_avg(x)                        # moving_mean : [series_len, n_vars]
        res = x - moving_mean                                   # res : [series_len, n_vars]
        return res.numpy(), moving_mean.numpy()

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size, stride=1):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=stride)

    def forward(self, x):
        # x : [bsz, seq_len, n_vars]
        moving_mean = self.moving_avg(x)                        # moving_mean : [bsz, seq_len, n_vars]
        res = x - moving_mean                                   # res : [bsz, seq_len, n_vars]
        return res, moving_mean
    

class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)
    
    def extrapolate(self, x_freq, f, t_0):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t_0, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = x_freq.abs()*2/t_0
        amp = rearrange(amp, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')
    
    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))  # 计算 top_k 的值
        # torch.topk
        #   largest: if True，按照大到小排序； if False，按照小到大排序
        #   sorted: 返回的结果按照顺序返回
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple