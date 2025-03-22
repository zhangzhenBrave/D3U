import torch
import torch.nn as nn
from layers.Embed import DataEmbedding
import yaml
import argparse
from model9_NS_transformer.diffusion_models.diffusion_utils import *
from model9_NS_transformer.denoise_models.PatchDN import PatchDN
from model9_NS_transformer.denoise_models.MLP import MLP
from model9_NS_transformer.denoise_models.CNN import CNN_DiffusionUnet

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

class Model(nn.Module):
    """
    Vanilla Transformer
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        with open(configs.diffusion_config_dir, "r") as f:
            config = yaml.unsafe_load(f)
            diffusion_config = dict2namespace(config)

        diffusion_config.diffusion.timesteps = configs.timesteps
        diffusion_config.diffusion.sampling_timesteps = configs.sampling_timesteps
        self.args = configs
        self.diffusion_config = diffusion_config

        self.model_var_type = diffusion_config.model.var_type
        self.num_timesteps = diffusion_config.diffusion.timesteps
        self.sampling_timesteps=diffusion_config.diffusion.sampling_timesteps
        self.vis_step = diffusion_config.diffusion.vis_step           #100
        self.num_figs = diffusion_config.diffusion.num_figs           #10
        self.dataset_object = None

        #beta_schedule: linear ,beta_start: 0.0001, beta_end: 0.02
        betas = make_beta_schedule(schedule=diffusion_config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=diffusion_config.diffusion.beta_start, end=diffusion_config.diffusion.beta_end)
        betas = self.betas = betas.float()
        self.betas=betas
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_cumprod= alphas_cumprod
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if diffusion_config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod=alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        self.sqrt_one_minus_betas_sqrt = torch.sqrt(1./alphas)
        self.sqrt_recip_alphas_cumprod =torch.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod=torch.sqrt(1. / alphas_cumprod - 1)

        self.posterior_mean_coef1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        self.posterior_mean_coef3 = (
                betas /torch.sqrt(1 - alphas_cumprod) / torch.sqrt( alphas)
        )

        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped=torch.log(posterior_variance.clamp(min=1e-20))

        self.posterior_variance = posterior_variance
        self.logvar = betas.log()



        self.tau = None  # precision fo test NLL computation

        # CATE MLP
        if self.args.denoise_model=='MLP':
            self.diffussion_model = MLP(diffusion_config, self.args)
        elif self.args.denoise_model=='PatchDN':
            self.diffussion_model = PatchDN(MTS_args=self.args,depth=self.args.depth, mlp_ratio=1.0)
        elif self.args.denoise_model=='CNN':
            self.diffussion_model = CNN_DiffusionUnet(diffusion_config, self.args)

    
        a = 0



    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t) * noise
        )
    

    # Forward functions
    def q_sample(self, y, t, noise=None):
        #y=y_0(真实), y_T_mean（NSformer学出来的）, self.model.alphas_bar_sqrt,self.model.one_minus_alphas_bar_sqrt, t（时间步骤), noise=e（高斯噪声）
        """
        y_0_hat: prediction of pre-trained guidance model; can be extended to represent
            any prior mean setting at timestep T.
        """
        #y[batch,len,var]
        if noise is None:
            noise = torch.randn_like(y)
        sqrt_alpha_bar_t = extract(self.alphas_bar_sqrt, t, y).to(y.device)
        
        sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, y).to(y.device)
        
        # q(y_t | y_0, x)
        y_t = sqrt_alpha_bar_t * y + sqrt_one_minus_alpha_bar_t * noise

        return y_t

    @torch.no_grad()
    def fast_sample(self,y_T_mean,enc_out, y_shape,eta):
            device=enc_out.device
            batch= y_shape[0]
            alphas_cumprod,sqrt_recip_alphas_cumprod,sqrt_recipm1_alphas_cumprod=self.alphas_cumprod.to(device),self.sqrt_recip_alphas_cumprod.to(device),self.sqrt_recipm1_alphas_cumprod.to(device)

            # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = torch.linspace(-1,  self.num_timesteps - 1, steps=self.sampling_timesteps + 1)

            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            img = torch.randn(y_shape, device=device)
            # y_p_seq = [img.detach().cpu()]
            for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
                time_cond = torch.full((batch,), time,device=device, dtype=torch.long)
                    
                y_t=img                                                    # y_t: [bs x nvars x pred_len]
                pred_noise = self.diffussion_model(y_t, time_cond,enc_out)                    # dec_out: [bs x nvars x pred_len]                 # dec_out: [bs x nvars x pred_len]
                # pred_noise=(extract(sqrt_recip_alphas_cumprod, time_cond, img) * img - x_start) /extract(sqrt_recipm1_alphas_cumprod, time_cond, img)
                x_start= self.predict_start_from_noise(y_t,time_cond, pred_noise)
                # x_start=(extract(sqrt_recip_alphas_cumprod, time_cond, img) * img -extract(sqrt_recipm1_alphas_cumprod, time_cond, img) * pred_noise)
                
                if time_next < 0:
                    img = x_start
                    # y_p_seq.append(img.detach().cpu()+enc_out.detach().cpu())
                    continue

                alpha = alphas_cumprod[time]
                alpha_next = alphas_cumprod[time_next]
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(img)
                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise
                # y_p_seq.append(img.detach().cpu()+enc_out.detach().cpu())

            return img

    def p_sample(self, enc_out,  y_t,  t):
        """
        Reverse diffusion process sampling -- one time step.

        y: sampled y at time step t, y_t.
        y_0_hat: prediction of pre-trained guidance model.
        y_T_mean: mean of prior distribution at timestep T.
        We replace y_0_hat with y_T_mean in the forward process posterior mean computation, emphasizing that 
            guidance model prediction y_0_hat = f_phi(x) is part of the input to eps_theta network, while 
            in paper we also choose to set the prior mean at timestep T y_T_mean = f_phi(x).
        """
        device = enc_out.device
        batch= y_t.shape[0]
        #预测x_0
        # posterior_mean_coef1=self.posterior_mean_coef1.to(device)
        # posterior_log_variance_clipped=self.posterior_log_variance_clipped.to(device)
        # posterior_mean_coef2=self.posterior_mean_coef2.to(device)
        #预测噪声
        sqrt_one_minus_betas_sqrt=self.sqrt_one_minus_betas_sqrt.to(device)
        posterior_log_variance_clipped=self.posterior_log_variance_clipped.to(device)
        posterior_mean_coef3=self.posterior_mean_coef3.to(device)

        z = torch.randn_like(y_t).to(device)   if t > 0 else torch.zeros_like(y_t)

        # y_t_m_1 posterior mean component coefficients
        time_cond = torch.full((batch,), t,device=device, dtype=torch.long)

        pred_noise = self.diffussion_model(y_t, time_cond,enc_out)                    # dec_out: [bs x nvars x pred_len]
        # y_start = self.diffussion_model(enc_out, y_t, time_cond) 
     
        # posterior mean
        #预测噪声
        posterior_mean = (
                (extract(sqrt_one_minus_betas_sqrt, time_cond, y_t)) * y_t  -
                extract(posterior_mean_coef3, time_cond, y_t) * pred_noise
        ).to(device)

        #预测x_0
        # posterior_mean = (
        #         extract(posterior_mean_coef1, time_cond, y_t) * y_start +
        #         extract(posterior_mean_coef2, time_cond, y_t) * y_t
        # ).to(device)

        # posterior variance
        posterior_log_variance_clipped = extract(posterior_log_variance_clipped, time_cond, y_t).to(device)
        #reparameterization
        y_t_m_1 = posterior_mean + (0.5 * posterior_log_variance_clipped).exp()* z
        return y_t_m_1


    # Reverse function -- sample y_0 given y_1

    def p_sample_loop(self, y_T_mean, enc_out,  y_shape):
        device=enc_out.device
        z = torch.randn(*y_shape).to(device)

        cur_y = z   # sample y_T
        # y_p_seq = [cur_y.detach().cpu()+y_T_mean.detach().cpu()]
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            y_t = cur_y
            cur_y = self.p_sample( enc_out, y_t, t)  # y_{t-1}
            # y_p_seq.append(cur_y.detach().cpu()+y_T_mean.detach().cpu())
        # assert len(y_p_seq) == self.num_timesteps+1

        return cur_y

    def forward(self,  y_t, t, enc_out):
        #enc_out:[bs x seq_len x nvars]  , y_t:[bs x pred_len x nvars]  , t:[bs ]  

        dec_out = self.diffussion_model( y_t, t, enc_out)                    # dec_out: [bs x nvars x pred_len]
        return dec_out

