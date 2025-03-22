from data_provider.data_factory import data_provider

from utils.tools import EarlyStopping
from utils.metrics import metric
import gc
# from model9_NS_transformer.ns_models import ns_Transformer
from model9_NS_transformer.exp.exp_basic import Exp_Basic
from model9_NS_transformer.diffusion_models import diffuMTS
from model9_NS_transformer.diffusion_models.diffusion_utils import *
import torch.distributed as dist
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from model9_NS_transformer.samplers.dpm_sampler import DPMSolverSampler
import os
import time
from utils.metrics import  calc_quantile_CRPS_sum
from multiprocessing import Pool
import CRPS.CRPS as pscore
from layers.Decompose import moving_avg
import warnings
from layers.Decompose import series_decomp, FourierLayer

warnings.filterwarnings('ignore')


def ccc(id, pred, true):

    res_box = np.zeros(len(true))

    for i in range(len(true)):
        res = pscore(pred[i], true[i]).compute()
        res_box[i] = res[0]
    return res_box


def log_normal(x, mu, var):
    """Logarithm of normal distribution with mean=mu and variance=var
       log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

    Args:
       x: (array) corresponding array containing the input
       mu: (array) corresponding array containing the mean
       var: (array) corresponding array containing the variance

    Returns:
       output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
    """
    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    # return -0.5 * torch.sum(
    #     np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)
    return 0.5 * torch.mean(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)


def calculate_crps_sum_worker(args):
        pred, true = args
        p_in = np.sum(pred, axis=-1).T
        t_in = np.sum(true, axis=-1).reshape(-1)
        crps = ccc(8, p_in, t_in)
        return crps.mean()
        
def calculate_crps_worker(args):

        pred, true = args
        p_in = pred.transpose(1, 0, 2)
        t_in = true
        all_res = []
        for i in range(pred.shape[-1]):
            crps = ccc(8, p_in[:,:,i], t_in[:,i])
            all_res.append(crps)
        all_res= np.array(all_res)
        if isinstance(all_res, np.ndarray):
            return np.mean(all_res, axis=0).mean()
        else:
            return all_res


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.moving_avg = moving_avg(7, stride=1)
        self.decomp = series_decomp(kernel_size=15) 
        self.seasonal = FourierLayer(d_model=128, factor=1) 
        

    def _build_model(self):
        model = diffuMTS.Model(self.args).float()

        cond_pred_model =self.cond_model_dict[self.args.model].Model(self.args).float()
        condition_path=os.path.join(self.args.pretrain_checkpoints, self.args.model)
        if self.args.decomposition:
           best_condition_model_path = condition_path +'/' +str('decomposition') +'/' + self.args.data_name+'/' +str(self.args.pred_len)+ '/'+ 'checkpoint.pth'  # 指定模型检查点的路径
        else:
           best_condition_model_path = condition_path+'/' +str('all') +'/' + self.args.data_name+'/' +str(self.args.pred_len)+ '/'+ 'checkpoint.pth'  # 指定模型检查点的路径
        print(best_condition_model_path)

        if self.args.from_scrach==False:
            cond_pred_model.load_state_dict(torch.load(best_condition_model_path, map_location=self.device))   
            #
            #cond_pred_model
            if self.args.cond_pred_model_requires_grad:
                for param in cond_pred_model.parameters():
                        param.requires_grad = True
            else:
                for param in cond_pred_model.parameters():
                        param.requires_grad = False

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            cond_pred_model = nn.DataParallel(cond_pred_model, device_ids=self.args.device_ids)

        return model, cond_pred_model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, mode='Model'):
        if mode == 'Model':
            model_optim = optim.Adam([{'params': self.model.parameters()}, {'params': self.cond_pred_model.parameters()}], lr=self.args.learning_rate)
        else:
            model_optim = None
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.cond_pred_model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder

                n = batch_x.size(0)
                t = torch.randint(
                    low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]



                y_T_mean,_,enc_out= self.cond_pred_model(batch_x, None, dec_inp, None)  
                if self.args.use_pretraining_condition:
                    enc_out=y_T_mean

                #y_T_mean: [bs x  pred_len x nvars] enc_out: [ bsz x n_vars x patch_num  x d_model ]

                batch_y=batch_y[:, -self.args.pred_len:, :]
                if self.args.bias :
                    y_0=batch_y-y_T_mean #y_bias

                    if self.args.bias_y_0:
                        res, trend = self.decomp(batch_y) # trend/res : [bsz, seq_len, n_vars]
                        seasonal = self.seasonal(res)
                        y_0=res-seasonal
              
                else:
                    y_0=batch_y
                
                
                e = torch.randn_like(y_0).to(self.device)
                

                y_t = self.model.q_sample( y_0, t, noise=e)

                output= self.model( y_t, t,enc_out)


                f_dim = -1 if self.args.features == 'MS' else 0
                output = output[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                if self.args.parameterization == "noise":
                    target = e
                elif self.args.parameterization == "x_start":
                    target = y_0

                loss = criterion(output,target)

                loss = loss.detach().cpu()
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        condition_path=os.path.join(self.args.pretrain_checkpoints, self.args.data)

        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(condition_path):
            os.makedirs(condition_path)


        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)


        model_optim = self._select_optimizer()


        criterion = self._select_criterion()
           
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):

            if self.args.scrach_10_stop:
                if epoch==10:
                     for param in self.cond_pred_model.parameters():
                        param.requires_grad = False
            # Training the diffusion part
            epoch_time = time.time()

            iter_count = 0
            train_loss = []
            self.model.train()
            self.cond_pred_model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder

                n = batch_x.size(0)
                t = torch.randint(
                    low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)

                t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n] .to(self.device)             #t: [batch]

                y_T_mean,_,enc_out= self.cond_pred_model(batch_x, None, dec_inp, None)  
                if self.args.use_pretraining_condition:
                    enc_out=y_T_mean



                #y_T_mean: [bs x  pred_len x nvars] enc_out: [ bsz x n_vars x patch_num  x d_model ]

   
                batch_y=batch_y[:, -self.args.pred_len:, :]
                if self.args.bias :
                    y_0=batch_y-y_T_mean #y_bias
                    
                    if self.args.bias_y_0:
                        res, trend = self.decomp(batch_y) # trend/res : [bsz, seq_len, n_vars]
                        seasonal = self.seasonal(res)
                        y_0=res-seasonal

                    
                else:
                    y_0=batch_y
                
                
                e = torch.randn_like(y_0).to(self.device)
                

                y_t = self.model.q_sample( y_0, t, noise=e)

                output= self.model( y_t, t,enc_out)


                f_dim = -1 if self.args.features == 'MS' else 0
                output = output[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                if self.args.parameterization == "noise":
                    target = e
                elif self.args.parameterization == "x_start":
                    target = y_0
                loss = (target - output).square().mean() 

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                a = 0

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss= self.vali(vali_data, vali_loader, criterion)
            test_loss= self.vali(test_data, test_loader, criterion)


            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping( vali_loss, self.model, path)

            if (math.isnan(train_loss)):
                break

            if early_stopping.early_stop:
                print("Early stopping")
                break

        
        return self.model


    def test(self, setting, test=0):
        #####################################################################################################
        ########################## local functions within the class function scope ##########################

        def exact_y_0(config, config_diff, y_tile_seq):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
           
            y_0= y_tile_seq.reshape(-1,int(config_diff.testing.n_z_samples / config_diff.testing.n_z_samples_depart),
                                                config.pred_len,
                                                config.c_out)
            return y_0
        
        def store_gen_y_at_step_t(config, config_diff, y_tile_seq,y_true,x_true,fast_sample):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            if fast_sample:
                gen_y_by_batch_list = [[] for _ in range(self.model.sampling_timesteps + 1)]
                y_true_by_batch_list = [[] for _ in range(self.model.sampling_timesteps + 1)]
                x_true_by_batch_list = [[] for _ in range(self.model.sampling_timesteps + 1)]
                for current_t  in range(self.model.sampling_timesteps + 1):
                    gen_y = y_tile_seq[current_t].reshape(-1,
                                                    int(config_diff.testing.n_z_samples / config_diff.testing.n_z_samples_depart),
                                                    config.pred_len,
                                                    config.c_out).cpu().numpy()
                    #print('gen_y',gen_y.shape)(8, 10, 240, 7)
                    # directly modify the dict value by concat np.array instead of append np.array gen_y to list
                    # reduces a huge amount of memory consumption
                    true_y=y_true.cpu().numpy()
                    true_x=x_true.cpu().numpy()
                    if len(gen_y_by_batch_list[current_t]) == 0:
                        gen_y_by_batch_list[current_t] = gen_y
                        y_true_by_batch_list[current_t] = true_y
                        x_true_by_batch_list[current_t] = true_x
                    else:
                        gen_y_by_batch_list[current_t] = np.concatenate([gen_y_by_batch_list[current_t], gen_y], axis=0)
                        y_true_by_batch_list[current_t] = np.concatenate([y_true_by_batch_list[current_t], true_y], axis=0)
                        x_true_by_batch_list[current_t] = np.concatenate([x_true_by_batch_list[current_t], true_x], axis=0)
            else:
                gen_y_by_batch_list = [[] for _ in range(self.model.num_timesteps + 1)]
                y_true_by_batch_list = [[] for _ in range(self.model.num_timesteps + 1)]
                x_true_by_batch_list = [[] for _ in range(self.model.num_timesteps + 1)]

                for current_t  in range(self.model.num_timesteps + 1):
                    gen_y = y_tile_seq[current_t].reshape(-1,
                                                    int(config_diff.testing.n_z_samples / config_diff.testing.n_z_samples_depart),
                                                    config.pred_len,
                                                    config.c_out).cpu().numpy()
                    #print('gen_y',gen_y.shape)(8, 10, 240, 7)
                    # directly modify the dict value by concat np.array instead of append np.array gen_y to list
                    # reduces a huge amount of memory consumption
                    true_y=y_true.cpu().numpy()
                    true_x=x_true.cpu().numpy()
                    if len(gen_y_by_batch_list[current_t]) == 0:
                        gen_y_by_batch_list[current_t] = gen_y
                        y_true_by_batch_list[current_t] = true_y
                        x_true_by_batch_list[current_t] = true_x
                    else:
                        gen_y_by_batch_list[current_t] = np.concatenate([gen_y_by_batch_list[current_t], gen_y], axis=0)
                        y_true_by_batch_list[current_t] = np.concatenate([y_true_by_batch_list[current_t], true_y], axis=0)
                        x_true_by_batch_list[current_t] = np.concatenate([x_true_by_batch_list[current_t], true_x], axis=0)
            return  gen_y_by_batch_list,y_true_by_batch_list,x_true_by_batch_list

        
            """
            Another coverage metric.
            """
            low, high = config.testing.PICP_range
            CI_y_pred = np.percentile(all_gen_y.squeeze(), q=[low, high], axis=1)
            # compute percentage of true y in the range of credible interval
            y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
            coverage = y_in_range.mean()
            if return_CI:
                return coverage, CI_y_pred, low, high
            else:
                return coverage, low, high

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))
            condition_path=os.path.join(self.args.pretrain_checkpoints, self.args.model)
            if self.args.decomposition:
               best_condition_model_path = condition_path +'/' +str('decomposition') +'/' + self.args.data_name+'/' +str(self.args.pred_len)+ '/'+ 'checkpoint.pth'  # 指定模型检查点的路径
            else:
                best_condition_model_path = condition_path+'/' +str('all') +'/' + self.args.data_name+'/' +str(self.args.pred_len)+ '/'+ 'checkpoint.pth'  # 指定模型检查点的路径
       
            print(best_condition_model_path)
            
            self.cond_pred_model.load_state_dict(torch.load(best_condition_model_path, map_location=self.device))
            self.cond_pred_model = self.cond_pred_model.to(self.device)
        preds = []
        trues = []
        folder_path = '../test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        minibatch_sample_start = time.time()

        self.model.eval()
        self.cond_pred_model.eval()
        self.sampler = DPMSolverSampler(self.model,self.device,self.args.parameterization)
        total_mse=0.0
        total_mae=0.0
        total_samples=0.0
        sum_crps = 0.0
        sum_crps_sum =0.0 
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
 
                y_T_mean,_,enc_out= self.cond_pred_model(batch_x, None, dec_inp, None)  

                #y_T_mean: [bs x  pred_len x nvars] enc_out:  [ bsz x n_vars x patch_num  x d_model ]
                y_true_bias=batch_y[:, -self.args.pred_len:, :]-y_T_mean

                gen_y_box = []
                gen_y_bias_box=[]
                for _ in range(self.model.diffusion_config.testing.n_z_samples_depart):

                    # sample
                    repeat_n = int(
                        self.model.diffusion_config.testing.n_z_samples / self.model.diffusion_config.testing.n_z_samples_depart)
                    x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
                    x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)                           
                    #x_tile: [bs*sample, seq_len, var]
                    
                    enc_out_tile = enc_out.repeat(repeat_n, 1, 1, 1, 1)
                    enc_out_tile = enc_out_tile.transpose(0, 1).flatten(0, 1).to(self.device)             #enc_out_tile:[100*batch, nvar, patch_num, d_model_c]


                    y_T_mean_tile = y_T_mean.repeat(repeat_n, 1, 1, 1)
                    y_T_mean_tile = y_T_mean_tile.transpose(0, 1).flatten(0, 1).to(self.device) 

                    y_shape=(x_tile.shape[0],self.args.pred_len,x_tile.shape[-1])                                    #y_shape: [bs*sample, pred_len, var]
                    
                    if self.args.use_pretraining_condition:
                            enc_out_tile=y_T_mean_tile


                    #
                    if self.args.type_sampler == "none":
                        y_tile_seq = self.model.p_sample_loop(y_T_mean_tile,enc_out_tile, y_shape)#(如果预测x_0重新更改)
                    elif self.args.type_sampler == "DDIM":
                        y_tile_seq =self.model.fast_sample(y_T_mean_tile,enc_out_tile, y_shape,self.args.eta)#(如果预测x_0重新更改)
                    elif self.args.type_sampler == "DPM_solver":
                        
                        y_tile_seq = self.sampler.sample(S=self.args.DPMsolver_step,
                                        conditioning=enc_out_tile,
                                        shape=y_shape,
                                        verbose=False,
                                        unconditional_guidance_scale=1.0,
                                        unconditional_conditioning=None,
                                        eta=0.,
                                        x_T=None)
                    
                    y_tile_bias=y_tile_seq
                    if self.args.bias :
                    
                        y_tile_seq=y_tile_seq+y_T_mean_tile
                    else:
                        y_tile_seq=y_tile_seq


                    
                    gen_y_bias = exact_y_0(config=self.model.args,\
                                                    config_diff=self.model.diffusion_config,\
                                                    y_tile_seq=y_tile_bias)        #gen_y: [bs,sample, pred_len, var]
                    gen_y = exact_y_0(config=self.model.args,\
                                                    config_diff=self.model.diffusion_config,\
                                                    y_tile_seq=y_tile_seq)        #gen_y: [bs,sample, pred_len, var]

                    gen_y_bias_box.append(gen_y_bias.cpu().numpy())
                    gen_y_box.append(gen_y.cpu().numpy())
                outputs = np.concatenate(gen_y_box, axis=1)
                outputs_bias= np.concatenate(gen_y_bias_box, axis=1)


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                batch_crps = self.calculate_batch_crps(pred, true)
                sum_crps += batch_crps


                pred_ns =np.mean(pred,axis=1)
                print('test2 shape:', pred_ns.shape, true.shape)
                mae, mse, rmse, mape, mspe = metric(pred_ns, true)
                print('mae_mse',mae, mse)
                total_mse += mse * pred_ns.shape[0]
                total_mae += mae * pred_ns.shape[0]
                total_samples += pred_ns.shape[0]

                preds.append(pred.sum(-1))                     
                trues.append(true.sum(-1))
                del outputs 
                gc.collect() 

                if i % 1 == 0 and i != 0:
                    print('Testing: %d/%d cost time: %f min' % (
                        i, len(test_loader), (time.time() - minibatch_sample_start) / 60))
                    minibatch_sample_start = time.time()


        print('total_samples',total_samples)
        avg_crps = sum_crps / total_samples
        mse_total = total_mse / total_samples
        mae_total = total_mae / total_samples
        print('NT metrc: CRPS:{:.4f}'.format(avg_crps))
        print('NT metrc: mse:{:.4f}, mae:{:.4f} '.format(mse_total, mae_total))
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds_save = np.array(preds)
        trues_save = np.array(trues)
        crps_sum=calc_quantile_CRPS_sum(preds,trues)
        print('NT metrc: CRPS_sum:{:.4f}'.format(crps_sum))


        np.save(folder_path + 'pred.npy', preds_save)
        np.save(folder_path + 'true.npy', trues_save)
       