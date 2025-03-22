# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
import argparse

# from model9_NS_transformer.ns_layers.RevIN import RevIN



class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        t=t.long()
        out = self.lin(x)
        gamma = self.embed(t)
        # out = gamma.view(-1, self.num_out) * out

        out = gamma.view(t.size()[0], -1, self.num_out) * out
        return out


class MLP(nn.Module):
    def __init__(self, config, MTS_args):
        super().__init__()
        n_steps = config.diffusion.timesteps + 1
        data_dim = MTS_args.enc_in*2
        self.lin0 = nn.Linear(12 * MTS_args.d_model_c, MTS_args.pred_len)
        self.lin1 = ConditionalLinear(data_dim, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, MTS_args.enc_in)


    def forward(self, y_t ,t,enc_out):
        #enc_out_tile:[100*batch, nvar, patch_num, d_model_c
        batch, nvar, patch_num, d_model_c= enc_out.shape
        enc_out=enc_out.reshape(batch, nvar, patch_num * d_model_c)
        enc_out =  self.lin0(enc_out).permute(0,2,1)
        eps_pred = torch.cat((y_t, enc_out), dim=-1)
        eps_pred = F.softplus(self.lin1(eps_pred, t))
        eps_pred = F.softplus(self.lin2(eps_pred, t))
        eps_pred = F.softplus(self.lin3(eps_pred, t))
        eps_pred = self.lin4(eps_pred)
        return eps_pred


# deterministic feed forward neural network
class DeterministicFeedForwardNeuralNetwork(nn.Module):

    def __init__(self, dim_in, dim_out, hid_layers,
                 use_batchnorm=False, negative_slope=0.01, dropout_rate=0):
        super(DeterministicFeedForwardNeuralNetwork, self).__init__()
        self.dim_in = dim_in  # dimension of nn input
        self.dim_out = dim_out  # dimension of nn output
        self.hid_layers = hid_layers  # nn hidden layer architecture
        self.nn_layers = [self.dim_in] + self.hid_layers  # nn hidden layer architecture, except output layer
        self.use_batchnorm = use_batchnorm  # whether apply batch norm
        self.negative_slope = negative_slope  # negative slope for LeakyReLU
        self.dropout_rate = dropout_rate
        layers = self.create_nn_layers()
        self.network = nn.Sequential(*layers)

    def create_nn_layers(self):
        layers = []
        for idx in range(len(self.nn_layers) - 1):
            layers.append(nn.Linear(self.nn_layers[idx], self.nn_layers[idx + 1]))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.nn_layers[idx + 1]))
            layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
            layers.append(nn.Dropout(p=self.dropout_rate))
        layers.append(nn.Linear(self.nn_layers[-1], self.dim_out))
        return layers

    def forward(self, x):
        return self.network(x)



# if __name__ == '__main__':
#     diffussion_model = DiT(depth=1,in_channels=7, hidden_size=1152, patch_size=2, num_heads=8,learn_sigma=False)
#     x=torch.randn((10,7,32,32))
#     t=torch.randn((10))
#     y=torch.randn((10))
#     output=diffussion_model(x,t,y)
    