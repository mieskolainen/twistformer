# Lipschitz MLP
#
# https://arxiv.org/abs/2202.08345
#
# mikael.mieskolainen@cern.ch (2026)

import torch
import torch.nn as nn
import math
import numpy as np

from .dmlp import *


class LipschitzLinear(torch.nn.Module):
    """ Lipschitz linear layer
    """
    def __init__(self, in_features, out_features):

        super(LipschitzLinear, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.weight       = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias         = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c            = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus     = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        # Lipschitz constant of initial weight
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data.fill_(float(torch.log(torch.expm1(W_abs_row_sum.max()))))

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc  = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1).clamp_min(torch.finfo(self.weight.dtype).tiny)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)


class LZMLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, mlp_dim = [128, 64], activation='relu',
                 layer_norm=False, batch_norm=False, dropout=0.0, last_tanh=False,
                 last_tanh_scale=10.0, act_after_norm=True, **kwargs):
        """
        Lipschitz MLP
        """
        super(LZMLP, self).__init__()

        layers = []

        ## First layer
        layers.append(LipschitzLinear(in_dim, mlp_dim[0]))

        if not act_after_norm:
            layers.append(get_act(activation))

        if layer_norm:  layers.append(nn.LayerNorm(mlp_dim[0]))
        if batch_norm:  layers.append(nn.BatchNorm1d(mlp_dim[0]))
        if dropout > 0: layers.append(nn.Dropout(dropout, inplace=False))

        if act_after_norm:
            layers.append(get_act(activation))

        ## Hidden layers
        if len(mlp_dim) >= 2:

            for i in range(len(mlp_dim)-1):
                layers.append(LipschitzLinear(mlp_dim[i], mlp_dim[i+1]))

                if not act_after_norm:
                    layers.append(get_act(activation))

                if layer_norm:  layers.append(nn.LayerNorm(mlp_dim[i+1]))
                if batch_norm:  layers.append(nn.BatchNorm1d(mlp_dim[i+1]))
                if dropout > 0: layers.append(nn.Dropout(dropout, inplace=False))

                if act_after_norm:
                    layers.append(get_act(activation))

        ## Output layer
        layers.append(LipschitzLinear(mlp_dim[-1], out_dim))

        self.mlp = nn.Sequential(*layers)

        # Add extra final squeezing activation and post-scale aka "soft clipping"
        if last_tanh:
            self.mlp.add_module("tanh",  nn.Tanh())
            self.mlp.add_module("scale", Multiply(last_tanh_scale))

    def get_lipschitz_loss(self):
        """ Lipschitz regularization loss
        """
        lc = 1.0
        for i in range(len(self.mlp)):
            if type(self.mlp[i]) is LipschitzLinear:
                lc = lc * self.mlp[i].get_lipschitz_constant()
        return lc

    def forward(self, x):
        return self.mlp(x)
