# Copyright (c) 2023-present, Royal Bank of Canada.
# Copyright (c) 2022, Tung Nguyen
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the TNP (https://arxiv.org/abs/2201.12740) implementation
# from https://github.com/tung-nd/TNP-pytorch by Tung Nguyen 
####################################################################################

import torch
import torch.nn as nn

from models.modules import build_mlp

from torch import nn
import torch.nn.functional as F
from attrdict import AttrDict

from torch.distributions.normal import Normal

from models.lbanp_modules import LBANPEncoderLayer, LBANPEncoder, NPDecoderLayer, NPDecoder
from inference.ar_custom import ar_log_likelihood, no_cheat_ar_log_likelihood
from data.gp import RBFKernel

class PerfectGP(nn.Module):
    """
     Perfect GP model which cheats and uses ground truth covariance matrix
    """
    def __init__(
        self,
        kernel = RBFKernel()
    ):
        super(PerfectGP, self).__init__()
        self.kernel = kernel
    
    def predict(self, xc, yc, xt):
        # expects xc, yc, xt to be of shape (batch_size, num_points, dim_x or dim_y)
        # dim_x and dim_y must both be 1
        assert xc.shape[2] == 1 and yc.shape[2] == 1 and xt.shape[2] == 1
        mean = torch.zeros(xc.shape[0], xt.shape[1], device=xt.device)
        cov = self.kernel(xt)

        return [torch.distributions.MultivariateNormal(mean[i], cov[i]) for i in range(xc.shape[0])]


    def forward(self, batch, reduce_ll=True, ):
        outs = AttrDict() 
        pred_targets = self.predict(batch.xc, batch.yc, batch.xt)

        if reduce_ll:
            outs.tar_ll = torch.stack([pred_target.log_prob(batch.yt[i,:,0]).sum(-1).mean() for i, pred_target in enumerate(pred_targets)])
        else:
            outs.tar_ll = torch.stack([pred_target.log_prob(batch.yt[i,:,0]).sum(-1) for i, pred_target in enumerate(pred_targets)])
        outs.loss = torch.stack([- (outs.tar_ll[i]) for i in range(len(outs.tar_ll))])

        return outs
