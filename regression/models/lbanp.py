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

class LBANP(nn.Module):
    """
        Latent Bottlenecked Attentive Neural Process (LBANPs)
    """
    def __init__(
        self,
        num_latents,
        dim_x,
        dim_y,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        norm_first=True,
        bound_std = False
    ):
        super(LBANP, self).__init__()


        self.latent_dim = d_model
        self.latents = nn.Parameter(torch.randn(num_latents, self.latent_dim), requires_grad=True) # Learnable latents! 

        # Context Related
        self.embedder = build_mlp(dim_x + dim_y, d_model, d_model, emb_depth)

        encoder_layer = LBANPEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=norm_first) 
        self.encoder = LBANPEncoder(encoder_layer, num_layers)


        # Query Related
        self.query_embedder = build_mlp(dim_x, d_model, d_model, emb_depth)

        decoder_layer = NPDecoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=norm_first)
        self.decoder = NPDecoder(decoder_layer, num_layers)

        # Predictor Related
        self.bound_std = bound_std

        self.norm_first = norm_first
        if self.norm_first:
            self.norm = nn.LayerNorm(d_model)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y*2)
        )

    

    def get_context_encoding(self, batch):
        # Perform Encoding
        x_y_ctx = torch.cat((batch.xc, batch.yc), dim=-1)
        context_embeddings = self.embedder(x_y_ctx)
        context_encodings = self.encoder(context_embeddings, self.latents)
        return context_encodings

    def get_predict_encoding(self, batch, context_encodings=None):

        if context_encodings is None:
            context_encodings = self.get_context_encoding(batch)

        # Perform Decoding
        query_embeddings = self.query_embedder(batch.xt)
        encoding = self.decoder(query_embeddings, context_encodings)
        # Make predictions
        if self.norm_first:
            encoding = self.norm(encoding)
        return encoding

    def predict(self, xc, yc, xt, context_encodings=None, num_samples=None):
        # expects xc, yc, xt to be of shape (batch_size, num_points, dim_x or dim_y)
        batch = AttrDict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt

        encoding = self.get_predict_encoding(batch, context_encodings=context_encodings)

        out = self.predictor(encoding)

        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        return Normal(mean, std)


    def forward(self, batch, num_samples=None, reduce_ll=True, use_ar=False):
        outs = AttrDict()
        if use_ar:
            ll = no_cheat_ar_log_likelihood(self, batch.xc, batch.yc, batch.xt, batch.yt, num_samples=50, seed=None, smooth=False, covariance_est="bayesian", batch_size_targets=4, nu_p=50)
            if reduce_ll:
                outs.tar_ll = ll.mean()
            else:
                outs.tar_ll = ll
            outs.loss = - (outs.tar_ll)
        else:   
            pred_tar = self.predict(batch.xc, batch.yc, batch.xt)

            if reduce_ll:
                outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1).mean()
            else:
                outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1)
            outs.loss = - (outs.tar_ll)

        return outs
