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

from einops import rearrange, repeat

from torch.distributions.normal import Normal

from models.lbanp_modules import LBANPCausalEncoderLayer, CausalLBANPEncoder, LBANPCausalDecoderLayer, CausalNPDecoder


class LBANP_AR(nn.Module):
    """
        Latent Bottlenecked Attentive Neural Process (LBANPs), that supports efficent AR rollouts
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
        super(LBANP_AR, self).__init__()


        self.latent_dim = d_model
        self.latents = nn.Parameter(torch.randn(num_latents, self.latent_dim), requires_grad=True) # Learnable latents! 

        # Context Related
        self.embedder = build_mlp(dim_x + dim_y, d_model, d_model, emb_depth)

        encoder_layer = LBANPCausalEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=norm_first) 
        self.encoder = CausalLBANPEncoder(encoder_layer, num_layers)


        # Query Related
        self.query_embedder = build_mlp(dim_x, d_model, d_model, emb_depth)

        decoder_layer = LBANPCausalDecoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first=norm_first)
        self.decoder = CausalNPDecoder(decoder_layer, num_layers)

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

    def reset_stored_qkv(self):
        for layer in self.encoder.layers:
            layer.cross_attn.reset_last()
            layer.latent_self_attn.reset_last()
            layer.cross_ff.reset_last()
            layer.latent_ff.reset_last()
        

    def get_context_encoding(self, batch, rollout=False):
        # Perform Encoding
        x_y_ctx = torch.cat((batch.xc, batch.yc), dim=-1)
        context_embeddings = self.embedder(x_y_ctx)
        context_encodings = self.encoder(context_embeddings, self.latents, rollout=rollout)
        return context_encodings

    def get_predict_encoding(self, batch, context_encodings=None, rollout=False):

        if context_encodings is None:
            context_encodings = self.get_context_encoding(batch, rollout=rollout)
        # Perform Decoding
        query_embeddings = self.query_embedder(batch.xt)
        encoding = self.decoder(query_embeddings, context_encodings, rollout=rollout)
        # Make predictions
        if self.norm_first:
            encoding = self.norm(encoding)
        return encoding


    def sample_ar(self, xc, yc, xt, context_encodings=None, num_samples=20, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        else:
            torch.manual_seed(42)
        batch = AttrDict()
        batch.xc = repeat(xc, 'b j d -> (s b) j d', s=num_samples) # allow for multiple sample generation
        batch.yc = repeat(yc, 'b j d -> (s b) j d', s=num_samples) # allow for multiple sample generation
        xt_drawn = repeat(xt, 'b j d -> (s b) j d', s=num_samples)
        # store original sizes of xc
        tgt_from = xc.shape[1]
        pred_dists = []
        # first run a forward pass with the first target point in xt without rollout, then the following points with rollout
        for i in range(xt.shape[1]):
            if i == 0:
                rollout = False
            else:
                rollout = True 
            batch.xt =  xt_drawn[:, i:i+1, :].clone()
            encoding = self.get_predict_encoding(batch, context_encodings=context_encodings, rollout=rollout)
            

            out = self.predictor(encoding)
            mean, std = torch.chunk(out, 2, dim=-1)
            if self.bound_std:
                std = 0.05 + 0.95 * F.softplus(std)
            else:
                std = torch.exp(std)
            pred_tar = Normal(mean, std)
            # now we need to sample from the distribution
            samples = pred_tar.sample()
            print("lbanp_ar", samples)
            # add new context
            batch.xc = torch.cat((batch.xc, batch.xt.clone()), dim=1)
            batch.yc = torch.cat((batch.yc, samples.clone()), dim=1)
            pred_dists.append(pred_tar)
        
        # xt_out = rearrange(batch.xc, '(s b) j d -> s b j d', s=num_samples)[:, :, tgt_from:, :]
        yt_out = rearrange(batch.yc, '(s b) j d -> s b j d', s=num_samples)[:, :, tgt_from:, :]
        # calculate empirical std and mean of yt_out across samples dimension
        yt_out_std = yt_out.std(dim=0)
        yt_out_mean = yt_out.mean(dim=0)

        # get the noiseless predictions
        # pred_dist = self.predict(batch.xc, batch.yc, xt_drawn.clone(), rollout=False)
        # yt_out_noiseless = rearrange(pred_dist.mean, '(s b) j d -> s b j d', s=num_samples)
        return yt_out_mean, yt_out_std, None, pred_dists

    def predict(self, xc, yc, xt, context_encodings=None, num_samples=None, rollout=False):
        batch = AttrDict()
        batch.xc = xc
        batch.yc = yc
        batch.xt = xt

        encoding = self.get_predict_encoding(batch, context_encodings=context_encodings, rollout=rollout)

        out = self.predictor(encoding)

        mean, std = torch.chunk(out, 2, dim=-1)
        if self.bound_std:
            std = 0.05 + 0.95 * F.softplus(std)
        else:
            std = torch.exp(std)

        return Normal(mean, std)


    def forward(self, batch, num_samples=None, reduce_ll=True):
  
        pred_tar = self.predict(batch.xc, batch.yc, batch.xt, rollout=False)

        outs = AttrDict()
        if reduce_ll:
            outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1).mean()
        else:
            outs.tar_ll = pred_tar.log_prob(batch.yt).sum(-1)
        outs.loss = - (outs.tar_ll)

        return outs
