# Copyright (c) 2023-present, Royal Bank of Canada.
# Copyright (c) 2022, Tung Nguyen
# Copyright (c) 2021, Phil Wang
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the TNP (https://arxiv.org/abs/2201.12740) implementation
# from https://github.com/tung-nd/TNP-pytorch by Tung Nguyen 
# and the Perceiver (https://arxiv.org/abs/2103.03206) implementation
# from https://github.com/lucidrains/Perceiver-pytorch by Phil Wang
####################################################################################

import torch
import torch.nn as nn

from torch import nn, einsum
from torch.nn import ModuleList
import copy
from einops import rearrange, repeat
import torch.nn.functional as F

def default(val, d):
    return val if exists(val) else d

def exists(val):
    return val is not None


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def reset_last(self):
        self.fn.reset_last()

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs) + x

class PostNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    
    def reset_last(self):
        self.fn.reset_last()

    def forward(self, x, **kwargs):
        x = x + self.fn(x, **kwargs)

        return self.norm(x)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_feedforward=128, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim_feedforward * 2),
            GEGLU(),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class CachedFeedForward(nn.Module):
    def __init__(self, dim, dim_feedforward=128, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim_feedforward * 2),
            GEGLU(),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout)
        )
        self.last_out = None
    
    def reset_last(self):
        self.last_out = None

    def forward(self, x, rollout=False):
        if rollout == False:
            out = self.net(x)
            self.last_out = out.clone()
            return out
        else:
            # only execute net on the last latent dimension for encoder rollout, last target embedding for decoder rollout
            last = self.net(x[:, -1:, :])
            out = self.last_out
            out[:, -1:, :] = last
            return out
        # return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        # mask = torch.ones((x.size(0)*h, q.size(1), k.size(1)), device=x.device, dtype=torch.bool)
        # last latent dimension only attented to by itself
        # mask[:,:,-1] = False
        # mask[:,-1,-1] = True

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class LBANPEncoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0,
                 norm_first: bool = True):
        super(LBANPEncoderLayer, self).__init__()
        self.latent_dim = d_model
        self.d_model = d_model

        assert (self.latent_dim % nhead == 0)


        if norm_first:
            self.latent_self_attn = PreNorm(self.latent_dim, Attention(self.latent_dim, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout))
            self.latent_ff = PreNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))
            # Self Attention performs the linear operations
            self.cross_attn = PreNorm(self.latent_dim, Attention(self.latent_dim, context_dim=self.d_model, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout), context_dim = self.d_model)
            self.cross_ff = PreNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))
        else:
            self.latent_self_attn = PostNorm(self.latent_dim, Attention(self.latent_dim, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout))
            self.latent_ff = PostNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))
            # Self Attention performs the linear operations
            self.cross_attn = PostNorm(self.latent_dim, Attention(self.latent_dim, context_dim=self.d_model, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout), context_dim = self.d_model)
            self.cross_ff = PostNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))

    def forward(self, context_encodings, latents):
        x = latents
        x = self.cross_attn(x, context = context_encodings)
        x = self.cross_ff(x)

        x = self.latent_self_attn(x)
        x = self.latent_ff(x)

        return x

class CausalAttention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.last_k = None
        self.last_v = None
        self.last_qkv_out = None
        self.last_qkv_before_out = None
        self.last_q = None
        self.last_sim = None
        self.rollout_num = None

    def reset_last(self):
        self.last_qkv_out = None
        self.last_q = None
        self.last_k = None
        self.last_v = None
        self.last_qkv_before_out = None
        self.last_sim = None
        self.rollout_num = None

    def forward(self, x, context = None, rollout=False, debug=False):  
        # for the encoder, x is just the latents/representation from previous layer, of dimension (b, l, H)

        # for the decoder, x is the query, of dimension (b, N, H) where N is the number of targets
        # we jsut use normal attention for this, so we don't use this class

        h = self.heads
        context = default(context, x)

        if rollout == False:
            q = self.to_q(x) # q shape: (b, l, H) H is dim_head

            k, v = self.to_kv(context).chunk(2, dim = -1) # k, v shape: (b, N, H) where N is number of context points

            # rearrange so that multiplication happens seperately per head
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

            # if debug and self.last_q is not None:
            #     print("prev last_q", self.last_q.shape, self.last_q[:,:,0])
            #     print("now q", q.shape, q[:,:,0])
            #     print("processed last q", rearrange(self.to_q(x[:,-1:,:]), 'b n (h d) -> (b h) n d', h = h)[:,:,0])
            
            if self.last_q is not None and debug:
                print("last q", self.last_q.shape, self.last_q[:,:,0])
                print("now q", q.shape, q[:,:,0])

            # store for future rollout use
            self.last_q = q.clone()
            self.last_k = k.clone()
            self.last_v = v.clone()
            

            # train with mask
            mask = torch.ones((x.size(0)*h, q.size(1), k.size(1)), device=x.device, dtype=torch.bool)
            #mask = torch.tril(mask, diagonal=k.size(1)-q.size(1))
            for i in range(q.size(1)):
                for j in range(k.size(1)):
                    mask[:,i,j] = i-j % q.size(1) == 0

            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale # shape: (b, l, H)

            
            
            if exists(mask):
                # mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                #mask = repeat(mask, 'b j -> (b h) () j', h = h)
                sim = sim.masked_fill_(~mask, max_neg_value)
            
            self.last_sim = sim.clone()

            # attention, what we cannot get enough of
            attn = sim.softmax(dim = -1)
            attn = self.dropout(attn)

            out = einsum('b i j, b j d -> b i d', attn, v)
            
            # rearrange back to the original shape for mlp layer
            out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
            if debug and self.last_qkv_before_out is not None:
                print("prev last_qkv_before_out", self.last_qkv_before_out.shape, self.last_qkv_before_out[:,:,0])
                print("now qkv before out", out.shape, out[:,:,0])
            self.last_qkv_before_out = out.clone()
            


            out = self.to_out(out)
            
            
            self.last_qkv_out = out.clone()
        else:
            # context should be not None and of shape (b, N, H)
            # Optimize by only computing for the last latent dimension
            
            # Calculate new query for one latent dimension
            latent_i = (context.size(1)-1) % x.size(1)
            if debug:
                print("latent_i", latent_i)
            new_q_vec = self.to_q(x[:,latent_i:latent_i+1,:])  # shape: (b, 1, H)
            
            # Calculate new key and value for last context dimension
            
            new_k_vec, new_v_vec = self.to_kv(context[:,-1:,:]).chunk(2, dim=-1)
            
            # Rearrange tensors for multi-head attention
            new_q_vec, new_k_vec, new_v_vec = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (new_q_vec, new_k_vec, new_v_vec))
            if debug:
                print("q_vec rollout", new_q_vec.shape, new_q_vec[:,:,0])
            # new_q_vec shape: (b*h, 1, d)
            # full q before would be (b*h, L, d)
            # new_k_vec shape: (b*h, 1, d)
            # new_v_vec shape: (b*h, 1, d)
            
            # Concatenate with previous keys and values
            appended_k = torch.cat((self.last_k.clone(), new_k_vec.clone()), dim=1)  # shape: (b*h, N+1, d)
            appended_v = torch.cat((self.last_v.clone(), new_v_vec.clone()), dim=1)  # shape: (b*h, N+1, d)

            
            # caching kv
            self.last_k = appended_k.clone()
            self.last_v = appended_v.clone()

            
            # Compute attention scores
            attn = einsum('b i d, b j d -> b i j', new_q_vec, appended_k) * self.scale  # shape: (b*h, 1, N+1)
            

            mask = torch.ones((x.size(0)*h, 1, context.size(1)), device=x.device, dtype=torch.bool)
            #mask = torch.tril(mask, diagonal=k.size(1)-q.size(1))
            for j in range(context.size(1)):
                mask[:,0,j] = j % x.size(1) == 0

            if exists(mask):
                # mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(attn.dtype).max
                #mask = repeat(mask, 'b j -> (b h) () j', h = h)
                if debug:
                    print("mask shape", mask.shape)
                    print("attn shape", attn.shape)
                attn.masked_fill_(~mask, max_neg_value)
            
            # Apply softmax to get attention weights
            attn = attn.softmax(dim=-1)
            
            # Apply attention weights to values
            new_qkv = einsum('b i j, b j d -> b i d', attn, appended_v)  # shape: (b*h, 1, d)
            
            # Rearrange back to original shape
            new_qkv = rearrange(new_qkv, '(b h) n d -> b n (h d)', h=h)

            if debug and self.last_qkv_before_out is not None:
                print("prev last_qkv_before_out", self.last_qkv_before_out.shape, self.last_qkv_before_out[:,:,0])
                print("now qkv before out", new_qkv.shape, new_qkv[:,:,0])
            
            # Apply output projection
            new_qkv = self.to_out(new_qkv)
            
            
            # Update only the last position in the output
            out = self.last_qkv_out.clone()
            out[:,latent_i:latent_i+1,:] = new_qkv.clone()
            self.last_qkv_out = out.clone()

        return out


class LBANPCausalEncoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0,
                 norm_first: bool = True):
        super(LBANPCausalEncoderLayer, self).__init__()
        self.latent_dim = d_model
        self.d_model = d_model

        assert (self.latent_dim % nhead == 0)


        if norm_first:
            self.latent_self_attn = PreNorm(self.latent_dim, CausalAttention(self.latent_dim, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout))
            self.latent_ff = PreNorm(self.latent_dim, CachedFeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))
            # Self Attention performs the linear operations
            self.cross_attn = PreNorm(self.latent_dim, CausalAttention(self.latent_dim, context_dim=self.d_model, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout), context_dim = self.d_model)
            self.cross_ff = PreNorm(self.latent_dim, CachedFeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))
        else:
            self.latent_self_attn = PostNorm(self.latent_dim, CausalAttention(self.latent_dim, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout))
            self.latent_ff = PostNorm(self.latent_dim, CachedFeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))
            # Self Attention performs the linear operations
            self.cross_attn = PostNorm(self.latent_dim, CausalAttention(self.latent_dim, context_dim=self.d_model, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout), context_dim = self.d_model)
            self.cross_ff = PostNorm(self.latent_dim, CachedFeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))

    def forward(self, context_encodings, latents, rollout=False, debug=False):
        x = latents
        x = self.cross_attn(x, context = context_encodings, rollout=rollout, debug=debug)
        x = self.cross_ff(x, rollout=rollout)

        x = self.latent_self_attn(x, rollout=rollout)
        x = self.latent_ff(x, rollout=rollout)

        return x

class LBANPCausalDecoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0,
                 norm_first: bool = True):
        super(LBANPCausalDecoderLayer, self).__init__()
        self.latent_dim = d_model
        self.d_model = d_model

        assert (self.latent_dim % nhead == 0)
        # Self Attention performs  the linear operations
        if norm_first:
            self.cross_attn = PreNorm(self.latent_dim, Attention(self.latent_dim, self.d_model, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout), context_dim = self.latent_dim)
            self.cross_ff = PreNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))
        else:
            self.cross_attn = PostNorm(self.latent_dim, Attention(self.latent_dim, self.d_model, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout), context_dim = self.latent_dim)
            self.cross_ff = PostNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))


    def forward(self, query_encodings, context, rollout=False):
        
        x = query_encodings
        x = self.cross_attn(x, context = context)
        x = self.cross_ff(x)

        return x



class LBANPEncoder(nn.Module):
    """
        Iterative Attention-based model that encodes context datapoints into a list of embeddings
    """
    def __init__(self, encoder_layer, num_layers, return_only_last=False):
        super(LBANPEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_only_last = return_only_last

    def forward(self, context_encodings, latents):
        b, *axis = context_encodings.shape
        latents = repeat(latents, 'n d -> b n d', b = b)

        layer_outputs = []
        last_layer_output = None
        for layer in self.layers:
            latents = layer(context_encodings, latents)
            layer_outputs.append(latents)
            last_layer_output = latents
        if self.return_only_last:
            return [last_layer_output]
        else:
            return layer_outputs
        
class CausalLBANPEncoder(nn.Module):
    """
        Iterative Attention-based model that encodes context datapoints into a list of embeddings
    """
    def __init__(self, encoder_layer, num_layers, return_only_last=False):
        super(CausalLBANPEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_only_last = return_only_last

    def forward(self, context_encodings, latents, rollout=False):
        b, *axis = context_encodings.shape
        latents = repeat(latents, 'n d -> b n d', b = b)

        layer_outputs = []
        last_layer_output = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                debug = True
            else:
                debug = False   
            latents = layer(context_encodings, latents, rollout=rollout, debug=debug)
            layer_outputs.append(latents)
            last_layer_output = latents
        if self.return_only_last:
            return [last_layer_output]
        else:
            return layer_outputs



class NPDecoderLayer(nn.Module):
    def __init__(self, 
                 d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0,
                 norm_first: bool = True):
        super(NPDecoderLayer, self).__init__()
        self.latent_dim = d_model
        self.d_model = d_model

        assert (self.latent_dim % nhead == 0)
        # Self Attention performs  the linear operations
        if norm_first:
            self.cross_attn = PreNorm(self.latent_dim, Attention(self.latent_dim, self.d_model, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout), context_dim = self.latent_dim)
            self.cross_ff = PreNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))
        else:
            self.cross_attn = PostNorm(self.latent_dim, Attention(self.latent_dim, self.d_model, heads = nhead, dim_head = self.latent_dim // nhead, dropout = dropout), context_dim = self.latent_dim)
            self.cross_ff = PostNorm(self.latent_dim, FeedForward(self.latent_dim, dim_feedforward=dim_feedforward, dropout = dropout))


    def forward(self, query_encodings, context):
        
        x = query_encodings
        x = self.cross_attn(x, context = context)
        x = self.cross_ff(x)

        return x

class NPDecoder(nn.Module):
    """
        Attention-based model that retrieves information via the context encodings to make predictions for the query/target datapoints
    """
    def __init__(self, decoder_layer, num_layers):
        super(NPDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, query_encodings, context_encodings):
        # assert len(context_encodings) == self.num_layers

        x = query_encodings
        for layer, context_enc in zip(self.layers, context_encodings):
            x = layer(x, context=context_enc)

        out = x
        return out

class CausalNPDecoder(nn.Module):
    """
        Attention-based model that retrieves information via the context encodings to make predictions for the query/target datapoints
    """
    def __init__(self, decoder_layer, num_layers):
        super(CausalNPDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, query_encodings, context_encodings, rollout=False):
        # assert len(context_encodings) == self.num_layers

        x = query_encodings
        for layer, context_enc in zip(self.layers, context_encodings):
            x = layer(x, context=context_enc, rollout=rollout)

        out = x
        return out



class TransformerEncoder(nn.Module):
    """
        Transformer-based model that encodes context datapoints into a list of embeddings
    """
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        output = src
        layer_outputs = [output]
        for layer in self.layers:
            output = layer(output)
            layer_outputs.append(output)
        return layer_outputs




class TransformerEncoderLayer(nn.Module):
    """
        Typical Transformer layer
    """
    def __init__(self, 
                 d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.0, 
                 norm_first: bool = True):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model

        assert (self.d_model % nhead == 0)

        if norm_first:
            self.dataset_self_attn = PreNorm(self.d_model, Attention(self.d_model, heads = nhead, dim_head = self.d_model // nhead, dropout = dropout))
            self.ff = PreNorm(self.d_model, FeedForward(self.d_model, dim_feedforward=dim_feedforward, dropout = dropout))
        else:
            self.dataset_self_attn = PostNorm(self.d_model, Attention(self.d_model, heads = nhead, dim_head = self.d_model // nhead, dropout = dropout))
            self.ff = PostNorm(self.d_model, FeedForward(self.d_model, dim_feedforward=dim_feedforward, dropout = dropout))

    def forward(self, context):
        x = context
        x = self.dataset_self_attn(x)
        x = self.ff(x)
        return x