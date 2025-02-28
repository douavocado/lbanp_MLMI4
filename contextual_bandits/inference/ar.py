import numpy as np
import torch
from typing import List, Tuple, Optional, Callable
from attrdict import AttrDict
from torch.distributions.normal import Normal

def inv_perm(perm: torch.Tensor) -> torch.Tensor:
    """Compute inverse permutation."""
    inv_perm = torch.zeros_like(perm)
    inv_perm[perm] = torch.arange(len(perm))
    return inv_perm


def _determine_order(
    generator: torch.Generator,
    xt: List[torch.Tensor],
    yt: Optional[List[torch.Tensor]],
    order: str,
) -> Tuple:
    """Determine the order of autoregressive prediction."""
    
    # Compute the given ordering
    pairs = []
    for i_xt, xti in enumerate(xt):
        for i_x in range(xti.shape[-2]):
            pairs.append((i_xt, i_x))

    if order in {"random", "given"}:
        if order == "random":
            # Randomly permute
            perm = torch.randperm(len(pairs), generator=generator)
            pairs = [pairs[i] for i in perm]

        # For every output, compute the inverse permutation
        perms = [[] for _ in range(len(xt))]
        for i_xt, i_x in pairs:
            perms[i_xt].append(i_x)
        inv_perms = [inv_perm(torch.tensor(perm)) for perm in perms]
        
        def unsort(y: torch.Tensor) -> List[torch.Tensor]:
            """Undo the sorting. Assumes vector of shape (num_samples, batch_size, num_contexts, num_features)"""
            # Put every output in its bucket
            buckets = [[] for _ in range(len(xt))]
            for i_y, (i_xt, _) in zip(range(y.shape[-2]), pairs):
                buckets[i_xt].append(y[..., i_y:i_y + 1, :])
            # Sort the buckets
            buckets = [[bucket[j] for j in p] for bucket, p in zip(buckets, inv_perms)]
            # Concatenate and return
            return [torch.cat(bucket, dim=-2) for bucket in buckets]

        return xt, yt, pairs, unsort

    elif order == "left-to-right":
        raise NotImplementedError("Left-to-right ordering not implemented.")
        # if len(xt) != 1:
        #     raise ValueError("Left-to-right ordering only works for a single output.")

        # # Make a copy since we'll modify
        # xt_data = xt[0].clone()
        # if yt is not None:
        #     yt = [y.clone() for y in yt]

        # # Sort the targets
        # perms = [torch.argsort(torch.argsort(batch, dim=1), dim=1) for batch in xt_data]
        # for i, perm in enumerate(perms):
        #     xt_data[i, :, :] = xt_data[i, :, perm]
        #     if yt is not None:
        #         yt[0][i, :, :] = yt[0][i, :, perm]

        # # Compute inverse permutations
        # inv_perms = [inv_perm(perm) for perm in perms]

        # def unsort(z: torch.Tensor) -> List[torch.Tensor]:
        #     """Undo the sorting."""
        #     z = z.clone()
        #     for i, perm in enumerate(inv_perms):
        #         z[..., i, :, :] = z[..., i, :, perm]
        #     return [z]

        # # Pack the one output again
        # xt = [xt_data]

        # return xt, yt, pairs, unsort

    else:
        raise RuntimeError(f'Invalid ordering "{order}".')

def ar_predict(
    generator: torch.Generator,
    model: Callable,
    xc: List[torch.Tensor],
    yc: List[torch.Tensor],
    xt: List[torch.Tensor],
    num_samples: int = 50,
    order: str = "random"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autoregressive sampling using torch.
    
    Args:
        generator: torch random number generator
        model: callable that takes (contexts, inputs) and returns predictions
        xc: list of 3D tensors for context inputs
        yc: list of 3D tensors for context targets
        xt: list of 3D tensors for target points
        num_samples: number of samples to generate
        order: ordering strategy ("random", "given", or "left-to-right")
    
    Returns:
        mean: marginal mean predictions
        var: marginal variance predictions
        yt: noisy samples
    """
    # Perform sorting
    xt_ordered, _, pairs, unsort = _determine_order(generator, xt, None, order)

    # Tile to produce multiple samples through batching
    xc_tiled = [x.unsqueeze(0).repeat(num_samples, 1, 1, 1) for x in xc]
    yc_tiled = [y.unsqueeze(0).repeat(num_samples, 1, 1, 1) for y in yc]
    xt_ordered = [x.unsqueeze(0).repeat(num_samples, 1, 1, 1) for x in xt_ordered]

    # Predict autoregressively
    preds = []
    yt = []
    for i_xt, i_x in pairs:
        xti = xt_ordered[i_xt][..., i_x:i_x + 1, :].reshape(-1, 1, xt_ordered[i_xt].shape[-1]) # collapse batch and num_samples       
        
        xci = xc_tiled[i_xt].reshape(-1, xc_tiled[i_xt].shape[-2], xc_tiled[i_xt].shape[-1])
        yci = yc_tiled[i_xt].reshape(-1, yc_tiled[i_xt].shape[-2], yc_tiled[i_xt].shape[-1])
        
        pred = model.predict(xci, yci, xti)
        yti = pred.sample()
        # reshape to original shape
        yti = yti.reshape(num_samples, xt_ordered[i_xt].shape[-3], 1, yti.shape[-1])
        mean = pred.mean.reshape(num_samples, xt_ordered[i_xt].shape[-3], 1, yti.shape[-1])
        variance = pred.variance.reshape(num_samples, xt_ordered[i_xt].shape[-3], 1, yti.shape[-1])

        xti = xti.reshape(num_samples, xt_ordered[i_xt].shape[-3], 1, xt_ordered[i_xt].shape[-1])

        yt.append(yti)
        preds.append(Normal(mean, variance))

        xc_tiled[i_xt] = torch.cat([xc_tiled[i_xt], xti], dim=-2)
        yc_tiled[i_xt] = torch.cat([yc_tiled[i_xt], yti], dim=-2)
    yt = unsort(torch.cat(yt, dim=-2))
    # Compute predictive statistics
    m1 = torch.mean(torch.cat([p.mean for p in preds], dim=-2), dim=0)
    m2 = torch.mean(torch.cat([p.variance + p.mean**2 for p in preds], dim=-2), dim=0)
    mean, var = unsort(m1), unsort(m2 - m1**2)

    # Get noiseless samples
    noise_less_preds = [model.predict(xc_tiled[i].reshape(-1, xc_tiled[i].shape[-2], xc_tiled[i].shape[-1]), yc_tiled[i].reshape(-1, yc_tiled[i].shape[-2], yc_tiled[i].shape[-1]), xt_ordered[i].reshape(-1, xt_ordered[i].shape[-2], xt_ordered[i].shape[-1])) for i in range(len(xc))]
    ft = [noise_less_preds[i].mean.reshape(num_samples, xt_ordered[i].shape[-3], xt_ordered[i].shape[-2], yc_tiled[i].shape[-1]) for i in range(len(xc))]
    # pred = model(contexts, xt)
    # ft = pred.mean

    return mean, var, yt, ft

def ar_predict_for_loop(
    generator: torch.Generator,
    model: Callable,
    xc: List[torch.Tensor],
    yc: List[torch.Tensor],
    xt: List[torch.Tensor],
    num_samples: int = 50,
    order: str = "random"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autoregressive sampling using torch.
    
    Args:
        generator: torch random number generator
        model: callable that takes (contexts, inputs) and returns predictions
        xc: list of 3D tensors for context inputs
        yc: list of 3D tensors for context targets
        xt: list of 3D tensors for target points
        num_samples: number of samples to generate
        order: ordering strategy ("random", "given", or "left-to-right")
    
    Returns:
        mean: marginal mean predictions
        var: marginal variance predictions
        yt: noisy samples
    """
    # Perform sorting
    xt_ordered, _, pairs, unsort = _determine_order(generator, xt, None, order)
    # Tile to produce multiple samples through batching
    xc_tiled = [x.unsqueeze(0).repeat(num_samples, 1, 1, 1) for x in xc]
    yc_tiled = [y.unsqueeze(0).repeat(num_samples, 1, 1, 1) for y in yc]
    xt_ordered = [x.unsqueeze(0).repeat(num_samples, 1, 1, 1) for x in xt_ordered]

    # Predict autoregressively
    preds = []
    yt = []
    for i_xt, i_x in pairs:
        new_append_xt = torch.zeros(num_samples, xt_ordered[i_xt].shape[1], 1, xt_ordered[i_xt].shape[3])
        new_append_yt = torch.zeros(num_samples, yc_tiled[i_xt].shape[1], 1, yc_tiled[i_xt].shape[3])
        new_append_means = torch.zeros(num_samples, xt_ordered[i_xt].shape[-3],  1, yc_tiled[i_xt].shape[-1])
        new_append_vars = torch.zeros(num_samples, xt_ordered[i_xt].shape[-3],  1, yc_tiled[i_xt].shape[-1])
        for sample_i in range(num_samples):        # todo get rid of this loop
            xti = xt_ordered[i_xt][sample_i][..., i_x:i_x + 1, :]            
            # Predict and sample
            xci = xc_tiled[i_xt][sample_i]
            yci = yc_tiled[i_xt][sample_i]
            
            pred = model.predict(xci, yci, xti)
            yti = pred.sample()
            yt.append(yti)
            new_append_xt[sample_i] = xti
            new_append_yt[sample_i] = yti
            new_append_means[sample_i] = pred.mean
            new_append_vars[sample_i] = pred.variance

        preds.append(Normal(new_append_means, new_append_vars))
        xc_tiled[i_xt] = torch.cat([xc_tiled[i_xt], new_append_xt], dim=-2)
        yc_tiled[i_xt] = torch.cat([yc_tiled[i_xt], new_append_yt], dim=-2)
    yt = unsort(torch.cat(yt, dim=-2))
    # Compute predictive statistics
    m1 = torch.mean(torch.cat([p.mean for p in preds], dim=-2), dim=0)
    m2 = torch.mean(torch.cat([p.variance + p.mean**2 for p in preds], dim=-2), dim=0)
    mean, var = unsort(m1), unsort(m2 - m1**2)

    # Get noiseless samples
    # preds = [model.predict(xc_tiled[i])]
    # ft = [pred.mean for pred in ]
    # pred = model(contexts, xt)
    # ft = pred.mean

    return mean, var, yt #ft