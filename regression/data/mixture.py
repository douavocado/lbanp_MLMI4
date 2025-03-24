# MIT License

# Copyright (c) 2022 Tung Nguyen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from attrdict import AttrDict
from .sawtooth import SawtoothSampler
from .gp import GPSampler, RBFKernel, Matern52Kernel, WeaklyPeriodicKernel


__all__ = ['MixtureSampler']

class MixtureSampler(object):
    def __init__(self, seed=None):
        """
        A sampler that randomly selects between sawtooth, RBF GP, Matern52 GP, and weakly periodic GP.
        Each distribution is selected with equal probability (25%).
        
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.seed = seed
        
        # Initialize individual samplers
        self.sawtooth_sampler = SawtoothSampler(seed=seed)
        self.rbf_gp_sampler = GPSampler(RBFKernel(), seed=seed)
        self.matern52_gp_sampler = GPSampler(Matern52Kernel(),seed=seed)
        self.weakly_periodic_gp_sampler = GPSampler(WeaklyPeriodicKernel(), seed=seed)
        
    def sample(self,
            batch_size=16,
            num_ctx=None,
            num_tar=None,
            max_num_points=50,
            x_range=(-2, 2),
            xt_range=(-2, 2),
            add_noise=True,
            noise_std=0.05,
            device='cpu'):
        """
        Sample from a randomly selected distribution.
        
        Args:
            batch_size (int): Number of samples to generate.
            num_ctx (int, optional): Number of context points. If None, randomly selected.
            num_tar (int, optional): Number of target points. If None, randomly selected.
            max_num_points (int): Maximum number of total points.
            x_range (tuple): Range for context input points.
            xt_range (tuple): Range for target input points.
            add_noise (bool): Whether to add noise to sawtooth samples.
            noise_std (float): Standard deviation of noise for sawtooth.
            device (str): Device to use for computations.
            
        Returns:
            AttrDict: A batch of samples with context and target points.
        """
        # Randomly select which distribution to sample from
        sampler_idx = torch.randint(0, 4, (1,)).item()
        
        if sampler_idx == 0:
            # Sawtooth
            batch = self.sawtooth_sampler.sample(
                batch_size=batch_size,
                num_ctx=num_ctx,
                num_tar=num_tar,
                max_num_points=max_num_points,
                x_range=x_range,
                xt_range=xt_range,
                add_noise=add_noise,
                noise_std=noise_std,
                device=device
            )
        elif sampler_idx == 1:
            # RBF GP
            batch = self.rbf_gp_sampler.sample(
                batch_size=batch_size,
                num_ctx=num_ctx,
                num_tar=num_tar,
                max_num_points=max_num_points,
                x_range=x_range,
                xt_range=xt_range,
                device=device
            )
        elif sampler_idx == 2:
            # Matern52 GP
            batch = self.matern52_gp_sampler.sample(
                batch_size=batch_size,
                num_ctx=num_ctx,
                num_tar=num_tar,
                max_num_points=max_num_points,
                x_range=x_range,
                xt_range=xt_range,
                device=device
            )
        else:
            # Weakly Periodic GP
            batch = self.weakly_periodic_gp_sampler.sample(
                batch_size=batch_size,
                num_ctx=num_ctx,
                num_tar=num_tar,
                max_num_points=max_num_points,
                x_range=x_range,
                xt_range=xt_range,
                device=device
            )
            
        return batch

