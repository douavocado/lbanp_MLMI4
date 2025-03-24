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


__all__ = ['SawtoothSampler']

class SawtoothSampler(object):
    def __init__(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.seed = seed

    def sample(self,
            batch_size=16,
            num_ctx=None,
            num_tar=None,
            max_num_points=50,
            x_range=(-2, 2),
            xt_range=(-2, 2),
            freq_range=(2, 4),
            add_noise=False,
            noise_std=0.05,
            device='cpu'):

        batch = AttrDict()
        num_ctx = num_ctx or torch.randint(low=0, high=30, size=[1]).item()  # Nc
        num_tar = num_tar or 100 # 100 used in AR paper")

        # Generate the context points
        batch.xc = x_range[0] + (x_range[1] - x_range[0]) * torch.rand([batch_size, num_ctx, 1], device=device)
        batch.xt = xt_range[0] + (xt_range[1] - xt_range[0]) * torch.rand([batch_size, num_tar, 1], device=device)
        batch.x = torch.cat([batch.xc, batch.xt], dim=1)
        num_points = num_ctx + num_tar  # N = Nc + Nt
        
        # Sample frequency from uniform distribution
        freq = freq_range[0] + (freq_range[1] - freq_range[0]) * torch.rand([batch_size, 1, 1], device=device)
        
        # Sample random direction (in 1D case, this is just +1 or -1)
        direction = torch.randn([batch_size, 1, 1], device=device)
        norm = torch.sqrt(torch.sum(direction * direction, dim=2, keepdim=True))
        direction = direction / norm
        
        # Sample random offset (scaled by frequency)
        offset = torch.rand([batch_size, 1, 1], device=device) / freq
        
        # Compute the sawtooth function
        # For each point x: f(x) = (freq * (direction·x - offset)) % 1
        projected_x = torch.matmul(direction, batch.x.transpose(1, 2))  # [B,1,N]
        f = (freq * (projected_x - offset)) % 1  # [B,1,N]
        
        # Add noise
        if add_noise:
            noise = noise_std * torch.randn(f.shape, device=device)
            y = f + noise
        else:
            y = f
        
        # Reshape to match expected dimensions [B,N,1]
        y = y.transpose(1, 2)
        
        # Split into context and target
        batch.y = y
        batch.yc = y[:, :num_ctx, :]
        batch.yt = y[:, num_ctx:, :]
        
        return batch
