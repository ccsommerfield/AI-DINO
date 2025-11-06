from config import torch, np, device, dtype

import torch.nn as nn
import numpy as np
from typing import Dict
from ode import ODE



# Note for self --> we want to only have Ti atoms moves, so take this into account with scattering later
class Kuramoto(ODE):
    def __init__(self, args, method='dopri5', dtype=torch.float32):   
        super(Kuramoto, self).__init__(method, adjoint=False, requires_grad=False, dtype=dtype)
        
        default_args = {'N': 50,
                        'K': 0.2,
                        's': 1.
                       }
        
        for k, v in default_args.items():
            setattr(self, k, args[k] if k in args else v)

        # Get the true kernel
        kernel = self.K * self.laplacian_of_gaussian_kernel()[None, None]
        #kernel = self.K * self.boring_kernel()[None, None]
        #kernel = self.K * self.exp_kernel()[None, None]

        # Convolution operation with kernel
        self.conv = nn.Conv3d(1, 1, kernel.shape[-1], bias=False, padding='same', padding_mode='circular')
        self.conv.weight = nn.Parameter(kernel, requires_grad=False)
    
    def laplacian_of_gaussian_kernel(self):
        # Extend kernel to three standard deviations
        d = int(np.ceil(3 * self.s))
        d_range = torch.arange(-d, d+1, dtype=self.dtype)

        # Create spatial grid over which to evaluate kernel
        x = torch.stack(torch.meshgrid(d_range, d_range, d_range, indexing='xy'))
        n = x.shape[0]

        # Evaluate laplacian of gaussian over grid
        kernel = torch.exp(-(x**2).sum(dim=0) / (2 * self.s**2))
        kernel *= -((x**2).sum(dim=0) / self.s**2 - n) / np.sqrt((2 * np.pi)**n) / self.s**(n + 2)
        kernel[d,d,d] = 0.
        
        return kernel #output is [kernel size,kernel size,kernel size]

    def boring_kernel(self):
        d = int(np.ceil(3 * self.s))
        k = 2 * d + 1
        # fill the kernel directly
        kernel =  torch.full((k, k, k), 0.46, dtype=self.dtype, device=device)
        return(kernel)
        
    def exp_kernel(self):
        d = int(np.ceil(3 * self.s))
        d_range = torch.arange(-d, d+1, dtype=self.dtype)

        # Create spatial grid over which to evaluate kernel
        x = torch.stack(torch.meshgrid(d_range, d_range, d_range, indexing='xy'))
        kernel = torch.exp((-x**2).sum(dim=0))
        coords = torch.meshgrid(
        torch.arange(-d, d+1),
        torch.arange(-d, d+1),
        torch.arange(-d, d+1),
        indexing='ij'
        )
        x, y, z = coords
        dist = x.abs() + y.abs() + z.abs()   # Manhattan (L1) distance
    
        # Zero out the center octahedron of radius r
        r = 1   # or 2, etc., to control size
        kernel[dist <= r] = 0.
    
        return kernel
        
    def init_state(self, M=1, seed=12):
        torch.manual_seed(seed)
        return 2 * torch.pi * torch.rand((M, 1, self.N, self.N, self.N), dtype=self.dtype).flatten(start_dim=-3)
    
    def forward(self, t, y):
        y = y.view((-1, 1, self.N, self.N, self.N))
        cosy = torch.cos(y)
        siny = torch.sin(y)
        conv_cosy = self.conv(cosy)
        conv_siny = self.conv(siny)
        return (cosy * conv_siny - siny * conv_cosy).flatten(start_dim=-3)