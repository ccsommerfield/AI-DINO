from config import torch, np, device, dtype

import torch.nn as nn
import numpy as np
from typing import Dict
from ode import ODE

class CahnHilliard(ODE):
    """
    Quasi-2D Cahn-Hilliard model where a 2D operator is replicated along the z-dimension.
    
    This class implements the Cahn-Hilliard equation as a stack of independent 2D problems
    with identical spatial operators in the x-y plane. The equation evolves concentration
    fields according to:
    
    dc/dt = -0.5 * D * ∇²(c - c³ + γ * ∇²c)
    
    where the Laplacian operator ∇² is 2D (acts only in x-y plane) and is applied
    independently to each z-slice.
    
    Attributes
    ----------
    Nx, Ny : int
        Grid sizes for the 2D spatial domain in x and y directions.
    Nz : int
        Number of independent z-slices.
    L : float
        Physical length of the square domain.
    D : float
        Diffusion coefficient controlling the rate of evolution.
    g : float
        Gradient energy coefficient controlling the length of transition regions.
    laplacian : nn.Conv3d
        Convolution layer implementing the 2D Laplace operator replicated in z.
    """
    
    def __init__(self, args, method='dopri5', dtype=torch.float32):
        """
        Initialize the quasi-2D Cahn-Hilliard model with specified parameters.
        
        Parameters
        ----------
        args : dict
            Dictionary containing model parameters:
            - 'Nx' : x-dimension size (default: 128)
            - 'Ny' : y-dimension size (default: 128)
            - 'Nz' : z-dimension size (default: 50)
            - 'L' : physical domain length (default: 2.0)
            - 'D' : diffusion coefficient (default: 1e-4)
            - 'g' : gradient energy coefficient (default: 1e-4)
        method : str, optional
            ODE integration method. Defaults to 'dopri5'.
        dtype : torch.dtype, optional
            Data type for computations. Defaults to torch.float32.
        """
        super(CahnHilliard, self).__init__(method, adjoint=False, requires_grad=False, dtype=dtype)
        
        default_args = {
            'Nx': 128,
            'Ny': 128,
            'Nz': 50,
            'L': 2.,
            'D': 1e-4,
            'g': 1e-4
        }
        
        for k, v in default_args.items():
            setattr(self, k, args[k] if k in args else v)

        # Five-point stencil for the 2D Laplace operator ∇²
        # This is a 2D stencil that will be applied to each z-slice independently
        h = self.L / max(self.Nx, self.Ny)
        stencil_2d = (1./h) ** 2 * torch.tensor([[0, 1, 0.],
                                                   [1, -4, 1],
                                                   [0, 1, 0]], dtype=self.dtype)
        
        # Convert to 3D kernel with singleton z dimension
        # Shape: (3, 3) -> (3, 3, 1) -> (1, 1, 3, 3, 1)
        kernel_3d = stencil_2d[..., None]  # Add z dimension: (3, 3, 1)
        kernel_3d = kernel_3d[None, None, ...]  # Add channel dims: (1, 1, 3, 3, 1)
        
        # Conv3d with kernel size 1 in z-direction applies 2D conv to each z-slice independently
        self.laplacian = nn.Conv3d(1, 1, kernel_3d.shape[-3:], 
                                   bias=False, padding='same', padding_mode='circular')
        self.laplacian.weight = nn.Parameter(kernel_3d, requires_grad=False)
            
    def init_state(self, M=1, seed=12, sigma=0.01):
        """
        Initialize random concentration field with identical patterns across all z-slices.
        
        Parameters
        ----------
        M : int, optional
            Batch size (number of initial conditions). Defaults to 1.
        seed : int, optional
            Random seed for reproducible initialization. Defaults to 12.
        sigma : float, optional
            Amplitude scaling factor for the random noise. Defaults to 0.01.
            
        Returns
        -------
        torch.Tensor
            Initial state tensor of shape (M, Nx*Ny*Nz) with random noise,
            identical across all z-slices.
        """
        torch.manual_seed(seed)
        
        # Generate 2D initial state
        c0_2d = sigma * torch.randn((M, 1, self.Nx, self.Ny), dtype=self.dtype)
        
        # Replicate across z-dimension
        c0 = c0_2d.unsqueeze(-1).expand(-1, -1, -1, -1, self.Nz)
        return c0.flatten(start_dim=-3)
        
    def forward(self, t, c):
        """
        Compute the time derivative for the quasi-2D Cahn-Hilliard equation.
        
        This method implements the Cahn-Hilliard equation with 2D spatial operators
        applied independently to each z-slice:
        dc/dt = -0.5 * D * ∇²(c - c³ + γ * ∇²c)
        
        Parameters
        ----------
        t : torch.Tensor
            Current time.
        c : torch.Tensor
            Current state vector of shape (M, Nx*Ny*Nz) where M is batch size.
            
        Returns
        -------
        torch.Tensor
            Time derivative dc/dt with same shape as input c, representing
            the rate of change according to the Cahn-Hilliard dynamics.
        """
        c = c.view(-1, 1, self.Nx, self.Ny, self.Nz)
        return -0.5 * self.D * self.laplacian(c - c ** 3 + self.g * self.laplacian(c)).flatten(start_dim=-3)