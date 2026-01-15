from config import torch, np, device, dtype

import torch.nn as nn
import numpy as np
from typing import Dict
from ode import ODE

class NeuralODE_simple(ODE):
    """
    The Cahn-Hilliard equation is a fourth-order partial differential equation
    that models phase separation in binary alloys and other systems. This implementation
    solves the equation on a 2D periodic domain using finite differences and 
    convolutional operations for spatial derivatives.
    
    The equation takes the form:
    dc/dt = -0.5 * D * ∇²(c - c³ + γ * ∇²c)
    
    where c represents the concentration field, D is the diffusion parameter,
    and ∇² is the Laplace operator.
    
    Attributes
    ----------
    N : int
        Grid size for the 3D spatial domain (N x N grid). Default is 128.
    L : float
        Physical length of the square domain. Default is 2.0.
    D : float
        Diffusion coefficient controlling the rate of evolution. Default is 1e-4.
    g: float
        Gradient energy coefficient controlling the length of transition regions between the domains. Default is 1e-4.
    laplacian : torch.nn.Conv3d
        Convolutional layer implementing the discrete Laplace operator
        with periodic boundary conditions.
    """
    def __init__(self, args, method='dopri5', adjoint=False, dtype=torch.float32):
        """
        Initialize the 3D Cahn Hilliard model with specified parameters.
        
        Parameters
        ----------
        args : dict
            Dictionary containing model parameters. Missing parameters will use defaults.
        method : str, optional
            ODE integration method. Defaults to 'dopri5'.
        dtype : torch.dtype, optional
            Data type for computations. Defaults to torch.float32.
        """
        super(NeuralODE_simple, self).__init__(method, adjoint=adjoint, requires_grad=True, dtype=dtype)
        
        default_args = {'N': 128,
                       }
        
        for k, v in default_args.items():
            setattr(self, k, args[k] if k in args else v)

        # Five-point stencil for the 2D Laplace operator ∇²
     
        self.laplacian = nn.Conv3d(1, 1, 3, bias=False, padding='same', padding_mode='circular')
        self.D = nn.Parameter(torch.tensor(3.5e-4), requires_grad=True)
        self.g = nn.Parameter(torch.tensor(3.5e-4), requires_grad=True)
        #self.laplacian.weight = nn.Parameter(stencil, requires_grad=False)
        
    def forward(self, t, c):
        """
        Compute the time derivative for the Cahn-Hilliard equation.
        
        This method implements the 2D Cahn-Hilliard equation:
        dc/dt = -0.5 * D * ∇²(c - c³ + γ * ∇²c)
        
        Parameters
        ----------
        t : torch.Tensor
            Current time.
        c : torch.Tensor
            Current state vector of shape (M, N*N) where M is batch size
            and N*N represents the flattened 2D spatial grid.
            
        Returns
        -------
        torch.Tensor
            Time derivative dc/dt with same shape as input c, representing
            the rate of change according to the Cahn-Hilliard dynamics.
        """
        c = c.view(-1, 1, self.N, self.N, self.N)

        return -0.5 * self.D * self.laplacian(c - c ** 3 + self.g * self.laplacian(c)).flatten(start_dim=-3)
