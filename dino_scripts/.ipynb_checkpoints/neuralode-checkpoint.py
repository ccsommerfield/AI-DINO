from config import torch, np
import math
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint, odeint_adjoint
from ode import ODE


class NeuralODE(ODE):
    def __init__(self, args, method='dopri5', adjoint=False, dtype=torch.float32):   
        super(NeuralODE, self).__init__(method, adjoint=adjoint, requires_grad=True, dtype=dtype)
        
        default_args = {'N': 50,
                        'kernel_size': 11}
        
        for k, v in default_args.items():
            setattr(self, k, args[k] if k in args else v)

        # Convolution operation with trainable kernel (self.conv.weight should have requires_grad=True; print to confirm!)
        self.conv = nn.Conv3d(1, 1, self.kernel_size, bias=False, padding='same', padding_mode='circular')
        nn.init.normal_(self.conv.weight, mean=0, std=0.01) # Initialize the kernel weights as random samples from a normal distribution
    
    def forward(self, t, y):
        y = y.view((-1, 1, self.N, self.N, self.N))
        cosy = torch.cos(y)
        siny = torch.sin(y)
        conv_cosy = self.conv(cosy)
        conv_siny = self.conv(siny)
        return (cosy * conv_siny - siny * conv_cosy).flatten(start_dim=-3)