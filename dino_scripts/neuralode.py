from config import torch, np, device, dtype
import math
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint, odeint_adjoint
from ode import ODE
'''
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
'''#### CHILL
class NeuralODE(ODE):
    """
    A Neural ODE implementation using a U-Net architecture for learning spatiotemporal dynamics.
    """
    
    def __init__(self, args, method='dopri5', adjoint=False, dtype=torch.float32):
        """
        Initialize the Neural ODE with U-Net architecture.
        
        Parameters
        ----------
        args : dict
            Dictionary containing model parameters. Supported keys:
            - 'N': Grid size for 2D spatial domain (default: 128)
            - 'num_layers': Number of encoder/decoder layers (default: 4)
        method : str, optional
            ODE integration method. Defaults to 'dopri5'.
        adjoint : bool, optional
            Whether to use adjoint method for gradient computation. Defaults to False.
        dtype : torch.dtype, optional
            Data type for computations. Defaults to torch.float32.
        """
        super(NeuralODE, self).__init__(method, adjoint=adjoint, requires_grad=True, dtype=dtype)
        
        default_args = {'N': 32,
                        'num_layers': 4}
        
        for k, v in default_args.items():
            setattr(self, k, args[k] if k in args else v)
        
        # Channel progression: 16 -> 32 -> 64 -> 128 -> ...
        start_channels = 16
        channels = [start_channels * (2 ** i) for i in range(self.num_layers + 1)]
        
        # Initial convolution: 2 -> 16 channels
        self.input_conv = nn.Sequential(
            nn.Conv3d(1, start_channels, kernel_size=3, padding='same', padding_mode='circular'),
            nn.ReLU(),
            nn.Conv3d(start_channels, start_channels, kernel_size=3, padding='same', padding_mode='circular'),
            nn.ReLU()
        )
        
        # Encoder (downsampling path)
        self.encoders = nn.ModuleList()
        for i in range(self.num_layers):
            self.encoders.append(nn.Sequential(
                nn.MaxPool3d(2),
                nn.Conv3d(channels[i], channels[i+1], kernel_size=3, padding='same', padding_mode='circular'),
                nn.ReLU(),
                nn.Conv3d(channels[i+1], channels[i+1], kernel_size=3, padding='same', padding_mode='circular'),
                nn.ReLU()
            ))
        
        # Decoder (upsampling path)
        
        self.decoders = nn.ModuleList()
        for i in range(self.num_layers):

            self.decoders.append(nn.Sequential(
                nn.ConvTranspose3d(channels[self.num_layers-i], channels[self.num_layers-i-1], kernel_size=2, stride=2),
                nn.Conv3d(channels[self.num_layers-i], channels[self.num_layers-i-1], kernel_size=3, padding='same', padding_mode='circular'),
                #nn.Conv3d(2*channels[self.num_layers-i-1], channels[self.num_layers-i-1], kernel_size=3, padding='same', padding_mode='circular'),
                nn.ReLU(),
                nn.Conv3d(channels[self.num_layers-i-1], channels[self.num_layers-i-1], kernel_size=3, padding='same', padding_mode='circular'),
                nn.ReLU()
            ))
    

        
        # Final output convolution: 16 -> 1 channel
        self.output_conv = nn.Conv3d(start_channels, 1, kernel_size=1)
    
    def forward(self, t, u):
        """
        Compute the time derivative using the U-Net neural network.
        
        Parameters
        ----------
        t : torch.Tensor
            Current time (not used in computation but required for ODE interface).
        u : torch.Tensor
            Current state tensor of shape (batch_size, N*N*N) representing
            flattened 3D spatial fields.
            
        Returns
        -------
        torch.Tensor
            Time derivative du/dt with same shape as input u, computed by
            the neural network to represent learned spatiotemporal dynamics.
        """

        u = u.view((-1, 1, self.N, self.N, self.N))
        
        # Store skip connections
        skip_connections = []
        
        # Initial convolution
        u = self.input_conv(u)
        skip_connections.append(u)
        
        # Encoder path
        for encoder in self.encoders:
            u = encoder(u)
            skip_connections.append(u)
        
        # Remove the last skip connection (bottleneck)
        skip_connections.pop()
        
        # Decoder path
        for i, decoder in enumerate(self.decoders):
            # Upsample
            u = decoder[0](u)  # ConvTranspose3d
            
            # Concatenate with skip connection
            skip = skip_connections.pop()
            u = torch.cat([skip, u], dim=1)
            
            # Apply remaining convolutions
            u = decoder[1:](u)
        
        # Final output
        return self.output_conv(u).flatten(start_dim=-3)
