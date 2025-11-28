from config import torch, np
import math
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint, odeint_adjoint


class ODE(nn.Module):
    def __init__(self, method='dopri5', adjoint=False, requires_grad=True, dtype=torch.float32):
        super(ODE, self).__init__()

        self.method = method
        self.adjoint = adjoint if requires_grad else False
        self.odeint = odeint_adjoint if self.adjoint else odeint
        self.dtype = dtype


    def solve(self, t, y0, device='cpu', rtol=1e-7, atol=1e-9):
        if self.training:
            return self.odeint(self.to(device), y0.to(device), t.to(device), method=self.method,
                               rtol=rtol, atol=atol, options={'dtype':self.dtype})
        else:
            with torch.no_grad():
                return odeint(self.to(device), y0.to(device), t.to(device), method=self.method,
                              rtol=rtol, atol=atol, options={'dtype':self._dtype})
                
    def get_batch(self, t, y, batch_time, batch_size):
        """
        Create batches from trajectory data.
        
        Parameters
        ----------
        t : torch.Tensor
            Time points. Shape: (T,).
        u : torch.Tensor
            Trajectory data. Shape: (T, M, D) where:
            - T: number of time points
            - M: number of trajectories  
            - D: state dimension
        batch_time : int
            Length of each batch sequence (number of time steps). If batch_time == T, returns full trajectories.
        batch_size : int
            Number of sequences in the batch.
        
        Returns
        -------
        t_batch : torch.Tensor
            Time points for the batch. Shape: (batch_time,).
        y0_batch : torch.Tensor
            Initial conditions for each batch sequence. Shape: (batch_size, D).
        y_batch : torch.Tensor
            Batch trajectory data. Shape: (batch_time, batch_size, D).
        """
        '''
        T, M = y.shape[:2]
        t_batch = t[:batch_time]
    
        if batch_time == T:
            # Return full trajectories for randomly sampled initial conditions
            replace = batch_size > M  # Allow replacement if we need more samples than available trajectories
            traj_indices = np.random.choice(M, batch_size, replace=replace)
            
            y0_batch = y[0, traj_indices]  # Initial conditions from selected trajectories
            y_batch = y[:, traj_indices]   # Full trajectories for selected trajectories
            
        else:
            # Sample subsequences of trajectories for randomly sampled initial conditions
            # Generate all possible (time_start, trajectory_index) combinations
            c = [[i, j] for i in range(T - batch_time) for j in range(M)]
            sampled_indices = np.random.choice(len(c), batch_size, replace=False)
            
            # Extract time and trajectory indices
            time_indices = [c[i][0] for i in sampled_indices]
            traj_indices = [c[i][1] for i in sampled_indices]
            
            # Extract initial conditions
            y0_batch = y[time_indices, traj_indices]
            
            # Extract trajectory subsequences
            batch_indices = torch.arange(batch_time)[:, None] + torch.tensor(time_indices)[None, :]
            y_batch = y[batch_indices, traj_indices]
        
        return t_batch, y0_batch, y_batch
        '''

    def get_batch(self, t, y0, y, batch_time, batch_size):
        T, M = y.shape[:2]
        D = y.shape[-1]
        t_batch = t[:batch_time]
    
        c = [[i,j] for i in range(T - batch_time) for j in range(M)]
        b = [c[i] for i in np.random.choice(len(c), batch_size, replace=False)]
    
        for i in range(len(b)):
            if i==0:
                y0_batch = y[b[i][0], b[i][1]][None,:]
                y_batch = torch.stack([y[b[i][0]+j, b[i][1]] for j in range(batch_time)], dim=0)[:,None,:]
            else:
                y0_batch = torch.cat((y0_batch, y[b[i][0], b[i][1]][None,:]))
                y_batch = torch.cat((y_batch,
                    torch.stack([y[b[i][0]+j, b[i][1]] for j in range(batch_time)], dim=0)[:,None,:]), dim=1)
        return t_batch, y0_batch, y_batch



# swarm.solve will return a tensor of shape [n_steps, N, 4]
