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