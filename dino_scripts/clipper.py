from config import torch, np, device, dtype

class AdaptiveGradClipper:
    def __init__(self, beta=0.9, scale_factor=2.0):
        self.beta = beta
        self.scale_factor = scale_factor
        self.running_norm = None
    
    def clip_gradients(self, parameters):
        # Compute current gradient norm
        total_norm = 0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        current_norm = total_norm ** 0.5
        
        # Update running average
        if self.running_norm is None:
            self.running_norm = current_norm
        else:
            self.running_norm = self.beta * self.running_norm + (1 - self.beta) * current_norm
        
        # Clip based on running average
        max_norm = self.scale_factor * self.running_norm
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        
        return current_norm, max_norm
    
    def state_dict(self):
        """Return state dictionary for saving."""
        return {
            'beta': self.beta,
            'scale_factor': self.scale_factor,
            'running_norm': self.running_norm
        }
    
    def load_state_dict(self, state_dict):
        """Load state from dictionary."""
        self.beta = state_dict['beta']
        self.scale_factor = state_dict['scale_factor']
        self.running_norm = state_dict['running_norm']
