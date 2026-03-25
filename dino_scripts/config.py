import torch
import numpy as np

# Select CUDA if available
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32