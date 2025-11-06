from config import torch, np, device, dtype
import math
from torch import Tensor



## final steps, rescale by magnitude (bc fully elastic scattering)
##magnitde is 2pi/wavelength so add wavelength into detector class
## rescale kfp and K_i bc qvector is kfp-K_i
class Detector:
    def __init__(self,X,Y,d_xy,R,wavelength,dtype=dtype,device=device):
        self.X = X #number of pixels
        self.Y = Y #number of pixels
        self.i_grid, self.j_grid = torch.meshgrid(torch.linspace(0,X-1,X, dtype = dtype, device=device), torch.linspace(Y-1,0,Y, dtype = dtype, device=device), indexing="ij") #specific pixle

        self.R = R #detector distance
        self.d_xy = d_xy #detector pixel size

        self.wavelength = wavelength #wavelength

    def get_alpha_i(self):
        alpha_i = torch.atan(self.d_xy/self.R*(self.i_grid-self.X/2))
        return alpha_i


    def get_alpha_j(self):
        alpha_j = torch.atan(self.d_xy/self.R*(self.j_grid-self.Y/2))
        return alpha_j


    def get_M1(self, K_i, K_f):

        self.K_i = K_i #wave vector incedent
        self.K_f = K_f #outgoing wave vector to center

        self.k_f_expand = K_f.view(1,1,3).expand(self.X, self.Y, 3)
        self.k_i_expand = K_i.view(1,1,3).expand(self.X, self.Y, 3)

        a_i = self.get_alpha_i()

        k_1 = torch.linalg.cross(self.K_i, self.K_f) / torch.norm(torch.linalg.cross(self.K_i, self.K_f))
        k_1_expand = k_1.view(1,1,3).expand(self.X, self.Y, 3)

    #defining the operations
        cos_ai = torch.cos(a_i).unsqueeze(-1)
        cross_k1_kf = torch.linalg.cross(k_1_expand, self.k_f_expand, dim = 2)
        sin_ai = torch.sin(a_i).unsqueeze(-1)
        dot_k1_kf = torch.dot(k_1, self.K_f)

        M1 = self.k_f_expand * cos_ai + cross_k1_kf * sin_ai + k_1_expand * dot_k1_kf * (1 - cos_ai)
        return M1


    def get_kfp(self, K_i, K_f):
        v = self.get_M1(K_i, K_f)
        k_2 = (self.K_f - self.K_i) / torch.norm(self.K_f - self.K_i)
        k_2_expand = k_2.view(1,1,3).expand(self.X, self.Y, 3)
        a_j = self.get_alpha_j()
  
  #defining the operations
        dot_prod = torch.sum(k_2_expand * v, dim = 2, keepdim = True)
        cos_aj = torch.cos(a_j).unsqueeze(-1)
        cross_k2_v = torch.linalg.cross(k_2_expand, v, dim=2)
        sin_aj = torch.sin(a_j).unsqueeze(-1)

        kfp = v * cos_aj + cross_k2_v * sin_aj + k_2 * dot_prod * (1 - cos_aj)
        return kfp
  
    def get_q(self, K_i, K_f):
        kfp = self.get_kfp(K_i, K_f)
        kfp_mag = kfp * 2 * torch.pi / self.wavelength
        K_i_mag = K_i * 2 * torch.pi / self.wavelength
        q = kfp_mag - K_i_mag
        return q

    @staticmethod
    def round_in_base(x: float, digits: int = 2, base: int = 10) -> float:
        """
        Round a number to the specified significant digits in the given base.
        Parameters:
        -----------
        x: float
            Number to round
        digits: int
            Number of significant digits to maintain when rounding
        base: int
            Base in which to round to the nearest power
        Returns:
        --------
        float:
            Rounded number
        """
        if x == 0:
            return 0
        else:
            exponent = np.floor(math.log(abs(x), base))
            scale = base ** (exponent - digits + 1)
            rounded = round(x / scale) * scale
            return type(x)(rounded)
        
    '''
    def calculate_resolution(self, K_i, K_f) -> float:
        q = self.get_q(K_i, K_f)
        q_max = q.max()
        res = 2 * torch.pi/q_max
        return res
    '''
