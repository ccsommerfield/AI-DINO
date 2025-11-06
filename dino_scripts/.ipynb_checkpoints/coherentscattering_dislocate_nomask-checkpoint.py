from config import torch, np, device, dtype
from torch import Tensor
from dynamicmotion import Dynamic_Motion

class CoherentScattering_dislocate_nomask:
    def __init__(self, crystal, K_i, K_f, detector, dtype=dtype, device=device): #model = model
        self.K_i = K_i
        self.K_f = K_f
        self.detector = detector
        self.q_vectors = self.detector.get_q(K_i,K_f)
        self.crystal = crystal
        
        
    def calculate_sum(self, d, scale, model, index=None, tf=None, dt=None, args=None) -> Tensor:

        '''
        --------------------------------------
        d: supercell size
        scale: how large are your dislocations
        model: what your dynamics are 
               (Currently just Swarmalators)
        tf: final time that dyanmics run until
        dt: size of time step
        args: A B J K vals for Swarmalators
        if you want to use manual input, model = disp
        --------------------------------------

        '''

        
        q_vectors_flat = self.q_vectors.reshape(-1, 3)  # [batch_size, 3]
        
        batch_size_original = self.q_vectors.shape[:-1]

        # Supercell positions R_nd
        d1, d2, d3 = d
        n1, n2, n3 = self.crystal.crystal_size

        with torch.no_grad():
            i_indices = torch.arange(0, n1, d1, dtype=dtype, device=device)
            j_indices = torch.arange(0, n2, d2, dtype=dtype, device=device)
            k_indices = torch.arange(0, n3, d3, dtype=dtype, device=device)

            # Compute a grid of all indices

            i, j, k = torch.meshgrid(i_indices, j_indices, k_indices, indexing='xy')
 
            # Reshape to [n_supercells, 3]

            supercell_indices = torch.stack([i.flatten(), j.flatten(), k.flatten()], dim=1)

            # Calculate positions in real space --> this is R_nd

        #This is our R_nd comp

            supercell_positions = torch.matmul(supercell_indices, self.crystal.lattice_vectors) ## full grid of R_nd


        #This is our r_m comp
       

        r_m = self.crystal.atom_positions
        
        
        n_atoms = r_m.shape[0]

        if model == 'Swarmalators':
       
            u_comp =  Dynamic_Motion(self.crystal, device, index, dtype)
            vals = u_comp.solve(Swarmalators, args, d, tf, dt) 
            u_avg = vals[..., :3] * scale                                                              # [time_steps ,n_supercells, n_atoms, 3]
        else:
            motion = Dynamic_Motion(self.crystal, device, index, dtype)
            u_avg = motion.manual_input(disp=model, d = d, index = index) * scale ## model = disp  [time_steps ,n_supercells, n_atoms, 3]

        # Form factors (broadcasted)
        q_magnitude = torch.norm(q_vectors_flat, dim=1, keepdim=True) * 1e-10               # [batch_size, 1]
        
        
        f_q = self.crystal.calculate_form_factors(q_magnitude)                               # [batch_size, n_atoms]
        f_q = f_q[None, :, None, :]                                                               # [1, batch_size, 1, n_atoms]
        
        
        r = r_m[None, None, :, :]                                                     # [1, 1, n_atoms, 3]
        u = u_avg                                                                    # [steps, n_supercells, n_atoms, 3]
        t_step = u.shape[0]

        #Summing the m dependant components
        
        m_dep_pos = r.to(device) + u.to(device)                                                             # [steps, n_supercells, n_atoms, 3]
    

        q_dot_m = torch.einsum('bi,tsai->tbsa',q_vectors_flat, m_dep_pos)              # [time_steps, batch_size, n_supercells, n_atoms]
       
        m_dep_phase = torch.exp(-1j*q_dot_m)
        m_structure_factor = f_q * m_dep_phase

        m_sum = torch.sum(m_structure_factor, dim = -1)                                # [batch_size, n_supercells]

        with torch.no_grad():
            q_dot_R = torch.matmul(q_vectors_flat, supercell_positions.T)             #[batch_size, n_supercells]
        
            R_phase = torch.exp(-1j * q_dot_R)

        ## Summing over Rnd

        tot_sum = torch.sum(m_sum * R_phase[None,...], dim = 2) * d1 * d2 * d3 ## might be -1
        
        return tot_sum.view(t_step, *batch_size_original)
        
    def get_intensity(self, d, scale, model, lambda_max, index=None, tf=None, dt=None, args=None, poisson = False):

    
        A = self.calculate_sum(d, scale, model, index, tf, dt, args) 

        I = A.abs()**2

        if poisson == True:

            I_scale = I * lambda_max/I.max()

            I_scale = torch.poisson(I_scale)
            
        else:

            I_scale = I * lambda_max/I.max()

        return I_scale

        '''
        
        if poisson == True:

            scale = lambda_max/I.max()

            I_scale = I * scale

            I_scale_poisson = torch.poisson(I_scale)

            I_poisson = torch.poisson(I_scale)/scale

            return I_poisson
            
        else:
            return I
        '''



    def get_amplitude(self, d, scale, model, lambda_max, index=None, tf=None, dt=None, args=None):

    
        A = self.calculate_sum(d, scale, model, index, tf, dt, args) 

        A_scale = A * np.sqrt(lambda_max)/A.abs().max()

        return A_scale

         

    def two_time_correlation(self, I, r_in, r_out):  # normalize to run from zero to one -- check paper for exact norm
         #ave I over annulus in detector and do I ave over that
        
        # Get time-dependent intensities: shape [time_steps, X, Y]
        #A = self.get_intensity(d, scale, model, tf, dt, args)   
        I_fluctuation = I-I.mean()

        # Reshape to [T, N] where N = number of pixels
        
        T = I.shape[0]   # Number of Time steps
        MX, MY = I.shape[1], I.shape[2]
        center_x, center_y = MX // 2, MY // 2
        yy, xx = torch.meshgrid(torch.arange(MX), torch.arange(MY), indexing="ij")
        dist_from_center = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        mask = torch.logical_and(dist_from_center >= r_in, dist_from_center <= r_out).int()

        ## Extract I values from mask --> like actually extract them
        masked_I = I_fluctuation * mask.to(I.device)

        I_no_zero = masked_I[masked_I != 0]
        
        I_flat = I_no_zero.view(T, -1)   # flatten detector pixels

        
        # Compute mean intensity per frame [T, 1] -> should be averaged over q, but is this inherently in q?
        I_mean = I_flat.sum(dim=1, keepdim=True)/mask.sum()

        # Normalize intensities to remove baseline
        I_norm = I_flat / I_mean ## Do I keep U mean?

        # Compute g2 by matrix multiplication (normalized correlation)
        g2 = (I_norm @ I_norm.T) / I_norm.shape[1]
        '''
        valid_pixels = mask.sum()
        I_flat = I_flat[:, :valid_pixels]

        # Cartesian product (matrix multiply) over pixels
        g2 = (I_flat @ I_flat.T) / valid_pixels
        '''
        return g2

   
    def autocorrelation(self, I, r_in, r_out): # normalize to run from one to two
        # I is intensity fluctuation, so maybe subtract from I.mean
        #changing it so you have to run seprately

        #A = self.get_intensity(d, scale, model, tf, dt, args)

        I_fluctuation = I-I.mean()

        T = I.shape[0]

        tau_list = torch.arange(0,T, dtype=I.dtype, device=I.device)



        # create annulus shaped mask here, then ave over this
        MX, MY = I.shape[1], I.shape[2]
        center_x, center_y = MX // 2, MY // 2
        yy, xx = torch.meshgrid(torch.arange(MX), torch.arange(MY), indexing="ij")
        dist_from_center = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        mask = torch.logical_and(dist_from_center >= r_in, dist_from_center <= r_out).int()

        ## Extract I values from mask
        masked_I = I_fluctuation * mask.to(I.device)
        
        
        I_sum = masked_I.view(I.shape[0], -1).sum(dim=1)     # [Time steps]
        N_pixels = mask.sum()                                # number of pixels in annulus

        # Average intensity in annulus for each time step
        I_avg = I_sum / N_pixels  

        I_mean = I_avg.mean()

        product = I_avg[None,:] * I_avg[:, None]

        g2_vals = torch.tensor([product.diagonal(offset=int(tau)).sum() / (T + 1 - tau) for tau in tau_list], device=I.device) / (I_mean**2)

        
        return tau_list, g2_vals