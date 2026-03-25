#start by generating a random list for spacing, and set form facter to 1 -> f
#u_tilde is ave displacement of atom m in the supercell at pos R_nd... we get these from the sample class above
# d_i are pos consant int values for each index i
#d_i goes 1, 2, 3
#d is size of super cell in each dim
#ni is diatance from or origin to lattice point
#pretend rm tilda for now
# Im just going to assume we know u_ave, R^m_dn and R_nd
from config import torch, np, device, dtype
from torch import Tensor

class CoherentScattering:
    def __init__(self, crystal, K_i, K_f, b_size, detector, dtype=dtype, device=device):
        self.K_i = K_i
        self.K_f = K_f
        self.detector = detector
        self.q_vectors = self.detector.get_q(K_i,K_f)
        self.crystal = crystal
        self.b_size = b_size
    
    #def animate_displacemet(self):
        #start with a cos function that evolves with time
        #loop simulation over the time  

    def bool_mask(self, d, shape_mask=None, r_out=None, start_coord=None, end_coord=None, chill=None):
        """
        Creating different shaped grains
        """

        d1, d2, d3 = d
        n1, n2, n3 = self.crystal.crystal_size

        i_indices = torch.arange(0, n1, d1, dtype=dtype, device=device)
        j_indices = torch.arange(0, n2, d2, dtype=dtype, device=device)
        k_indices = torch.arange(0, n3, d3, dtype=dtype, device=device)

            # Compute a grid of all indices

        i, j, k = torch.meshgrid(i_indices, j_indices, k_indices, indexing='xy')
 
            # Reshape to [n_supercells, 3]

        supercell_indices = torch.stack([i.flatten(), j.flatten(), k.flatten()], dim=1)

            # Calculate positions in real space --> this is R_nd

        supercell_positions = torch.matmul(supercell_indices, self.crystal.lattice_vectors) ## full grid of R_nd


        
        if shape_mask == 'circle':
            center_i = (n1 - 1) / 2 
            center_j = (n2 - 1) / 2 
            center_k = (n3 - 1) / 2 

            dist_from_center = torch.sqrt((i - center_i) ** 2 + 
                                          (j - center_j) ** 2 + 
                                          (k - center_k) ** 2)

            mask_bool = (dist_from_center < r_out).flatten() # shape is product of supercells in each direction
            
        elif shape_mask == 'rectangle':

            x0, y0, z0 = start_coord #indicies
            x1, y1, z1 = end_coord #indicies
            '''
            mask_bool = (
                (supercell_indices[:, 0] >= x0) & (supercell_indices[:, 0] < x1) &
                (supercell_indices[:, 1] >= y0) & (supercell_indices[:, 1] < y1) &
                (supercell_indices[:, 2] >= z0) & (supercell_indices[:, 2] < z1)
            )
            '''
            mask_bool = (
                (supercell_positions[:, 0] >= x0) & (supercell_positions[:, 0] < x1) &
                (supercell_positions[:, 1] >= y0) & (supercell_positions[:, 1] < y1) &
                (supercell_positions[:, 2] >= z0) & (supercell_positions[:, 2] < z1)
            )
            


        elif shape_mask == 'combo':
            # rectangle
            mask_rect = torch.zeros((n1, n2, n3), device=device)
            z_start, y_start, x_start = start_coord
            z_end, y_end, x_end = end_coord

            z0, z1 = max(0, z_start), min(n1, z_end)
            y0, y1 = max(0, y_start), min(n2, y_end)
            x0, x1 = max(0, x_start), min(n3, x_end)

            mask_rect[z0:z1, y0:y1, x0:x1] += 1


            # circle
            center_i = (n1 - 1) / 2
            center_j = (n2 - 1) / 2
            center_k = (n3 - 1) / 2


            # Compute distance from center for each grid point
            dist_from_center = torch.sqrt((i - center_i) ** 2 + (j - center_j) ** 2 + (k - center_k) ** 2)

            

            mask1 = (dist_from_center < r_out).to(dtype=dtype)  # [n1, n2, n3]

            mask_tot = mask1 + mask_rect
            mask_no_zero = mask_tot[mask_tot != 0]
            mask_bool = mask_no_zero.bool().flatten().unsqueeze(-1)
            '''
            center_i = (n1 - 1) / 2 / d1
            center_j = (n2 - 1) / 2 / d2
            center_k = (n3 - 1) / 2 / d3
            dist_from_center = torch.sqrt((i - center_i) ** 2 +
                                          (j - center_j) ** 2 +
                                          (k - center_k) ** 2)
            mask_circle = (dist_from_center < r_out)

            mask_bool = (mask_rect | mask_circle).flatten().unsqueeze(-1)
            '''

        elif shape_mask == 'chill': ## add a shape check and if it's at supercell res you dont have to reshape
            mask_bool = chill 
            if chill.shape[-1] == n1 * n2 * n3:
                s1, s2, s3, nt = chill.shape
                mask_bool = mask_bool.view(-1,n1,n2,n3) #shape is (init_cond * time * channel, n1, n2, n3)
                b = mask_bool.shape[0]
                mask_bool = mask_bool.reshape(
                    b,
                    n1 // d1, d1,
                    n2 // d2, d2,
                    n3 // d3, d3)
            
                mask_bool = mask_bool.mean(dim=(2,4,6))  # Might not work well for bool. (batch, channel, nx/dx, ny/dy, nz/dz)
                mask_bool = mask_bool.view(b, -1).to(device) #(init_cond * time * channel, nx/dx * ny/dy * nz/dz)
            else:
                s1, s2, s3, nt = chill.shape
                mask_bool = mask_bool.view(-1,n1//d1,n2//d2,n3//d3) #shape is (init_cond * time * channel, n1, n2, n3)
                b = mask_bool.shape[0]
                mask_bool = mask_bool.view(b, -1).to(device) #(init_cond * time * channel, nx/dx * ny/dy * nz/dz)

            

        else:
            mask_bool = torch.ones((n1 * n2 * n3,), dtype=torch.bool, device=device).unsqueeze(-1)

        return mask_bool



    def calculate_structure_factor(self) -> Tensor:


            # Store original shape of q_vectors for later reshaping

        batch_size_original = self.q_vectors.shape[:-1]

            # Reshape q_vectors to [batch_size, 3] for matrix multiplication

        q_vectors_flat = self.q_vectors.reshape(-1, 3)

        batch_size = q_vectors_flat.shape[0]

            # Calculate q·r for each atom in the unit cell and each q-vector

            # q_vectors_flat shape: [batch_size, 3]

            # atom_positions shape: [n_atoms, 3]

            # result shape: [batch_size, n_atoms]
        
        q_dot_r = torch.matmul(q_vectors_flat, self.crystal.atom_positions.T)

            # Calculate |q| for form factor (convert to Å for typical form factor formulas)

        q_magnitude = torch.norm(q_vectors_flat, dim=1, keepdim=True) * 1e-10  # [batch_size, 1]

 
            # Vectorized calculation of form factors for all atoms and all q values

        form_factors = self.crystal.calculate_form_factors(q_magnitude)

            # Calculate e^(-iq·r) for each atom and each q-vector

            # Result shape: [batch_size, n_atoms]

        phase_factors = torch.exp(-1j * q_dot_r)

            # Multiply by form factors and sum over atoms

            # result shape: [batch_size]

        structure_factor = torch.sum(form_factors * phase_factors, dim=1)

            # Reshape back to original q_vectors shape

        structure_factor = structure_factor.view(batch_size_original)

        return(structure_factor)

    def R_sum(self,d, shape_mask = None, r_out = None, start_coord = None, end_coord = None, chill=None):  # d and n should b tuples with 3 values

        
        d1, d2, d3 = d
        n1, n2, n3 = self.crystal.crystal_size

            # Generate supercell indices

        i_indices = torch.arange(0, n1, d1, dtype=dtype, device=device)
        j_indices = torch.arange(0, n2, d2, dtype=dtype, device=device)
        k_indices = torch.arange(0, n3, d3, dtype=dtype, device=device)

            # Compute a grid of all indices

        i, j, k = torch.meshgrid(i_indices, j_indices, k_indices, indexing='xy')
 
            # Reshape to [n_supercells, 3]

        supercell_indices = torch.stack([i.flatten(), j.flatten(), k.flatten()], dim=1)

            # Calculate positions in real space --> this is R_nd

        supercell_positions = torch.matmul(supercell_indices, self.crystal.lattice_vectors) ## full grid of R_nd
      
            # Resizing q
    
        batch_size_original = self.q_vectors.shape[:-1]
        q_vectors_flat = self.q_vectors.reshape(-1, 3)
        batch_size = q_vectors_flat.shape[0]

            # Calculating dot_prod
        '''
        Adding masks for phase field mapping
        Chill is cahn hilliard

        '''
        if shape_mask == 'chill': #--> will need to change this to fit as well. Should just be the same final R value
            s1, s2, s3, nt = chill.shape
            
            mask_bool = self.bool_mask(d, shape_mask, chill = chill)
            r_vals = []
            for i in range(0, q_vectors_flat.shape[0], self.b_size):
                q_vec_batch = q_vectors_flat[i:i+self.b_size]
                #mask_batch = mask_bool[i:i+self.b_size]
                dot_prod_R = torch.matmul(q_vec_batch, supercell_positions.T)
            
                phase_R = torch.exp(-1j * dot_prod_R).unsqueeze(dim=0)  * mask_bool.unsqueeze(dim = 1)

                R_comp = torch.sum(phase_R, dim=2)
                #print(R_comp.shape)
                #d_0,d_1,_ = R_comp.shape
                d_2, d_3 = batch_size_original
                r_vals.append(R_comp)
            R_comp = torch.cat(r_vals, dim = 1)
            
            final_R = R_comp.view(s1,s2,s3,d_2,d_3)*d1*d2*d3 #(batch, time, channel, batch_size_original)
            return(final_R) 
        if shape_mask == None:
            r_vals = []
            for i in range(0, q_vectors_flat.shape[0], self.b_size):
                q_vec_batch = q_vectors_flat[i:i+self.b_size]
                dot_prod_R = torch.matmul(q_vec_batch, supercell_positions.T)

                # Finding the exponential component

                phase_R = torch.exp(-1j * dot_prod_R)

                R_comp = torch.sum(phase_R, dim=1)
                r_vals.append(R_comp)
            R_comp = torch.cat(r_vals, dim = 0)
            final_R = R_comp.view(batch_size_original)*d1*d2*d3

            return(final_R) 
        else:
            r_vals = []
            for i in range(0, q_vectors_flat.shape[0], self.b_size):
                mask_bool = self.bool_mask(d, shape_mask, r_out, start_coord, end_coord)
                q_vec_batch = q_vectors_flat[i:i+self.b_size]
                #mask_batch = mask_bool[i:i+self.b_size]
                dot_prod_R = torch.matmul(q_vec_batch, supercell_positions.T)

                # Finding the exponential component

                phase_R = torch.exp(-1j * dot_prod_R) * mask_bool.unsqueeze(dim = 0)

                R_comp = torch.sum(phase_R, dim=1)
                r_vals.append(R_comp)
            R_comp = torch.cat(r_vals, dim = 0)

            final_R = R_comp.view(batch_size_original)*d1*d2*d3

            return(final_R) 
        
    def global_phase_factor(self):
        # Add global position phase shift

        # Calculate q·R_g for each q-vector

        # grain_position shape: [3]

        # q_vectors_flat shape: [n_pixels, 3]

        # Result shape: [n_pixels]

        batch_size_original = self.q_vectors.shape[:-1]

        q_vectors_flat = self.q_vectors.reshape(-1, 3)

        q_dot_global_position = torch.matmul(q_vectors_flat, self.crystal.position)

        # Calculate e^(-iq·R_g) for each q-vector

        global_phase_factor = torch.exp(-1j * q_dot_global_position)

        return(global_phase_factor.view(batch_size_original))
 

    def get_intensity_chill(self,d,chill,shape_mask = 'chill', r_out = None, start_coord = None, end_coord = None):

        q = self.q_vectors

        A = self.R_sum(d, shape_mask, r_out, start_coord, end_coord, chill) *self.calculate_structure_factor().unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)*self.global_phase_factor().unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
        return A.abs()**2
         
    def get_intensity(self,d,shape_mask = None, r_out = None, start_coord = None, end_coord = None):

        q = self.q_vectors

        A = self.R_sum(d, shape_mask, r_out, start_coord, end_coord)*self.calculate_structure_factor()*self.global_phase_factor()
        return A.abs()**2
    def get_amplitude(self,d,shape_mask = None, r_out = None, start_coord = None, end_coord = None):

        q = self.q_vectors

        A = self.R_sum(d, shape_mask, r_out, start_coord, end_coord)*self.calculate_structure_factor()*self.global_phase_factor()
        return A
    def get_amplitude_chill(self,d,chill,shape_mask = 'chill', r_out = None, start_coord = None, end_coord = None):

        q = self.q_vectors

        A = self.R_sum(d, shape_mask, r_out, start_coord, end_coord, chill) *self.calculate_structure_factor().unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)*self.global_phase_factor().unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0)
        return A

    '''

    ## This max_intensity DOES NOT calculate R from scratch

    def max_intensity(self, R, d_xy, beta, rotated_structure):

        self.sample.calculate_sample_properties(rotated_structure) 

        N = self.sample.n_cells

        return(self.calculate_structure_factor()**2 * N**2 / R**2)

    ## The max_intensity function below calculates R from scratch, this should be used if we do not have a defined R

   '''     

    def max_intensity(self, rotated_structure):

        R = self.detector.R
        
        self.crystal.calculate_crystal_properties(rotated_structure) 

        N = self.crystal.n_cells

        S = self.calculate_structure_factor()
        
        return(S.abs()**2 * N**2 / R**2)

