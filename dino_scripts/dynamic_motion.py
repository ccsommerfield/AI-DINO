import torch


class Dynamic_Motion:
    def __init__(self, crystal, device, index, dtype):
        self.crystal = crystal
        self.device = device
        self.index = index
        self.dtype = dtype
    def solve(self, model, args, d, tf, dt):
        r_m = self.crystal.atom_positions
        n_atoms = r_m.shape[0]
        nx, ny, nz = self.crystal.crystal_size
        dx, dy, dz = d
        
        vol = nx//dx * ny//dy * nz//dz
        N = vol * n_atoms

        ode = model(N, args, self.crystal, method='dopri5', default_type=self.dtype).to(self.device)


        y0= 2*torch.rand(vol * n_atoms,4, device = self.device)-1
        t = torch.arange(0, tf, dt)
        y = ode.solve(t, y0, device=self.device, rtol=1e-6, atol=1e-6)

        return((y.view(len(t), vol ,n_atoms,4))) #shape [#steps, sup_vol, n_Atoms, 4]

    def manual_input(self, disp, d, index):
        '''
        disp should be of shape [batch_size, channels, nx, ny, nz, 3] 
        '''
        batch_t, channel, nx, ny, nz, _ = disp.shape
        dx, dy, dz = d

        r_m = self.crystal.atom_positions
        n_atoms = r_m.shape[0]
        
        disp_sq = disp.squeeze(1)

        mask = torch.zeros(n_atoms, device=disp_sq.device)
        mask[self.index] = 1

        # Reshape mask so it can broadcast over disp_sq
        mask = mask.view(1, 1, 1, 1, n_atoms, 1)   # [1,1,1,1,n_atoms,1]

        u = disp_sq.unsqueeze(4) * mask

        u = u.view(
            batch_t,
            nx // dx, dx,
            ny // dy, dy,
            nz // dz, dz,
            n_atoms, 3
        )
        u_avg = u.mean(dim=(2, 4, 6))                                                       # [t,sx, sy, sz, n_atoms, 3]
        u_avg = u_avg.reshape(batch_t,-1, n_atoms, 3)                                               # [t,n_supercells, n_atoms, 3]

        return(u_avg)
        
## OK, so we want to replace   u_avg = [sx, sy, sz, n_atoms, 3] or u_avg = [n_supercells, n_atoms, 3] instead of u_full