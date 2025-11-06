from config import torch, np, dtype, device
import os
import json
from typing import Tuple, List
from torch import Tensor
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.transformations.standard_transformations import RotationTransformation

class Crystal:
    """
    Class representing a crystal sample.
    """
    def __init__(
        self,
        cif_path: str,
        crystal_size: Tuple[int, int, int],
        dtype: torch.dtype = torch.float32,
        device: str = 'mps'
    ):
        """
        Initialize a Sample object.
        Parameters:
        -----------
        cif_path: str
            Path to cif file
        crystal_size: tuple
            Tuple of integers (n1, n2, n3) specifying crystal size in unit cells
        dtype: torch.dtype
            torch data type
        device: str
            torch device ('cuda' or 'cpu')
        """
        self.cif_path = cif_path
        self.crystal_size = crystal_size
        self.dtype = dtype
        self.device = device

        # Parse cif file
        self._parse_cif_file()
        
        # Parse atom types
        try:
            # If oxidation state is given, atom type is stored in element field of each specie
            self.atom_types = list(map(str, map(lambda x: x.element, self.structure.species)))
        except:
            # No oxidation state given
            self.atom_types = list(map(str, self.structure.species))

        # Get atomic form factors
        self.atomic_form_factors = self._load_atomic_form_factors()
        self._get_atomic_form_factor_coefficients()

        # Set global position (default is origin)
        self._position = torch.zeros(3, dtype=self.dtype, device=self.device)

    def _parse_cif_file(self):
        """
        Parse the CIF file and extract the structure object.
        """
        parser = CifParser(self.cif_path)
        self.structure = parser.parse_structures(primitive=False)[0]

    def _load_atomic_form_factors(self):
        """
        Load atomic form factor coefficients from JSON file.
        
        Parameters:
        -----------
        filename : str
            Path to the JSON file containing atomic form factor coefficients
            
        Returns:
        --------
        dict : Dictionary with element symbols as keys and coefficients as list of values
               Each coefficient list contains [a1, b1, a2, b2, a3, b3, a4, b4, c]
        """

        #with open(os.path.join(os.path.dirname(__file__), 'resources/atomic_form_factors.json'), 'r') as f:
            #data = json.load(f)
        file_path = os.path.join(os.getcwd(), "resources", "atomic_form_factors.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def _get_atomic_form_factor_coefficients(self):
        """
        Extract the coefficients of the atomic form factors for each atom in the structure.
        """
        
        self.coeff_a = torch.zeros((1, self.n_atoms, 4), dtype=self.dtype, device=self.device)
        self.coeff_b = torch.zeros((1, self.n_atoms, 4), dtype=self.dtype, device=self.device)
        self.coeff_c = torch.zeros((1, self.n_atoms), dtype=self.dtype, device=self.device)
        
        for i, atom_type in enumerate(self.atom_types):
            self.coeff_a[:,i] = torch.tensor(self.atomic_form_factors[atom_type][0:-1:2], dtype=self.dtype, device=self.device)
            self.coeff_b[:,i] = torch.tensor(self.atomic_form_factors[atom_type][1:-1:2], dtype=self.dtype, device=self.device)
            self.coeff_c[:,i] = torch.tensor(self.atomic_form_factors[atom_type][-1], dtype=self.dtype, device=self.device)

    def calculate_form_factors(self, q_magnitude: Tensor) -> Tensor:
        """
        Calculate the approximate atomic form factors as a sum of gaussians for each atom over the given q values.
        
        Parameters:
        -----------
        q_magnitude : torch.Tensor
            q vector magnitude as a tensor of shape [batch_size, 1]

        Returns:
        --------
        torch.Tensor:
            Approximated form factor amplitudes of shape [batch_size, n_atoms]
        """
        
        result = self.coeff_a.view(1,-1) * torch.exp(-self.coeff_b.view(1,-1) * (q_magnitude / (4 * torch.pi)) ** 2)
        
        return result.view(-1, self.n_atoms, 4).sum(dim=-1) + self.coeff_c

    @property
    def position(self) -> Tensor:
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @position.deleter
    def position(self):
        del self._position
    
    @property
    def lattice_vectors(self) -> Tensor:
        return torch.tensor(self.structure.lattice.matrix * 1e-10, dtype=self.dtype, device=self.device)

    @property
    def atom_positions(self) -> Tensor:
        return torch.tensor(self.structure.cart_coords * 1e-10, dtype=self.dtype, device=self.device)

    @property
    def n_atoms(self) -> int:
        return len(self.atom_types)
        
    @property
    def n_cells(self) -> int:
        return self.crystal_size[0] * self.crystal_size[1] * self.crystal_size[2]

    @property
    def cell_volume(self) -> float:
        return torch.det(self.lattice_vectors).abs()

    @property
    def crystal_volume(self) -> float:
        return self.cell_volume * self.n_cells
            

    '''
    def calculate_crystal_properties(self, structure):
        self.lattice_vectors = torch.tensor(structure.lattice.matrix * 1e-10, dtype=self.dtype, device=self.device)
        self.atom_positions = torch.tensor(structure.cart_coords * 1e-10, dtype=self.dtype, device=self.device)
        try: self.atom_types = list(map(str, map(lambda x: x.element, structure.species)))
        except: self.atom_types = list(map(str, structure.species))
        self.n_atoms = len(self.atom_types)
        self.n_cells = self.crystal_size[0] * self.crystal_size[1] * self.crystal_size[2]
        self.cell_volume = torch.det(self.lattice_vectors).abs()
        self.sample_volume = self.cell_volume * self.n_cells
    '''

    def unit_cell_lattice(self, structure): #finish
        lattice_vectors = torch.tensor(structure.lattice.matrix * 1e-10, dtype=self.dtype, device=self.device)
        
        n1, n2, n3 = self.crystal_size

        
        i_indices = torch.arange(0, n1, 1, dtype=dtype, device=device)
        j_indices = torch.arange(0, n2, 1, dtype=dtype, device=device)
        k_indices = torch.arange(0, n3, 1, dtype=dtype, device=device)

            # Compute a grid of all indices

        i, j, k = torch.meshgrid(i_indices, j_indices, k_indices, indexing='xy')
 
            # Reshape to [n_unitcells, 3]

        unitcell_indices = torch.stack([i.flatten(), j.flatten(), k.flatten()], dim=1)

            # Calculate positions in real space --> this is R_n

        unitcell_positions = torch.matmul(unitcell_indices, lattice_vectors) ## full grid of R_nd


        atom_pos_unit_cell = self.crystal.atom_positions #initial atomic positions (Ask) I dont think this is right
        shape = atom_pos_unit_cell.shape

        total_pos = unitcell_pos[:,none,:] + atom_pos_unit_cell[none,:,:]

        return(total_pos.view(n1,n2,n3,shape))

    def align_miller_plane_to_axis(self, miller_indices, target_axis='x'):
        """
        Rotate a crystal structure so that the normal to a specified Miller plane
        is aligned with a target axis.
        
        Parameters:
        -----------
        miller_indices : tuple
            Miller indices (h, k, l) of the plane
        target_axis : str
            Target axis ('x', 'y', or 'z')
        """
        
        # Define target direction vectors
        target_directions = {
            'x': np.array([1, 0, 0]),
            'y': np.array([0, 1, 0]),
            'z': np.array([0, 0, 1])
        }
        
        if target_axis not in target_directions:
            raise ValueError("target_axis must be 'x', 'y', or 'z'")
        
        target_direction = target_directions[target_axis]

        # Save the original structure
        if not hasattr(self, 'original_structure'):
            self.original_structure = self.structure.copy()
        structure = self.original_structure.copy()
        
        # Get the reciprocal lattice
        reciprocal_lattice = structure.lattice.reciprocal_lattice
        
        # Calculate the normal vector to the Miller plane
        # Normal to (h,k,l) plane = h*a* + k*b* + l*c* (reciprocal lattice vectors)
        h, k, l = miller_indices
        miller_normal = (h * reciprocal_lattice.matrix[0] + 
                         k * reciprocal_lattice.matrix[1] + 
                         l * reciprocal_lattice.matrix[2])
        
        # Normalize the normal vector
        miller_normal = miller_normal / np.linalg.norm(miller_normal)
        
        # Calculate the rotation axis (cross product)
        rotation_axis = np.linalg.cross(miller_normal, target_direction)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-10:  # Already aligned or anti-aligned
            if np.dot(miller_normal, target_direction) > 0:
                # Already aligned, no rotation needed
                return
            else:
                # Anti-aligned, need 180° rotation
                # Choose a perpendicular axis for 180° rotation
                rotation_axis = np.roll(target_direction, shift = 1)
                rotation_angle = np.pi
        else:
            rotation_axis = rotation_axis / rotation_axis_norm
            
            # Calculate rotation angle
            rotation_angle = np.arccos(np.clip(np.dot(miller_normal, target_direction), -1.0, 1.0))
        
        # Apply rotation transformation
        transformation = RotationTransformation(rotation_axis, rotation_angle, angle_in_radians=True)
        rotated_structure = transformation.apply_transformation(structure)

        self.structure = rotated_structure


    def misalign_about_axis(self, rotation_angle=0., rotation_axis='y'):
        """
        Rotate a crystal structure to misalign it.
        
        Parameters:
        -----------
        rotation angle : float
            Angle in degrees by which to misalign the structure
        rotation_axis : str
            Rotation axis ('x', 'y', or 'z')
        """

        # Define rotation axis vectors
        rotation_axis_directions = {
            'x': np.array([1, 0, 0]),
            'y': np.array([0, 1, 0]),
            'z': np.array([0, 0, 1])
        }

        rotation_axis = rotation_axis_directions[rotation_axis]
        
        # Apply rotation transformation
        transformation = RotationTransformation(rotation_axis, rotation_angle, angle_in_radians=False)
        self.structure = transformation.apply_transformation(self.structure)
    '''

    def _load_atomic_form_factors(self):
        """
        Load atomic form factor coefficients from JSON file.
        
        Parameters:
        -----------
        filename : str
            Path to the JSON file containing atomic form factor coefficients
            
        Returns:
        --------
        dict : Dictionary with element symbols as keys and coefficients as list of values
               Each coefficient list contains [a1, b1, a2, b2, a3, b3, a4, b4, c]
        """
        with open('atomic_form_factors.json', 'r') as f:
            data = json.load(f)
        
        return data
        
    def get_atomic_form_factors(self) -> List[Tensor]:
        self.atomic_form_factors = {
            'O': [3.0485, 13.2771, 2.2868, 5.7011, 1.5463, 0.3239, 0.867, 32.9089, 0.2508],
            'Zn': [14.0743, 3.2655, 7.0318, 0.2333, 5.1652, 10.3163, 2.41, 58.7097, 1.3041],
            'Ba': [20.3361, 3.216, 19.297, 0.2756, 10.888, 20.2073, 2.6959, 167.202, 2.7731],
            'Ti': [9.7595, 7.8508, 7.3558, 0.5, 1.6991, 35.6338, 1.9021, 116.105, 1.2807],
            'Na': [4.7626, 3.285, 3.1736, 8.8422, 1.2674, 0.3136, 1.1128, 129.424, 0.676],
            'B':[2.0545, 23.2185, 1.3326, 1.021, 1.0979, 60.3498, 0.7068, 0.1403, -0.1932],
            'H':[0.489918, 20.6593, 0.262003, 7.74039, 0.196767, 49.5519, 0.049879, 2.20159, 0.001305]
        }
 
        self.coeff_a = torch.zeros((1, self.n_atoms, 4), dtype=self.dtype, device=self.device)
        self.coeff_b = torch.zeros((1, self.n_atoms, 4), dtype=self.dtype, device=self.device)
        self.coeff_c = torch.zeros((1, self.n_atoms), dtype=self.dtype, device=self.device)
        for i, atom_type in enumerate(self.atom_types):
            self.coeff_a[:,i] = torch.tensor(self.atomic_form_factors[atom_type][0:-1:2], dtype=self.dtype, device=self.device)
            self.coeff_b[:,i] = torch.tensor(self.atomic_form_factors[atom_type][1:-1:2], dtype=self.dtype, device=self.device)
            self.coeff_c[:,i] = torch.tensor(self.atomic_form_factors[atom_type][-1], dtype=self.dtype, device=self.device)
    def calculate_form_factors(self, q_magnitude: Tensor) -> Tensor:
        gaussians = self.coeff_a.view(1,-1) * torch.exp(-self.coeff_b.view(1,-1) * (q_magnitude / (4 * torch.pi)) ** 2)
        return gaussians.view(-1, self.n_atoms, 4).sum(dim=-1) + self.coeff_c
    '''

    def get_XRD(self, structure, xrd_wavelength='CuKa'):
        xrd_calc = XRDCalculator(wavelength=xrd_wavelength)
        pattern = xrd_calc.show_plot(structure)
        val = xrd_calc.get_pattern(structure)
        return pattern, val

    def get_dislocations(self,scale):
        #scale should be size in m
        u = torch.randn(self.crystal_size + self.atom_positions.shape, device = device) * scale ##this creates a dislocation field
        return u
    
    def apply_alternating_dislocation(self, scale: float):
        """
        Apply alternating dislocations along the (1,2) component of the displacement field.

        Parameters:
        -----------
        scale : float
            Magnitude of the displacement to apply
        """
        u = torch.zeros_like(self.get_dislocations(scale))
        nx, ny, nz, _, _ = self.get_dislocations(scale).shape

        # Create meshgrid of indices
        i, j, k = torch.meshgrid(
            torch.arange(nx, device=u.device),
            torch.arange(ny, device=u.device),
            torch.arange(nz, device=u.device),
            indexing="ij"
        )

    
        
        # Apply displacement
        u[:, :, :, 1, 0] = torch.where(i>j, scale, -scale)
        u[:, :, :, 1, 0] = torch.where(i==j, 0, u[:,:,:,1,0])
        
        '''
        parity = (i + j + k) % 2 == 1  # shape [nx, ny, nz]

        # Expand to [nx, ny, nz, n_atoms]
        parity = parity[:, :, :, None].expand(-1, -1, -1, u.shape[3])

        # Apply displacements in the x-direction (index 0)
        u[:, :, :, :, 0] = torch.where(parity, scale, -scale)
        '''
        return u

    def animate_displacement(self, w, t, scale):
        return scale * torch.cos(w*t) + self.get_dislocations(scale)
        #start with a cos function that evolves with time
        #loop simulation over the time

    def scale_displacement(self, w, t, scale):
        return torch.cos(w*t) * self.apply_alternating_dislocation(scale)

    def grain_shape(self, structure, r_out):
        #return mask for shape
        lattice_vectors = torch.tensor(structure.lattice.matrix * 1e-10, dtype=self.dtype, device=self.device)
        
        n1, n2, n3 = self.crystal_size

        
        i_indices = torch.arange(0, n1, 1, dtype=dtype, device=device)
        j_indices = torch.arange(0, n2, 1, dtype=dtype, device=device)
        k_indices = torch.arange(0, n3, 1, dtype=dtype, device=device)

            # Compute a grid of all indices

        i, j, k = torch.meshgrid(i_indices, j_indices, k_indices, indexing='xy')

        center_i = n1 / 2
        center_j = n2 / 2
        center_k = n3 / 2

        dist_from_center = torch.sqrt((i - center_i)**2 + (j - center_j)**2 + (k - center_k)**2)

        mask = torch.ones((n1, n2, n3), dtype=dtype, device=device)

        mask[dist_from_center >= r_out] = 0
        
        

 
            # Reshape to [n_unitcells, 3]

        mask_flat = torch.stack([i.flatten(), j.flatten(), k.flatten()], dim=1)

        return mask_flat
