from config import np, torch
from typing import Tuple, List, Union
from crystal import Crystal


class Polycrystal:
    """Collection of crystals (grains) that scatter coherently"""
    
    def __init__(
        self,
        cif_path: str,
        bounding_box: Tuple[float, float, float],
        crystal_sizes: Union[Tuple[int, int, int], List[Tuple[int, int, int]]],
        crystal_misorientations: List[Tuple[float, str]],
        miller_indices: Tuple[int, int, int],
        target_axis: str ='x',
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda'
    ):
        """
        Initialize a Polycrystal object.
        
        Parameters:
        -----------
        cif_path: str
            Path to cif file
        bounding_box: tuple
            Tuple of floats delimiting the bounding box for all crystallites, in meters
        crystal_sizes: list
            Tuple of integers or list of tuples of integers (n1, n2, n3) specifying crystal sizes in unit cells.
            If tuple, all crystals will be considered the same size.
        miller_indices : tuple
            Miller indices (h, k, l) of the plane
        crystal_misorientations: list
            List of tuples of (angle, axis) specifying the rotation angle and axis by which to misalign crystal
        target_axis : str
            Target axis ('x', 'y', or 'z')
        dtype: torch.dtype
            torch data type
        device: str
            torch device ('cuda' or 'cpu')
        """

        self.bounding_box = torch.tensor(bounding_box, dtype=dtype, device=device)

        if isinstance(crystal_sizes, tuple):
            crystal_sizes = [crystal_sizes] * len(crystal_misorientations)
            
        self.grains = []
        for crystal_size, rotation in zip(crystal_sizes, crystal_misorientations):
            # Create crystal object
            self.grains.append(Crystal(cif_path, crystal_size, dtype, device))

            # Assign random global position within bounding box
            self.grains[-1].position = self.bounding_box * torch.rand(3, dtype=dtype, device=device)
            
            # Align Miller plane to axis
            self.grains[-1].align_miller_plane_to_axis(miller_indices, target_axis=target_axis)

            # Add slight misalignment
            self.grains[-1].misalign_about_axis(rotation_angle=rotation[0], rotation_axis=rotation[1])

    @property
    def n_grains(self) -> int:
        return len(self.grains)