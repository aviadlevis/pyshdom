"""
A medium object used for atmospheric rendering and inversion.
"""

import core
import numpy as np
from enum import Enum
import warnings
from shdom import Grid, BoundingBox, ScalarField


class Phase(object):
    """
    TODO: add documentation

    Parameters
    ----------
    
    Notes
    -----
    """
    def __init__(self):
        pass
    

class Medium(object):
    """
    TODO: add documentation

    Parameters
    ----------
    extinction: ScalarField
    phase: VectorField
    
    Notes
    -----
    Different grids for phase and extinction is not supported.
    """
    def __init__(self, extinction, phase):
        self._ext = extinction
        self._phase = phase
        
        assert extinction.grid == phase.grid, 'Different grids for phase and extinction is not supported.'
        self._grid = extinction.grid
        
    @property
    def extinction(self):
        return self._ext
    
    @property
    def phase(self):
        return self._phase    
        
    @property
    def grid(self):
        return self._grid
    

def load_les_from_csv(path_to_csv):
    """ 
    A utility function to load Large Eddy Simulated clouds.
    
    Parameters
    ----------
    path_to_csv: str
         Path to file. 

    Returns
    -------
    lwc: ScalarField
         a ScalarField object contatining the liquid water content of the LES cloud.
    reff: ScalarField
          a ScalarField object contatining the effective radius of the LES cloud.
    
    Notes
    -----
    CSV format should be as follows:
    
    #name=name of les file
    #original_cloud_data=path to original 
    #resampled_cloud_data_grid_size=grid resolution in meters
    nx ny nz
    dz dy dz     z_levels[0]     z_levels[1] ...  z_levels[nz-1]
    ix iy iz     lwc[ix, iy, iz]    reff[ix, iy, iz]
    .
    .
    .
    ix iy iz     lwc[ix, iy, iz]    reff[ix, iy, iz]
    """ 
    
    nx, ny, nz = np.genfromtxt(path_to_csv, max_rows=1, dtype=int) 
    dx, dy = np.genfromtxt(path_to_csv, max_rows=1, usecols=(0, 1), dtype=float, skip_header=4)
    z_levels = np.genfromtxt(path_to_csv, max_rows=1, usecols=range(2, 2 + nz), dtype=float, skip_header=4)
    grid_index = np.genfromtxt(path_to_csv, usecols=(0, 1, 2), dtype=int, skip_header=5)
    lwc = np.genfromtxt(path_to_csv, usecols=3, dtype=float, skip_header=5)
    reff = np.genfromtxt(path_to_csv, usecols=4, dtype=float, skip_header=5)
    
    particle_levels = np.array([z in grid_index[:, 2] for z in range(nz)], dtype=int)
    lwc_3d  = np.zeros(shape=(nx, ny, nz), dtype=np.float32)
    reff_3d = np.zeros(shape=(nx, ny, nz), dtype=np.float32)
    lwc_3d[grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]]  = lwc
    reff_3d[grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]] = reff
    
    bounding_box = BoundingBox(xmin=0.0, 
                               ymin=0.0, 
                               zmin=z_levels.min(), 
                               xmax=(nx - 1) * dx, 
                               ymax=(ny - 1) * dy, 
                               zmax=z_levels.max())
    
    grid = Grid(bounding_box, nx, ny, nz, z_levels)
    return ScalarField(grid, lwc_3d), ScalarField(grid, reff_3d)