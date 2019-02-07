"""
Spherical Harmonic Discrete Ordinate Method for 3D Atmospheric Radiative Transfer.
Here all the necessary functions used to solve the RT using SHDOM are wrapped.
"""

import core
import numpy as np
from enum import Enum
import warnings
import sys


class BoundaryCondition(Enum):
    open = 1       # open boundary conditions mean that exiting radiance is lost.
    periodic = 2   # periodic boundary conditions mean that exiting radiance returns from the opposite side.
    

class PhaseFunctionType(Enum):
    # TODO: Add documentation
    tabulated = 1 
    
    
class SolarSource(object):
    """ 
    Solar source object.
    
    Parameters
    ----------
    flux: float
        Top of medium solar flux on a horizontal surface (any units).
        For k-distribution this is a multiplier for the solar flux in the CKD file (i.e. normally should be 1).
    azimuth: float,
        Solar beam azimuthal angle (photon direction); specified in degrees but immediately converted to radians for use in code. 
        0 is beam going in positive X direction (North), 90 is positive Y (East).
    zenith: 
        Solar beam zenith angle; Specified in degrees but immediately converted to the cosine of the angle (mu).
        This angle represents the direction of travel of the solar beam, so is forced to be negative although it can be specified positive.
    
    Notes
    -----
    Paramters: skyrad, units, wavenumber, are used for Thermal source and are not supported. 
    """
    def __init__(self, flux, azimuth, zenith):
        self.flux = flux
        self.azimuth = azimuth
        self.zenith = zenith
        self.units = 'R'
        self.skyrad = 0.0
        self.wavenumber = [10000, 10001]
        
    
class Surface(object):
    """ 
    An abstract sufrace class to be inhirted by different surface types.
    """
    def __init__(self):
        self.type = 'AbstractSurfaceClass'
       
        
class LambertianSurface(Surface):
    """
    A Lambertian surface, defined by a single parameter.
    
    Parameters
    ----------
    albedo: float, optional
        Bottom surface Lambertian albedo the range [0, 1] where 0 means completly absorbing.
        
    Notes
    -----
    Parameter: ground_temperature, is used for Thermal source and is not supported. 
    """
    def __init__(self, albedo):
        self.type = 'Lambertian'
        self.albedo = albedo
        self.ground_temperature = 298.15
        
class NumericalParameters(object):
    """
    TODO: add description

    Parameters
    ----------
    """    
    def __init__(self):
        self.maxnewphase = 200
        self.phase_function_type = PhaseFunctionType.tabulated
        
        # TODO: check these values
        self.asymetric_tolerance = 0.1
        self.frational_phase_tolerance = 0.1
        self.num_mu_bins = 8
        self.num_phi_bins = 16
        self.split_accuracy = 0.1
        self.delta_m = True
        self.spherical_harmonics_accuracy = 0.003
        self.solution_accuracy = 0.0001
        self.acceleration_flag = True
        self.max_total_mb = 100000.0
        self.adapt_grid_factor = 5
        self.num_sh_term_factor = 5
        self.cell_to_point_ratio = 1.5


class SceneParameters(object):
    """
    TODO: add description

    Parameters
    ----------
    """    
    def __init__(self): 
        self.wavelength = 0.670
        self.surface = LambertianSurface(albedo=0.05)
        self.source = SolarSource(flux=np.pi, azimuth=0.0, zenith=180.0)
        self.boundary_conditions = {'x': BoundaryCondition.open, 
                                    'y': BoundaryCondition.open}

        
class RteSolver(object):
    """
    TODO: add documentation

    Parameters
    ----------
    scene_params: SceneParameters
    numerical_params: NumericalParameters
    
    Notes
    -----
    k-distribution not supported.
    """
    
    def __init__(self, scene_params, numerical_params):

        # Start mpi (if available).
        self._masterproc = core.start_mpi()

        # Link to the properties array module.
        self._pa = core.shdom_property_arrays

        # Assign scene parameters to shdom internal parameters.
        self.set_scene_parameters(scene_params)
        
        # Assign numerical parameters to shdom internal parameters.
        self.set_numerical_parameters(numerical_params)
        
        
    def __del__(self):
        
        try:
            core.end_shdom_mpi(
                gridpos=self._gridpos,
                npx=self._pa.npx,
                npy=self._pa.npy,
                xstart=self._pa.xstart,
                ystart=self._pa.ystart,
                delx=self._pa.delx,
                dely=self._pa.dely,
                npxt=self._npxt,
                npyt=self._npyt,
                propfile=self._propfile
            )
        except Exception as e:
            warnings.warn(repr(e))

    def set_scene_parameters(self, scene_params):
        """
        Set the scene related parameters: 
          wavelength, source, surface, boundary conditions, k-distribution 
    
        Parameters
        ----------
        scene_params : SceneParameters
        
        Returns
        -------
        None
    
        Notes
        -----
        k-distribution not supported.
    
        """  
        
        # Wavelength
        self._wavelen = scene_params.wavelength
        
        # Source parameters
        self._solarflux = scene_params.source.flux
        self._solarmu = np.cos(np.deg2rad(scene_params.source.zenith))
        self._solaraz = np.deg2rad(scene_params.source.azimuth)
        self._skyrad = scene_params.source.skyrad
        self._units = scene_params.source.units
        self._waveno = scene_params.source.wavenumber
        
        # Surface parameters
        self._maxsfcpars = 4
        if scene_params.surface.type == 'Lambertian':
            self._sfctype = 'FL'  # Fixed Lambertian
            self._gndalbedo = scene_params.surface.albedo
            self._gndtemp = scene_params.surface.ground_temperature
            self._nxsfc = 0
            self._nysfc = 0
            self._delxsfc = 0
            self._delysfc = 0
            self._nsfcpar = 1
            self._sfcparms = []
            self._sfcgridparms = []  
        else:
            raise NotImplementedError('Only Lambertian surface is currently supported')
    
        # Boundary conditions
        self._bcflag = 0
        if scene_params.boundary_conditions['x'] == BoundaryCondition.open:
            self._bcflag += 1
        if scene_params.boundary_conditions['y'] == BoundaryCondition.open:
            self._bcflag += 2
        
        # TODO: Independent pixel not supported yet
        self._ipflag = 0
        
        # TODO: k-distribution not supported yet
        self._kdist = False
        self._ng = 1
        self._delg = np.ones(1, order='F')
        self._pa.nzckd = 0
        self._baseout = False


    def set_numerical_parameters(self, numerical_params):
        """
        Set the numerical parameters of the SHDOM forward solver.
    
        Parameters
        ----------
        numerical_params : NumericalParameters

        Returns
        -------
        None
    
        Notes
        -----
        None
        """ 
        self._maxnewphase         = numerical_params.maxnewphase
        self._proptype            = numerical_params.phase_function_type
        self._asymtol             = numerical_params.asymetric_tolerance   
        self._fracphasetol        = numerical_params.frational_phase_tolerance
        self._nmu                 = max(2, 2 * int((numerical_params.num_mu_bins + 1) / 2))
        self._nphi                = max(1, numerical_params.num_phi_bins)
        self._deltam              = numerical_params.delta_m
        self._max_total_mb        = numerical_params.max_total_mb
        self._adapt_grid_factor   = numerical_params.adapt_grid_factor
        self._accelflag           = numerical_params.acceleration_flag
        self._num_sh_term_factor  = numerical_params.num_sh_term_factor
        self._cell_to_point_ratio = numerical_params.cell_to_point_ratio
        self._splitacc            = numerical_params.split_accuracy
        self._shacc               = numerical_params.spherical_harmonics_accuracy
        self._solacc              = numerical_params.solution_accuracy
    
        
    def init_medium(self, medium):
        """
        Initilize SHDOM internal grid structures according to the Medium.
    
        Parameters
        ----------
        medium: Medium

        Returns
        -------
        None
    
        Notes
        -----
        """
   
        
        def ibits(val, bit, ret_val):
            if val & 2**bit:
                return ret_val
            return 0 
        
        grid = medium.grid
        
        # Set shdom property array
        self._pa.npx = grid.nx
        self._pa.npy = grid.ny    
        self._pa.npz = grid.nz
        self._pa.xstart = grid.bounding_box.xmin
        self._pa.ystart = grid.bounding_box.ymin
        self._pa.delx = grid.dx
        self._pa.dely = grid.dy
        self._pa.zlevels = grid.z_levels
        
        # Initialize shdom internal grid sizes to property array grid
        self._nx = self._pa.npx
        self._ny = self._pa.npy
        self._nz = self._pa.npz
        self._maxpg  = self._nx * self._ny * self._nz
        self._maxnz = self._pa.npz
        self._maxleg = medium.phase.depth
        
        # Set the full domain grid sizes (variables end in t)
        self._nxt, self._nyt, self._npxt, self._npyt = self._nx, self._ny, self._pa.npx, self._pa.npy
        self._nx, self._ny, self._nz = max(1, self._nx), max(1, self._ny), max(2, self._nz)
        
        # gridtype = 'P': Z levels taken from property file
        self._gridtype = 'P'        

        # Set up base grid point actual size (NX1xNY1xNZ)
        self._nx1, self._ny1 = self._nx+1, self._ny+1
        if self._bcflag & 5:
            self._nx1 -= 1
        if self._bcflag & 7:
            self._ny1 -= 1
        self._nbpts = self._nx1 * self._ny1 * self._nz 
        
        # Calculate the number of base grid cells depending on the BCFLAG
        self._nbcells = (self._nz - 1) * (self._nx + ibits(self._bcflag, 0, 1) - \
                        ibits(self._bcflag, 2, 1)) * (self._ny + ibits(self._bcflag, 1, 1) - ibits(self._bcflag, 3, 1))
        
        # Make ml and mm from nmu and nphi
        # ML is the maximum meridional mode, MM is the maximum azimuthal mode,
        # and NCS is the azimuthal mode flag (|NCS|=1 for cosine only, |NCS|=2 for 
        # sines and cosines).
        # nphi0max: The maximum number of azimuth angles actually used;
        # for NCS=1 (cosine modes only) NPHI0=INT((NPHI+2)/2),
        # otherwise NPHI0=NPHI.
        self._ml = self._nmu - 1
        self._mm = max(0, int(self._nphi / 2) - 1)
        self._nlm = (2 * self._mm + 1) * (self._ml + 1) - self._mm * (self._mm + 1)
        self._ncs = 2
        self._nphi0max = self._nphi
        self._nleg = self._ml
    
        if self._deltam:
            self._nleg += 1
    
        self._memword = self._nmu*(2 + 2 * self._nphi + 2 * self._nlm + 2 * 33 *32)
    
        # Guess maximum number of grid points, cells, SH vector size needed
        # but don't let MAX_TOTAL_MB be exceeded
        if self._max_total_mb*1024**2 > 1.75*sys.maxsize:
            self._max_total_mb > 1.75*sys.maxsize/1024**2 
            print 'MAX_TOTAL_MB reduced to fit memory model: ', self._max_total_mb
    
        if self._splitacc < 0.0:
            self._adapt_grid_factor = 1.0
    
        self._big_arrays = 3
        if not self._accelflag:
            self._big_arrays = 2        
    
        wantmem = self._adapt_grid_factor * self._nbpts * (
            28 + 16.5 * self._cell_to_point_ratio + self._nphi0max + self._num_sh_term_factor * self._nlm * self._big_arrays)                
    
        REDUCE = min(1.0, ((self._max_total_mb * 1024**2) / 4 - self._memword) / wantmem)
        self._adapt_grid_factor *= REDUCE
        assert self._adapt_grid_factor > 1.0, 'max_total_mb memory limit exceeded with just base grid.'
    
        if REDUCE < 1.0:
            print 'adapt_grid_factor reduced to ', self._adapt_grid_factor
    
        wantmem *= REDUCE
        assert wantmem < sys.maxsize, 'number of words of memory exceeds max integer size: %d'% wantmem
        self._maxig = int(self._adapt_grid_factor*self._nbpts)
        self._maxic = int(self._cell_to_point_ratio*self._maxig)
        self._maxiv = int(self._num_sh_term_factor*self._nlm*self._maxig)
        self._maxido = self._maxig*self._nphi0max
    
        if self._pa.numphase > 0:
            self._maxigl = self._pa.numphase*(self._maxleg+1)
        else:
            self._maxigl = self._maxig*(self._maxleg+1)
    
        assert 4.0*(self._maxiv+self._maxig) < sys.maxsize, 'size of big sh arrays (maxiv) probably exceeds max integer number of bytes: %d' % self._maxiv
        assert 4.0*8.0*self._maxic <= sys.maxsize, 'size of gridptr array (8*maxic) probably exceeds max integer number of bytes: %d' % 8*self._maxic
    
        self._maxnbc = int(self._maxig*3/self._nz)
        self._maxsfcpars = 4
        if self._sfctype.endswith('L'):
            self._maxbcrad = 2*self._maxnbc
        else:
            self._maxbcrad = int((2+self._nmu*self._nphi0max/2)*self._maxnbc)
    
        self._sfcgridparms = np.empty(self._maxsfcpars*self._maxnbc,dtype=np.float32)               
        self._bcptr = np.empty(shape=(self._maxnbc, 2), dtype=np.int32, order='F')
        self._bcrad = np.empty(shape=(self._maxbcrad,), dtype=np.float32, order='F')
    
        # Array allocation
        self._mu = np.empty(shape=(self._nmu,), dtype=np.float32, order='F')
        self._wtdo = np.empty(shape=(self._nmu*self._nphi,), dtype=np.float32, order='F')
        self._phi = np.empty(shape=(self._nmu*self._nphi,), dtype=np.float32, order='F')
        self._phi0 = np.empty(shape=(self._nmu,), dtype=np.int32, order='F')
        self._cmu1 = np.empty(shape=(self._nlm*self._nmu,), dtype=np.float32, order='F')
        self._cmu2 = np.empty(shape=(self._nlm*self._nmu,), dtype=np.float32, order='F')
        self._cphi1 = np.empty(shape=(33*32*self._nmu,), dtype=np.float32, order='F')
        self._cphi2 = np.empty(shape=(33*32*self._nmu,), dtype=np.float32, order='F')
        self._temp = np.empty(shape=(self._maxig,), dtype=np.float32, order='F')
        self._planck = np.empty(shape=(self._maxig,), dtype=np.float32, order='F')
        self._extinct = np.empty(shape=(self._maxig,), dtype=np.float32, order='F')
        self._albedo = np.empty(shape=(self._maxig,), dtype=np.float32, order='F')
        self._iphase = np.empty(shape=(self._maxig,), dtype=np.int32, order='F')
        self._legen = np.empty(shape=(self._maxigl,), dtype=np.float32, order='F')
        self._gridptr = np.empty(shape=(8, self._maxic), dtype=np.int32, order='F')
        self._neighptr = np.empty(shape=(6, self._maxic), dtype=np.int32, order='F')
        self._treeptr = np.empty(shape=(2, self._maxic), dtype=np.int32, order='F')
        self._cellflags = np.empty(shape=(self._maxic,), dtype=np.int16, order='F')
        self._gridpos = np.empty(shape=(3, self._maxig), dtype=np.float32, order='F')
        self._rshptr = np.empty(shape=(self._maxig+2,), dtype=np.int32, order='F')
        self._shptr = np.empty(shape=(self._maxig+1,), dtype=np.int32, order='F')
        self._oshptr = np.empty(shape=(self._maxig+1,), dtype=np.int32, order='F')
        self._radiance = np.empty(shape=(self._maxiv+self._maxig,), dtype=np.float32, order='F')
        self._source = np.empty(shape=(self._maxiv,), dtype=np.float32, order='F')
        self._delsource = np.empty(shape=(self._maxiv,), dtype=np.float32, order='F')
        self._fluxes = np.empty(shape=(2, self._maxig,), dtype=np.float32, order='F')
        self._fluxes1 = np.empty(shape=(4, self._maxig,), dtype=np.float32, order='F')
        self._dirflux = np.empty(shape=(self._maxig,), dtype=np.float32, order='F')
        self._pa.extdirp = np.empty(shape=(self._maxpg,), dtype=np.float32, order='F')
        self.work = np.empty(shape=(self._maxido,), dtype=np.float32, order='F')
        self.work1 = np.empty(shape=(8*self._maxig,), dtype=np.int32, order='F')
        self.work2 = np.empty(shape=(self._maxig,), dtype=np.float32, order='F')        
        
        self._xgrid, self._ygrid, self._zgrid = core.new_grids(
            bcflag=self._bcflag,
            gridtype=self._gridtype,
            npx=self._pa.npx,
            npy=self._pa.npy,
            nx=self._nx,
            ny=self._ny,
            nz=self._nz,
            xstart=self._pa.xstart,
            ystart=self._pa.ystart,
            delxp=self._pa.delx,
            delyp=self._pa.dely,
            zlevels=self._pa.zlevels
        )        

        self._npts, self._ncells, self._gridpos, self._gridptr, self._neighptr, \
        self._treeptr, self._cellflags = core.init_cell_structure(
            bcflag=self._bcflag,
            ipflag=self._ipflag,
            nx=self._nx,
            ny=self._ny,
            nz=self._nz,
            nx1=self._nx1,
            ny1=self._ny1,
            xgrid=self._xgrid,
            ygrid=self._ygrid,
            zgrid=self._zgrid,
            gridpos=self._gridpos,
            gridptr=self._gridptr,
            neighptr=self._neighptr,
            treeptr=self._treeptr,
            cellflags=self._cellflags
        )
        
            