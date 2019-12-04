"""
Camera, Sensor and Projection related objects used for rendering.
"""
import numpy as np
import itertools
import dill as pickle
from joblib import Parallel, delayed
import shdom 
from shdom import core

norm = lambda x: x / np.linalg.norm(x, axis=0)
    
class Sensor(object):
    """
    A sensor class to be inherited by specific sensor types (e.g. Radiance, Polarization).
    This class defines the render method which preforms ray-tracing across the medium.
    """
    def __init__(self):
        self._type = 'Sensor'
    
    def render(self, rte_solver, projection):
        """
        The core rendering method.
        
        Parameters
        ----------
        rte_solver: shdom.RteSolver
            A solver with all the associated parameters and the solution to the RTE
        projection: shdom.Projection 
            A projection model which specified the position and direction of each and every pixel 
        """

        if isinstance(projection.npix, list):
            total_pix = np.sum(projection.npix)
        else:
            total_pix = projection.npix 
            
        output = core.render(
            nstphase=rte_solver._nstphase,
            ylmsun=rte_solver._ylmsun,
            phasetab=rte_solver._phasetab,
            nscatangle=rte_solver._nscatangle,
            ncs=rte_solver._ncs,
            nstokes=rte_solver._nstokes,
            nstleg=rte_solver._nstleg,
            camx=projection.x,
            camy=projection.y,
            camz=projection.z,
            cammu=projection.mu,
            camphi=projection.phi,
            npix=total_pix,             
            nx=rte_solver._nx,
            ny=rte_solver._ny,
            nz=rte_solver._nz,
            bcflag=rte_solver._bcflag,
            ipflag=rte_solver._ipflag,   
            npts=rte_solver._npts,
            ncells=rte_solver._ncells,
            ml=rte_solver._ml,
            mm=rte_solver._mm,
            nlm=rte_solver._nlm,
            numphase=rte_solver._pa.numphase,
            nmu=rte_solver._nmu,
            nphi0max=rte_solver._nphi0max,
            nphi0=rte_solver._nphi0,
            maxnbc=rte_solver._maxnbc,
            ntoppts=rte_solver._ntoppts,
            nbotpts=rte_solver._nbotpts,
            nsfcpar=rte_solver._nsfcpar,
            gridptr=rte_solver._gridptr,
            neighptr=rte_solver._neighptr,
            treeptr=rte_solver._treeptr,             
            shptr=rte_solver._shptr,
            bcptr=rte_solver._bcptr,
            cellflags=rte_solver._cellflags,
            iphase=rte_solver._iphase[:rte_solver._npts],
            deltam=rte_solver._deltam,
            solarmu=rte_solver._solarmu,
            solaraz=rte_solver._solaraz,
            gndtemp=rte_solver._gndtemp,
            gndalbedo=rte_solver._gndalbedo,
            skyrad=rte_solver._skyrad,
            waveno=rte_solver._waveno,
            wavelen=rte_solver._wavelen,
            mu=rte_solver._mu,
            phi=rte_solver._phi.reshape(rte_solver._nmu, -1),
            wtdo=rte_solver._wtdo.reshape(rte_solver._nmu, -1),
            xgrid=rte_solver._xgrid,
            ygrid=rte_solver._ygrid,
            zgrid=rte_solver._zgrid,
            gridpos=rte_solver._gridpos,
            sfcgridparms=rte_solver._sfcgridparms,
            bcrad=rte_solver._bcrad,
            extinct=rte_solver._extinct[:rte_solver._npts],
            albedo=rte_solver._albedo[:rte_solver._npts],
            legen=rte_solver._legen,            
            dirflux=rte_solver._dirflux,
            fluxes=rte_solver._fluxes,
            source=rte_solver._source,          
            srctype=rte_solver._srctype,
            sfctype=rte_solver._sfctype,
            units=rte_solver._units,
            total_ext=rte_solver._total_ext[:rte_solver._npts],
            npart=rte_solver._npart)    
        
        return output

    @property
    def type(self):
        return self._type
    
    
class RadianceSensor(Sensor):
    """
    A Radiance sensor measures monochromatic radiances.
    """
    def __init__(self):
        super().__init__()
        self._type = 'RadianceSensor'
        
    def render(self, rte_solver, projection, n_jobs=1, verbose=0):
        """
        The render method integrates a pre-computed in-scatter field (source function) J over the projection gemoetry.
        The source code for this function is in src/unoplarized/shdomsub4.f. 
        It is a modified version of the original SHDOM visualize_radiance subroutine in src/unpolarized/shdomsub2.f.
        
        If n_jobs>1 than parallel rendering is used with pixels distributed amongst all workers
        
        
        Parameters
        ----------
        rte_solver: shdom.RteSolver object
            The RteSolver with the precomputed radiative transfer solution (RteSolver.solve method).
        projection: shdom.Projection object
            The Projection specifying the sensor camera geomerty.
        n_jobs: int, default=1
            The number of jobs to divide the rendering.
        verbose: int, default=0
            How much verbosity in the parallel rendering proccess.
            
        Returns
        -------
        radiance: np.array(shape=(projection.resolution), dtype=np.float32)
            The rendered radiances.
        
        Notes
        -----
        For a small amout of pixels parallel rendering is slower due to communication overhead.
        """
        
        # If rendering several atmospheres (e.g. multi-spectral rendering)
        if isinstance(rte_solver, shdom.RteSolverArray):
            num_channels = rte_solver.num_solvers  
            rte_solvers = rte_solver
        else:
            num_channels = 1
            rte_solvers = [rte_solver]
            
        # Pre-computation of phase-function for all solvers.
        for rte_solver in rte_solvers:
            rte_solver.precompute_phase()

        # Parallel rendering using multithreading (threadsafe Fortran)
        if n_jobs > 1:
            radiance = Parallel(n_jobs=n_jobs, backend="threading", verbose=verbose)(
                delayed(super(RadianceSensor, self).render, check_pickle=False)(
                    rte_solver=rte_solver,
                    projection=projection) for rte_solver, projection in 
                itertools.product(rte_solvers, projection.split(n_jobs)))
            
        # Sequential rendering
        else:
            radiance = [super(RadianceSensor, self).render(rte_solver, projection) for rte_solver in rte_solvers]

        radiance = np.concatenate(radiance) 
        images = self.make_images(radiance, projection, num_channels)
        return images

    def make_images(self, radiance, projection, num_channels):
        """
        Split radiances into Multiview, Multi-channel images (channel last)
        
        Parameters
        ----------
        radiance: np.array(dtype=np.float32)
            A 1D array of radiances
        projection: shdom.Projection
            The projection geometry
        num_channels: int
            The number of channels
            
        Returns
        -------
        radiance: np.array(dtype=np.float32)
            An array of radiances with the shape (H,W,C) or (H,W) for a single channel.
        """
        multiview = isinstance(projection, shdom.MultiViewProjection)
        multichannel = num_channels > 1
        
        if multichannel:
            radiance = np.array(np.split(radiance, num_channels)).T

        if multiview:
            split_indices = np.cumsum(projection.npix[:-1])
            radiance = np.split(radiance, split_indices)

            if multichannel:
                radiance = [
                    image.reshape(resolution + [num_channels], order='F')
                    for image, resolution in zip(radiance, projection.resolution)
                ]
            else:
                radiance = [
                    image.reshape(resolution, order='F')
                    for image, resolution in zip(radiance, projection.resolution) 
                ]                  
        else:
            new_shape = projection.resolution
            if multichannel:
                new_shape.append(num_channels)       
            radiance = radiance.reshape(new_shape, order='F')
                
        return radiance         
    
 
class StokesSensor(Sensor):
    """
    A StokesSensor measures monochromatic stokes vector [I, U, Q, V].
    """
    def __init__(self):
        super().__init__()
        self._type = 'StokesSensor'
    
    def render(self, rte_solver, projection, n_jobs=1, verbose=0):
        """      
        The render method integrates a pre-computed stokes vector in-scatter field (source function) J over the sensor geometry.
        The source code for this function is in src/polarized/shdomsub4.f. 
        It is a modified version of the original SHDOM visualize_radiance subroutine in src/polarized/shdomsub2.f.
        
        If n_jobs > 1 than parallel rendering is used where all pixels are distributed amongst all workers
        
        Parameters
        ----------
        rte_solver: shdom.RteSolver object
            The RteSolver with the precomputed radiative transfer solution (RteSolver.solve method).
        projection: shdom.Projection object
            The Projection specifying the sensor camera geomerty.
        n_jobs: int, default=1
            The number of jobs to divide the rendering into.
        verbose: int, default=0
            How much verbosity in the parallel rendering proccess.

        Returns
        -------
        stokes: np.array(shape=(nstokes, sensor.resolution), dtype=np.float32)
            The rendered radiances.

        Notes
        -----
        For a small amout of pixels parallel rendering is slower due to communication overhead.
        """
        # If rendering several atmospheres (e.g. multi-spectral rendering)
        if isinstance(rte_solver, shdom.RteSolverArray):
            num_channels = rte_solver.num_solvers
            rte_solvers = rte_solver
        else:
            num_channels = 1
            rte_solvers = [rte_solver]
        
        # Pre-computation of phase-function for all solvers.
        for rte_solver in rte_solvers:
            rte_solver._phasetab = core.precompute_phase_check(
                negcheck=True,
                nscatangle=rte_solver._nscatangle,
                numphase=rte_solver._pa.numphase,
                nstphase=rte_solver._nstphase,
                nstokes=rte_solver._nstokes,
                nstleg=rte_solver._nstleg,
                nleg=rte_solver._nleg,
                ml=rte_solver._ml,
                nlm=rte_solver._nlm,
                legen=rte_solver._legen,
                deltam=rte_solver._deltam
            )
            
        # Parallel rendering using multithreading (threadsafe Fortran)
        if n_jobs > 1:
            stokes = Parallel(n_jobs=n_jobs, backend="threading", verbose=verbose)(
                delayed(super(StokesSensor, self).render, check_pickle=False)(
                    rte_solver=rte_solver,
                    projection=projection) for rte_solver, projection in 
                itertools.product(rte_solvers, projection.split(n_jobs)))  
            
        # Sequential rendering
        else:      
            stokes = [super(StokesSensor, self).render(rte_solver, projection) for rte_solver in rte_solvers]
          
        stokes = np.hstack(stokes) 
        images = self.make_images(stokes, projection, num_channels)
        return images
        
    def make_images(self, stokes, projection, num_channels):
        """
        Split into Multiview, Multi-channel Stokes images (channel last)
        
        Parameters
        ----------
        stokes: np.array(dtype=np.float32)
            A 2D array of stokes pixels (number of stokes is first dimension)
        projection: shdom.Projection
            The projection geometry
        num_channels: int
            The number of channels
            
        Returns
        -------
        stokes: np.array(dtype=np.float32)
            An array of stokes pixels with the shape (NSTOKES,H,W,C) or (NSTOKES,H,W) for a single channel.
        """
        multiview = isinstance(projection, shdom.MultiViewProjection)
        multichannel = num_channels > 1        

        if multichannel:
            stokes = np.array(np.split(stokes, num_channels, axis=-1)).transpose([1, 2, 0]).squeeze()      
    
        if multiview: 
            split_indices = np.cumsum(projection.npix[:-1])        
            stokes = np.split(stokes, split_indices, axis=1)
    
            if multichannel:
                stokes = [
                    image.reshape([image.shape[0]] + resolution + [num_channels], order='F')
                    for image, resolution in zip(stokes, projection.resolution)
                ]
            else:
                stokes = [
                    image.reshape([image.shape[0]] + resolution, order='F')
                    for image, resolution in zip(stokes, projection.resolution) 
                ]                  
        else:
            new_shape = [stokes.shape[0]] + projection.resolution
            if multichannel:
                new_shape.append(num_channels)       
            stokes = stokes.reshape(new_shape, order='F')
            
        return stokes


class DolpAolpSensor(StokesSensor):
    """
    A DolpAolp measures monochromatic Degree and Angle of Linear Polarization.
    """
    def __init__(self):
        super().__init__()
        self._type = 'DolpAolpSensor'
    
    def render(self, rte_solver, projection, n_jobs=1, verbose=0):
        """   
        The render method integrates a pre-computed stokes vector in-scatter field (source function) J over the sensor geometry.
        The source code for this function is in src/polarized/shdomsub4.f. 
        It is a modified version of the original SHDOM subroutine in src/polarized/shdomsub2.f.
        
        If n_jobs>1 than parallel rendering is used where all pixels are distributed amongst all workers
        
        Parameters
        ----------
        rte_solver: shdom.RteSolver object
            The RteSolver with the precomputed radiative transfer solution (RteSolver.solve method).
        projection: shdom.Projection object
            The Projection specifying the sensor camera geomerty.
        n_jobs: int, default=1
            The number of jobs to divide the rendering into.
        verbose: int, default=0
            How much verbosity in the parallel rendering proccess.
            
            
        Returns
        -------
        dolp: np.array(shape=(sensor.resolution), dtype=np.float32)
            Degree of Linear Polarization
        aolp: np.array(shape=(sensor.resolution), dtype=np.float32)
            Angle of Linear Polarization
        """
        stokes = super().render(rte_solver, projection, n_jobs, verbose)
        
        indices = stokes[0] > 0.0
        dolp = np.zeros_like(stokes[0])
        aolp = np.zeros_like(stokes[0])
        i, q, u = stokes[0][indices], stokes[1][indices], stokes[2][indices]
        dolp[indices] = np.sqrt(q**2 + u**2) / i
        aolp[indices] = (180.0/np.pi) * 0.5 * np.arctan2(u, q)
        
        # Choose the best range for the angle of linear polarization (-90 to 90 or 0 to 180)
        aolp1 = aolp.reshape(-1, aolp.shape[-1])
        aolp2 = aolp1.copy()
        aolp2[aolp2 < 0.0] += 180.0
        std1 = np.std(aolp1, axis=0)
        std2 = np.std(aolp2, axis=0)
        aolp[..., std2 < std1] = aolp2.reshape(aolp.shape)[..., std2 < std1]
        
        return dolp, aolp


class Projection(object):
    """
    Abstract Projection class to be inherited by the different types of projections.
    Each projection defines an arrays of pixel locations (x,y,z) in km and directions (phi, mu).
    
    Parameters
    ----------
    x: np.array(np.float32)
        Locations in global x coordinates [km] (North)
    y: np.array(np.float32)
        Locations in global y coordinates [km] (East)
    z: np.array(np.float32)
        Locations in global z coordinates [km] (Up)
    mu: np.array(np.float64)
        Cosine of the zenith angle of the measurements (direction of photons)
    phi: np.array(np.float64)
        Azimuth angle [rad] of the measurements (direction of photons)
    resolution: list
        Resolution is the number of pixels in each dimension (H,W) used to reshape arrays into images.
        
    Notes
    -----
    All input arrays are raveled and should be of the same size.
    """
    def __init__(self, x=None, y=None, z=None, mu=None, phi=None, resolution=None):
        self._x = x
        self._y = y
        self._z = z
        self._mu = mu
        self._phi = phi
        self._npix = None
        if type(x)==type(y)==type(z)==type(mu)==type(phi)==np.ndarray:
            assert x.size==y.size==z.size==mu.size==phi.size, 'All input arrays must be of equal size'
            self._npix = x.size
        self._resolution = resolution
        
    def __getitem__(self, val):
        projection = Projection(
            x=np.array(self._x[val]), 
            y=np.array(self._y[val]),
            z=np.array(self._z[val]), 
            mu=np.array(self._mu[val]), 
            phi=np.array(self._phi[val]),
        )
        return projection
    
    def split(self, n_parts):
        """
        Split the projection geometry.
        
        Parameters
        ----------
        n_parts: int
            The number of parts to split the projection geometry to 
        
        Returns
        -------
        projections: list
            A list of projections each with n_parts
            
        Notes
        -----
        An even split doesnt always exist, in which case some parts will have slightly more pixels.
        """
        x_split = np.array_split(self.x, n_parts) 
        y_split = np.array_split(self.y, n_parts) 
        z_split = np.array_split(self.z, n_parts) 
        mu_split = np.array_split(self.mu, n_parts) 
        phi_split = np.array_split(self.phi, n_parts)        
        projections = [
            Projection(x, y, z, mu, phi) for 
            x, y, z, mu, phi in zip(x_split, y_split, z_split, mu_split, phi_split)
        ]
        return projections
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y
    
    @property
    def z(self):
        return self._z
    
    @property
    def mu(self):
        return self._mu
    
    @property
    def phi(self):
        return self._phi
    
    @property
    def zenith(self):
        return np.rad2deg(np.arccos(self.mu))
 
    @property
    def azimuth(self):
        return np.rad2deg(self.phi)
    
    @property 
    def npix(self):
        return self._npix    
    
    @property
    def resolution(self):
        if self._resolution is None:
            return [self.npix, ]
        else:
            return self._resolution  
    
    
class HomographyProjection(Projection):
    """
    A Homography has a projective tansformation that relates 3D coordinates to pixels.
    """
    def __init__(self):
        super().__init__()
        
    def project(self, projection_matrix, point_array):
        """
        Project 3D coordinates according to the sensor projection matrix
        
        Parameters
        ----------
        projection matrix: np.array(shape=(3,4), dtype=float)
            The sensor projection matrix K.
        point_array: np.array(shape=(3, num_points), dtype=float)
            An array of num_points 3D points (x,y,z) [km]
        """
        homogenic_point_array = np.pad(point_array,((0,1),(0,0)),'constant', constant_values=1)
        return np.dot(projection_matrix, homogenic_point_array)  
  

class OrthographicProjection(HomographyProjection):
    """
    A parallel ray projection. 
    
    Parameters
    ----------
    bounding_box: shdom.BoundingBox object
        The bounding box is used to compute a projection that will make the entire bounding box visible.
    x_resolution: float
        Pixel resolution [km] in x axis (North)
    y_resolution: float
        Pixel resolution [km] in y axis (East)
    azimuth: float
        Azimuth angle [deg] of the measurements (direction of photons)
    zenith: float
        Zenith angle [deg] of the measurements (direction of photons)
    altitude: float or 'TOA' (default)
       1. 'TOA': Top of the atmosphere.
       2. float: Altitude of the  measurements.    
    """
    
    def __init__(self, bounding_box, x_resolution, y_resolution, azimuth, zenith, altitude='TOA'):
        super().__init__()
        self._x_resolution = x_resolution
        self._y_resolution = y_resolution
        
        mu = np.cos(np.deg2rad(zenith))
        phi = np.deg2rad(azimuth)
        if altitude == 'TOA':
            self._altitude = bounding_box.zmax
        else:
            assert (type(altitude) == float or type(altitude) == int), 'altitude of incorrect type'
            self._altitude = altitude        
            
        # Project the bounding box onto the image plane
        alpha = np.sqrt(1 - mu**2) * np.cos(phi) / mu
        beta  = np.sqrt(1 - mu**2) * np.sin(phi) / mu
        projection_matrix = np.array([
            [1, 0, -alpha, alpha* self.altitude],
            [0, 1,  -beta, beta * self.altitude],
            [0, 0,      0,        self.altitude]
        ])
        
        self.projection_matrix = projection_matrix
        bounding_box_8point_array = np.array(list(itertools.product([bounding_box.xmin, bounding_box.xmax], 
                                                                    [bounding_box.ymin, bounding_box.ymax],
                                                                    [bounding_box.zmin, bounding_box.zmax]))).T
        projected_bounding_box = self.project(projection_matrix, bounding_box_8point_array)
            
        # Use projected bounding box to define image sampling
        x_s, y_s = projected_bounding_box[:2,:].min(axis=1)
        x_e, y_e = projected_bounding_box[:2,:].max(axis=1)
        x = np.arange(x_s, x_e+1e-6, self.x_resolution)
        y = np.arange(y_s, y_e+1e-6, self.y_resolution)
        z = self.altitude
        self._x, self._y, self._z, self._mu, self._phi = np.meshgrid(x, y, z, mu, phi)
        self._x = self._x.ravel().astype(np.float32)
        self._y = self._y.ravel().astype(np.float32)
        self._z = self._z.ravel().astype(np.float32)
        self._mu = self._mu.ravel().astype(np.float64)
        self._phi = self._phi.ravel().astype(np.float64)
        self._npix = self.x.size
        self._resolution = [x.size, y.size]

    @property 
    def altitude(self):
        return self._altitude
    
    @property 
    def x_resolution(self):
        return self._x_resolution
     
    @property 
    def y_resolution(self):
        return self._y_resolution
    
    
class PerspectiveProjection(HomographyProjection):
    """
    A Perspective trasnormation (pinhole camera).
    
    Parameters
    ----------
    fov: float
        Field of view [deg]
    nx: int
        Number of pixels in camera x axis
    ny: int
        Number of pixels in camera y axis
    x: float
        Location in global x coordinates [km] (North)
    y: float
        Location in global y coordinates [km] (East)
    z: float
        Location in global z coordinates [km] (Up)
    """
    def __init__(self, fov, nx, ny, x, y, z):
        super().__init__()
        self._resolution = [nx, ny]
        self._npix = nx*ny
        self._position = np.array([x, y, z], dtype=np.float32)
        self._fov = fov
        self._focal = 1.0 / np.tan(np.deg2rad(fov) / 2.0)
        self._k = np.array([[self._focal, 0, 0],
                            [0, self._focal, 0],
                            [0, 0, 1]], dtype=np.float32)
        self._inv_k = np.linalg.inv(self._k)
        self._rotation_matrix = np.eye(3)
        x_c, y_c, z_c = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), 1.0)        
        self._homogeneous_coordinates = np.stack([x_c.ravel(), y_c.ravel(), z_c.ravel()])
        self.update_global_coordinates()

    def update_global_coordinates(self):
        """
        This is an internal method which is called upon when a rotation matrix is computed to update the global camera coordinates.
        """
        x_c, y_c, z_c = norm(np.matmul(
            self._rotation_matrix, np.matmul(self._inv_k, self._homogeneous_coordinates)))

        self._mu = -z_c.astype(np.float64)
        self._phi = (np.arctan2(y_c, x_c) + np.pi).astype(np.float64)
        self._x = np.full(self.npix, self.position[0], dtype=np.float32)
        self._y = np.full(self.npix, self.position[1], dtype=np.float32)
        self._z = np.full(self.npix, self.position[2], dtype=np.float32)

    def look_at_transform(self, point, up):
        """
        A look at transform is defined with a point and an up vector.
        
        Parameters
        ----------
        point: np.array(shape=(3,), dtype=float)
            A point in 3D space (x,y,z) coordinates in [km]
        up: np.array(shape=(3,), dtype=float)
            The up vector determines the roll of the camera.
        """
        up = np.array(up)
        direction = np.array(point) - self.position
        zaxis = norm(direction)
        xaxis = norm(np.cross(up, zaxis))
        yaxis = np.cross(zaxis, xaxis)
        self._rotation_matrix = np.stack((xaxis, yaxis, zaxis), axis=1)
        self.update_global_coordinates()
        
    def rotate_transform(self, axis, angle):
        """
        Rotate the camera with respect to one of it's (local) axis
        
        Parameters
        ----------
        axis: 'x', 'y' or 'z'
            The rotation axis
        angle: float
            The angle of rotation [deg]
            
        Notes
        -----
        The axis are in the camera coordinates
        """
        assert axis in ['x', 'y', 'z'], 'axis parameter can only recieve "x", "y" or "z"'

        angle = np.deg2rad(angle)
        if axis == 'x':
            rot = np.array([[1, 0, 0],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle)]], dtype=np.float32)
        elif axis == 'y':
            rot = np.array([[np.cos(angle), 0, np.sin(angle)],
                            [0, 1, 0],
                            [-np.sin(angle), 0, np.cos(angle)]], dtype=np.float32)
        elif axis == 'z':
            rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1]], dtype=np.float32)        
        
        self._rotation_matrix = np.matmul(self._rotation_matrix, rot)
        self.update_global_coordinates()

    def plot(self, ax, xlim, ylim, zlim, length=0.1):
        """
        Plot the cameras and their orientation in 3D space using matplotlib's quiver.
        
        Parameters
        ----------
        ax: matplotlib.pyplot.axis
           and axis for the plot
        xlim: list
            [xmin, xmax] to set the domain limits
        ylim: list
            [ymin, ymax] to set the domain limits
        zlim: list
            [zmin, zmax] to set the domain limits 
        length: float, default=0.1
            The length of the quiver arrows in the plot
            
        Notes
        -----
        The axis are in the camera coordinates
        """        
        mu = -self.mu.reshape(self.resolution)[[0, -1, 0, -1],[0, 0, -1, -1]]
        phi = np.pi + self.phi.reshape(self.resolution)[[0, -1, 0, -1],[0, 0, -1, -1]]
        u = np.sqrt(1 - mu**2) * np.cos(phi)
        v = np.sqrt(1 - mu**2) * np.sin(phi)
        w = mu
        x = np.full(4, self.position[0], dtype=np.float32)
        y = np.full(4, self.position[1], dtype=np.float32)
        z = np.full(4, self.position[2], dtype=np.float32)
        ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.quiver(x, y, z, u, v, w, length=length, pivot='tail')

    @property
    def position(self):
        return self._position
    
    
class PrincipalPlaneProjection(Projection):
    """
    Measurments along the principal solar plane.
    
    Parameters
    ----------
    source: shdom.SolarSource
        The source azimuth is used to find the solar principal plane
    x: float
        Location in global x coordinates [km] (North)
    y: float
        Location in global y coordinates [km] (East)
    z: float
        Location in global z coordinates [km] (Up)
    resolution: float 
        Angular resolution of the measurements in [deg]
    """
    def __init__(self, source, x, y, z, resolution=1.0):
        super().__init__()
        self._angles = np.arange(-89.0, 89.0, resolution)
        self._npix = len(self._angles)
        self._x = np.full(self.npix, x, dtype=np.float32)
        self._y = np.full(self.npix, y, dtype=np.float32)
        self._z = np.full(self.npix, z, dtype=np.float32)
        self._mu = (np.cos(np.deg2rad(self._angles))).astype(np.float64)
        self._phi = np.deg2rad(180 * (self._angles < 0.0).astype(np.float64) + source.azimuth)
        self._source = source
    
    @property 
    def angles(self):
        return self._angles
        
        
class AlmucantarProjection(Projection):
    """
    Measurments along the solar almucantar.
    
    Parameters
    ----------
    source: shdom.SolarSource
        The source zenith is used to find the solar almucantar plane
    x: float
        Location in global x coordinates [km] (North)
    y: float
        Location in global y coordinates [km] (East)
    z: float
        Location in global z coordinates [km] (Up)
    resolution: float 
        Angular resolution of the measurements in [deg]
    """
    def __init__(self, source, x, y, z, resolution=1.0):
        super().__init__()
        self._phi = np.deg2rad(np.arange(180.0, 360.0, resolution)).astype(np.float64)
        self._npix = len(self._phi)
        self._mu = np.full(self.npix, np.cos(np.deg2rad(source.zenith - 180)), dtype=np.float64)
        self._x = np.full(self.npix, x, dtype=np.float32)
        self._y = np.full(self.npix, y, dtype=np.float32)
        self._z = np.full(self.npix, z, dtype=np.float32)


class HemisphericProjection(Projection):
    """
    Measurments on a hemisphere.
    
    Parameters
    ----------
    source: shdom.SolarSource
        The source zenith is used to find the solar almucantar plane
    x: float
        Location in global x coordinates [km] (North)
    y: float
        Location in global y coordinates [km] (East)
    z: float
        Location in global z coordinates [km] (Up)
    resolution: float 
        Angular resolution of the measurements in [deg]
    """
    def __init__(self, x, y, z, resolution=5.0):
        super().__init__()
        mu = np.cos(np.deg2rad(np.arange(0.0, 80.0+resolution, resolution)))
        phi = np.deg2rad(np.arange(0.0, 360.0+resolution, resolution))
        self._x, self._y, self._z, self._mu, self._phi = np.meshgrid(x, y, z, mu, phi)
        self._x = self._x.ravel().astype(np.float32)
        self._y = self._y.ravel().astype(np.float32)
        self._z = self._z.ravel().astype(np.float32)
        self._mu = self._mu.ravel().astype(np.float64)
        self._phi = self._phi.ravel().astype(np.float64)
        self._npix = self.x.size
        self._resolution = [phi.size, mu.size]


class Measurements(object):
    """
    A Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
    It can be initialized with a Camera and images or pixels.
    Alternatively is can be loaded from file.

    Parameters
    ----------
    camera: shdom.Camera
        The camera model used to take the measurements
    images: list of images, optional
        A list of images (multiview camera)
    pixels: np.array(dtype=float)
        pixels are a flattened version of the image list where the channel dimension is kept (1 for monochrome).
    """
    def __init__(self, camera=None, images=None, pixels=None, wavelength=None):
        self._camera = camera
        self._images = images
        self._wavelength = np.atleast_1d(wavelength)

        num_channels = 0
        if images is not None:

            if type(images) is not list:
                self._images = [images]

            pixels = []
            for image in self.images:
                if camera.sensor.type == 'RadianceSensor':
                    num_channels = image.shape[-1] if image.ndim == 3 else 1
                    pixels.append(image.reshape((-1, num_channels), order='F'))

                elif camera.sensor.type == 'StokesSensor':
                    num_channels = image.shape[-1] if image.ndim == 4 else 1
                    pixels.append(image.reshape((image.shape[0], -1, num_channels), order='F'))

                else:
                    raise AttributeError('Error image dimensions: {}'.format(image.ndim))
            pixels = np.concatenate(pixels, axis=-2)

        elif pixels is not None:
            if (pixels.ndim == 1 and camera.sensor.type == 'RadianceSensor') or (pixels.ndim == 2 and camera.sensor.type == 'StokesSensor'):
                num_channels = 1
                pixels = pixels[..., None]
            elif (pixels.ndim == 2 and camera.sensor.type == 'RadianceSensor') or (pixels.ndim == 3 and camera.sensor.type == 'StokesSensor'):
                num_channels = pixels.shape[-1]
            else:
                raise AttributeError('Pixels should be a flat along spatial dimensions while maintaining channels/stokes dimension')

        if num_channels > 1:
            assert num_channels == len(self._wavelength), 'Number of channels = {} differs from len(wavelength)={}'.format(num_channels, len(self._wavelength))
        self._pixels = pixels
        self._num_channels = num_channels

    def save(self, path):
        """
        Save Measurements to file.
    
        Parameters
        ----------
        path: str,
            Full path to file. 
        """
        file = open(path, 'wb')
        file.write(pickle.dumps(self.__dict__, -1))
        file.close()

    def load(self, path):
        """
        Load Measurements from file.

        Parameters
        ----------
        path: str,
            Full path to file. 
        """        
        file = open(path, 'rb')
        data = file.read()
        file.close()
        self.__dict__ = pickle.loads(data)
    
    def split(self, n_parts):
        """
        Split the measurements and projection.
        
        Parameters
        ----------
        n_parts: int
            The number of parts to split the measurements to 
        
        Returns
        -------
        measurements: list
            A list of measurements each with n_parts
            
        Notes
        -----
        An even split doesnt always exist, in which case some parts will have slightly more pixels.
        """
        projections = self.camera.projection.split(n_parts)
        pixels = np.array_split(self.pixels, n_parts)
        measurements = [shdom.Measurements(
            camera=shdom.Camera(self.camera.sensor, projection), 
            pixels=pixel) for projection, pixel in zip(projections, pixels)
        ]
        return measurements

    @property
    def camera(self):
        return self._camera
    
    @property
    def pixels(self):
        return self._pixels
    
    @property
    def images(self):
        return self._images

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def wavelength(self):
        if self.num_channels == 1:
            return self._wavelength[0]
        else:
            return self._wavelength


class Camera(object):
    """
    An Camera object ecapsulates both sensor and projection.
    
    Parameters
    ----------
    sensor: shdom.Sensor
        A sensor object
    projection: shdom.Projection
        A projection geometry
    """
    def __init__(self, sensor=Sensor(), projection=Projection()):
        self.set_sensor(sensor)
        self.set_projection(projection)
        
    def set_projection(self, projection):
        """
        Add a projection.
        
        Parameters
        ----------
        projection: shdom.Projection
            A projection geomtry
        """
        self._projection = projection
        
    def set_sensor(self, sensor):
        """
        Add a sensor.
        
        Parameters
        ----------
        sensor: shdom.Sensor
            A sensor object
            
        Notes
        -----
        This method also updates the docstring of the render method according to the specific sensor
        """        
        self._sensor = sensor
        
        # Update function docstring
        if sensor.render.__doc__ is not None:
            self.render.__func__.__doc__ += sensor.render.__doc__

    def render(self, rte_solver, n_jobs=1, verbose=0):
        """
        Render an image according to the render function defined by the sensor.
        
        Notes
        -----
        This is a dummy docstring that is overwritten when the set_sensor method is used.
        """
        return self.sensor.render(rte_solver, self.projection, n_jobs, verbose)

    @property
    def projection(self):
        return self._projection
    
    @property
    def sensor(self):
        return self._sensor
    

class MultiViewProjection(Projection):
    """
    A MultiViewProjection object encapsulate several projection geometries for multi-view imaging of a domain.
    
    Parameters
    ----------
    projection_list: list, optional
        A list of Sensor objects
    """
    def __init__(self, projection_list=None):
        super().__init__()
        self._num_projections = 0
        self._projection_list = []
        self._names = []
        if projection_list:
            for projection in projection_list:
                self.add_projection(projection)
    
    def add_projection(self, projection, name=None):
        """
        Add a projection to the projection list
        
        Parameters
        ----------
        projection: Projection object
            A Projection object to add to the MultiViewProjection
        name: str, optional
            An ID for the projection. 
        """
        # Set a default name for the projection
        if name is None:
            name = 'View{}'.format(self.num_projections)

        attributes = ['x', 'y', 'z', 'mu', 'phi']
        
        if self.num_projections == 0:
            for attr in attributes:
                self.__setattr__('_' + attr, projection.__getattribute__(attr))
            self._npix = [projection.npix]
            self._resolution = [projection.resolution]
            self._names = [name]
        else:
            for attr in attributes:
                self.__setattr__('_' + attr, np.concatenate((self.__getattribute__(attr), 
                                                             projection.__getattribute__(attr))))
            self._npix.append(projection.npix)
            self._names.append(name)  
            self._resolution.append(projection.resolution)
                                    
        self._projection_list.append(projection)
        self._num_projections += 1 

    @property
    def projection_list(self):
        return self._projection_list
    
    @property
    def num_projections(self):
        return self._num_projections


class Noise(object):
    """
    An abstract noise object to be inherited by specific noise models

    Parameters
    ----------
    full_well: integer
      full well in electrons. Translates radiance measurements and electron counts.
    quantum_efficiency: float
      in range [0,1]. Translates electrons and photon counts.
    """

    def __init__(self, full_well, quantum_efficiency):
        self.full_well = full_well
        self.qe = quantum_efficiency

    def apply(self, measurements):
        """
        Dummy function to apply noise to measurements
        """
        return None


class AirMSPINoise(Noise):
    """
    AirMSPI noise modeled accroding to:
        [1] Van Harten, G., Diner, D.J., Daugherty, B.J., Rheingans, B.E., Bull, M.A., Seidel, F.C.,
            Chipman, R.A., Cairns, B., Wasilewski, A.P. and Knobelspiesse, K.D., 2018.
            Calibration and validation of airborne multiangle spectropolarimetric imager (AirMSPI) polarization
            measurements. Applied optics, 57(16), pp.4499-4513.
        [2] Diner, D.J., Davis, A., Hancock, B., Geier, S., Rheingans, B., Jovanovic, V., Bull, M.,
            Rider, D.M., Chipman, R.A., Mahler, A.B. and McClain, S.C., 2010.
            First results from a dual photoelastic-modulator-based polarimetric camera.
            Applied optics, 49(15), pp.2929-2946.

    Notes
    -----
    The maximum radiance measurement is transformed to 0.85*max_well (200K electrons) before Poisson noise is applied
    35% Quantum effciency at 660nm according to [2]
    """

    def __init__(self):
        super().__init__(full_well=200000, quantum_efficiency=0.35)

        from scipy.special import j0, jv

        self.polarized_bands = [0.47, 0.66, 0.865]  # microns
        num_subframes = 23
        p = np.linspace(0.0, 1.0, num_subframes + 1)
        p1 = p[0:-1]
        p2 = p[1:]
        x = 0.5 * (p1 + p2 - 1)
        delta0_list = [4.472, 3.081, 2.284]
        r = 0.0
        eta = 0.009

        self.pq, self.pu, self.w = dict(), dict(), dict()
        for wavelength, delta0 in zip(self.polarized_bands, delta0_list):
            # Define z'(x_n) (Eq. 8in [1])
            z_idx = np.pi * x != eta
            z = np.full_like(x, r)
            z[z_idx] = -2 * delta0 * np.sin(np.pi * x - eta)[z_idx] * np.sqrt(
                1 + r ** 2 / np.tan(np.pi * x - eta)[z_idx])

            # Define s_n (Eq. 9 in [1])
            s = np.ones(shape=(num_subframes))
            s_idx = z_idx if r == 0 else np.ones_like(x, dtype=np.bool)
            s[s_idx] = (np.tan(np.pi * x - eta) ** 2 - r)[s_idx] / (np.tan(np.pi * x - eta) ** 2 + r)[s_idx]

            # Define F(x_n) (Eq. 7 in [1])
            f = j0(z) + (1 / 3) * (np.pi * (p2 - p1) / 2) ** 2 * delta0 ** 2 * (1 - r ** 2) * (
                    s * jv(2, z) - np.cos(2 * (np.pi * x - eta)) * j0(z))

            # P modulation matrix for I, Q, U with and idealized modulator (without the linear correction factor)
            # Eq. 15 of [1]
            self.pq[wavelength] = np.vstack((np.ones_like(x), f, np.zeros_like(x))).T
            self.pu[wavelength] = np.vstack((np.ones_like(x), np.zeros_like(x), f)).T

            # W demodulation matrix (Eq. 16 of [1])
            self.w[wavelength] = np.linalg.pinv(np.vstack((self.pq[wavelength], self.pu[wavelength])))

    def apply(self, measurements):
        """
        Apply the AirMSPI poisson noise model according to the modulation and de-modulation matrices.

        Parameters
        ----------
        measurements: shdom.Measurements
            input clean measurements

        Returns
        -------
        noisy_measurements: shdom.Measurements
            output noisy measurements
        """
        images = []
        for view in measurements.images:
            multi_spectral_image = []
            for i, wavelength in enumerate(measurements.wavelength):
                image = view[..., i]
                if isinstance(measurements.camera.sensor, shdom.StokesSensor):
                    if wavelength not in self.polarized_bands:
                        raise AttributeError('wavelength {} is not in AirMSPI polarized channels ({})'.format(
                            wavelength, self.polarized_bands)
                        )

                    image0 = np.matmul(self.pq[wavelength], np.rollaxis(image, 1))
                    image45 = np.matmul(self.pu[wavelength], np.rollaxis(image, 1))
                    image0_sign = np.sign(image0)
                    image45_sign = np.sign(image45)
                    image0_mag = np.abs(image0)
                    image45_mag = np.abs(image45)

                    # Gain
                    g0 = (1.0 / self.qe) * 0.85 * self.full_well / image0_mag.max()
                    g45 = (1.0 / self.qe) * 0.85 * self.full_well / image45_mag.max()

                    # Digital number d with Poisson noise
                    d0 = (self.qe * image0_mag.max() / (0.85 * self.full_well)) * \
                         np.random.poisson(np.round(g0 * image0_mag)).astype(np.float32) * image0_sign
                    d45 = (self.qe * image45_mag.max() / (0.85 * self.full_well)) * \
                          np.random.poisson(np.round(g45 * image45_mag)).astype(np.float32) * image45_sign
                    d = np.concatenate((d0, d45), axis=1)

                    # Back to I, Q, U using W demodulation matrix
                    noisy_image = np.rollaxis(np.matmul(self.w[wavelength], d), 1)
                else:
                    g = 0.85 * self.full_well / image.max()
                    noisy_image = (image.max() / (0.85 * self.full_well)) * np.random.poisson(
                        np.round(g * image)).astype(np.float32)
                multi_spectral_image.append(noisy_image)

            images.append(np.stack(multi_spectral_image, axis=-1))
        noisy_measurements = shdom.Measurements(measurements.camera, images=images, wavelength=measurements.wavelength)
        return noisy_measurements