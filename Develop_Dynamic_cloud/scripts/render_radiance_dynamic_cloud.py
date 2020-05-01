import numpy as np
import argparse
import shdom


class RenderScript(object):
    """
    Render: Radiance at Top of the Atmosphere (TOA)
    -----------------------------------------------
    Forward rendering of an atmospheric medium with at multiple spectral bands with an orthographic sensor measuring
    exiting radiance at the top of the the domain. This sensor is an approximation (somewhat crude) for far observing satellites where the rays are parallel.

    As with all `render` scripts a `generator` needs to be specified using the generator flag (--generator).
    The Generator defines the medium parameters: Grid, Extinction, Single Scattering Albedo and Phase function with it's own set of command-line flags.

    For example usage see the README.md

    For information about the command line flags see:
      python scripts/render/render_radiance_toa.py --help

    For a tutorial overview of how to operate the forward rendering model see the following notebooks:
     - notebooks/Make Mie Table.ipynb
     - notebooks/Radiance Rendering [Single Image].ipynb
     - notebooks/Radiance Rendering [Multiview].ipynb
     - notebooks/Radiance Rendering [Multispectral].ipynb

    Parameters
    ----------
    scatterer_name: str
        The name of the scatterer that will be optimized.
    """
    def __init__(self, scatterer_name='cloud'):
        self.scatterer_name = scatterer_name
        self.num_stokes = 1
        self.sensor = shdom.RadianceSensor()

    def solution_args(self, parser):
        """
        Add common RTE solution arguments that may be shared across scripts.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.

        Returns
        -------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.
        """
        parser.add_argument('output_dir',
                            help='Path to an output directory where the measurements and model parameters will be saved. \
                                  If the folder doesnt exist, it will be created.')
        parser.add_argument('wavelength',
                            nargs='+',
                            type=np.float32,
                            help='Wavelengths for the measurements [micron].')
        parser.add_argument('--solar_zenith',
                            default=135.0,
                            type=np.float32,
                            help='(default value: %(default)s) Solar zenith [deg]. This is the direction of the photons in range (90, 180]')
        parser.add_argument('--solar_azimuth',
                            default=65.0,
                            type=np.float32,
                            help='(default value: %(default)s) Solar azimuth [deg]. This is the direction of the photons')
        parser.add_argument('--n_jobs',
                            default=1,
                            type=int,
                            help='(default value: %(default)s) Number of jobs for parallel rendering. n_jobs=1 uses no parallelization')
        parser.add_argument('--surface_albedo',
                            default=0.05,
                            type=float,
                            help='(default value: %(default)s) The albedo of the Lambertian Surface.')
        parser.add_argument('--num_mu',
                            default=8,
                            type=int,
                            help='(default value: %(default)s) The number of discrete ordinates in the zenith direction.\
                            See rte_solver.NumericalParameters for more details.')
        parser.add_argument('--num_phi',
                            default=16,
                            type=int,
                            help='(default value: %(default)s) The number of discrete ordinates in the azimuthal direction.\
                            See rte_solver.NumericalParameters for more details.')
        parser.add_argument('--split_accuracy',
                            default=0.03,
                            type=float,
                            help='(default value: %(default)s) The cell splitting accuracy for SHDOM. \
                            See rte_solver.NumericalParameters for more details.')
        parser.add_argument('--solution_accuracy',
                            default=1e-4,
                            type=float,
                            help='(default value: %(default)s) The SHDOM solution criteria (accuracy). \
                            See rte_solver.NumericalParameters for more details.')
        parser.add_argument('--adapt_grid_factor',
                            default=10,
                            type=float,
                            help='(default value: %(default)s) The radio of adaptive (internal) grid points property array grid points \
                            See rte_solver.NumericalParameters for more details.')
        parser.add_argument('--solar_spectrum',
                            action='store_true',
                            help='Use solar spectrum flux for each wavelength. \
                            If not used, the solar flux is normalized to be 1.0. \
                            See shdom.SolarSpectrum for more details.')
        return parser

    def rendering_args(self, parser):
        """
        Add common rendering arguments that may be shared across scripts.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.

        Returns
        -------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.
        """
        parser.add_argument('--x_res',
                            default=0.01,
                            type=np.float32,
                            help='(default value: %(default)s) Radiance sampling resolution in x axis (North)')
        parser.add_argument('--y_res',
                            default=0.01,
                            type=np.float32,
                            help='(default value: %(default)s) Radiance sampling resolution in y axis (East)')
        parser.add_argument('--azimuth',
                            default=[0.0],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Azimuth angles for the radiance measurements [deg]' \
                                 '90 is for measuring radiance exiting along Y axis (East)')
        parser.add_argument('--zenith',
                            default=[0.0],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Zenith angles for the radiance measurements [deg].' \
                                 '0 is for measuring radiance exiting directly up.')
        parser.add_argument('--cloud_velocity',
                            default=[0.0, 0.0, 0.0],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Cloud velocity vector[m/sec].' \
                                 '[0.0, 0.0, 0.0] is for stationary cloud.')
        parser.add_argument('--camera_velocity',
                            default=45,
                            type=np.float32,
                            help='(default value: %(default)) Camera speed in y-axis direction[m/sec].' \
                                 '0 is for stationary camera.')
        parser.add_argument('--camera_height',
                            default=['TOA'],
                            type=list,
                            help='(default value: %(default)) Camera height in z-axis direction[km].' \
                                 'TOA is for camera at the Top of the Atmosphere.')
        parser.add_argument('--projection_type',
                            default='Perspective',
                            help='(default value: %(default)) rendering Projection type (Perspective, Orthographic etc.).' \
                                 'Perspective is for Perspective projection images rendering.')
        parser.add_argument('--mie_base_path',
                            default='mie_tables/polydisperse/Water_<wavelength>nm.scat',
                            help='(default value: %(default)s) Mie table base file name. '\
                                 '<wavelength> will be replaced by the corresponding wavelengths.')
        return parser

    def parse_arguments(self):
        """
        Handle all the argument parsing needed for this script.
        """
        parser = argparse.ArgumentParser()
        parser = self.solution_args(parser)
        parser = self.rendering_args(parser)

        subparser = argparse.ArgumentParser(add_help=False)
        subparser.add_argument('--generator')
        subparser.add_argument('--add_rayleigh', action='store_true')
        parser.add_argument('--generator',
                            help='Name of the generator used to generate the atmosphere. \
                                  or additional generator arguments: python scripts/render_radiance_toa.py --generator GENERATOR --help. \
                                  See generate.py for more documentation.')
        parser.add_argument('--add_rayleigh',
                            action='store_true',
                            help='Overlay the atmosphere with (known) Rayleigh scattering due to air molecules. \
                                  Temperature profile is taken from AFGL measurements of summer mid-lat.')

        add_rayleigh = subparser.parse_known_args()[0].add_rayleigh
        generator = subparser.parse_known_args()[0].generator

        CloudGenerator = None
        if generator:
            CloudGenerator = getattr(shdom.Generate, generator)
            parser = CloudGenerator.update_parser(parser)

        AirGenerator = None
        if add_rayleigh:
            AirGenerator = shdom.Generate.AFGLSummerMidLatAir
            parser = AirGenerator.update_parser(parser)

        self.args = parser.parse_args()
        self.cloud_generator, self.air_generator = self.init_generators(CloudGenerator, AirGenerator)

    def init_generators(self, CloudGenerator, AirGenerator):
        """
        Initialize the medium generators. The cloud generator also loads the Mie scattering
        tables for the given wavelengths at this point.

        Parameters
        -------
        CloudGenerator: a shdom.Generator class object.
            Creates the cloudy medium.
        AirGenerator: a shdom.Air class object
            Creates the scattering due to air molecules

        Returns
        -------
        cloud_generator: a shdom.CloudGenerator object.
            Creates the cloudy medium. The loading of Mie tables takes place at this point.
        air_generator: a shdom.AirGenerator object
            Creates the scattering due to air molecules
        """
        cloud_generator = CloudGenerator(self.args)
        for wavelength in self.args.wavelength:
            table_path = self.args.mie_base_path.replace('<wavelength>', '{}'.format(shdom.int_round(wavelength)))
            cloud_generator.add_mie(table_path)

        air_generator = None
        if self.args.add_rayleigh:
            air_generator = AirGenerator(self.args)
        return cloud_generator, air_generator

    def calccenterofmass(self, scatterer):
        lwc = scatterer.lwc.data
        mx = np.sum(np.sum(lwc, 2), 1)
        my = np.sum(np.sum(lwc, 2), 0)
        mz = np.sum(np.sum(lwc, 0), 0)
        com_x = sum(mx * scatterer.grid.x) / sum(mx)
        com_y = sum(my * scatterer.grid.y) / sum(my)
        com_z = sum(mz * scatterer.grid.z) / sum(mz)
        return com_x, com_y, com_z

    def get_camera_parm(self, scatterer):

        self.args.camera_height = self.args.camera_height[0]
        if self.args.camera_height == 'TOA':
            self.args.camera_height = scatterer.grid.z[-1]
        else:
            self.args.camera_height = float(self.args.camera_height)

        com_x, com_y, com_z = self.calccenterofmass(scatterer=scatterer)
        look_at_point = np.array([com_x, com_y, com_z])
        L_list = (np.sign(self.args.azimuth)) * (self.args.camera_height - com_z) * np.tan(np.deg2rad(self.args.zenith))#km
        time_list = L_list * 1e3 / self.args.camera_velocity  # sec
        camera_position_list = np.zeros([L_list.shape[0], 3])

        # move the camera instead of the cloud
        camera_position_list[:, 0] = com_x
        camera_position_list[:, 1] = com_y + np.asarray(L_list)
        camera_position_list[:, 2] = self.args.camera_height
        return look_at_point, camera_position_list, time_list

    def get_dynamic_medium(self, cloud, time_list):
        """
        Generate an atmospheric domain

        returns
        -------
        atmosphere: shdom.Medium object.
            Creates the atmospheric medium.
        """
        dynamic_scatterer = shdom.DynamicScatterer()
        dynamic_scatterer.generate_dynamic_scatterer(scatterer=cloud, time_list=time_list,
                                                     scatterer_velocity_list=np.asarray(self.args.cloud_velocity))

        if self.args.add_rayleigh:
            air = self.air_generator.get_scatterer(cloud.wavelength)

        else:
            assert True, 'is not supported'

        dynamic_medium = shdom.DynamicMedium(dynamic_scatterer, air=air)

        return dynamic_medium

    def get_solver(self, dynamic_medium):
        """
        Define an RteSolverArray object

        Parameters
        ----------
        medium: shdom.Medium
            The atmospheric Medium for which the RTE is solved

        Returns
        -------
        rte_solver: shdom.RteSolverArray object
            A solver array initialized to the input medium and numerical and scene arguments
        """
        if self.args.solar_spectrum:
            solar_spectrum = shdom.SolarSpectrum()
            solar_fluxes = solar_spectrum.get_monochrome_solar_flux(self.args.wavelength)
            solar_fluxes = solar_fluxes / max(solar_fluxes)
        else:
            solar_fluxes = np.full_like(self.args.wavelength, 1.0)

        numerical_params = shdom.NumericalParameters(
            num_mu_bins=self.args.num_mu,
            num_phi_bins=self.args.num_phi,
            split_accuracy=self.args.split_accuracy,
            adapt_grid_factor=self.args.adapt_grid_factor,
            solution_accuracy=self.args.solution_accuracy
        )
        for wavelength, solar_flux in zip([dynamic_medium.wavelength], solar_fluxes):
            scene_params = shdom.SceneParameters(
                wavelength=wavelength,
                source=shdom.SolarSource(self.args.solar_azimuth, self.args.solar_zenith, solar_flux),
                surface=shdom.LambertianSurface(albedo=self.args.surface_albedo)
            )
        dynamic_solver = shdom.DynamicRteSolver(scene_params=scene_params, numerical_params=numerical_params)
        dynamic_solver.set_dynamic_medium(dynamic_medium)

        dynamic_solver.solve(maxiter=100)

        return dynamic_solver

    def get_projections(self, camera_position_list, look_at_point):

        projections = shdom.MultiViewProjection()

        for camera_azimuth, camera_zenith, camera_position in zip(self.args.azimuth, self.args.zenith,
                                                                  camera_position_list):
            projection = shdom.PerspectiveProjection(fov=30,
                                            nx=200, ny=200, x=camera_position[0], y=camera_position[1],
                                            z=camera_position[2])
            projection.look_at_transform(point=look_at_point, up=[1.0, 0.0, 0.0])

            projections.add_projection(projection)
        return projections

    def get_dynamic_medium(self, dynamic_scatterer):
        """
        Generate an atmospheric domain

        returns
        -------
        atmosphere: shdom.Medium object.
            Creates the atmospheric medium.
        """
        atmosphere = shdom.DynamicMedium()
        if self.args.add_rayleigh:
            air = self.air_generator.get_scatterer(dynamic_scatterer.wavelength)
            atmosphere = shdom.DynamicMedium(dynamic_scatterer, air=air)
        else:
            NotImplemented()
        return atmosphere

    def render(self,projections, dynamic_solver,time_list):
        """
        Define a sensor and render an orthographic image at the top domain.

        Parameters
        ----------
        bounding_box: shdom.BoundingBox object
            Used to compute the projection that will see the entire bounding box.
        rte_solver: shdom.RteSolverArray
            A solver that contains the solution to the RTE in the medium

        Returns
        -------
        measurements: shdom.Measurements object
            Encapsulates the measurements and sensor geometry for later optimization
        """

        dynamic_camera = shdom.DynamicCamera(self.sensor, projections)
        images = dynamic_camera.render(dynamic_solver, self.args.n_jobs)
        measurements = shdom.DynamicMeasurements(dynamic_camera, images=images, wavelength=dynamic_solver.wavelength,time_list=time_list)
        return measurements

    def main(self):
        """
        Main forward rendering script.
        """
        self.parse_arguments()
        cloud = self.cloud_generator.get_scatterer()
        look_at_point, camera_position_list, time_list = self.get_camera_parm(cloud)
        dynamic_scatterer = shdom.DynamicScatterer()
        dynamic_scatterer.generate_dynamic_scatterer(scatterer=cloud, time_list=time_list,
                                                     scatterer_velocity_list=self.args.cloud_velocity)
        dynamic_medium = self.get_dynamic_medium(dynamic_scatterer)

        rte_solvers = self.get_solver(dynamic_medium)
        projections = self.get_projections(camera_position_list, look_at_point)
        measurements = self.render(projections, rte_solvers,time_list)

        # Save measurements, medium and solver parameters
        shdom.save_dynamic_forward_model(self.args.output_dir, dynamic_medium, rte_solvers, measurements)


if __name__ == "__main__":
    script = RenderScript(scatterer_name='cloud')
    script.main()
