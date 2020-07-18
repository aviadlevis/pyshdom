import os, time
import numpy as np
import argparse
import shdom
import scipy.io as sio


class ReadReasultsScript(object):
    def __init__(self, scatterer_name='cloud'):
        self.scatterer_name = scatterer_name

    def read_reasults_args(self, parser):
        """
        Add common optimization arguments that may be shared across scripts.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.

        Returns
        -------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.
        """
        parser.add_argument('--input_dir',
                            help='Path to an input directory where the forward modeling parameters are be saved. \
                                  This directory will be used to save the optimization results and progress.')
        parser.add_argument('--load_path',
                            help='Reload an optimizer or checkpoint and continue optimizing from that point.')
        parser.add_argument('--log',
                            help='Write intermediate TensorBoardX results. \
                                  The provided string is added as a comment to the specific run.')
        parser.add_argument('--use_forward_grid',
                            action='store_true',
                            help='Use the same grid for the reconstruction. This is a sort of inverse crime which is \
                                  usefull for debugging/development.')
        parser.add_argument('--use_forward_mask',
                            action='store_true',
                            help='Use the ground-truth cloud mask. This is an inverse crime which is \
                                  usefull for debugging/development.')
        parser.add_argument('--add_noise',
                            action='store_true',
                            help='currently only supports AirMSPI noise model. \
                                  See shdom.AirMSPINoise object for more info.')
        parser.add_argument('--n_jobs',
                            default=1,
                            type=int,
                            help='(default value: %(default)s) Number of jobs for parallel rendering. n_jobs=1 uses no parallelization')
        parser.add_argument('--globalopt',
                            action='store_true',
                            help='Global optimization with basin-hopping.'
                                 'For more info see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html')
        parser.add_argument('--maxiter',
                            default=1000,
                            type=int,
                            help='(default value: %(default)s) Maximum number of L-BFGS iterations.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--maxls',
                            default=30,
                            type=int,
                            help='(default value: %(default)s) Maximum number of line search steps (per iteration).'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--disp',
                            choices=[True, False],
                            default=True,
                            type=np.bool,
                            help='(default value: %(default)s) Display optimization progression.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--gtol',
                            default=1e-16,
                            type=np.float32,
                            help='(default value: %(default)s) Stop criteria for the maximum projected gradient.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--ftol',
                            default=1e-16,
                            type=np.float32,
                            help='(default value: %(default)s) Stop criteria for the relative change in loss function.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--stokes_weights',
                            nargs=4,
                            default=[1.0, 0.0, 0.0, 0.0],
                            type=float,
                            help='(default value: %(default)s) Loss function weights for stokes vector components [I, Q, U, V]')
        parser.add_argument('--loss_type',
                            choices=['l2', 'normcorr'],
                            default='l2',
                            help='Different loss functions for optimization. Currently only l2 is supported.')
        return parser

    def medium_args(self, parser):
        """
        Add common medium arguments that may be shared across scripts.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.

        Returns
        -------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.
        """
        parser.add_argument('--use_forward_cloud_velocity',
                            action='store_true',
                            help='Use the ground truth cloud velocity.')
        parser.add_argument('--use_forward_albedo',
                            action='store_true',
                            help='Use the ground truth albedo.')
        parser.add_argument('--use_forward_phase',
                            action='store_true',
                            help='Use the ground-truth phase reconstruction.')
        parser.add_argument('--radiance_threshold',
                            default=[0.02],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Threshold for the radiance to create a cloud mask.' \
                            'Threshold is either a scalar or a list of length of measurements.')
        parser.add_argument('--mie_base_path',
                            default='mie_tables/polydisperse/Water_<wavelength>nm.scat',
                            help='(default value: %(default)s) Mie table base file name. ' \
                                 '<wavelength> will be replaced by the corresponding wavelength.')

        return parser


    def parse_arguments(self):
        """
        Handle all the argument parsing needed for this script.

        Returns
        -------
        args: arguments from argparse.ArgumentParser()
            Arguments required for this script.
        cloud_generator: a shdom.CloudGenerator object.
            Creates the cloudy medium. The loading of Mie tables takes place at this point.
        air_generator: a shdom.AirGenerator object
            Creates the scattering due to air molecules
        """
        parser = argparse.ArgumentParser()
        parser = self.read_reasults_args(parser)
        parser = self.medium_args(parser)

        # Additional arguments to the parser
        subparser = argparse.ArgumentParser(add_help=False)
        subparser.add_argument('--init')
        subparser.add_argument('--add_rayleigh', action='store_true')
        parser.add_argument('--init',
                            default='Homogeneous',
                            help='(default value: %(default)s) Name of the generator used to initialize the atmosphere. \
                                  for additional generator arguments: python scripts/optimize_extinction_lbgfs.py --generator GENERATOR --help. \
                                  See generate.py for more documentation.')
        parser.add_argument('--add_rayleigh',
                            action='store_true',
                            help='Overlay the atmosphere with (known) Rayleigh scattering due to air molecules. \
                                  Temperature profile is taken from AFGL measurements of summer mid-lat.')

        init = subparser.parse_known_args()[0].init
        add_rayleigh = subparser.parse_known_args()[0].add_rayleigh

        CloudGenerator = None
        if init:
            CloudGenerator = getattr(shdom.generate, init)
            parser = CloudGenerator.update_parser(parser)

        AirGenerator = None
        if add_rayleigh:
            AirGenerator = shdom.generate.AFGLSummerMidLatAir
            parser = AirGenerator.update_parser(parser)

        self.args = parser.parse_args()
        self.cloud_generator = CloudGenerator(self.args) if CloudGenerator is not None else None
        self.air_generator = AirGenerator(self.args) if AirGenerator is not None else None

    def get_medium_estimator(self, measurements, ground_truth):
        """
        """
        wavelength = measurements.wavelength


        # Find a cloud mask for non-cloudy grid points
        if self.args.use_forward_mask:
            mask_list = ground_truth.get_mask(threshold=1.0)
        else:
            # carver = shdom.SpaceCarver(measurements)
            # mask = carver.carve(grid, agreement=0.9, thresholds=self.args.radiance_threshold)
            mask_list = []

        # Define the known albedo and phase: either ground-truth or specified, but it is not optimized.
        # if self.args.use_forward_albedo is False or self.args.use_forward_phase is False:
        #     table_path = self.args.mie_base_path.replace('<wavelength>', '{}'.format(shdom.int_round(wavelength)))
        #     self.cloud_generator.add_mie(table_path)
        # else:
        #     NotImplemented()


        albedo = ground_truth.get_albedo()


        phase = ground_truth.get_phase()

        # phase = self.cloud_generator.get_phase(wavelength, phase.grid)

        extinction = shdom.DynamicGridDataEstimator(ground_truth.get_extinction(),
                                             min_bound=1e-3,
                                             max_bound=2e2)
        cloud_estimator = shdom.DynamicScattererEstimator(wavelength, extinction, albedo, phase,measurements.time_list)
        cloud_estimator.set_mask(mask_list)

        # Create a medium estimator object (optional Rayleigh scattering)
        air = self.air_generator.get_scatterer(wavelength)
        medium_estimator = shdom.DynamicMediumEstimator(cloud_estimator, air)

        return medium_estimator

    def get_summary_writer(self, measurements, ground_truth):
        """
        Define a SummaryWriter object

        Parameters
        ----------
        measurements: shdom.Measurements object
            The acquired measurements.
        ground_truth: shdom.Scatterer
            The ground-truth scatterer for monitoring

        Returns
        -------
        writer: shdom.SummaryWriter object
            A logger for the TensorboardX.
        """
        writer = None
        if self.args.log is not None:
            log_dir = os.path.join(self.args.input_dir, 'logs', self.args.log + '-' + time.strftime("%d-%b-%Y-%H:%M:%S"))
            writer = shdom.SummaryWriter(log_dir)
            writer.save_checkpoints(ckpt_period=20 * 60)
            writer.monitor_loss()
            writer.monitor_shdom_iterations()
            writer.monitor_images(measurements=measurements, ckpt_period=5 * 60)

            # Compare estimator to ground-truth
            writer.monitor_scatterer_error(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_domain_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_scatter_plot(estimator_name=self.scatterer_name, ground_truth=ground_truth, dilute_percent=0.4)
            writer.monitor_horizontal_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth, ground_truth_mask=ground_truth.get_mask(threshold=0.0001))

        return writer

    def load_forward_model(self, input_directory):
        """
        Load the ground-truth medium, rte_solver and measurements which define the forward model

        Parameters
        ----------
        input_directory: str
            The input directory where the forward model is saved

        Returns
        -------
        ground_truth: shdom.DynamicScatterer
            The ground truth scatterer
        rte_solver: shdom.RteSolverArray
            The rte solver with the numerical and scene parameters
        measurements: shdom.Measurements
            The acquired measurements
        """
        # Load forward model and measurements
        dynamic_medium, dynamic_solver, measurements = shdom.load_dynamic_forward_model(input_directory)

        # Get optical medium ground-truth
        dynamic_scatterer = dynamic_medium.get_dynamic_scatterer()
        if dynamic_scatterer.type == 'MicrophysicalScatterer':
            ground_truth = dynamic_scatterer.get_dynamic_optical_scatterer(measurements.wavelength)
        return ground_truth, dynamic_solver, measurements

    def get_results(self):
        """
        """
        self.parse_arguments()
        ground_truth, dynamic_solver, measurements = self.load_forward_model(self.args.input_dir)
        wavelength = measurements.wavelength

        # Define the grid for reconstruction
        if self.args.use_forward_grid:
            dynamic_grid = []
            for i in range(ground_truth.num_scatterers):
                dynamic_grid.append(ground_truth.get_extinction()[i].grid)
            grid = dynamic_grid[0]
            grid = shdom.Grid(x = grid.x - grid.xmin, y = grid.y - grid.ymin, z = grid.z)
        else:
            extinction_grid = albedo_grid = phase_grid = self.cloud_generator.get_grid()
            grid = extinction_grid

        if self.args.use_forward_cloud_velocity:
            cloud_velocity = ground_truth.get_velocity()
            cloud_velocity = cloud_velocity[0]*1000
            cloud_velocity[1] = -6
        else:
            cloud_velocity = None

        # Find a cloud mask for non-cloudy grid points
        if self.args.use_forward_mask:
            mask_list = ground_truth.get_mask(threshold=0.001)
        else:
            dynamic_carver = shdom.DynamicSpaceCarver(measurements)
            mask_list, dynamic_grid, cloud_velocity = dynamic_carver.carve(grid, agreement=0.8,
                                time_list = measurements.time_list, thresholds=self.args.radiance_threshold,
                                vx_max = 5, vy_max=0, gt_velocity = cloud_velocity)
            show_mask=1
            if show_mask:
                a = (mask_list[0].data).astype(int)
                b = ((ground_truth.get_mask(threshold=0.001)[4].resample(dynamic_grid[4]).data)).astype(int)
                print(np.sum((a > b)))
                print(np.sum((a < b)))
                shdom.cloud_plot(a)
                shdom.cloud_plot(b)

        if self.args.use_forward_albedo:
            albedo = ground_truth.get_albedo()
        else:
            # albedo = self.cloud_generator.get_albedo(wavelength, albedo_grid)
            NotImplemented()

        if self.args.use_forward_phase:
            phase = ground_truth.get_phase()
        else:
            NotImplemented()
        # phase = self.cloud_generator.get_phase(wavelength, phase.grid)
        extinction = shdom.DynamicGridDataEstimator(ground_truth.get_extinction(dynamic_grid=dynamic_grid),
                                                    min_bound=1e-3,
                                                    max_bound=2e2)

        # if self.args.use_forward_grid:
        #     extinction = shdom.DynamicGridDataEstimator(ground_truth.get_extinction(),
        #                                          min_bound=1e-3,
        #                                          max_bound=2e2)
        # else:
        #     if self.args.use_forward_mask:
        #         grid = ground_truth.get_extinction()[0].grid
        #         grid = shdom.Grid(x=grid.x - grid.xmin, y=grid.y - grid.ymin, z=grid.z)
        #         dynamic_carver = shdom.DynamicSpaceCarver(measurements)
        #         _, dynamic_grid, _ = dynamic_carver.carve(grid, agreement=0.8,
        #                                                   time_list=ground_truth.time_list,
        #                                                   thresholds=self.args.radiance_threshold)
        #     else:
        #         dynamic_extinction = []
        #         for grid, ext in zip(dynamic_grid,ground_truth.get_extinction()):
        #             dynamic_extinction.append(shdom.GridData(grid.grid,ext.data))
        #         extinction = shdom.DynamicGridDataEstimator(dynamic_extinction,
        #                                          min_bound=1e-3,
        #                                          max_bound=2e2)

        cloud_estimator = shdom.DynamicScattererEstimator(wavelength, extinction, albedo, phase,time_list=measurements.time_list)
        cloud_estimator.set_mask(mask_list)

        # Create a medium estimator object (optional Rayleigh scattering)
        air = self.air_generator.get_scatterer(wavelength)
        medium_estimator = shdom.DynamicMediumEstimator(cloud_estimator, air,cloud_velocity)

        return medium_estimator

    def extinction_compare(self,ground_truth, estimated_dynamic_medium):
        gt_extinction_stack = []
        estimated_extinction_stack = []

        for gt_extinction, medium_estimator in zip(ground_truth.get_extinction(), estimated_dynamic_medium):
            estimated_extinction = medium_estimator.scatterers[self.scatterer_name].extinction.data
            gt_extinction_stack.append(gt_extinction.data)
            estimated_extinction_stack.append(estimated_extinction)

        gt_extinction_stack = np.stack(gt_extinction_stack, axis=3)
        estimated_extinction_stack = np.stack(estimated_extinction_stack, axis=3)

        sio.savemat('gt_extinction.mat', {'gt_extinction': gt_extinction_stack})
        sio.savemat('estimated_extinction.mat', {'estimated_extinction': estimated_extinction_stack})



    def main(self):
        """
        Main optimization script
        """
        ground_truth, estimated_dynamic_medium = self.get_results()
        self.extinction_compare(ground_truth, estimated_dynamic_medium)





if __name__ == "__main__":
    script = ReadReasultsScript(scatterer_name='cloud')
    script.main()




