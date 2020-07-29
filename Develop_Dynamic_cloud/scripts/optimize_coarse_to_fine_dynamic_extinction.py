import os, time
import numpy as np
import argparse
import shdom
import scipy.ndimage as sci


class OptimizationScript(object):
    """
    Optimize: Extinction
    --------------------
    Estimate the extinction coefficient based on monochrome radiance measurements.
    In this script, the phase function, albedo and rayleigh scattering are assumed known and are not estimated.

    Measurements are simulated measurements using a forward rendering script
    (e.g. scripts/render_radiance_toa.py).

    For example usage see the README.md

    For information about the command line flags see:
      python scripts/optimize_extinction_lbfgs.py --help

    Parameters
    ----------
    scatterer_name: str
        The name of the scatterer that will be optimized.
    """
    def __init__(self, scatterer_name='cloud'):
        self.scatterer_name = scatterer_name

    def optimization_args(self, parser):
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
        parser.add_argument('--reload_path',
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
        parser.add_argument('--reg_const',
                            default=0,
                            type=float,
                            help='(default value: %(default)s) Regularization constant. reg_const=0 uses no regularization')
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
        parser.add_argument('--use_forward_cloud_velocity',
                            action='store_true',
                            help='Use the ground truth cloud velocity.')
        parser.add_argument('--num_of_mediums_list',
                            default=[1, 3, 9],
                            type=int,
                            help='(default value: %(default)s) Loss function weights for stokes vector components [I, Q, U, V]')
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
        parser = self.optimization_args(parser)
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
            CloudGenerator = getattr(shdom.dynamic_scene, init)
            parser = CloudGenerator.update_parser(parser)

        AirGenerator = None
        if add_rayleigh:
            AirGenerator = shdom.generate.AFGLSummerMidLatAir
            parser = AirGenerator.update_parser(parser)

        self.args = parser.parse_args()
        self.cloud_generator = CloudGenerator(self.args) if CloudGenerator is not None else None
        self.air_generator = AirGenerator(self.args) if AirGenerator is not None else None

    def get_medium_estimator(self, measurements, ground_truth, num_of_mediums, coarse_extinction=None):
        """
        """
        wavelength = measurements.wavelength

        # Define the grid for reconstruction
        if self.args.use_forward_grid:
            #TODO
            dynamic_grid = []
            for i in range(0, ground_truth.num_scatterers, int(ground_truth.num_scatterers/num_of_mediums)):
                combined_grid = ground_truth.get_extinction()[i].grid
                for j in  range(1, int(ground_truth.num_scatterers/num_of_mediums)):
                    combined_grid = combined_grid + ground_truth.get_extinction()[i+j].grid
                dynamic_grid.append(combined_grid)
            grid = dynamic_grid[0]
            grid = shdom.Grid(x = grid.x - grid.xmin, y = grid.y - grid.ymin, z = grid.z)
        else:
            extinction_grid = albedo_grid = phase_grid = self.cloud_generator.get_grid()
            grid = extinction_grid + albedo_grid + phase_grid
            dynamic_grid = [grid] * num_of_mediums

        if self.args.use_forward_cloud_velocity:
            if ground_truth.num_scatterers > 1:
                cloud_velocity = ground_truth.get_velocity()
                cloud_velocity = cloud_velocity[0]*1000 #km/sec to m/sec
            else:
                cloud_velocity = [0,0,0]
        else:
            cloud_velocity = None

        # Find a cloud mask for non-cloudy grid points
        if self.args.use_forward_mask:
            mask_list = ground_truth.get_mask(threshold=0.000001)
            a = (mask_list[0].data).astype(int)
            shdom.cloud_plot(a)
        else:
            dynamic_carver = shdom.DynamicSpaceCarver(measurements)
            mask_list, dynamic_grid, cloud_velocity = dynamic_carver.carve(grid, agreement=0.9,
                                time_list = measurements.time_list, thresholds=self.args.radiance_threshold,
                                vx_max = 5, vy_max=0, gt_velocity = cloud_velocity)
            show_mask=1
            if show_mask:
                a = (mask_list[0].data).astype(int)
                b = ((ground_truth.get_mask(threshold=0.0000001)[0].resample(dynamic_grid[0]).data)).astype(int)
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
        # extinction = shdom.DynamicGridDataEstimator(ground_truth.get_extinction(dynamic_grid=dynamic_grid),
        #                                             init_val=self.args.extinction,
        #                                             min_bound=1e-5,
        #                                             max_bound=2e2)

        if coarse_extinction is None:
            extinction = shdom.DynamicGridDataEstimator(self.cloud_generator.get_extinction(measurements.wavelength, dynamic_grid),
                                                        min_bound=1e-5,
                                                        max_bound=2e2)
        else:
            previous_num_of_mediums = len(coarse_extinction)
            extinction_list = []
            for ext in coarse_extinction:
                for i in range(int(num_of_mediums/previous_num_of_mediums)):
                    extinction_list.append(ext)

            extinction = shdom.DynamicGridDataEstimator(
                extinction_list,
                min_bound=1e-5,
                max_bound=2e2)


        kw_optical_scatterer = {"extinction": extinction, "albedo": albedo, "phase": phase}

        cloud_estimator = shdom.DynamicScattererEstimator(wavelength=wavelength, time_list=measurements.time_list, **kw_optical_scatterer)
        cloud_estimator.set_mask(mask_list)

        # Create a medium estimator object (optional Rayleigh scattering)
        air = self.air_generator.get_scatterer(wavelength)
        medium_estimator = shdom.DynamicMediumEstimator(cloud_estimator, air.resample(grid),cloud_velocity)

        return medium_estimator

    def get_summary_writer(self, measurements, ground_truth, num_of_mediums):
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
            log_dir = os.path.join(self.args.input_dir, 'logs', self.args.log + 'num_of_mediums_{}'.format(num_of_mediums) + '-' + time.strftime("%d-%b-%Y-%H:%M:%S"))
            writer = shdom.DynamicSummaryWriter(log_dir)
            writer.save_checkpoints(ckpt_period=20 * 60)
            writer.monitor_loss()
            writer.monitor_shdom_iterations()
            writer.monitor_images(measurements=measurements, ckpt_period=5 * 60)
            # writer.monitor_time_smoothness()

            # Compare estimator to ground-truth
            writer.monitor_scatterer_error(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_domain_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_scatter_plot(estimator_name=self.scatterer_name, ground_truth=ground_truth, dilute_percent=0.8)
            writer.monitor_horizontal_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth, ground_truth_mask=ground_truth.get_mask(threshold=0.000001))

            # save parse_arguments
            self.save_args(log_dir)
        return writer

    def save_args(self,log_dir):
        text_file = open(log_dir+"/Input_args.txt", "w")
        for data in self.args.__dict__:
            text_file.write("{} : {}\n".format(data, self.args.__dict__[data]))
        text_file.close()

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
        assert dynamic_scatterer.type == 'MicrophysicalScatterer'
        ground_truth = dynamic_scatterer

        return ground_truth, dynamic_solver, measurements

    def get_optimizer(self, ground_truth, dynamic_solver, measurements , num_of_mediums, coarse_extinction):
        """
        Define an Optimizer object

        Returns
        -------
        optimizer: shdom.Optimizer object
            An optimizer object.
        """

        # Initialize a Medium Estimator
        medium_estimator = self.get_medium_estimator(measurements, ground_truth, num_of_mediums,coarse_extinction=coarse_extinction)

        # Rte solver adjustment for coarse to fine optimization assuming all solvers are the same
        dynamic_solver.set_dynamic_medium(medium_estimator)

        # Initialize TensorboardX logger
        writer = self.get_summary_writer(measurements, ground_truth, num_of_mediums)
        regularization_const = self.args.reg_const

        # Initialize a LocalOptimizer
        options = {
            'maxiter': self.args.maxiter,
            'maxls': self.args.maxls,
            'disp': self.args.disp,
            'gtol': self.args.gtol,
            'ftol': self.args.ftol,
        }
        optimizer = shdom.DynamicLocalOptimizer('L-BFGS-B', options=options, n_jobs=self.args.n_jobs,
                                                regularization_const=regularization_const)
        optimizer.set_measurements(measurements)
        optimizer.set_dynamic_solver(dynamic_solver)
        optimizer.set_medium_estimator(medium_estimator)
        optimizer.set_writer(writer)

        # Reload previous state
        if self.args.reload_path is not None:
            optimizer.load_state(self.args.reload_path)
        return optimizer

    def get_current_ground_truth(self, ground_truth, num_of_mediums):
        time_list = np.mean(np.split(np.array(ground_truth.time_list), num_of_mediums), 1)
        temporary_scatterer_list = []
        for temporary_scatterer in ground_truth.temporary_scatterer_list:
            temporary_scatterer_list.append(temporary_scatterer.scatterer)
        avg = len(temporary_scatterer_list) / float(num_of_mediums)
        splited_temporary_scatterer_list = []
        last = 0.0

        while last < len(temporary_scatterer_list):
            splited_temporary_scatterer_list.append(temporary_scatterer_list[int(last):int(last + avg)])
            last += avg
        new_ground_truth = shdom.DynamicScatterer()
        new_temporary_scatterer_list = []
        for temporary_scatterer_chunk, time in zip(splited_temporary_scatterer_list, time_list):
            averaged_temporary_scatterer = shdom.TemporaryScatterer(self.average_scatterers(temporary_scatterer_chunk), time)
            new_temporary_scatterer_list.append(averaged_temporary_scatterer)
        new_ground_truth.add_temporary_scatterer(new_temporary_scatterer_list)
        return new_ground_truth

    def average_scatterers(self, scatterer_list):
        first = True
        for scatterer in scatterer_list:
            if first:
                lwc = scatterer.lwc
                reff = scatterer.reff
                veff = scatterer.veff
                wavelength = scatterer.wavelength
                first = False
            else:
                assert wavelength == scatterer.wavelength
                lwc = lwc + scatterer.lwc
                reff = reff + scatterer.reff
                veff = veff + scatterer.veff

        lwc._data /= len(scatterer_list)
        reff._data /= len(scatterer_list)
        veff._data /= len(scatterer_list)

        scatterer = shdom.MicrophysicalScatterer(lwc, reff, veff)
        mie = shdom.MiePolydisperse()
        table_path = 'mie_tables/polydisperse/Water_{}nm.scat'.format(shdom.int_round(wavelength))
        mie.read_table(table_path)
        reff._data[reff._data<mie.size_distribution.reff.min()] = mie.size_distribution.reff.min()
        veff._data[veff._data<mie.size_distribution.veff.min()] = mie.size_distribution.veff.min()
        scatterer.add_mie(mie)
        return scatterer.get_optical_scatterer(wavelength)

    def main(self):
        """
        Main optimization script
        """
        self.parse_arguments()
        coarse_extinction = None
        ground_truth, dynamic_solver, measurements = self.load_forward_model(self.args.input_dir)
        # Add noise (currently only supports AirMSPI noise model)
        if self.args.add_noise:
            measurements.set_noise(shdom.AirMSPINoise())

        for num_of_mediums in self.args.num_of_mediums_list:

            current_measurements = measurements.downsample_viewed_mediums(num_of_mediums)
            current_ground_truth = self.get_current_ground_truth(ground_truth, num_of_mediums)
            local_optimizer = self.get_optimizer(current_ground_truth, dynamic_solver, current_measurements ,num_of_mediums,coarse_extinction)

            # Optimization process
            num_global_iter = 1
            if self.args.globalopt:
                global_optimizer = shdom.GlobalOptimizer(local_optimizer=local_optimizer)
                result = global_optimizer.minimize(niter_success=20, T=1e-3)
                num_global_iter = result.nit
                result = result.lowest_optimization_result
                local_optimizer.set_state(result.x)
            else:
                result = local_optimizer.minimize()

            print('\n------------------ Optimization Finished ------------------\n')
            print('Number global iterations: {}'.format(num_global_iter))
            print('Success: {}'.format(result.success))
            print('Message: {}'.format(result.message))
            print('Final loss: {}'.format(result.fun))
            print('Number iterations: {}'.format(result.nit))

            # Save optimizer state
            save_dir = local_optimizer.writer.dir if self.args.log is not None else self.args.input_dir
            local_optimizer.save_state(os.path.join(save_dir, '{}_mediums_final_state.ckpt'.format(num_of_mediums)))
            coarse_extinction = []
            for tem_scat in local_optimizer.medium.dynamic_scatterer_estimator.temporary_scatterer_estimator_list:
                coarse_extinction.append(tem_scat.get_scatterer().extinction)


if __name__ == "__main__":
    script = OptimizationScript(scatterer_name='cloud')
    script.main()




