import os, time
import numpy as np
import argparse
import shdom
import scipy.io as sio
from Develop_Dynamic_cloud.scripts.optimize_dynamic_extinction_AirMSPI_lbfgs import OptimizationScript as ExtinctionOptimizationScript

def is_float(value):
  try:
    float(value)
    return True
  except:
    return False

def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

class ReadReasultsScript(ExtinctionOptimizationScript):
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
        parser.add_argument('--load_path',
                            help='loading path.')
        parser.add_argument('--ckpt',
                            help='check point file to read.')

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
        self.args = parser.parse_args()
        self.load_args(self.args.load_path)


        init = self.args.init
        add_rayleigh = self.args.add_rayleigh

        CloudGenerator = None
        if init:
            CloudGenerator = getattr(shdom.dynamic_scene, init)
            # parser = CloudGenerator.update_parser(parser)

        AirGenerator = None
        if add_rayleigh:
            AirGenerator = shdom.generate.AFGLSummerMidLatAir
            # parser = AirGenerator.update_parser(parser)

        if self.args.lwc == 'None':
            self.args.lwc = 1
        self.cloud_generator = CloudGenerator(self.args) if CloudGenerator is not None else None
        self.air_generator = AirGenerator(self.args) if AirGenerator is not None else None

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


    def get_results(self):
        """
        Define an Optimizer object

        Returns
        -------
        optimizer: shdom.Optimizer object
            An optimizer object.
        """
        measurements = shdom.AirMSPIDynamicMeasurements()
        measurements.load_airmspi_measurements(self.args.input_dir)
        cv_index = self.args.use_cross_validation
        self.args.num_mediums = int(self.args.num_mediums)
        self.args.use_cross_validation = int(self.args.use_cross_validation)
        if int(self.args.num_mediums) < 0:
            self.args.num_mediums = len(measurements.time_list)
            if cv_index >= 0:
                self.args.num_mediums -= 1

        # Initialize a Medium Estimator
        medium_estimator = self.get_medium_estimator(measurements)


        if cv_index >= 0:
            cv_measurement, measurements = measurements.get_cross_validation_measurements(cv_index)
        measurements = measurements.downsample_viewed_mediums(self.args.num_mediums)


        optimizer = shdom.DynamicLocalOptimizer('L-BFGS-B')
        optimizer.set_measurements(measurements)
        optimizer.set_medium_estimator(medium_estimator)
        if cv_index >= 0:
            optimizer.set_cross_validation_param(None, cv_measurement, cv_index)
        # Reload previous state
        log_dir = os.path.join(self.args.load_path + '/' + self.args.ckpt)
        optimizer.load_results(log_dir)
        return optimizer

    def save3d(self, optimizer):
        estimated_extinction_stack = []
        estimated_dynamic_medium = optimizer.medium.medium_list
        for medium_estimator in estimated_dynamic_medium:
            estimated_extinction = medium_estimator.scatterers[self.scatterer_name].extinction
            estimated_extinction_stack.append(estimated_extinction.data)

        estimated_extinction_stack = np.stack(estimated_extinction_stack, axis=3)
        try:
            dx = estimated_extinction.grid.dx
            dy = estimated_extinction.grid.dy
        except AttributeError:
            dx = -1
            dy = -1

        nz = estimated_extinction.grid.nz
        dz = (estimated_extinction.grid.zmax - estimated_extinction.grid.zmin) / nz
        sio.savemat(os.path.join(self.args.load_path, 'FINAL_3D_{}.mat'.format('extinction')),
                    {'estimated_extinction': estimated_extinction_stack, 'dx': dx, 'dy': dy, 'dz': dz})

    def load_args(self, load_path):
        text_file = open(load_path+"/Input_args.txt", "r")
        while True:
            line = text_file.readline().split(' : ')
            if line == ['']:
                break
            line[1] = line[1].rstrip('\n')
            if is_float(line[1]):
                line[1] = num(line[1])
            elif line[1][0]=='[' and line[1][-1]==']':
                line[1] = [num(i) for i in line[1][1:-1].split(",")]
            elif line[1] == 'True':
                line[1] = True
            elif line[1] == 'False':
                line[1] = False
            elif line[1] == 'None':
                line[1] = None
            self.args.__dict__[line[0]] = line[1]
        text_file.close()

    def main(self):
        """
        Main optimization script
        """
        self.parse_arguments()
        optimizer = self.get_results()
        self.save3d(optimizer)
        # self.extinction_compare(ground_truth, estimated_dynamic_medium)


if __name__ == "__main__":
    script = ReadReasultsScript(scatterer_name='cloud')
    script.main()




