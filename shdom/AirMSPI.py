import h5py, os
import numpy as np
import glob
import shdom
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pymap3d
from ipywidgets import widgets
import IPython.display as Disp




class AirMSPIMeasurements(shdom.DynamicMeasurements):
    """
      A AirMSPI Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
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
    def __init__(self, camera=None, images=None, pixels=None, wavelength=None, time_list=None):
        super().__init__(camera, images, pixels, wavelength,uncertainties=None, time_list=time_list)
        self._region_of_interest = None
        self._paths = None
        self._set_valid_wavelength_range = None
        if images is not None:
            self._region_of_interest = [0, images[0].shape[0],0, images[0].shape[1]]
        if wavelength is not None:
            self._valid_wavelength_range = [np.min(wavelength), np.max(wavelength)]

    @classmethod
    def select_region_of_interest(cls, data_dir,index):
        format_ = '*.hdf'  # load
        path = sorted(glob.glob(data_dir + '/' + format_))[index]
        f = h5py.File(path, 'r')
        channels_data = f['HDFEOS']['GRIDS']
        image = np.array(channels_data['660nm_band']['Data Fields']['I'])
        image = np.dstack((image,channels_data['555nm_band']['Data Fields']['I']))
        image = np.dstack((image, channels_data['445nm_band']['Data Fields']['I']))
        image[image==-999] = 0
        image -= image.min()
        image /= image.max()

        # Select ROI
        fromCenter = False
        scale_percent = 25  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(image, dim)
        roi = cv2.selectROI(resized, fromCenter)
        x, y, w, h = tuple([int(r * 100/scale_percent) for r in roi])
        roi = [y , y+h, x, x + w]
        cv2.destroyAllWindows()
        return roi

    @classmethod
    def imshow(cls, data_dir):
        format_ = '*.hdf'  # load
        paths = sorted(glob.glob(data_dir + '/' + format_))
        images = []
        for path in paths:
            f = h5py.File(path, 'r')
            channels_data = f['HDFEOS']['GRIDS']
            image = np.array(channels_data['660nm_band']['Data Fields']['I'])
            image = np.dstack((image,channels_data['555nm_band']['Data Fields']['I']))
            image = np.dstack((image, channels_data['445nm_band']['Data Fields']['I']))
            image[image==-999] = 0
            images.append(np.array(image))
        f, axarr = plt.subplots(1, len(images), figsize=(20, 20))
        if isinstance(axarr, plt.Axes):
            axarr = [axarr]
        for ax, image in zip(axarr, images):
            image -= image.min()
            ax.imshow(image / image.max())


    def save_airmspi_measurements(self, directory):
        """
            Save the AirMSPI measurements parameters for reconstruction.

            Parameters
            ----------
            directory: str
                Directory path where the forward modeling parameters are saved.
                If the folder doesnt exist it will be created.

            """
        if not os.path.isdir(directory):
            os.makedirs(directory)
        path = os.path.join(directory, 'measurements')
        self.save(path)

    def load_airmspi_measurements(self,directory):
        # Load shdom.Measurements object (sensor geometry and radiances)
        measurements_path = os.path.join(directory, 'measurements')
        assert os.path.exists(measurements_path), 'No measurements file in directory: {}'.format(directory)
        self.load(path=measurements_path)

    def set_sun_params(self):
        for path in self._paths:
            f = h5py.File(path, 'r')
            channels_data = f['HDFEOS']['GRIDS']
            sun_azimuth_list = []
            sun_zenith_list =[]
            for param_name, param in channels_data.items():
                if "band" in param_name:
                    if int(param_name[0:3]) in self._valid_wavelength:
                        sun_azimuth = param['Data Fields']['Sun_azimuth'][self._region_of_interest[0]:self._region_of_interest[1],
                                 self._region_of_interest[2]:self._region_of_interest[3]]
                        sun_zenith = param['Data Fields']['Sun_zenith'][self._region_of_interest[0]:self._region_of_interest[1],
                                 self._region_of_interest[2]:self._region_of_interest[3]]
                        sun_azimuth_list.append(np.mean(sun_azimuth))
                        sun_zenith_list.append(180 - np.mean(sun_zenith))

            self._sun_azimuth_list = sun_azimuth_list
            self._sun_zenith_list = sun_zenith_list

    def load_from_hdf(self, data_dir, region_of_interest=[1500, 1700, 1500, 1700], valid_wavelength=[355, 380, 445, 470, 555, 660, 865, 935], type='Radiance'):
        assert len(region_of_interest) == 4, 'region of interest should be list of size 4: ' \
                                             '[pixel_x_start, pixel_x_end,pixel_y_start, pixel_y_end]'
        AirMSPI_wavelength = [355, 380, 445, 470, 555, 660, 865, 935]
        assert all(item in AirMSPI_wavelength for item in valid_wavelength), \
            'valid_wavelength can only contain AirMSPI wavelengths:  [355, 380, 445, 470, 555, 660, 865, 935]'

        assert type =='Radiance' or type =='Polarize', 'type should be Radiance or Polarize'

        format_ = '*.hdf'  # load
        paths = sorted(glob.glob(data_dir + '/' + format_))
        assert len(paths)>0, 'there are no hdf files in the given directory, try to add ../ at the beginning'
        self._paths = paths
        self._type = type
        self._valid_wavelength = valid_wavelength
        self.set_valid_wavelength_index()
        self.set_region_of_interest(region_of_interest)
        self.set_camera()
        self.set_wavelengths()
        self.set_times()
        self.set_images()
        self._pixels = self.images_to_pixels(self.images)
        self.set_sun_params()

    def set_region_of_interest(self,region_of_interest):
        for path in self._paths:
            f = h5py.File(path, 'r')
            channels_data = f['HDFEOS']['GRIDS']
            for param_name, param in channels_data.items():
                if "band" in param_name:
                    if int(param_name[0:3]) in self._valid_wavelength:
                        mask = np.array(param['Data Fields']['I.mask'][region_of_interest[0]:region_of_interest[1],
                                   region_of_interest[2]:region_of_interest[3]])
                        assert np.all(mask==1), 'Invalid region of interest'
        self._region_of_interest = region_of_interest

    def set_valid_wavelength_index(self):
        AirMSPI_wavelength = [355, 380, 445, 470, 470, 470, 555, 660, 660, 660, 865, 865, 865, 935]
        self._valid_wavelength_index = np.array(list(map(lambda wl: wl in self._valid_wavelength, AirMSPI_wavelength)))
        if self._type == 'Radiance':
            self._valid_wavelength_index[[4,5,8,9,11,12]] = False

    def get_general_info(self, f):
        general_info = {}
        data_general_info = f['HDFEOS']['ADDITIONAL']['FILE_ATTRIBUTES'].attrs
        epoch_time = data_general_info['Epoch (UTC)'].decode('utf-8')
        crop = epoch_time.find('.') -1
        crop_begining = epoch_time.find('T') + 1
        general_info['epoch_time'] = datetime.strptime(epoch_time[:crop],'%Y-%m-%dT%H:%M:%S')
        general_info['epoch_date'] = datetime.strptime(epoch_time[0 : crop_begining - 1], '%Y-%m-%d')
        general_info['aircraft_heading'] = data_general_info['Aircraft heading (degrees)']
        general_info['radiance_units'] = data_general_info['Radiance units'].decode('utf-8')
        assert general_info['radiance_units'] == 'W * (sr^-1) * (m^-2) * (nm^-1) : Watts per steradian per square meter per nanometer'
        general_info['resolution'] = data_general_info['Resolution']

        return general_info

    def build_projection_old(self, f):
        region_of_interest = self._region_of_interest
        channels_data_ancillary = f['HDFEOS']['GRIDS']['Ancillary']['Data Fields']
        x_ground = np.array(channels_data_ancillary['XDim']) / 1000  # meter to km
        x_ground = x_ground[region_of_interest[0]:region_of_interest[1]]
        y_ground = np.array(channels_data_ancillary['YDim']) / 1000  # meter to km
        y_ground = y_ground[region_of_interest[2]:region_of_interest[3]]
        # xv_ground, yv_ground = np.meshgrid(y_ground,x_ground)
        yv_ground, xv_ground  = np.meshgrid(y_ground,x_ground)


        z_ground = np.array(channels_data_ancillary['Elevation']) / 1000  # meter to km
        z_ground = z_ground[region_of_interest[0]:region_of_interest[1],region_of_interest[2]:region_of_interest[3]]

        resolution = list(z_ground.shape)
        # z = np.full(resolution,0)
        channels_data = f['HDFEOS']['GRIDS']['355nm_band']['Data Fields']
        mu = np.cos(np.deg2rad(channels_data['View_zenith'])
                    [region_of_interest[0]:region_of_interest[1],region_of_interest[2]:region_of_interest[3]])
        phi = np.deg2rad(np.array(channels_data['View_azimuth'])
                         [region_of_interest[0]:region_of_interest[1],region_of_interest[2]:region_of_interest[3]])

        u = np.sqrt(1 - mu ** 2) * np.cos(phi)
        v = np.sqrt(1 - mu ** 2) * np.sin(phi)
        w = mu

        airmspi_flight_altitude = 20.0
        z = np.full(resolution,airmspi_flight_altitude)
        t = (z - z_ground) / w

        x = xv_ground + t * u
        y = yv_ground + t * v
        if self._relative_coordinates is None:
            self._relative_coordinates = [x.min(), y.min(), z_ground]
        x -= self._relative_coordinates[0]
        y -= self._relative_coordinates[1]
        z -= self._relative_coordinates[2]
        return shdom.Projection(x=np.ravel(x,'F'), y=np.ravel(y,'F'), z=np.ravel(z,'F'), mu=np.ravel(mu,'F'), phi=np.ravel(phi,'F'), resolution=resolution)

    def build_projection(self, f):
        region_of_interest = self._region_of_interest

        channels_data_ancillary = f['HDFEOS']['GRIDS']['Ancillary']['Data Fields']
        latitude = np.array(channels_data_ancillary['Latitude'])
        latitude = latitude[region_of_interest[0]:region_of_interest[1],region_of_interest[2]:region_of_interest[3]]
        longitude = np.array(channels_data_ancillary['Longitude'])
        longitude = longitude[region_of_interest[0]:region_of_interest[1],region_of_interest[2]:region_of_interest[3]]
        elevation = np.array(channels_data_ancillary['Elevation']) /1000 # [Km]
        elevation = elevation[region_of_interest[0]:region_of_interest[1],region_of_interest[2]:region_of_interest[3]]

        resolution = list(elevation.shape)
        #-------------- Registration to 20 km altitude -------------
        # LLA(Latitude - Longitude - Altitude) to Flat surface(meters)
        airmspi_flight_altitude = 20.0 # km
        lla = [latitude, longitude, elevation]
        if self._relative_coordinates is None:
            llo = [latitude.min(), longitude.min()]  # Origin of lat - long coordinate system
            self._relative_coordinates = [llo[0], llo[1], 0]
        else:
            llo = self._relative_coordinates[:-1]
        psio = 0 # Angle between X axis and North
        href = 0 # Reference height
        flat_earth_pos = (np.array(self.lla2flat(lla, llo, psio, href))/1000) #[Km] N-E coordinates
        # flat_earth_pos = np.array(pymap3d.geodetic2ned(latitude,longitude,elevation,latitude.min()
        #                                                ,longitude.min(),elevation.min()))/1000 #[Km] N-E coordinates

        channels_data = f['HDFEOS']['GRIDS']['355nm_band']['Data Fields']
        theta = np.deg2rad(channels_data['View_zenith']
                    [region_of_interest[0]:region_of_interest[1],region_of_interest[2]:region_of_interest[3]])
        mu = np.cos(theta)
        phi = np.deg2rad(np.array(channels_data['View_azimuth'])
                         [region_of_interest[0]:region_of_interest[1],region_of_interest[2]:region_of_interest[3]])

        xTranslation = airmspi_flight_altitude * np.tan(theta) * np.cos(phi) # x - North
        yTranslation = airmspi_flight_altitude * np.tan(theta) * np.sin(phi) # Y - East

        x = flat_earth_pos[0] + xTranslation
        y = flat_earth_pos[1] + yTranslation
        z = np.full(resolution,airmspi_flight_altitude)

        # x -= self._relative_coordinates[0]
        # y -= self._relative_coordinates[1]
        # z -= self._relative_coordinates[2]
        return shdom.Projection(x=x.ravel('F'), y=y.ravel('F'), z=z.ravel('F'), mu=mu.ravel('F'), phi=phi.ravel('F'), resolution=resolution)


    def set_camera(self):
        projections = shdom.MultiViewProjection()
        self._relative_coordinates = None
        for path in self._paths:
            f = h5py.File(path, 'r')
            projection = self.build_projection(f)
            projections.add_projection(projection)
        self._camera = shdom.DynamicCamera(shdom.RadianceSensor(), projections)

    def set_wavelengths(self):
        first = True
        tol = 1.01
        for path in self._paths:
            f = h5py.File(path, 'r')
            if first:
                wavelength = np.array(f['Channel_Information']['Center_wavelength'])
                first = False
            else:
                assert np.array_equal(wavelength,np.array(f['Channel_Information']['Center_wavelength'])), \
                    'All wavelengths in the given files must be equal'
        # wavelength.sort()
        # d = np.append(True, np.diff(wavelength))
        # d=np.array(d>tol)
        # d[0] = True
        # in_range = np.logical_and(wavelength >= self._valid_wavelength_range[0], wavelength<= self._valid_wavelength_range[1])
        # d = np.logical_and(d, in_range)
        # wavelength = wavelength[d]

        self._wavelength = np.round(wavelength[self._valid_wavelength_index]/1000,3).tolist()
        self._num_channels = len(self._wavelength)

    def set_times(self):
        time_list = []
        first = True
        for path in self._paths:
            f = h5py.File(path, 'r')
            time_in_epoch = np.array(f['HDFEOS']['GRIDS']['355nm_band']['Data Fields']['Time_in_seconds_from_epoch'])
            time_in_epoch = time_in_epoch[self._region_of_interest[0]:self._region_of_interest[1],
                            self._region_of_interest[2]:self._region_of_interest[3]]
            relative_time = np.mean(time_in_epoch) #sec
            general_info = self.get_general_info(f)
            epoch_time = general_info['epoch_time']
            if first:
                self._absolute_time = epoch_time
                self._acquisition_date = general_info['epoch_date']
                first = False
            assert self._acquisition_date == general_info['epoch_date']
            time_list.append((epoch_time - self._absolute_time).total_seconds() + relative_time)
        self._time_list = np.array(time_list)

    def radiance2brf(self, radiance, sun_distance, solar_irradiance, sun_zenith):
        brf = radiance * np.pi * sun_distance ** 2 / solar_irradiance / np.cos(np.deg2rad(sun_zenith))
        return brf


    def set_images(self):
        images = []
        for path in self._paths:
            f = h5py.File(path, 'r')
            channels_data = f['HDFEOS']['GRIDS']
            solar_irradiance_table = np.array(f['Channel_Information']['Solar_irradiance_at_1_AU'])[self._valid_wavelength_index]
            sun_distance = f['HDFEOS']['ADDITIONAL']['FILE_ATTRIBUTES'].attrs['Sun distance']
            first = True
            image = None
            num_valid_wavelength = 0
            for param_name, param in channels_data.items():
                if "band" in param_name:
                    if int(param_name[0:3]) in self._valid_wavelength:
                        radiance = param['Data Fields']['I'][self._region_of_interest[0]:self._region_of_interest[1],
                                     self._region_of_interest[2]:self._region_of_interest[3]]

                        solar_irradiance = solar_irradiance_table[num_valid_wavelength]
                        sun_zenith = param['Data Fields']['Sun_zenith'][self._region_of_interest[0]:self._region_of_interest[1],
                                     self._region_of_interest[2]:self._region_of_interest[3]]
                        brf = self.radiance2brf(radiance, sun_distance, solar_irradiance, sun_zenith)

                        if first:
                            image = radiance
                            first = False
                        else:
                            image = np.dstack((image,radiance))
                        num_valid_wavelength += 1

            assert num_valid_wavelength == len(self._wavelength), 'invalid wavelength_range, try to increase it'
            if image is not None:
                images.append(np.array(image))
        self._images = images

    def images_to_pixels(self, images):
        """
        Set image list.

        Parameters
        ----------
        images: list of images,
            A list of images (multiview camera)

        Returns
        -------
        pixels: a flattened version of the image list
        """
        pixels = []

        if type(images) is not list:
            images = [images]

        for image in images:
            if self.camera.sensor.type == 'RadianceSensor':
                if len(image.shape) == 2:
                    num_channels = 1
                else:
                    num_channels = image.shape[-1]
                pixels.append(image.reshape((-1, num_channels), order='F'))

            elif self.camera.sensor.type == 'StokesSensor':
                NotImplemented()
                num_channels = image.shape[-1] if image.ndim == 4 else 1
                pixels.append(image.reshape((image.shape[0], -1, num_channels), order='F'))

            else:
                raise AttributeError('Error image dimensions: {}'.format(image.ndim))
        pixels = np.concatenate(pixels, axis=-2)
        return pixels

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
        # ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        first = True
        for projection in self.camera.projection.projection_list:
            position_x = projection.x.reshape(projection.resolution)[[0, -1, 0, -1],[0, 0, -1, -1]]
            position_y = projection.y.reshape(projection.resolution)[[0, -1, 0, -1],[0, 0, -1, -1]]
            position_z = projection.z.reshape(projection.resolution)[[0, -1, 0, -1],[0, 0, -1, -1]]
            mu = -projection.mu.reshape(projection.resolution)[[0, -1, 0, -1],[0, 0, -1, -1]]
            phi = np.pi + projection.phi.reshape(projection.resolution)[[0, -1, 0, -1],[0, 0, -1, -1]]
            if first:
                u = np.sqrt(1 - mu**2) * np.cos(phi)
                v = np.sqrt(1 - mu**2) * np.sin(phi)
                w = mu
                x = np.full(4, position_x, dtype=np.float32)
                y = np.full(4, position_y, dtype=np.float32)
                z = np.full(4, position_z, dtype=np.float32)
                first = False
            else:
                u = np.vstack((u, np.sqrt(1 - mu**2) * np.cos(phi)))
                v = np.vstack((v, np.sqrt(1 - mu**2) * np.sin(phi)))
                w = np.vstack((w, mu))
                x = np.vstack((x, np.full(4, position_x, dtype=np.float32)))
                y = np.vstack((y, np.full(4, position_y, dtype=np.float32)))
                z = np.vstack((z, np.full(4, position_z, dtype=np.float32)))
        ax.quiver(x, y, z, u, v, w, length=length, pivot='tail')

    def plot2(self, ax, xlim, ylim, zlim, length=0.1):
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
        # ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        first = True
        for projection in self.camera.projection.projection_list:
            position_x = projection.x.reshape(projection.resolution)
            position_y = projection.y.reshape(projection.resolution)
            position_z = projection.z.reshape(projection.resolution)
            mu = -projection.mu.reshape(projection.resolution)
            phi = np.pi + projection.phi.reshape(projection.resolution)
            if first:
                u = np.sqrt(1 - mu**2) * np.cos(phi)
                v = np.sqrt(1 - mu**2) * np.sin(phi)
                w = mu
                x = position_x
                y = position_y
                z = position_z
                first = False
            else:
                u = np.vstack((u, np.sqrt(1 - mu**2) * np.cos(phi)))
                v = np.vstack((v, np.sqrt(1 - mu**2) * np.sin(phi)))
                w = np.vstack((w, mu))
                x = np.vstack((x, position_x))
                y = np.vstack((y, position_y))
                z = np.vstack((z, position_z))
        ax.quiver(x, y, z, u, v, w, length=length, pivot='tail')

    def lla2flat(self, lla, llo, psio, href):
        '''
        lla  -- array of geodetic coordinates 
                (latitude, longitude, and altitude), 
                in [degrees, degrees, meters]. 
                Latitude and longitude values can be any value. 
                However, latitude values of +90 and -90 may return 
                unexpected values because of singularity at the poles.
        llo  -- Reference location, in degrees, of latitude and 
                longitude, for the origin of the estimation and 
                the origin of the flat Earth coordinate system.
        psio -- Angular direction of flat Earth x-axis 
                (degrees clockwise from north), which is the angle 
                in degrees used for converting flat Earth x and y 
                coordinates to the North and East coordinates.
        href -- Reference height from the surface of the Earth to 
                the flat Earth frame with regard to the flat Earth 
                frame, in meters.
        usage: print(lla2flat((0.1, 44.95, 1000.0), (0.0, 45.0), 5.0, -100.0))
        '''
        R = 6378137.0  # Equator radius in meters
        f = 0.00335281066474748071  # 1/298.257223563, inverse flattening
        
        Lat_p = lla[0] * np.pi / 180.0  # from degrees to radians
        Lon_p = lla[1] * np.pi / 180.0  # from degrees to radians
        Alt_p = lla[2]  # meters

        # Reference location (lat, lon), from degrees to radians
        Lat_o = llo[0] * np.pi / 180.0
        Lon_o = llo[1] * np.pi / 180.0

        psio = psio * np.pi / 180.0  # from degrees to radians

        dLat = Lat_p - Lat_o
        dLon = Lon_p - Lon_o

        ff = (2.0 * f) - (f ** 2)  # Can be precomputed

        sinLat = np.sin(Lat_o)

        # Radius of curvature in the prime vertical
        Rn = R / np.sqrt(1 - (ff * (sinLat ** 2)))

        # Radius of curvature in the meridian
        Rm = Rn * ((1 - ff) / (1 - (ff * (sinLat ** 2))))

        dNorth = (dLat) / np.arctan2(1, Rm)
        dEast = (dLon) / np.arctan2(1, (Rn * np.cos(Lat_o)))

        # Rotate matrice clockwise
        Xp = (dNorth * np.cos(psio)) + (dEast * np.sin(psio))
        Yp = (-dNorth * np.sin(psio)) + (dEast * np.cos(psio))
        Zp = -Alt_p - href

        return Xp, -Yp, Zp
    @property
    def sun_azimuth_list(self):
        return self._sun_azimuth_list

    @property
    def sun_zenith_list(self):
        return self._sun_zenith_list


class AirMSPIMeasurementsv3(shdom.DynamicMeasurements):
    """
      A AirMSPI Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
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

    def __init__(self, camera=None, images=None, pixels=None, wavelength=None, time_list=None):
        super().__init__(camera, images, pixels, wavelength, uncertainties=None, time_list=time_list)
        self._region_of_interest = None
        self._paths = None
        self._set_valid_wavelength_range = None
        if images is not None:
            self._region_of_interest = [0, images[0].shape[0], 0, images[0].shape[1]]
        if wavelength is not None:
            self._valid_wavelength_range = [np.min(wavelength), np.max(wavelength)]

    @classmethod
    def select_region_of_interest(cls, data_dir, index):
        format_ = '*.hdf'  # load
        path = sorted(glob.glob(data_dir + '/' + format_))[index]
        f = h5py.File(path, 'r')
        channels_data = f['HDFEOS']['GRIDS']
        image = np.array(channels_data['660nm_band']['Data Fields']['I'])
        image = np.dstack((image, channels_data['555nm_band']['Data Fields']['I']))
        image = np.dstack((image, channels_data['445nm_band']['Data Fields']['I']))
        image[image == -999] = 0
        image -= image.min()
        image /= image.max()

        # Select ROI
        fromCenter = False
        scale_percent = 25  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(image, dim)
        roi = cv2.selectROI('image',resized, fromCenter)
        x, y, w, h = tuple([int(r * 100 / scale_percent) for r in roi])
        roi = [y, y + h, x, x + w]
        cv2.destroyAllWindows()
        return roi

    @classmethod
    def imshow(cls, data_dir):
        format_ = '*.hdf'  # load
        paths = sorted(glob.glob(data_dir + '/' + format_))
        images = []
        for path in paths:
            f = h5py.File(path, 'r')
            channels_data = f['HDFEOS']['GRIDS']
            image = np.array(channels_data['660nm_band']['Data Fields']['I'])
            image = np.dstack((image, channels_data['555nm_band']['Data Fields']['I']))
            image = np.dstack((image, channels_data['445nm_band']['Data Fields']['I']))
            image[image == -999] = 0
            images.append(np.array(image))
        f, axarr = plt.subplots(1, len(images), figsize=(20, 20))
        if isinstance(axarr, plt.Axes):
            axarr = [axarr]
        for ax, image in zip(axarr, images):
            image -= image.min()
            ax.imshow(image / image.max())

    def save_airmspi_measurements(self, directory):
        """
            Save the AirMSPI measurements parameters for reconstruction.

            Parameters
            ----------
            directory: str
                Directory path where the forward modeling parameters are saved.
                If the folder doesnt exist it will be created.

            """
        if not os.path.isdir(directory):
            os.makedirs(directory)
        path = os.path.join(directory, 'measurements')
        self.save(path)

    def load_airmspi_measurements(self, directory):
        # Load shdom.Measurements object (sensor geometry and radiances)
        measurements_path = os.path.join(directory, 'measurements')
        assert os.path.exists(measurements_path), 'No measurements file in directory: {}'.format(directory)
        self.load(path=measurements_path)

    def set_sun_params(self):
        for path, roi in zip(self._paths,self._region_of_interest):
            f = h5py.File(path, 'r')
            channels_data = f['HDFEOS']['GRIDS']
            sun_azimuth_list = []
            sun_zenith_list = []
            for param_name, param in channels_data.items():
                if "band" in param_name:
                    if int(param_name[0:3]) in self._valid_wavelength:
                        sun_azimuth = param['Data Fields']['Sun_azimuth'][roi[0]:roi[1], roi[2]:roi[3]]
                        sun_zenith = param['Data Fields']['Sun_zenith'][roi[0]:roi[1], roi[2]:roi[3]]
                        sun_azimuth_list.append(np.mean(sun_azimuth))
                        sun_zenith_list.append(180 - np.mean(sun_zenith))

            self._sun_azimuth_list = sun_azimuth_list
            self._sun_zenith_list = sun_zenith_list

    def load_from_hdf(self, data_dir, region_of_interest=[1500, 1700, 1500, 1700],
                      valid_wavelength=[355, 380, 445, 470, 555, 660, 865, 935], type='Radiance'):

        assert np.array(region_of_interest).shape[-1] == 4, 'region of interest should be list of size 4: ' \
                                             '[pixel_x_start, pixel_x_end,pixel_y_start, pixel_y_end]'
        AirMSPI_wavelength = [355, 380, 445, 470, 555, 660, 865, 935]
        assert all(item in AirMSPI_wavelength for item in valid_wavelength), \
            'valid_wavelength can only contain AirMSPI wavelengths:  [355, 380, 445, 470, 555, 660, 865, 935]'

        assert type == 'Radiance' or type == 'Polarize', 'type should be Radiance or Polarize'

        format_ = '*.hdf'  # load
        paths = sorted(glob.glob(data_dir + '/' + format_))
        assert len(paths) > 0, 'there are no hdf files in the given directory, try to add ../ at the beginning'
        self._paths = paths
        self._type = type
        self._valid_wavelength = valid_wavelength
        self.set_valid_wavelength_index()
        self.set_region_of_interest(region_of_interest)
        self.set_camera()
        self.set_wavelengths()
        self.set_times()
        self.set_images()
        self._pixels = self.images_to_pixels(self.images)
        self.set_sun_params()

    def set_region_of_interest(self, region_of_interest):
        assert len(region_of_interest) == len(self._paths)
        for path, roi in zip(self._paths,region_of_interest):
            f = h5py.File(path, 'r')
            channels_data = f['HDFEOS']['GRIDS']
            for param_name, param in channels_data.items():
                if "band" in param_name:
                    if int(param_name[0:3]) in self._valid_wavelength:
                        mask = np.array(param['Data Fields']['I.mask'][roi[0]:roi[1],
                                        roi[2]:roi[3]])
                        assert np.all(mask == 1), 'Invalid region of interest'
        self._region_of_interest = region_of_interest

    def set_valid_wavelength_index(self):
        AirMSPI_wavelength = [355, 380, 445, 470, 470, 470, 555, 660, 660, 660, 865, 865, 865, 935]
        self._valid_wavelength_index = np.array(list(map(lambda wl: wl in self._valid_wavelength, AirMSPI_wavelength)))
        if self._type == 'Radiance':
            self._valid_wavelength_index[[4, 5, 8, 9, 11, 12]] = False

    def get_general_info(self, f):
        general_info = {}
        data_general_info = f['HDFEOS']['ADDITIONAL']['FILE_ATTRIBUTES'].attrs
        epoch_time = data_general_info['Epoch (UTC)'].decode('utf-8')
        crop = epoch_time.find('.') - 1
        crop_begining = epoch_time.find('T') + 1
        general_info['epoch_time'] = datetime.strptime(epoch_time[:crop], '%Y-%m-%dT%H:%M:%S')
        general_info['epoch_date'] = datetime.strptime(epoch_time[0: crop_begining - 1], '%Y-%m-%d')
        general_info['aircraft_heading'] = data_general_info['Aircraft heading (degrees)']
        general_info['radiance_units'] = data_general_info['Radiance units'].decode('utf-8')
        assert general_info[
                   'radiance_units'] == 'W * (sr^-1) * (m^-2) * (nm^-1) : Watts per steradian per square meter per nanometer'
        general_info['resolution'] = data_general_info['Resolution']

        return general_info

    def build_projection(self, f,region_of_interest):

        channels_data_ancillary = f['HDFEOS']['GRIDS']['Ancillary']['Data Fields']
        latitude = np.array(channels_data_ancillary['Latitude'])
        latitude = latitude[region_of_interest[0]:region_of_interest[1], region_of_interest[2]:region_of_interest[3]]
        longitude = np.array(channels_data_ancillary['Longitude'])
        longitude = longitude[region_of_interest[0]:region_of_interest[1], region_of_interest[2]:region_of_interest[3]]
        elevation = np.full(longitude.shape, 0)

        resolution = list(elevation.shape)
        # -------------- Registration to 20 km altitude -------------
        # LLA(Latitude - Longitude - Altitude) to Flat surface(meters)
        airmspi_flight_altitude = 20.0  # km
        lla = [latitude, longitude, elevation]
        if self._relative_coordinates is None:
            llo = [latitude.min(), longitude.min()]  # Origin of lat - long coordinate system
            self._relative_coordinates = [llo[0], llo[1], 0]
        else:
            llo = self._relative_coordinates[:-1]
        psio = 0  # Angle between X axis and North
        href = 0  # Reference height
        flat_earth_pos = (np.array(self.lla2flat(lla, llo, psio, href)) / 1000)  # [Km] N-E coordinates
        # flat_earth_pos = np.array(pymap3d.geodetic2ned(latitude,longitude,elevation,latitude.min()
        #                                                ,longitude.min(),elevation.min()))/1000 #[Km] N-E coordinates

        channels_data = f['HDFEOS']['GRIDS']['355nm_band']['Data Fields']
        theta = np.deg2rad(channels_data['View_zenith']
                           [region_of_interest[0]:region_of_interest[1], region_of_interest[2]:region_of_interest[3]])
        mu = np.cos(theta)
        phi = np.deg2rad(np.array(channels_data['View_azimuth'])
                         [region_of_interest[0]:region_of_interest[1], region_of_interest[2]:region_of_interest[3]])

        xTranslation = airmspi_flight_altitude * np.tan(theta) * np.cos(phi)  # x - North
        yTranslation = airmspi_flight_altitude * np.tan(theta) * np.sin(phi)  # Y - East

        x = flat_earth_pos[0] + xTranslation
        y = flat_earth_pos[1] + yTranslation
        z = np.full(resolution, airmspi_flight_altitude)

        # x -= self._relative_coordinates[0]
        # y -= self._relative_coordinates[1]
        # z -= self._relative_coordinates[2]
        return shdom.Projection(x=x.ravel('F'), y=y.ravel('F'), z=z.ravel('F'), mu=mu.ravel('F'), phi=phi.ravel('F'),
                                resolution=resolution)

    def set_camera(self):
        projections = shdom.MultiViewProjection()
        self._relative_coordinates = None
        for path, roi in zip(self._paths,self._region_of_interest):
            f = h5py.File(path, 'r')
            projection = self.build_projection(f,roi)
            projections.add_projection(projection)
        self._camera = shdom.DynamicCamera(shdom.RadianceSensor(), projections)

    def set_wavelengths(self):
        first = True
        tol = 1.01
        wavelength = np.array(self._valid_wavelength)
        self._wavelength = np.round(wavelength / 1000, 3).tolist()
        self._num_channels = len(self._wavelength)

    def set_times(self):
        time_list = []
        first = True
        for path, roi in zip(self._paths,self._region_of_interest):
            f = h5py.File(path, 'r')
            time_in_epoch = np.array(f['HDFEOS']['GRIDS']['355nm_band']['Data Fields']['Time_in_seconds_from_epoch'])
            time_in_epoch = time_in_epoch[roi[0]:roi[1], roi[2]:roi[3]]
            relative_time = np.mean(time_in_epoch)  # sec
            general_info = self.get_general_info(f)
            epoch_time = general_info['epoch_time']
            if first:
                self._absolute_time = epoch_time
                self._acquisition_date = general_info['epoch_date']
                first = False
            assert self._acquisition_date == general_info['epoch_date']
            time_list.append((epoch_time - self._absolute_time).total_seconds() + relative_time)
        self._time_list = np.array(time_list)

    def radiance2brf(self, radiance, sun_distance, solar_irradiance, sun_zenith):
        brf = radiance * np.pi * sun_distance ** 2 / solar_irradiance / np.cos(np.deg2rad(sun_zenith))
        return brf

    def set_images(self):
        images = []
        for path, roi in zip(self._paths,self._region_of_interest):
            f = h5py.File(path, 'r')
            channels_data = f['HDFEOS']['GRIDS']
            first = True
            image = None
            num_valid_wavelength = 0
            for param_name, param in channels_data.items():
                if "band" in param_name:
                    if int(param_name[0:3]) in self._valid_wavelength:
                        radiance = param['Data Fields']['I'][roi[0]:roi[1], roi[2]:roi[3]]
                        if first:
                            image = radiance
                            first = False
                        else:
                            image = np.dstack((image, radiance))
                        num_valid_wavelength += 1

            assert num_valid_wavelength == len(self._wavelength), 'invalid wavelength_range, try to increase it'
            if image is not None:
                images.append(np.array(image))
        self._images = images

    def images_to_pixels(self, images):
        """
        Set image list.

        Parameters
        ----------
        images: list of images,
            A list of images (multiview camera)

        Returns
        -------
        pixels: a flattened version of the image list
        """
        pixels = []

        if type(images) is not list:
            images = [images]

        for image in images:
            if self.camera.sensor.type == 'RadianceSensor':
                if len(image.shape) == 2:
                    num_channels = 1
                else:
                    num_channels = image.shape[-1]
                pixels.append(image.reshape((-1, num_channels), order='F'))

            elif self.camera.sensor.type == 'StokesSensor':
                NotImplemented()
                num_channels = image.shape[-1] if image.ndim == 4 else 1
                pixels.append(image.reshape((image.shape[0], -1, num_channels), order='F'))

            else:
                raise AttributeError('Error image dimensions: {}'.format(image.ndim))
        pixels = np.concatenate(pixels, axis=-2)
        return pixels

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
        # ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        first = True
        for projection in self.camera.projection.projection_list:
            position_x = projection.x.reshape(projection.resolution)[[0, -1, 0, -1], [0, 0, -1, -1]]
            position_y = projection.y.reshape(projection.resolution)[[0, -1, 0, -1], [0, 0, -1, -1]]
            position_z = projection.z.reshape(projection.resolution)[[0, -1, 0, -1], [0, 0, -1, -1]]
            mu = -projection.mu.reshape(projection.resolution)[[0, -1, 0, -1], [0, 0, -1, -1]]
            phi = np.pi + projection.phi.reshape(projection.resolution)[[0, -1, 0, -1], [0, 0, -1, -1]]
            if first:
                u = np.sqrt(1 - mu ** 2) * np.cos(phi)
                v = np.sqrt(1 - mu ** 2) * np.sin(phi)
                w = mu
                x = np.full(4, position_x, dtype=np.float32)
                y = np.full(4, position_y, dtype=np.float32)
                z = np.full(4, position_z, dtype=np.float32)
                first = False
            else:
                u = np.vstack((u, np.sqrt(1 - mu ** 2) * np.cos(phi)))
                v = np.vstack((v, np.sqrt(1 - mu ** 2) * np.sin(phi)))
                w = np.vstack((w, mu))
                x = np.vstack((x, np.full(4, position_x, dtype=np.float32)))
                y = np.vstack((y, np.full(4, position_y, dtype=np.float32)))
                z = np.vstack((z, np.full(4, position_z, dtype=np.float32)))
        ax.quiver(x, y, z, u, v, w, length=length, pivot='tail')
        x = self.camera.projection.projection_list[0].x[0]
        y = self.camera.projection.projection_list[0].y[0]
        z = self.camera.projection.projection_list[0].z[0]
        u = np.cos(np.deg2rad(270))
        v = np.sin(np.deg2rad(270))
        w=0
        ax.quiver(x, y, z, u, v, w, length=length*10, pivot='tail',color='r')

    def plot2(self, ax, xlim, ylim, zlim, length=0.1):
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
        # ax.set_aspect('equal')
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        first = True
        for projection in self.camera.projection.projection_list:
            position_x = projection.x.reshape(projection.resolution)
            position_y = projection.y.reshape(projection.resolution)
            position_z = projection.z.reshape(projection.resolution)
            mu = -projection.mu.reshape(projection.resolution)
            phi = np.pi + projection.phi.reshape(projection.resolution)
            if first:
                u = np.sqrt(1 - mu ** 2) * np.cos(phi)
                v = np.sqrt(1 - mu ** 2) * np.sin(phi)
                w = mu
                x = position_x
                y = position_y
                z = position_z
                first = False
            else:
                u = np.vstack((u, np.sqrt(1 - mu ** 2) * np.cos(phi)))
                v = np.vstack((v, np.sqrt(1 - mu ** 2) * np.sin(phi)))
                w = np.vstack((w, mu))
                x = np.vstack((x, position_x))
                y = np.vstack((y, position_y))
                z = np.vstack((z, position_z))
        ax.quiver(x, y, z, u, v, w, length=length, pivot='tail')

    def lla2flat(self, lla, llo, psio, href):
        '''
        lla  -- array of geodetic coordinates
                (latitude, longitude, and altitude),
                in [degrees, degrees, meters].
                Latitude and longitude values can be any value.
                However, latitude values of +90 and -90 may return
                unexpected values because of singularity at the poles.
        llo  -- Reference location, in degrees, of latitude and
                longitude, for the origin of the estimation and
                the origin of the flat Earth coordinate system.
        psio -- Angular direction of flat Earth x-axis
                (degrees clockwise from north), which is the angle
                in degrees used for converting flat Earth x and y
                coordinates to the North and East coordinates.
        href -- Reference height from the surface of the Earth to
                the flat Earth frame with regard to the flat Earth
                frame, in meters.
        usage: print(lla2flat((0.1, 44.95, 1000.0), (0.0, 45.0), 5.0, -100.0))
        '''
        R = 6378137.0  # Equator radius in meters
        f = 0.00335281066474748071  # 1/298.257223563, inverse flattening

        Lat_p = lla[0] * np.pi / 180.0  # from degrees to radians
        Lon_p = lla[1] * np.pi / 180.0  # from degrees to radians
        Alt_p = lla[2]  # meters

        # Reference location (lat, lon), from degrees to radians
        Lat_o = llo[0] * np.pi / 180.0
        Lon_o = llo[1] * np.pi / 180.0

        psio = psio * np.pi / 180.0  # from degrees to radians

        dLat = Lat_p - Lat_o
        dLon = Lon_p - Lon_o

        ff = (2.0 * f) - (f ** 2)  # Can be precomputed

        sinLat = np.sin(Lat_o)

        # Radius of curvature in the prime vertical
        Rn = R / np.sqrt(1 - (ff * (sinLat ** 2)))

        # Radius of curvature in the meridian
        Rm = Rn * ((1 - ff) / (1 - (ff * (sinLat ** 2))))

        dNorth = (dLat) / np.arctan2(1, Rm)
        dEast = (dLon) / np.arctan2(1, (Rn * np.cos(Lat_o)))

        # Rotate matrice clockwise
        Xp = (dNorth * np.cos(psio)) + (dEast * np.sin(psio))
        Yp = (-dNorth * np.sin(psio)) + (dEast * np.cos(psio))
        Zp = -Alt_p - href

        return Xp, -Yp, Zp

    @property
    def sun_azimuth_list(self):
        return self._sun_azimuth_list

    @property
    def sun_zenith_list(self):
        return self._sun_zenith_list


