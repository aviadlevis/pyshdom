import h5py, os, cv2, glob
import numpy as np
import shdom
from datetime import datetime
import matplotlib.pyplot as plt


class AirMSPIMeasurements(shdom.Measurements):
    """
      A AirMSPI Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
      It can be initialized with a Camera and images or pixels.
      Alternatively is can be loaded from AirMSPI data files.

      Parameters
      ----------
      camera: shdom.Camera
          The camera model used to take the measurements
      images: list of images, optional
          A list of images (multiview camera)
      pixels: np.array(dtype=float)
          pixels are a flattened version of the image list where the channel dimension is kept (1 for monochrome)
      wavelength: list
          A list of AirMSPI's wavelengths to add to the measurements object.
      """

    def __init__(self, camera=None, images=None, pixels=None, wavelength=None):
        super().__init__(camera, images, pixels, wavelength, uncertainties=None)
        self._region_of_interest = None
        self._paths = None
        self._set_valid_wavelength_range = None
        self.cloud_base = 0
        self.cloud_top = 1
        self._bb = None
        self._relative_coordinates = None
        if images is not None:
            self._region_of_interest = [0, images[0].shape[0], 0, images[0].shape[1]]
        if wavelength is not None:
            self._valid_wavelength_range = [np.min(wavelength), np.max(wavelength)]

    @classmethod
    def select_region_of_interest(cls, directory, index):
        """
            Crop region of interest in AirMSPI's image at specific index.

            Parameters
            ----------
            directory: str
                Directory path where the AirMSPI files are located
            index: int
                of iamge index to be crop

            Notes
            ----------
            The method uses OpenCV package with can have problem in certain versions.
        """
        format_ = '*.hdf'  # load
        path = sorted(glob.glob(directory + '/' + format_))[index]
        f = h5py.File(path, 'r')
        channels_data = f['HDFEOS']['GRIDS']
        image = np.array(channels_data['660nm_band']['Data Fields']['I'])
        image = np.dstack((channels_data['555nm_band']['Data Fields']['I'], image))
        image = np.dstack((channels_data['445nm_band']['Data Fields']['I'], image))
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
    def imshow(cls, directory, index=-1):
        """
            Show the AirMSPI images in the directory directory.

            Parameters
            ----------
            directory: str
                Directory path where the AirMSPI files are located.

            Notes
            ----------
            The RGB are images of the 660,555,445 nm wavelengths.

        """
        format_ = '*.hdf'
        paths = sorted(glob.glob(directory + '/' + format_))
        images = []
        if index > 0 and index < len(paths):
            paths=[paths[index]]
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
        """
            Load shdom.Measurements object (sensor geometry and radiances)

        """
        measurements_path = os.path.join(directory, 'measurements')
        assert os.path.exists(measurements_path), 'No measurements file in directory: {}'.format(directory)
        self.load(path=measurements_path)

    def set_sun_params(self):
        """
            Read sun's parameters for the optimization.

        """
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

    def load_from_hdf(self, data_dir, region_of_interest,
                      valid_wavelength=[355, 380, 445, 470, 555, 660, 865, 935], sensor_type='Radiance'):
        """
            Read AirMSPI data and build adequate projection according to SHDOM package.

            Parameters
            ----------
            data_dir: AirMSPI data directory
            region_of_interest: list
               [xmin, xmax, ymin, ymax] to set the image's domain limits
            valid_wavelength: list
               of AirMSPI wavelengths to be used. Should be from [355, 380, 445, 470, 555, 660, 865, 935]
            sensor_type: str
               of sensor type 'Radiance' or 'Stokes'
        """

        assert np.array(region_of_interest).shape[-1] == 4, 'region of interest should be list of size 4: ' \
                                             '[xmin, xmax, ymin, ymax]'
        AirMSPI_wavelength = [355, 380, 445, 470, 555, 660, 865, 935]
        assert all(item in AirMSPI_wavelength for item in valid_wavelength), \
            'valid_wavelength can only contain AirMSPI wavelengths:  [355, 380, 445, 470, 555, 660, 865, 935]'

        assert sensor_type == 'Radiance' or sensor_type == 'Stokes', 'type should be Radiance or Stokes'

        format_ = '*.hdf'  # load
        paths = sorted(glob.glob(data_dir + '/' + format_))
        assert len(paths) > 0, 'there are no hdf files in the given directory, try to add ../ at the beginning'
        self._paths = paths
        self._sensor_type = sensor_type
        self._valid_wavelength = valid_wavelength
        self.set_valid_wavelength_index()
        self.set_region_of_interest(region_of_interest)
        self.set_camera()
        self.set_wavelengths()
        self.set_images()
        self._pixels = self.images_to_pixels(self.images)
        self.set_sun_params()

    def set_region_of_interest(self, region_of_interest):
        """
            Set the region of interest for every image.

            Parameters
            ----------
            region_of_interest: list
                [xmin, xmax, ymin, ymax] to set the image's domain limits
        """
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
        """
            Set the valid wavelengths indices. AirMSPI wavelengths are
            [355, 380, 445, 470, 470, 470, 555, 660, 660, 660, 865, 865, 865, 935] nm.

            Note
            ----------
            The repeated wavelength are belongs to the polarized wavelength.
        """
        AirMSPI_wavelength = [355, 380, 445, 470, 470, 470, 555, 660, 660, 660, 865, 865, 865, 935]
        self._valid_wavelength_index = np.array(list(map(lambda wl: wl in self._valid_wavelength, AirMSPI_wavelength)))
        if self._sensor_type == 'Radiance':
            self._valid_wavelength_index[[4, 5, 8, 9, 11, 12]] = False

    def build_projection(self, f, region_of_interest):
        """
            Read AirMSPI data and build adaquate projection according to SHDOM package.

            Parameters
            ----------
            f: AirMSPI file to be read
            region_of_interest: list
                [xmin, xmax, ymin, ymax] to set the image's domain limits

            Returns
            -------
                shdom.Projection object
        """
        channels_data_ancillary = f['HDFEOS']['GRIDS']['Ancillary']['Data Fields']
        latitude = np.array(channels_data_ancillary['Latitude'])
        latitude = latitude[region_of_interest[0]:region_of_interest[1], region_of_interest[2]:region_of_interest[3]]
        longitude = np.array(channels_data_ancillary['Longitude'])
        longitude = longitude[region_of_interest[0]:region_of_interest[1], region_of_interest[2]:region_of_interest[3]]
        if len(f) == 2:
            elevation = np.full(longitude.shape, 0)
        elif len(f) == 3:
            elevation = np.array(channels_data_ancillary['Elevation']) / 1000  # [Km]
            elevation = elevation[region_of_interest[0]:region_of_interest[1],
                        region_of_interest[2]:region_of_interest[3]]
        else:
            assert 'Unsupported AirMSPI data version'

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

        channels_data = f['HDFEOS']['GRIDS']['355nm_band']['Data Fields']
        theta = np.deg2rad(channels_data['View_zenith']
                           [region_of_interest[0]:region_of_interest[1], region_of_interest[2]:region_of_interest[3]])
        mu = np.cos(theta)
        phi = np.deg2rad(np.array(channels_data['View_azimuth'])
                         [region_of_interest[0]:region_of_interest[1], region_of_interest[2]:region_of_interest[3]])
        I = np.array(channels_data['I'][region_of_interest[0]:region_of_interest[1],
                     region_of_interest[2]:region_of_interest[3]])
        cloud_com_x, cloud_com_y = self.center_of_mass(I,flat_earth_pos[0],flat_earth_pos[1])

        bb_x = flat_earth_pos[0] + self.cloud_base * np.tan(theta) * np.cos(phi) - cloud_com_x
        bb_y = flat_earth_pos[1] + self.cloud_base * np.tan(theta) * np.cos(phi) - cloud_com_y
        self.set_cloud_bounding_box(bb_x, bb_y)

        xTranslation = airmspi_flight_altitude * np.tan(theta) * np.cos(phi) - cloud_com_x # X - North
        yTranslation = airmspi_flight_altitude * np.tan(theta) * np.sin(phi) - cloud_com_y # Y - East

        x = flat_earth_pos[0] + xTranslation
        y = flat_earth_pos[1] + yTranslation
        z = np.full(resolution, airmspi_flight_altitude)

        return shdom.Projection(x=x.ravel('F'), y=y.ravel('F'), z=z.ravel('F'), mu=mu.ravel('F'), phi=phi.ravel('F'),
                                resolution=resolution)

    def set_cloud_bounding_box(self, x, y):
        """
            Set the bounding box for the cloud reconstruction.

            Parameters
            ----------
            x: np.array
                of cloud's base location in Km
            y: np.array
                of cloud's base location in Km.
        """
        if self._bb is None:
            self._bb = shdom.BoundingBox(x.min(), y.min(), self.cloud_base, x.max(), y.max(), self.cloud_top)
        else:
            bb = shdom.BoundingBox(x.min(), y.min(), self.cloud_base, x.max(), y.max(), self.cloud_top)
            self._bb = self._bb + bb

    def center_of_mass(self, I, x, y):
        """
            Calculate the Center of Mass of a radiance image for cloud's pre-processing.

            Returns
            -------
            com_x: scalar
                of center of mass at x axis
            com_y: scalar
                of center of mass at y axis
        """
        com_x = np.sum(I * x) / np.sum(I)
        com_y = np.sum(I * y) / np.sum(I)
        return com_x, com_y

    def get_projections_from_data(self):
        """
            Get the AirMSPI's projections.

            Returns
            -------
            projections: shdom.MultiViewProjection

            Notes
            -----
            The projections are centered that the axes are only positive.
        """

        projections = shdom.MultiViewProjection()
        self._relative_coordinates = None
        for path, roi in zip(self._paths, self._region_of_interest):
            f = h5py.File(path, 'r')
            projection = self.build_projection(f, roi)
            projections.add_projection(projection)

        centered_projections = shdom.MultiViewProjection()
        for projection in projections.projection_list:
            projection._x -= self.bb.xmin
            projection._y -= self.bb.ymin
            centered_projections.add_projection(projection)
        self._bb = shdom.BoundingBox(0, 0, self.bb.zmin, self.bb.xmax - self.bb.xmin,
                                     self.bb.ymax - self.bb.ymin, self.bb.zmax)
        return centered_projections

    def set_camera(self):
        """
            Set the AirMSPI's camera.
        """

        projections = self.get_projections_from_data()
        self._projections = projections
        if self._sensor_type == 'Radiance':
            sensor = shdom.RadianceSensor()
        else:
            NotImplemented()
        self._camera = shdom.Camera(sensor, projections)

    def set_wavelengths(self):
        """
            Set the AirMSPI's actual measured central wavelength of each valid band.

            Notes
            -----
            Checks that the wavelengths are the same for all the view-points.
        """
        first = True
        for path in self._paths:
            f = h5py.File(path, 'r')
            if len(f) == 2:
                wavelength = np.array(self._valid_wavelength)
            elif len(f) == 3:
                wavelength = np.array(f['Channel_Information']['Center_wavelength'][self._valid_wavelength_index])
            else:
                assert 'Unsupported AirMSPI data version'
            if first:
                self._wavelength = np.round(wavelength / 1000, 3).tolist()
                first = False
            else:
                assert np.array_equal(np.round(wavelength / 1000, 3).tolist(),self._wavelength), \
                    'All wavelengths in the given files must be equal'
        self._num_channels = len(self._wavelength)

    def radiance2brf(self, radiance, sun_distance, solar_irradiance, sun_zenith):
        """
        Calculate the brf according to the AirMSPI conversion equation.

        Parameters
        ----------
        radiance: np.array
            2D radiance array
        sun_distance: scaler
            Earth-Sun distance for use in calculation of BRF (AU)
        solar_irradiance:
            The extraterrestrial solar irradiance (in W m-2 nm-1) weighted by the total-band spectral
            response function for each channel at the nominal Earth-Sun distance (1 AU)
        sun_zenith:
            Solar zenith angle relative to overhead sun (0°)

        Returns
        -------
        brf: np.array
            of at-altitude bidirectional reflectance factor
        """
        brf = radiance * np.pi * sun_distance ** 2 / solar_irradiance / np.cos(np.deg2rad(sun_zenith))
        return brf

    def set_images(self):
        """
            Set the images of the valid wavelengths.

            Notes
            -----
            Only the region of interest of the is set
        """
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

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        first = True
        for projection in self._projections.projection_list:
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

    @property
    def bb(self):
        return self._bb


class AirMSPIDynamicMeasurements(AirMSPIMeasurements, shdom.DynamicMeasurements):
    """
      A AirMSPI Dynamic Measurements object bundles together the imaging geometry and sensor measurements for later optimization.
      It can be initialized with a Camera and images or pixels.
      Alternatively is can be loaded from AirMSPI files.

      Parameters
      ----------
      camera: shdom.Camera
          The camera model used to take the measurements, optional
      images: list of images, optional
          A list of images (multiview camera)
      pixels: np.array(dtype=float)
          pixels are a flattened version of the image list where the channel dimension is kept (1 for monochrome), optional
      wavelength: list
          A list of AirMSPI's wavelengths to add to the measurements object, optional
      time_list: list
          A list of approximated times in seconds of images acquisition, optional.
      """

    def __init__(self, camera=None, images=None, pixels=None, wavelength=None, time_list=None):
        super().__init__(camera, images, pixels, wavelength)
        self._time_list = time_list

    def load_from_hdf(self, data_dir, region_of_interest,
                      valid_wavelength=[355, 380, 445, 470, 555, 660, 865, 935], sensor_type='Radiance'):
        """
            Read AirMSPI data and build adequate projection according to SHDOM package.

            Parameters
            ----------
            data_dir: AirMSPI data directory
            region_of_interest: list
               [xmin, xmax, ymin, ymax] to set the image's domain limits
            valid_wavelength: list
               of AirMSPI wavelengths to be used. Should be from [355, 380, 445, 470, 555, 660, 865, 935]
            sensor_type: str
               of sensor type 'Radiance' or 'Stokes'

            Returns
            -------
            projection: shdom.Projection object
        """
        super().load_from_hdf(data_dir, region_of_interest,
                      valid_wavelength=valid_wavelength, sensor_type=sensor_type)
        self.set_times()

    def get_general_info(self, f):
        """
            Read AirMSPI's general information.

            Parameters
            ----------
            f: AirMSPI data file

            Returns
            -------
            general_info: Dict
              with information.
        """
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

    def set_times(self):
        """
            Set acquiring time for every image .

        """
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

    def set_camera(self):
        """
            Set the AirMSPI's Dynamic camera.
        """
        projections = self.get_projections_from_data()
        self._projections = projections
        if self._sensor_type == 'Radiance':
            sensor = shdom.RadianceSensor()
        else:
            NotImplemented()
        self._camera = shdom.DynamicCamera(sensor, shdom.DynamicProjection(projections.projection_list))

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

    @property
    def time_list(self):
        return self._time_list


