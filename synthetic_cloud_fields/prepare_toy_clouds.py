import numpy as np
import mayavi.mlab as mlab
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import shdom
from argparse import ArgumentParser

#SAMPLE_DIR = 'Medium_Samples'
#scat_file_path='../mie_tables/polydisperse/Water_672nm.scat'


def load_MAT_FILE(mat_path):
    if os.path.exists(mat_path):
        print("loading the 3D mat from: {}".format(mat_path))
    else:
        print("{} not exists".format(mat_path))    
    
    matrix3D = sio.loadmat(mat_path)
    return matrix3D    

    
def float_round(x):
    """Round a float or np.float32 to a 3 digits float"""
    if type(x) == np.float32:
        x = x.item()
    return round(x,3) 


def safe_mkdirs(path):
    """Safely create path, warn in case of race."""

    if not os.path.exists(path):
        print('No directory name {}, It will be created now!'.format(path))
        try:
            os.makedirs(path)
        except OSError as e:
            import errno
            if e.errno == errno.EEXIST:
                warnings.warn(
                    "Failed creating path: {path}, probably a race".format(path=path)
                )
    
    
    
    
    """generate grid info:
Evrey mediume mush have his grid discription
"""
    
class TOY_CLOUD_SAMPLE(object):
    def __init__(self,sample_name,Grid_bounding_box_data):
        safe_mkdirs(SAMPLE_DIR)# make the dir to save evrey thing
        
        self.sample_name = sample_name
        json_data_file = self.sample_name+'_sample.json'
        self.json_data_file = os.path.join(SAMPLE_DIR,json_data_file)
        
        # write description json file
        with open(self.json_data_file, 'w') as f:
            json.dump(Grid_bounding_box_data, f, indent=4)   
            
        # -----------------------------------------------------------------------------------    
        # read grid discription:
        # -----------------------------------------------------------------------------------
        
        with open(self.json_data_file, 'r') as f:
                grid_json = json.load(f)
                
        self.bounding_box_xmin = grid_json['Grid_bounding_box']['xmin']
        self.bounding_box_ymin = grid_json['Grid_bounding_box']['ymin']
        self.bounding_box_zmin = grid_json['Grid_bounding_box']['zmin']
        self.bounding_box_xmax = grid_json['Grid_bounding_box']['xmax']
        self.bounding_box_ymax = grid_json['Grid_bounding_box']['ymax']
        self.bounding_box_zmax = grid_json['Grid_bounding_box']['zmax']
        
        self.nx=grid_json['Grid_bounding_box']['nx']
        self.ny=grid_json['Grid_bounding_box']['ny']
        self.nz=grid_json['Grid_bounding_box']['nz']
        
        self.dx = grid_json['Grid_bounding_box']['dx'] 
        self.dy = grid_json['Grid_bounding_box']['dy'] 
        self.dz = grid_json['Grid_bounding_box']['dz']
        
        assert np.isclose(self.dx,((self.bounding_box_xmax- self.bounding_box_xmin)/(self.nx-1))), \
               'bad data discription'
        assert np.isclose(self.dy,((self.bounding_box_ymax- self.bounding_box_ymin)/(self.ny-1))), \
               'bad data discription'  
        assert np.isclose(self.dz,((self.bounding_box_zmax- self.bounding_box_zmin)/(self.nz-1))), \
               'bad data discription'    
         
        # -----------------------------------------------------------------------------------
        # ----CREAT THE GRID---------------------------------------------------------------------
        # -----------------------------------------------------------------------------------
        
        self.xgrid = np.linspace(self.bounding_box_xmin, self.bounding_box_xmax,self.nx)
        self.ygrid = np.linspace(self.bounding_box_ymin, self.bounding_box_ymax,self.ny)
        self.zgrid = np.linspace(self.bounding_box_zmin, self.bounding_box_zmax,self.nz)   
        self.X, self.Y, self.Z = np.meshgrid(self.xgrid, self.ygrid, self.zgrid, indexing='ij')     
        # defulte:
        
        # Mie scattering for water droplets
        self.mie = shdom.MiePolydisperse()
        self.mie.read_table(scat_file_path)          
        
        self.mask3D = np.zeros([self.nx,self.ny,self.nz])
        self.lwc = self.mask3D
        self.re = self.mask3D
        self.extinction = self.mask3D
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------

        
    def get_grid_XYZ(self):
        return self.X, self.Y, self.Z
    
    def set3Dmask(self,mask3D):
        self.mask3D = mask3D
        
    def generate_lwc(self,MIN_VAL,MAX_VAL):    
        # rho is how the value change with high.
        rho = np.linspace(MIN_VAL, MAX_VAL,self.nz)
        _,_,rho = np.meshgrid(self.xgrid, self.ygrid, rho, indexing='ij')
        self.lwc = self.mask3D*rho
        
    def generate_homogenous_reff(self, reff_homogenous = 10):# reff_homogenous is in um
        _, _, rho = np.meshgrid(self.xgrid, self.ygrid, reff_homogenous*np.ones_like(self.zgrid), indexing='ij')
        self.re = self.mask3D*rho

    def compute_extinction(self,veff=0.1):
        # extrarc grid data:
        bounding_box = shdom.BoundingBox(self.bounding_box_xmin,
                                             self.bounding_box_ymin,
                                             self.bounding_box_xmin,
                                             self.bounding_box_xmax,
                                             self.bounding_box_ymax,
                                             self.bounding_box_zmax)      
            
        grid = shdom.Grid(bounding_box=bounding_box,nx=self.nx,
                          ny=self.ny,
                          nz=self.nz)
        
        # self.extinction = np.zeros([self.nx,self.ny,self.nz])
        veff = veff * np.ones_like(self.re)
        lwc=shdom.GridData(grid, self.lwc).squeeze_dims()
        reff=shdom.GridData(grid, self.re).squeeze_dims() 
        veff=shdom.GridData(grid, veff).squeeze_dims() 
        extinction = self.mie.get_extinction(lwc, reff, veff) 
        self.extinction = extinction.data

    def visualize_3D_LWC(self):
        mlab.pipeline.volume(mlab.pipeline.scalar_field(self.lwc),vmin=0, vmax=self.lwc.max())
        mlab.colorbar(title='LWC [g/m^3]', orientation='vertical')
        mlab.show()   
        
    def _save(self):
        out_mat = self.sample_name+'_sample.mat'
        out_txt = self.sample_name+'_sample.txt'
                
        comment_line = self.sample_name+' sample by Vadim'
        out_mat = os.path.join(SAMPLE_DIR,out_mat)
        out_txt = os.path.join(SAMPLE_DIR,out_txt)
        
        # save mat files:
        sio.savemat(out_mat, dict(json=self.json_data_file, data_lwc = self.lwc,data_re=self.re, data_beta=self.extinction))  
        
        # save txt file:
        np.savetxt(out_txt, X=np.array([self.lwc.shape]), fmt='%d', header=comment_line)
        f = open(out_txt, 'ab') 
        np.savetxt(f, X=np.concatenate((np.array([self.dx, self.dy]), self.zgrid)).reshape(1,-1), fmt='%2.3f')
        y, x, z = np.meshgrid(range(self.ny), range(self.nx), range(self.nz))
        data = np.vstack((x.ravel(), y.ravel(), z.ravel(), self.lwc.ravel(), self.re.ravel())).T
        np.savetxt(f, X=data, fmt='%d %d %d %.5f %.3f')        
        f.close()    
        
        
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------
        
def long_blob(VIS_FLAG=False):
    data=OrderedDict()
    data['Grid_type'] = "3D"
    Grid_bounding_box=OrderedDict()
    
    Grid_bounding_box['nx'] = 50
    Grid_bounding_box['ny'] = 50
    Grid_bounding_box['nz'] = 50
    
    Grid_bounding_box['dz'] = 20*1e-3 # must be in km
    Grid_bounding_box['dx'] = 20*1e-3 # must be in km
    Grid_bounding_box['dy'] = 20*1e-3 # must be in km
    
    H = Grid_bounding_box['dz']*float(Grid_bounding_box['nz']) # must be in km
    Lx = Grid_bounding_box['dx']*float(Grid_bounding_box['nx']) # must be in km
    Ly = Grid_bounding_box['dy']*float(Grid_bounding_box['ny']) # must be in km
    
    H = round(H,5)
    Lx = round(Lx,5)
    Ly = round(Ly,5)

    Grid_bounding_box['xmin'] = float_round(-0.5*Lx) # must be in km
    Grid_bounding_box['xmax'] = float_round((0.5*Lx)-Grid_bounding_box['dx'])
    Grid_bounding_box['ymin'] = float_round(-0.5*Ly)
    Grid_bounding_box['ymax'] = float_round((0.5*Ly)-Grid_bounding_box['dy'])
    Grid_bounding_box['zmin'] = float_round(0)
    Grid_bounding_box['zmax'] = float_round(H-Grid_bounding_box['dz'])
    data['Grid_bounding_box'] = Grid_bounding_box
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # work with TOY_CLOUD_SAMPLE object:
    cloud_sample = TOY_CLOUD_SAMPLE('long_blob',data)
    
    # set 3D mask the define where is the cloud:
    X,Y,Z = cloud_sample.get_grid_XYZ()
    AtmWidth = 0.5*Lx
    AtmHeight = H
    Z = ((X + 0.6*AtmWidth/3)**2)/0.002 + ((Z - AtmHeight/2.5)**2)*450 + ((Y + 0*AtmWidth/3)**2)/0.012
    A = (Z<(4)**2)
    cloud_sample.set3Dmask(A)
    
    # generate lwc:
    MAX_LWC = (1.5178/1.44)*Grid_bounding_box['zmax'] #
    MIN_LWC = 0.0
    cloud_sample.generate_lwc(MIN_LWC,MAX_LWC)

    # generate re:
    reff_homogenous = 10 # um
    cloud_sample.generate_homogenous_reff(reff_homogenous)
    
    
    # generate extinction 1/km:
    cloud_sample.compute_extinction()
    
    # save evrey thing:
    cloud_sample._save()

    if(VIS_FLAG):
        cloud_sample.visualize_3D_LWC()

def single_voxel(VIS_FLAG=False):
    data=OrderedDict()
    data['Grid_type'] = "3D"
    Grid_bounding_box=OrderedDict()
    
    Grid_bounding_box['nx'] = 21
    Grid_bounding_box['ny'] = 21
    Grid_bounding_box['nz'] = 21
    
    Grid_bounding_box['dz'] = 50*1e-3 # must be in km
    Grid_bounding_box['dx'] = 50*1e-3 # must be in km
    Grid_bounding_box['dy'] = 50*1e-3 # must be in km
    
    H = Grid_bounding_box['dz']*float(Grid_bounding_box['nz']) # must be in km
    Lx = Grid_bounding_box['dx']*float(Grid_bounding_box['nx']) # must be in km
    Ly = Grid_bounding_box['dy']*float(Grid_bounding_box['ny']) # must be in km
    
    H = round(H,5)
    Lx = round(Lx,5)
    Ly = round(Ly,5)
    
    Grid_bounding_box['xmin'] = float_round(-0.5*Lx) # must be in km
    Grid_bounding_box['xmax'] = float_round((0.5*Lx)-Grid_bounding_box['dx'])
    Grid_bounding_box['ymin'] = float_round(-0.5*Ly)
    Grid_bounding_box['ymax'] = float_round((0.5*Ly)-Grid_bounding_box['dy'])
    Grid_bounding_box['zmin'] = float_round(0)
    Grid_bounding_box['zmax'] = float_round(H-Grid_bounding_box['dz'])
    data['Grid_bounding_box'] = Grid_bounding_box
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # work with TOY_CLOUD_SAMPLE object:
    cloud_sample = TOY_CLOUD_SAMPLE('single_voxel',data)
    
    # set 3D mask the define where is the cloud:
    X,Y,Z = cloud_sample.get_grid_XYZ()
    AtmWidth = 0.5*Lx
    AtmHeight = H
    
    bounding_box_xmin = data['Grid_bounding_box']['xmin']
    bounding_box_ymin = data['Grid_bounding_box']['ymin']
    bounding_box_zmin = data['Grid_bounding_box']['zmin']
    bounding_box_xmax = data['Grid_bounding_box']['xmax']
    bounding_box_ymax = data['Grid_bounding_box']['ymax']
    bounding_box_zmax = data['Grid_bounding_box']['zmax']
    dx = data['Grid_bounding_box']['dx'] 
    dy = data['Grid_bounding_box']['dy'] 
    dz = data['Grid_bounding_box']['dz']
    
    A = ((X-bounding_box_xmin)> 10*dx)*((X-bounding_box_xmin) <= 11*dx)*\
        ((Y-bounding_box_ymin) > 10*dy)*((Y-bounding_box_ymin) <= 11*dy)*((Z-bounding_box_zmin) > 10*dz)*((Z-bounding_box_zmin) <= 11*dz)
    cloud_sample.set3Dmask(A)
    
    # generate lwc:
    MAX_LWC = (1.5178/1.44)*Grid_bounding_box['zmax'] #
    MIN_LWC = 0.0
    cloud_sample.generate_lwc(MIN_LWC,MAX_LWC)
       
    # generate re:
    reff_homogenous = 10 # um
    cloud_sample.generate_homogenous_reff(reff_homogenous)

    # generate extinction 1/km:
    cloud_sample.compute_extinction()
    
    # save evrey thing:
    cloud_sample._save()
    
    if(VIS_FLAG):
        cloud_sample.visualize_3D_LWC()


def small_box(VIS_FLAG=False):
    data=OrderedDict()
    data['Grid_type'] = "3D"
    Grid_bounding_box=OrderedDict()
    
    Grid_bounding_box['nx'] = 21
    Grid_bounding_box['ny'] = 21
    Grid_bounding_box['nz'] = 21
    
    Grid_bounding_box['dz'] = 50*1e-3 # must be in km
    Grid_bounding_box['dx'] = 50*1e-3 # must be in km
    Grid_bounding_box['dy'] = 50*1e-3 # must be in km
    
    H = Grid_bounding_box['dz']*float(Grid_bounding_box['nz']) # must be in km
    Lx = Grid_bounding_box['dx']*float(Grid_bounding_box['nx']) # must be in km
    Ly = Grid_bounding_box['dy']*float(Grid_bounding_box['ny']) # must be in km
    
    H = round(H,5)
    Lx = round(Lx,5)
    Ly = round(Ly,5)
    
    Grid_bounding_box['xmin'] = float_round(-0.5*Lx) # must be in km
    Grid_bounding_box['xmax'] = float_round((0.5*Lx)-Grid_bounding_box['dx'])
    Grid_bounding_box['ymin'] = float_round(-0.5*Ly)
    Grid_bounding_box['ymax'] = float_round((0.5*Ly)-Grid_bounding_box['dy'])
    Grid_bounding_box['zmin'] = float_round(0)
    Grid_bounding_box['zmax'] = float_round(H-Grid_bounding_box['dz'])
    data['Grid_bounding_box'] = Grid_bounding_box
    # work with TOY_CLOUD_SAMPLE object:
    cloud_sample = TOY_CLOUD_SAMPLE('small_box',data)
    
    # set 3D mask the define where is the cloud:
    X,Y,Z = cloud_sample.get_grid_XYZ()
    AtmWidth = 0.5*Lx
    AtmHeight = H
    bounding_box_xmin = data['Grid_bounding_box']['xmin']
    bounding_box_ymin = data['Grid_bounding_box']['ymin']
    bounding_box_zmin = data['Grid_bounding_box']['zmin']
    bounding_box_xmax = data['Grid_bounding_box']['xmax']
    bounding_box_ymax = data['Grid_bounding_box']['ymax']
    bounding_box_zmax = data['Grid_bounding_box']['zmax']
    dx = data['Grid_bounding_box']['dx'] 
    dy = data['Grid_bounding_box']['dy'] 
    dz = data['Grid_bounding_box']['dz']
    
    A = ((X-bounding_box_xmin)> 5*dx)*((X-bounding_box_xmin) <= 14*dx)*\
        ((Y-bounding_box_ymin) > 5*dy)*((Y-bounding_box_ymin) <= 14*dy)*((Z-bounding_box_zmin) > 5*dz)*((Z-bounding_box_zmin) <= 14*dz)
    cloud_sample.set3Dmask(A)    
    
    # generate lwc:
    MAX_LWC = (1.5178/1.44)*Grid_bounding_box['zmax'] #
    MIN_LWC = 0.0
    cloud_sample.generate_lwc(MIN_LWC,MAX_LWC)

    # generate re:
    reff_homogenous = 10 # um
    cloud_sample.generate_homogenous_reff(reff_homogenous)
    
    
    # generate extinction 1/km:
    cloud_sample.compute_extinction()
    
    # save evrey thing:
    cloud_sample._save()

    if(VIS_FLAG):
        cloud_sample.visualize_3D_LWC()
        

def thick_layer(VIS_FLAG=False):
    data=OrderedDict()
    data['Grid_type'] = "3D"
    Grid_bounding_box=OrderedDict()
    
    Grid_bounding_box['nx'] = 21
    Grid_bounding_box['ny'] = 21
    Grid_bounding_box['nz'] = 21
    
    Grid_bounding_box['dz'] = 50*1e-3 # must be in km
    Grid_bounding_box['dx'] = 50*1e-3 # must be in km
    Grid_bounding_box['dy'] = 50*1e-3 # must be in km
    
    H = Grid_bounding_box['dz']*float(Grid_bounding_box['nz']) # must be in km
    Lx = Grid_bounding_box['dx']*float(Grid_bounding_box['nx']) # must be in km
    Ly = Grid_bounding_box['dy']*float(Grid_bounding_box['ny']) # must be in km
    
    H = round(H,5)
    Lx = round(Lx,5)
    Ly = round(Ly,5)
    
    Grid_bounding_box['xmin'] = float_round(-0.5*Lx) # must be in km
    Grid_bounding_box['xmax'] = float_round((0.5*Lx)-Grid_bounding_box['dx'])
    Grid_bounding_box['ymin'] = float_round(-0.5*Ly)
    Grid_bounding_box['ymax'] = float_round((0.5*Ly)-Grid_bounding_box['dy'])
    Grid_bounding_box['zmin'] = float_round(0)
    Grid_bounding_box['zmax'] = float_round(H-Grid_bounding_box['dz'])
    data['Grid_bounding_box'] = Grid_bounding_box
    
    # work with TOY_CLOUD_SAMPLE object:
    cloud_sample = TOY_CLOUD_SAMPLE('thick_layer',data)
    
    # set 3D mask the define where is the cloud:  
    X,Y,Z = cloud_sample.get_grid_XYZ()
    AtmWidth = 0.5*Lx
    AtmHeight = H
    bounding_box_xmin = data['Grid_bounding_box']['xmin']
    bounding_box_ymin = data['Grid_bounding_box']['ymin']
    bounding_box_zmin = data['Grid_bounding_box']['zmin']
    bounding_box_xmax = data['Grid_bounding_box']['xmax']
    bounding_box_ymax = data['Grid_bounding_box']['ymax']
    bounding_box_zmax = data['Grid_bounding_box']['zmax']
    dx = data['Grid_bounding_box']['dx'] 
    dy = data['Grid_bounding_box']['dy'] 
    dz = data['Grid_bounding_box']['dz']
    
    A = ((Z-bounding_box_zmin) > 5*dz)*((Z-bounding_box_zmin) <= 14*dz)
    cloud_sample.set3Dmask(A)
       
   # generate lwc:
    MAX_LWC = (1.5178/1.44)*Grid_bounding_box['zmax'] #
    MIN_LWC = 0.0
    cloud_sample.generate_lwc(MIN_LWC,MAX_LWC)

    # generate re:
    reff_homogenous = 10 # um
    cloud_sample.generate_homogenous_reff(reff_homogenous)
    
    
    # generate extinction 1/km:
    cloud_sample.compute_extinction()
    
    # save evrey thing:
    cloud_sample._save()

    if(VIS_FLAG):
        cloud_sample.visualize_3D_LWC()
        
 
def thin_layer(VIS_FLAG=False):
    data=OrderedDict()
    data['Grid_type'] = "3D"
    Grid_bounding_box=OrderedDict()
    
    Grid_bounding_box['nx'] = 21
    Grid_bounding_box['ny'] = 21
    Grid_bounding_box['nz'] = 21
    
    Grid_bounding_box['dz'] = 50*1e-3 # must be in km
    Grid_bounding_box['dx'] = 50*1e-3 # must be in km
    Grid_bounding_box['dy'] = 50*1e-3 # must be in km
    
    H = Grid_bounding_box['dz']*float(Grid_bounding_box['nz']) # must be in km
    Lx = Grid_bounding_box['dx']*float(Grid_bounding_box['nx']) # must be in km
    Ly = Grid_bounding_box['dy']*float(Grid_bounding_box['ny']) # must be in km
    
    H = round(H,5)
    Lx = round(Lx,5)
    Ly = round(Ly,5)
    
    Grid_bounding_box['xmin'] = float_round(-0.5*Lx) # must be in km
    Grid_bounding_box['xmax'] = float_round((0.5*Lx)-Grid_bounding_box['dx'])
    Grid_bounding_box['ymin'] = float_round(-0.5*Ly)
    Grid_bounding_box['ymax'] = float_round((0.5*Ly)-Grid_bounding_box['dy'])
    Grid_bounding_box['zmin'] = float_round(0)
    Grid_bounding_box['zmax'] = float_round(H-Grid_bounding_box['dz'])
    data['Grid_bounding_box'] = Grid_bounding_box
    
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # work with TOY_CLOUD_SAMPLE object:
    cloud_sample = TOY_CLOUD_SAMPLE('thin_layer',data)
    
    # set 3D mask the define where is the cloud:
    X,Y,Z = cloud_sample.get_grid_XYZ()
    AtmWidth = 0.5*Lx
    AtmHeight = H
    bounding_box_xmin = data['Grid_bounding_box']['xmin']
    bounding_box_ymin = data['Grid_bounding_box']['ymin']
    bounding_box_zmin = data['Grid_bounding_box']['zmin']
    bounding_box_xmax = data['Grid_bounding_box']['xmax']
    bounding_box_ymax = data['Grid_bounding_box']['ymax']
    bounding_box_zmax = data['Grid_bounding_box']['zmax']
    dx = data['Grid_bounding_box']['dx'] 
    dy = data['Grid_bounding_box']['dy'] 
    dz = data['Grid_bounding_box']['dz']
    
    A = ((Z-bounding_box_zmin) > 10*dz)*((Z-bounding_box_zmin) <= 11*dz)
    cloud_sample.set3Dmask(A)
       
    # generate lwc:
    MAX_LWC = (1.5178/1.44)*Grid_bounding_box['zmax'] #
    MIN_LWC = 0.0
    cloud_sample.generate_lwc(MIN_LWC,MAX_LWC)

    # generate re:
    reff_homogenous = 10 # um
    cloud_sample.generate_homogenous_reff(reff_homogenous)
    
    
    # generate extinction 1/km:
    cloud_sample.compute_extinction()
    
    # save evrey thing:
    cloud_sample._save()

    if(VIS_FLAG):
        cloud_sample.visualize_3D_LWC()


# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------

def main():

    parser = ArgumentParser()
    parser.add_argument('--scat', type=str,
                        dest='scat_file_path',
                        help='path to the scat file',
                        metavar='scat_file_path', required=True)
    
    parser.add_argument('--output', type=str,
                        dest='SAMPLE_DIR',
                        help='path sample directory',
                        metavar='SAMPLE_DIR', default='Mediume_Samples')
    
    opts = parser.parse_args()
    global SAMPLE_DIR
    global scat_file_path
    SAMPLE_DIR =opts.SAMPLE_DIR
    scat_file_path=opts.scat_file_path
    
    VIS_FLAG = False
    long_blob(VIS_FLAG)
    single_voxel(VIS_FLAG)
    small_box(VIS_FLAG)
    thin_layer(VIS_FLAG)
    thick_layer(VIS_FLAG)
    
if __name__ == '__main__':
    main()
    
    
