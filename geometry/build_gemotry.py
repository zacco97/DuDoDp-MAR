import odl  # https://github.com/odlgroup/odl
import numpy as np
from odl.contrib import torch as odl_torch

## 640geo
class initialization:
    def __init__(self):
        self.param = {}
        self.reso = 512 / 512 * 0.08

        # image
        self.param['nx_h'] = 512
        self.param['ny_h'] = 512
        self.param['sx'] = self.param['nx_h']*self.reso
        self.param['sy'] = self.param['ny_h']*self.reso

        ## view
        self.param['startangle'] = 0
        self.param['endangle'] = 2 * np.pi

        self.param['nProj'] = 640

        ## detector
        self.param['su'] = 2*np.sqrt(self.param['sx']**2+self.param['sy']**2)
        self.param['nu_h'] = 640
        self.param['dde'] = 1075*self.reso
        self.param['dso'] = 1075*self.reso
        self.param['u_water'] = 0.192 #0.0205

def imaging_geo(param):
    reco_space_h = odl.uniform_discr(
        min_pt=[-param.param['sx'] / 2.0, -param.param['sy'] / 2.0],
        max_pt=[param.param['sx'] / 2.0, param.param['sy'] / 2.0], shape=[param.param['nx_h'], param.param['ny_h']],
        dtype='float32')
    angle_partition = odl.uniform_partition(param.param['startangle'], param.param['endangle'],
                                            param.param['nProj'])

    detector_partition_h = odl.uniform_partition(-(param.param['su'] / 2.0), (param.param['su'] / 2.0),
                                                 param.param['nu_h'])

    geometry_h = odl.tomo.FanBeamGeometry(angle_partition, detector_partition_h,
                                          src_radius=param.param['dso'],
                                          det_radius=param.param['dde'])
    # geometry_h = odl.tomo.geometry.conebeam.FanFlatGeometry(angle_partition, detector_partition_h,
    #                                     src_radius=param.param['dso'],
    #                                     det_radius=param.param['dde'])

    ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl=None)  #https://github.com/astra-toolbox/astra-toolbox
    FBPOper_hh = odl.tomo.fbp_op(ray_trafo_hh, filter_type='Hamming', frequency_scaling=1.0)
    
    fp = odl_torch.OperatorModule(ray_trafo_hh)
    fbp = odl_torch.OperatorModule(FBPOper_hh) # fildered backprojection
    bp = odl_torch.OperatorModule(ray_trafo_hh.adjoint) # backprojection

    
    return fp, fbp, bp
