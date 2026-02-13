import odl
import torch
import numpy as np
import pandas as pd 

from scipy.interpolate import interp1d
from odl.contrib import torch as odl_torch

class initialization:
    def __init__(self, path="./dataset/xray_characteristic_data.csv", im_size=512, numproj=641):
        self.param = {}
        self.reso = 512 / im_size * 0.08 # says the pixel space

        # values for geometry
        # image
        self.param['nx_h'] = im_size
        self.param['ny_h'] = im_size
        self.param['sx'] = self.param['nx_h']*self.reso
        self.param['sy'] = self.param['ny_h']*self.reso

        ## view
        self.param['startangle'] = 0
        self.param['endangle'] = 2*np.pi

        self.param['nProj'] = numproj # for training network, otherwise 1000

        ## detector
        self.param['su'] = 2*np.sqrt(self.param['sx']**2+self.param['sy']**2)
        self.param['nu_h'] = numproj # for training network, otherwise 1001 
        self.param['dde'] = 1075*self.reso
        self.param['dso'] = 1075*self.reso

        # values for simulation not geometry
        self.param['mu_water'] = 0.193
        self.param['mu_air'] = 0
        self.param["mu_metal"] = 0.536
        self.param["bone"] = 0.234
        
        self.param["poly_order"] = 3
        self.param["T1"] = 100
        self.param["T2"] = 1500
        self.param["metal_density"] = 4.5
        self.param["noise_scale"] = 5

        self.param["intensity"] = 10 ** 7 # just for info (not used)
        self.param["xrays_energy"] = pd.read_csv(path)
        
def build_gemotry(param):
    """_summary_

    Args:
        param (_type_): _description_

    Returns:
        : fp, fbp
    """
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

    kwargs = {"impl":None} # , 'use_cache' :False}
    ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, **kwargs)
    FBPOper_hh = odl.tomo.fbp_op(ray_trafo_hh, filter_type='Hann', frequency_scaling=1.0)
    
    fp = odl_torch.OperatorModule(ray_trafo_hh)
    bfp = odl_torch.OperatorModule(FBPOper_hh) # fildered backprojection
    bp = odl_torch.OperatorModule(ray_trafo_hh.adjoint) # backprojection
    op_norm = odl.power_method_opnorm(ray_trafo_hh)
    
    # print("op_norm", op_norm)
    return fp, bfp, bp

def hu2mu(inp, mu_water, mu_air):
    out = inp/1000*(mu_water - mu_air) + mu_water
    return out

def mu2hu(inp, mu_water, mu_air):
    # out = inp/1000*(mu_water - mu_air) + mu_water
    out = 1000*(inp - mu_water)/(mu_water - mu_air)
    return out

def create_phantom(shape, radius, mu_water):
    x_size, y_size = shape
    x = np.arange(-(x_size - 1) / 2, (x_size - 1) / 2 + 1)
    y = np.arange(-(y_size - 1) / 2, (y_size - 1) / 2 + 1)
    X, Y = np.meshgrid(x, y)
    phantom = (X**2 + Y**2) < radius**2
    phantom = phantom.astype(np.float32)
    phantom = phantom * mu_water
    return phantom

def get_torch(inp):
    out_t = torch.tensor(inp[np.newaxis, np.newaxis, ...])
    return out_t

def untorch(torch_tensor):
    out = torch_tensor[0].detach().cpu().squeeze().numpy()
    return out

def __monochromatic(phantom, fp):
    phantom_mono = untorch(fp(get_torch(phantom))) # self.Fproj(self.phantom)
    
    y = np.exp(-phantom_mono)
    p = np.log(1/y)
    return p

def __polychromatic(phantom, fp, mu0_water, xrays_energy, param):
    phantom_poly = untorch(fp(get_torch(phantom)))
    
    total_intensity = 0
    v = np.zeros(shape=(phantom_poly.shape[0], phantom_poly.shape[1], len(xrays_energy)), dtype=np.float32)
    for i, row in xrays_energy.iterrows():
        m_water, m_bone, m_metal, intensity = row[["Water", "Bone", "Titanium", "Intensity"]]
        d_water_temp = phantom_poly * (m_water / mu0_water)
        DRR = d_water_temp
        y = intensity * np.exp(-DRR, dtype=np.float32)
        v[:, :, i] = y
        total_intensity += intensity
    poly_y = np.sum(v, 2)
    p = np.log(total_intensity/ poly_y)
    # p[p<10**-10] = 0
    return p

def water_correction(phantom, fp, param):
    # create phantom
    mu_water, poly_order, xrays_energy = param.param["mu_water"], param.param["poly_order"], param.param["xrays_energy"]
    order = poly_order
    mono = __monochromatic(phantom, fp)
    # print(mono.min(), mono.max())
    poly = __polychromatic(phantom, fp, mu_water, xrays_energy, param)

    coeff = np.polyfit(poly.flatten(), mono.flatten(), order)
    return coeff

def min_max_img():
    return 0, 1

def normalize(inp, minmax):
    vmin, vmax = minmax
    out = (inp - vmin) / (vmax - vmin)
    return out

def compute_correction(X, coef):
    return np.polyval(coef, X).astype(np.float32)

def thresholding(image, metal, T1, T2):
    if T1 is None or T2 is None:
        raise ValueError('Missing arguments')

    w_bone = (image - T1).astype(np.float32) / (T2 - T1)
    w_bone = np.clip(w_bone, 0, 1)
    bone = w_bone * image.astype(np.float32)

    w_water = (T2 - image).astype(np.float32) / (T2 - T1)
    w_water = np.clip(w_water, 0, 1)
    water = w_water * image.astype(np.float32)

    water[metal>0], bone[metal>0] = 0, 0

    return water, bone

def __poisson_noise(image):
    rng = None
    rng = np.random.default_rng(rng)
    if image.min() < 0:
        low_clip = -1.0
    else:
        low_clip = image.min()
    
    print("low_clip:", low_clip)
    
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    
    # Ensure image is exclusively positive
    if low_clip == -1.0:
        old_max = image.max()
        image = (image + 1.0) / (old_max + 1.0)

    # Generating noise for each unique value in image.
    out = rng.poisson(image * vals) / vals
    
    # Return image to original range if input was signed
    if low_clip == -1.0:
        out = out * (old_max + 1.0) - 1.0
    return out

def compute_intensity(d_water, d_bone, d_metal, param):
    xrays_energy = param.param["xrays_energy"]
    total_intensity = 0
    v = np.zeros(shape=(d_water.shape[0], d_water.shape[1], len(xrays_energy)), dtype=np.float32)

    for i, row in xrays_energy.iterrows():
        m_water, m_bone, m_metal, intensity = row[["Water","Bone","Titanium","Intensity"]]
        d_water_temp = d_water * (m_water/param.param['mu_water'])
        d_bone_temp = d_bone * (m_bone/param.param["bone"])
        d_metal_temp = d_metal * (m_metal/param.param["mu_metal"])

        DRR = d_water_temp + d_bone_temp + d_metal_temp
        y = intensity * np.exp(-DRR, dtype=np.float32)
        v[:,:,i] = y
        total_intensity = total_intensity + intensity
    poly_y = np.sum(v, 2)

    low_clip = (poly_y / (10**param.param["noise_scale"])).min()
    print("low Clip Old:", low_clip)
    #temp = np.round(np.exp(-poly_y)*total_intensity)
    #temp = temp + 20

    ProjPhoton = (10**param.param["noise_scale"]) * __poisson_noise(poly_y / (10**param.param["noise_scale"]))
    ProjPhoton[ProjPhoton<10**-10] = 10
    # ProjPhoton = np.clip(ProjPhoton, 1e-10, None)
    projkvpMetalNoise = -np.log(ProjPhoton / total_intensity, dtype=np.float32)


    # scale = 10 ** param.param["noise_scale"]
    #noisy_y = scale * np.random.poisson(poly_y / scale)
    #noisy_y = (10**param.param["noise_scale"]) * __poisson_noise(poly_y / (10**param.param["noise_scale"])) #param.param["noise_scale"]))
    #noisy_y = np.clip(noisy_y, 1e-10, None)
    #print("noisy info:", noisy_y.min(), noisy_y.max())
    #p = -np.log(noisy_y/total_intensity, dtype=np.float32)
    
    return ProjPhoton, projkvpMetalNoise


def metal_artifact_simulation(image, metal, param, fp, fbp, poly_coef=0):
    """return xma, sma, sgt"""
    
    T1 = hu2mu(param.param["T1"], param.param["mu_water"], param.param["mu_air"])
    T2 = hu2mu(param.param["T2"], param.param["mu_water"], param.param["mu_air"])
    # print(T1, T2)
    sgt = untorch(fp(get_torch(image)))
    
    x_water, x_bone = thresholding(image, metal, T1, T2)

    mu_metal0 = param.param["mu_metal"] * param.param["metal_density"]
    x_metal = metal.astype(np.float32) * mu_metal0
    
    # print(x_metal.min(), x_metal.max())

    d_water = untorch(fp(get_torch(x_water)))
    d_bone = untorch(fp(get_torch(x_bone)))
    d_metal = untorch(fp(get_torch(x_metal)))

    poly_y, y_int = compute_intensity(d_water, d_bone, d_metal, param)

    p = np.polyval(poly_coef, y_int).astype(np.float32)

    sim = untorch(fbp(get_torch(p)))
    sim[sim<0] = 0
    
    return sim, p, sgt