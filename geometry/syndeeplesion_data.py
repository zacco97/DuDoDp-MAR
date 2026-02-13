import os
import os.path
import argparse
import numpy as np
import torch
# import matplotlib.pyplot as plt
import h5py
from .simulation import build_gemotry, initialization, untorch, get_torch, create_phantom, metal_artifact_simulation, water_correction, hu2mu
from PIL import Image
import matplotlib.pyplot as plt 
import SimpleITK as sitk

# from .build_gemotry import initialization, imaging_geo
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    # data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    # data = data * 255.0
    data = data * 2. - 1.
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data

# param = initialization()
# ray_trafo, FBPOper, _ = imaging_geo(param)
def test_image(data_path, imag_idx, mask_idx, inner_dir):
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    test_mask = np.load(os.path.join(data_path, 'testmask.npy'))
    with open(txtdir, 'r') as f:
        mat_files = f.readlines()
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    abs_dir = os.path.join(data_path, inner_dir, data_file)
    gt_absdir = os.path.join(data_path, inner_dir, gt_dir[:-1])
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    file = h5py.File(abs_dir, 'r')
    Xma= file['ma_CT'][()]
    Sma = file['ma_sinogram'][()]
    XLI = file['LI_CT'][()]
    SLI = file['LI_sinogram'][()]
    Tr = file['metal_trace'][()]
    # Sgt = np.asarray(ray_trafo(Xgt))
    file.close()
    M512 = test_mask[:,:,mask_idx]
    M = np.array(Image.fromarray(M512).resize((416, 416), Image.Resampling.BILINEAR))
    Xma = normalize(Xma, image_get_minmax())  # *255
    Xgt = normalize(Xgt, image_get_minmax())
    XLI = normalize(XLI, image_get_minmax())
    Sma = normalize(Sma, proj_get_minmax())
    #Sgt = normalize(Sgt, proj_get_minmax())
    SLI = normalize(SLI, proj_get_minmax())
    Tr = 1 - Tr.astype(np.float32)
    
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(Tr, cmap="gray")
    axs[1].imshow(Sma[0][0], cmap="gray")
    plt.show()
    
    Tr = np.expand_dims(np.transpose(np.expand_dims(Tr, 2), (2, 0, 1)), 0)  # 1*1*h*w
    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)),0)
    return 0, 0, 0, 0, \
       torch.Tensor(Sma).float().cuda(), 0, 0, torch.Tensor(Tr).cuda()

# my dataset
def my_normalize(data, minmax):
    data_min, data_max = minmax
    # data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    # data = data * 255.0
    data = data * 2. - 1.
    data = data.astype(np.float32)
    # data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data

def read_real_images(p, slice):
    """_summary_

    Args:
        p (str): path to volume

    Returns:
        (np.ndarray): return one slice sampled randomly
    """
    name = p.split("\\")[-1].split("_")[-2]
    Image = sitk.ReadImage(p)
    image = sitk.GetArrayFromImage(Image)
    assert slice < image.shape[0]
    im = image[slice,:, :]
    im = np.clip(im, -1024, 3071)
    return im

def mydatasetCTPelvic(file, slice):
    param = initialization(path="./geometry/xray_characteristic_data.csv", im_size=512, numproj=640) 
    fp, fbp, _ = build_gemotry(param)

    image = read_real_images(file, slice)
    im_mu = hu2mu(image, 0.193, 0)
    sma = untorch(fp(get_torch(im_mu)))
    
    mask = np.zeros_like(image)
    mask[image>2500] = 1.0
    mask = mask > 0
    smt = untorch(fp(get_torch(mask)))
    smt = smt > 0
    
    Tr = 1 - smt.astype(np.float32)
    SMA = my_normalize(sma, (0, 10.0)) # [0, 10]

    fig, axs = plt.subplots(1,3)
    axs[0].imshow(sma, cmap="gray")
    axs[1].imshow(Tr, cmap="gray")
    axs[2].imshow(mask, cmap="gray")
    plt.savefig("./figure_t.jpg")
    
    print(SMA.min(), SMA.max())
    sma, smt = get_torch(SMA).float().cuda(), get_torch(Tr).float().cuda()
    return sma, smt, mask

def mydatasetsample(file, idx, fp):
    XGT = file["XGT"][idx] # ground truth
    XMA = file["XMA"][idx] # corrupted image
    M = file["M"][idx] # mask of corruption
    SGT = file["SGT"][idx] # sinogram gt
    SMA =  file["SMA"][idx] # sinogram ma
    
    param = initialization(path="./geometry/xray_characteristic_data.csv", im_size=512, numproj=640) 
    fp, fbp, _ = build_gemotry(param)
    # xgt = gt.copy()
    phantom = create_phantom((param.param["nx_h"], param.param["nx_h"]), 256, param.param["mu_water"])
    coef = water_correction(phantom, fp, param)
    XMA, SMA, SGT = metal_artifact_simulation(XGT, M, param, fp, fbp, coef)
    
    print(XGT.min(), XGT.max(), XMA.min(), XMA.max(), SMA.min(), SMA.max())
    
    #M = np.zeros_like(XMA)
    #M[XMA > hu2mu(2000, 0.193, 0)] = 1.0
    M = M > 0
    Tr = untorch(fp(get_torch(M)))
    Tr = Tr > 0
    # SMA = untorch(fp(get_torch(XMA)))
    
    Tr = 1 - Tr.astype(np.float32)

    #fig, axs = plt.subplots(2,3)
    #axs[0,0].imshow(M, cmap="gray")
    #axs[0,1].imshow(XGT, cmap="gray")
    #axs[0,2].imshow(XMA, cmap="gray")
    #axs[1,0].imshow(Tr, cmap="gray")
    #axs[1,1].imshow(SGT, cmap="gray")
    #axs[1,2].imshow(SMA, cmap="gray")
    #plt.savefig("./figure_t.jpg")
    
    # SMT = 1 - my_normalize(SMT, (0, 1.0)) # [0, 1]
    SMA = my_normalize(SMA, (0, 10.0)) # [0, 10]
    print(SMA.min(), SMA.max())
    sma, smt = get_torch(SMA).float().cuda(), get_torch(Tr).float().cuda()
    return sma, smt, M, XGT    
    
def imscale(im0, maxRef, maxVal):
    """Map image intensity range to [0, maxVal]."""
    im = im0 / maxRef * maxVal
    im = np.clip(im, 0, maxVal)
    return im

    
def ssim_psnr_rmse(imRef, imCor, metalBW, ROIx, ROIy, maxVal, maxRef, returnROI=False):
    """
    Compute SSIM, PSNR and RMSE for all images in a given ROI, masking out metal pixels.
    All metrics computed on range [-1000, 3071].
    metalBW: must be boolean
    imCor: must be in range of [-1000, 3071]
    imRef: must be in range of [-1000, 3071]
    ROIx, ROIy: np.arrange(x0,x1), np.arrange(y0,y1)
    
    Returns
    -------
    np.ndarray
        SSIM values for all images within the ROI (float array).
    if return_ROI == TRUE: return refROI and tempROI
    """    
    # Mask out metal in reference
    temp = imRef.copy()
    temp[metalBW==1] = 0 # set to minimum value
    
    # Scale and extract ROI
    imRefROI = imscale(temp[np.ix_(ROIx, ROIy)], maxRef, maxVal) # temp[np.ix_(ROIx, ROIy)]
    #rmse_ref = imRefROI.copy()
    
    temp = imCor.copy()
    temp[metalBW==1] = 0 # set to minimum value
    
    tempROI = imscale(temp[np.ix_(ROIx, ROIy)], maxRef, maxVal) # temp[np.ix_(ROIx, ROIy)]
    # rmse_ma = tempROI.copy()

    # compute ssim
    ssim_val = structural_similarity(
        imRefROI,
        tempROI,
        data_range=maxVal,
        gaussian_weights=True,  # optional: matches typical SSIM defaults
        use_sample_covariance=False
    )
    # compute psnr
    psnr_val = peak_signal_noise_ratio(imRefROI, 
                                       tempROI, 
                                       data_range=maxVal)
    
    rmse_val = 0
    
    # compute RMSE on the HU 
    pixSum = np.sum(1 - metalBW[np.ix_(ROIx, ROIy)])
    diff = imRefROI - tempROI
    rmse_val = np.sqrt(np.sum(diff**2) / pixSum)

    if returnROI:
        return ssim_val, psnr_val, rmse_val, imRefROI, tempROI
    else:
        return ssim_val, psnr_val, rmse_val   
        