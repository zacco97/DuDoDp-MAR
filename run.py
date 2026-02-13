import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from geometry.build_gemotry import initialization, imaging_geo
from geometry.syndeeplesion_data import test_image, mydatasetsample, ssim_psnr_rmse, mydatasetCTPelvic
import yaml
import argparse
import h5py
import pandas as pd
from patch_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    add_dict_to_dict,
    dict_to_dict
)
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
import imageio.v2 as imageio
import numpy as np
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info

def convert(inp, norm_values=(-1000, 3071)):
    inp = inp.clamp(-1, 1)
    out = inp * (norm_values[1] - norm_values[0]) + norm_values[0]
    return out

def mu2hu(inp, mu_water, mu_air):
    # out = inp/1000*(mu_water - mu_air) + mu_water
    out = 1000*(inp - mu_water)/(mu_water - mu_air)
    return out

if __name__ == "__main__":
    
    param = initialization()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/MAR.yaml',
                        help='yaml file for configuration')
    args = parser.parse_args()
    yaml_path = args.config
    
    metrics = ['PSNR', 'SSIM', 'RMSE']
    columns = pd.MultiIndex.from_product([["DUDODP"], metrics])
    full_columns = pd.MultiIndex.from_tuples(
    [('pixels', '')] + list(columns),
    names=['method', 'metric']
    )
    df = pd.DataFrame(columns=full_columns)
    
    with open(yaml_path) as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
    args_dict = add_dict_to_dict(args_dict, model_and_diffusion_defaults())

    a = args_dict['a']
    n = args_dict['n']
    delta_y = args_dict['delta_y']
    save_dir = args_dict['save_dir']
    data_path = args_dict['data_path']
    inner_dir = args_dict['inner_dir']
    n_img = args_dict['num_test_image']
    n_mask = args_dict['num_test_mask']

    model_names = ["../persistent/MAR/weights/dudodpmar/model150000.pt"]
    ct_pevic_dataset = "" # real dataset path
    synthetic_dataset = "" # 
    id_ct_pelvic = 208
    
    models = []
    for model_name in model_names:
        model, diffusion = create_model_and_diffusion(
            **dict_to_dict(args_dict, model_and_diffusion_defaults().keys())
        )
        add_dict_to_argparser(parser, args_dict)
        args = parser.parse_args()
        weights = torch.load(model_name, map_location="cpu", weights_only=False)
        model.load_state_dict(weights)
        
        model.to("cuda")
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        models.append(model)
    
    model_fns = models
    denoised_fn = None
    
    os.makedirs(save_dir, exist_ok=True)
    
    args.batch_size = 1
    resize1 = Resize(416)
    resize2 = Resize(512)
    
    fp, fbp, _  = imaging_geo(param) 
    
    maxVal = 1
    maxRef = None
    real = True
    
    if not real:
        file = h5py.File(synthetic_dataset)
    
    
    for imag_idx in tqdm(range(0, 1000, 1)): # 200 for all test data
        # print(imag_idx)
        # for mask_idx in tqdm(range(n_mask)): # 10
        #     _, _, _, _, Sma, _, _, Tr = test_image(data_path, imag_idx, mask_idx, inner_dir)
        # break
            # M = resize2(M)
            # _ = fp(Xgt) # Must use forward before calling backward or specify a volume
        if real:
            Sma, Tr, mask = mydatasetCTPelvic(ct_pevic_dataset, id_ct_pelvic) #"../temp/luca.zaccagna/CTPelvic/dataset7_CLINIC_metal_0000_data.nii.gz"
        else:
            Sma, Tr, metalBW, xgt = mydatasetsample(file, imag_idx, fp) # torch.randn(1,1,640, 640).to("cuda"), torch.randn(1,1,640, 640).to("cuda"), torch.randn(1,1,512, 512).to("cuda"), torch.randn(1,1,512, 512).to("cuda") # 
            mask = metalBW 
            xgtHU = mu2hu(xgt, 0.193, 0).clip(-1000, 3071)
            maxRef = float(xgtHU.max())
        
        model_kwargs = {}
        # print(Sma.shape, Tr.shape)
        
        sample_fn = diffusion.p_mar_loop
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        sample = sample_fn(
            model_fns,
            (args.batch_size, 1, args.image_size, args.image_size),
            Sma.view(1, 1, 640, 640),
            Tr.view(1, 1, 640, 640),
            fp,
            fbp,
            (a, n, delta_y),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            denoised_fn=denoised_fn,
            device=torch.device("cuda")
        )

        end_event.record()
        torch.cuda.synchronize()
    
        total_time_ms = start_event.elapsed_time(end_event)
        
        print(f"Total Algorithm Time: {total_time_ms:.2f} ms")
        # print(f"Average time per step: {total_time_ms / nfe:.2f} ms")
        
        print("sample", sample.min(), sample.max())
        sample = (sample + 1)/2
        sample = mu2hu(sample, 0.193, 0)
        # sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        
        # sample = sample /255. 
        # print(sample.min(), sample.max())
        # sample = mu2hu(sample, 0.192, 0)
        
        # print(np.min(sample.cpu().numpy()), np.max(sample.cpu().numpy()))
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.squeeze().cpu().detach().numpy()
        
        if not real:
            ROIx, ROIy = np.arange(0, 512), np.arange(0, 512)
            SSIM, PSNR, RMSE = ssim_psnr_rmse(xgtHU, sample, metalBW=metalBW, ROIx=ROIx, ROIy=ROIy, 
                                                        returnROI=False, maxVal=maxVal, maxRef=maxRef)
            print(f"{imag_idx}", SSIM, PSNR, RMSE)
        
        im = sample.clip(-175, 275)
        im = (im + 175) / 450
        # print(im.shape)
        im = np.stack([im, im, im], axis=-1)  # convert grayscale -> RGB
        mask_bool = mask == 1
        im[mask_bool] = [1, 0, 0]
        plt.imshow(im, cmap="gray")
        plt.axis("off")
        plt.show()
        # plt.savefig("figure_r1.jpg")
        
        
        # fig, axs = plt.subplots(1,2)
        # axs[0].imshow(sample, cmap="gray")
        # axs[1].imshow(xgtHU, cmap="gray")
        # plt.show()
        #df.loc[imag_idx, ("pixels", '')] = 0        
        #df.loc[imag_idx, ("DUDODP", 'PSNR')] = PSNR
        #df.loc[imag_idx, ("DUDODP", 'SSIM')] = SSIM
        #df.loc[imag_idx, ("DUDODP", 'RMSE')] = RMSE
        #if imag_idx == 20:
        #    break
    #df.to_csv("results/dudodp_res1.csv", index=False)
        #print(SSIM, PSNR, RMSE)
        #plt.imshow(sample, vmin=-175, vmax=275, cmap="gray")
        #plt.show()
    
            # imageio.imwrite(os.path.join(save_dir, '%03d_%03d.png'%(imag_idx, mask_idx)), (sample.squeeze().cpu().numpy() / 255. *65535.).astype(np.uint16))
