import argparse
import math
import os
import warnings
from glob import glob

import cv2
import lpips
import numpy as np
import torch
import yaml
from skimage import img_as_ubyte
from skimage import io
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import model
import utils
from data.data import get_validation_data
from utils.utils import img_pad

warnings.filterwarnings("ignore")


def getResult(type,name):
    with open('config.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    ## Model
    # model_restored = UBlock(base_channels=opt['TRAINOPTIM']['CHANNELS'])
    model_class = getattr(model, opt['MODEL']['ARCH'])
    model_args = opt['MODEL']['ARGS']
    model_restored = model_class(**model_args)

    ## Load yaml configuration file

    Test = opt['TESTING']
    test_dir = Test[f'TEST_DIR_{type.upper()}']
    model_restored.cuda()
    utils.mkdir(f"./mask_result_{type}")

    ## DataLoaders
    test_dataset = get_validation_data(test_dir, {'patch_size': Test['TEST_PS']})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2,
                             drop_last=True)
    # Weight_path

    model_path = f"./checkpoints/{opt['MODEL']['MODE']}/models/model_best{name}_{type.upper()}.pth"
    ## Evaluation (Validation)
    utils.load_checkpoint(model_restored, model_path)
    model_restored.eval()
    for ii, data_test in enumerate(test_loader, 0):
        target = data_test[0].cuda()
        input_ = data_test[1].cuda()
        b, c, h, w = input_.size()
        k = 16
        # pad image such that the resolution is a multiple of 32
        w_pad = (math.ceil(w / k) * k - w) // 2
        h_pad = (math.ceil(h / k) * k - h) // 2
        w_odd_pad = w_pad
        h_odd_pad = h_pad
        if w % 2 == 1:
            w_odd_pad += 1
        if h % 2 == 1:
            h_odd_pad += 1
        input_ = img_pad(input_, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)

        with torch.no_grad():
            restored,masks = model_restored(input_)
            mask = masks[2]
            # for res, tar in zip(restored, target):
            #     psnr_val_rgb.append(utils.torchPSNR(res, tar))
            #     ssim_val_rgb.append(utils.torchSSIM(restored, target))
            if h_pad != 0:
                mask = mask[:, :, h_pad:-h_odd_pad, :]
            if w_pad != 0:
                mask = mask[:, :, :, w_pad:-w_odd_pad]
        mask = torch.clamp(mask, 0, 1)
        mask = mask.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(mask)):
            mask_img = img_as_ubyte(mask[batch])
            cv2.imwrite(os.path.join(f"./mask_result_{type}", data_test[2][batch] + '.png'),
                        cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default='PSNR',choices=['PSNR', 'SSIM', 'LPIPS', 'Gpsnr','Spsnr'], help='which index to calculate')
    parser.add_argument('--type', type=str, default='real', choices=['real', 'syn'], help='which type of data to test')
    args = parser.parse_args()
    print(f"Testing {args.type} {args.index} model for datasets...")
    getResult("real",args.index)
    getResult("syn",args.index)