import math
import os
import torch
import yaml
import utils
from DataPro.data import get_test_data
from skimage import img_as_ubyte
import cv2
from torch.utils.data import DataLoader
from models.UFT import UFCNet
from utils.utils import img_pad
from torchvision.transforms import ToTensor
import argparse
from skimage import io
import numpy as np
from glob import glob
import lpips
import time
import warnings
warnings.filterwarnings("ignore")

model_restored = UFCNet()

## Load yaml configuration file
with open('config.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Test = opt['TESTING']
# test_dir = Test['TEST_DIR']
test_dir='/mnt/zbl/deflare/DataSet/test/dawn'
model_restored.cuda()
utils.mkdir("./rtest_result")
utils.mkdir("./rtest_mask_result")
## DataLoaders
test_dataset = get_test_data(test_dir,'')
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2,
                         drop_last=True)
#Weight_path
weight_root=Test['WEIGHT_ROOT']
weight_name=Test['WEIGHT_NAME']
weight_path=weight_root+weight_name
## Evaluation (Validation)
utils.load_checkpoint(model_restored, weight_path)
model_restored.eval()
for ii, data_test in enumerate(test_loader, 0):
    input_ = data_test[0].cuda()
    b, c, h, w = input_.size()
    k=16
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
        start_time = time.time()
        mask,restored = model_restored(input_)
        print((time.time() -start_time))  # seconds
    # for res, tar in zip(restored, target):
    #     psnr_val_rgb.append(utils.torchPSNR(res, tar))
    #     ssim_val_rgb.append(utils.torchSSIM(restored, target))
        if h_pad != 0:
           restored = restored[:, :, h_pad:-h_odd_pad, :]
        if w_pad != 0:
           restored = restored[:, :, :, w_pad:-w_odd_pad]
        if h_pad != 0:
            mask = mask[:, :, h_pad:-h_odd_pad, :]
        if w_pad != 0:
            mask = mask[:, :, :, w_pad:-w_odd_pad]
    restored = torch.clamp(restored, 0, 1)
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    for batch in range(len(restored)):
        restored_img = img_as_ubyte(restored[batch])
        cv2.imwrite(os.path.join('./rtest_result', data_test[1][batch] + '.png'), cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))
    mask = mask.sigmoid().data.cpu().numpy().squeeze()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    cv2.imwrite(os.path.join('./rtest_mask_result', data_test[1][0] + '.png'), mask * 255)


