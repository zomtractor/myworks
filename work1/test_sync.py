import math
import os
from model import UBlock
import torch
import yaml
import utils
from DataPro.data import get_validation_data
from skimage import img_as_ubyte
import cv2
from torch.utils.data import DataLoader
from utils.utils import img_pad
from torchvision.transforms import ToTensor
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage import io
import numpy as np
from glob import glob
import lpips
import warnings

warnings.filterwarnings("ignore")


def getResult():
    ## Model
    model_restored = UBlock()

    ## Load yaml configuration file
    with open('config.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    Test = opt['TESTING']
    test_dir = Test['TEST_DIR_SYN']
    model_restored.cuda()
    utils.mkdir("./test_result_syn")

    ## DataLoaders
    test_dataset = get_validation_data(test_dir, {'patch_size': Test['TEST_PS']})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2,
                             drop_last=True)
    # Weight_path
    weight_root = Test['WEIGHT_ROOT']
    weight_name = Test['WEIGHT_NAME']
    weight_path = weight_root + weight_name
    ## Evaluation (Validation)
    utils.load_checkpoint(model_restored, weight_path)
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
            restored = model_restored(input_)
            # for res, tar in zip(restored, target):
            #     psnr_val_rgb.append(utils.torchPSNR(res, tar))
            #     ssim_val_rgb.append(utils.torchSSIM(restored, target))
            if h_pad != 0:
                restored = restored[:, :, h_pad:-h_odd_pad, :]
            if w_pad != 0:
                restored = restored[:, :, :, w_pad:-w_odd_pad]
        restored = torch.clamp(restored, 0, 1)
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            cv2.imwrite(os.path.join('./test_result_syn', data_test[2][batch] + '.png'),
                        cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))


def compare_lpips(img1, img2, loss_fn_alex):
    to_tensor = ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)
    output_lpips = loss_fn_alex(img1_tensor.cuda(), img2_tensor.cuda())
    return output_lpips.cpu().detach().numpy()[0, 0, 0, 0]


def compare_score(img1, img2, img_seg):
    # Return the G-PSNR, S-PSNR, Global-PSNR and Score
    # This module is for the MIPI 2023 Challange: https://codalab.lisn.upsaclay.fr/competitions/9402
    mask_type_list = ['glare', 'streak', 'global']
    metric_dict = {'glare': 0, 'streak': 0, 'global': 0}
    for mask_type in mask_type_list:
        mask_area, img_mask = extract_mask(img_seg)[mask_type]
        if mask_area > 0:
            img_gt_masked = img1 * img_mask
            img_input_masked = img2 * img_mask
            input_mse = compare_mse(img_gt_masked, img_input_masked) / (255 * 255 * mask_area)
            input_psnr = 10 * np.log10((1.0 ** 2) / input_mse)
            metric_dict[mask_type] = input_psnr
        else:
            metric_dict.pop(mask_type)
    return metric_dict


def extract_mask(img_seg):
    # Return a dict with 3 masks including streak,glare,global(whole image w/o light source), masks are returned in 3ch.
    # glare: [255,255,0]
    # streak: [255,0,0]
    # light source: [0,0,255]
    # others: [0,0,0]
    mask_dict = {}
    streak_mask = (img_seg[:, :, 0] - img_seg[:, :, 1]) / 255
    glare_mask = (img_seg[:, :, 1]) / 255
    global_mask = (255 - img_seg[:, :, 2]) / 255
    mask_dict['glare'] = [np.sum(glare_mask) / (512 * 512),
                          np.expand_dims(glare_mask, 2).repeat(3, axis=2)]  # area, mask
    mask_dict['streak'] = [np.sum(streak_mask) / (512 * 512), np.expand_dims(streak_mask, 2).repeat(3, axis=2)]
    mask_dict['global'] = [np.sum(global_mask) / (512 * 512), np.expand_dims(global_mask, 2).repeat(3, axis=2)]
    return mask_dict


def calculate_metrics(args):
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    gt_folder = args['gt'] + '/*'
    input_folder = args['input'] + '/*'
    gt_list = sorted(glob(gt_folder))
    input_list = sorted(glob(input_folder))
    if args['mask'] is not None:
        mask_folder = args['mask'] + '/*'
        mask_list = sorted(glob(mask_folder))

    assert len(gt_list) == len(input_list)
    n = len(gt_list)

    ssim, psnr, lpips_val = 0, 0, 0
    score_dict = {'glare': 0, 'streak': 0, 'global': 0, 'glare_num': 0, 'streak_num': 0, 'global_num': 0}
    for i in range(n):
        img_gt = io.imread(gt_list[i])
        img_input = io.imread(input_list[i])
        ssim += compare_ssim(img_gt, img_input, multichannel=True)
        psnr += compare_psnr(img_gt, img_input, data_range=255)
        lpips_val += compare_lpips(img_gt, img_input, loss_fn_alex)
        if args['mask'] is not None:
            img_seg = io.imread(mask_list[i])
            metric_dict = compare_score(img_gt, img_input, img_seg)
            for key in metric_dict.keys():
                score_dict[key] += metric_dict[key]
                score_dict[key + '_num'] += 1
    ssim /= n
    psnr /= n
    lpips_val /= n
    print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips_val}")
    if args['mask'] is not None:
        for key in ['glare', 'streak', 'global']:
            if score_dict[key + '_num'] == 0:
                assert False, "Error, No mask in this type!"
            score_dict[key] /= score_dict[key + '_num']
        score_dict['score'] = 1 / 3 * (score_dict['glare'] + score_dict['global'] + score_dict['streak'])
        print(
            f"Score: {score_dict['score']}, G-PSNR: {score_dict['glare']}, S-PSNR: {score_dict['streak']}, Global-PSNR: {score_dict['global']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./test_result_syn')
    parser.add_argument('--gt', type=str, default='/mnt/zbl/deflare/DataSet/test/synthetic/gt')
    parser.add_argument('--mask', type=str, default=None)
    getResult()
    args = vars(parser.parse_args())
    calculate_metrics(args)
