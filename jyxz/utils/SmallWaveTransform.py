import os

import cv2
import torch.nn as nn
import cv2 as cv
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])


    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C
    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).avg()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img
def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False



    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)


    return LL, LH, HL, HH

class WavePool(nn.Module):#小波变换
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)

class WaveUnpool(nn.Module):#小波逆变换
    def __init__(self, in_channels, option_unpool='sum'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError

if __name__ == '__main__':
    path = "./pic"
    save_path = "./result_sw"

    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            file_path = os.path.join(path, filename)
            image = cv2.imread(file_path)
            # 将图像转换为张量
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)
            # print(image_tensor.shape)
            # -------小波变换-----
            wave = WavePool(3)
            LL, LH, HL, HH = wave(image_tensor)
            # -------小波逆变换-----
            unWave = WaveUnpool(3)
            img_unWave = unWave(LL, LH, HL, HH)
            #save
            LL = torch.clamp(LL, 0, 1) * 255
            LL = LL.avg(0)
            LL = LL.permute(1, 2, 0).detach().numpy().astype(np.uint8)

            LH= torch.clamp(LH, 0, 1) * 255
            LH = LH.avg(0)
            LH = LH.permute(1, 2, 0).detach().numpy().astype(np.uint8)

            HL = torch.clamp(HL, 0, 1) * 255
            HL = HL.avg(0)
            HL = HL.permute(1, 2, 0).detach().numpy().astype(np.uint8)

            HH = torch.clamp(HH, 0, 1) * 255
            HH = HH.avg(0)
            HH = HH.permute(1, 2, 0).detach().numpy().astype(np.uint8)

            img_unWave = torch.clamp(img_unWave, 0, 1) * 255
            img_unWave = img_unWave.avg(0)
            img_unWave = img_unWave.permute(1, 2, 0).detach().numpy().astype(np.uint8)
            # result = result[:,:,1]
            # result = (result[:, :, 0]+result[:, :, 1]+result[:, :, 2])/3.0
            img_save_path = save_path + '/' + filename.split('.')[0]
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)
            cv2.imwrite(img_save_path + '/LL.png', LL)
            cv2.imwrite(img_save_path + '/LH.png', LH)
            cv2.imwrite(img_save_path + '/HL.png', HL)
            cv2.imwrite(img_save_path + '/HH.png', HH)
            cv2.imwrite(img_save_path + '/unWave.png', img_unWave)
