import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import scipy.stats as st
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.transforms import ToTensor
from typing import Tuple
import pytorch_msssim
import torchvision
import torchvision.models as models
from torch.autograd import Variable#Evaluate
from math import exp

def compare_lpips(img1, img2, loss_fn_alex):
    to_tensor=ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)
    output_lpips = loss_fn_alex(img1_tensor, img2_tensor)
    return output_lpips.detach().numpy()[0,0,0,0]

def calculate_metrics(img_input,img_gt,loss_fn_alex):
    ssim = compare_ssim(img_gt, img_input, channel_axis=2)
    psnr = compare_psnr(img_gt, img_input, data_range=255)
    lpips_val= compare_lpips(img_gt, img_input, loss_fn_alex)
    return lpips_val,psnr,ssim

#Loss
class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam = lam
        self.lam_p = lam_p

    def forward(self, out1, out2, out3, gt1, feature_layers=[2]):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)

        loss1 = self.lam_p * self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam * F.l1_loss(out1, gt1)
        loss2 = self.lam_p * self.loss_fn(out2, gt2, feature_layers=feature_layers) + self.lam * F.l1_loss(out2, gt2)
        loss3 = self.lam_p * self.loss_fn(out3, gt3, feature_layers=feature_layers) + self.lam * F.l1_loss(out3, gt3)

        return loss1 + loss2 + loss3

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
class VGGLoss(nn.Module):
    def __init__(self, conv_index='54', rgb_range=1):
        super(VGGLoss, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
            self.vgg.cuda()
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])
            self.vgg.cuda()

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std).cuda()
        self.vgg.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss


class PSNRLoss(nn.Module):
    """
    reference from: https://github.com/megvii-model/HINet/blob/main/basicsr/models/losses/losses.py
    """
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4
        loss = -(self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean())
        return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM_loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)




class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
class ASLLayer(nn.Module):
    def __init__(self):
        super(ASLLayer, self).__init__()
        kernel_v = [[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]
        kernel_h = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernel_lu = [[-2, -1, 0],
                    [-1, 0, 1],
                    [0, 1, 2]]
        kernel_rd = [[0, -1, -2],
                    [1, 0, -1],
                    [2, 1, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        kernel_lu = torch.FloatTensor(kernel_lu).unsqueeze(0).unsqueeze(0)
        kernel_rd = torch.FloatTensor(kernel_rd).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
        self.weight_lu = nn.Parameter(data=kernel_lu, requires_grad=False)
        self.weight_rd = nn.Parameter(data=kernel_rd, requires_grad=False)
    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)
    def forward(self, x):
        if x.shape[1] == 3:
            x = self.get_gray(x)
        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x_lu = F.conv2d(x, self.weight_lu, padding=1)
        x_rd = F.conv2d(x, self.weight_rd, padding=1)
        return x_v,x_h,x_lu,x_rd
class ASLloss(nn.Module):

    def __init__(self):
        super(ASLloss, self).__init__()
        self.loss = nn.L1Loss()
        self.asl_layer = ASLLayer()
    def forward(self, output, gt_img):
        output_grad1,output_grad2,output_grad3,output_grad4 = self.asl_layer(output)
        gt_grad1,gt_grad2,gt_grad3,gt_grad4 = self.asl_layer(gt_img)
        return self.loss(output_grad1, gt_grad1)+self.loss(output_grad2, gt_grad2)+self.loss(output_grad3, gt_grad3)+self.loss(output_grad4, gt_grad4)
def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter
class Blur(nn.Module):
    def __init__(self, nc):
        super(Blur, self).__init__()
        self.nc = nc
        kernel = gauss_kernel(kernlen=21, nsig=3, channels=self.nc)
        kernel = torch.from_numpy(kernel).permute(2, 3, 0, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
    def forward(self, x):
        if x.size(1) != self.nc:
            raise RuntimeError(
                "The channel of input [%d] does not match the preset channel [%d]" % (x.size(1), self.nc))
        x = F.conv2d(x, self.weight, stride=1, padding=10, groups=self.nc)
        return x
class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
    def forward(self, x1, x2):
        return torch.mean(torch.pow((x1 - x2), 2)).div(2 * x1.size()[0])

#Padding
def img_pad(x, w_pad, h_pad, w_odd_pad, h_odd_pad):
    '''
    Here the padding values are determined by the average r,g,b values across the training set
    in FHDMi dataset. For the evaluation on the UHDM, you can also try the commented lines where
    the mean values are calculated from UHDM training set, yielding similar performance.
    '''
    x1 = F.pad(x[:, 0:1, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.3827)
    x2 = F.pad(x[:, 1:2, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.4141)
    x3 = F.pad(x[:, 2:3, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value=0.3912)
    # x1 = F.pad(x[:, 0:1, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.5165)
    # x2 = F.pad(x[:, 1:2, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.4952)
    # x3 = F.pad(x[:, 2:3, ...], (w_pad, w_odd_pad, h_pad, h_odd_pad), value = 0.4695)
    y = torch.cat([x1, x2, x3], dim=1)

    return y
    
    
    
    

  