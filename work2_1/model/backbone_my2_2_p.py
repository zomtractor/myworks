import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BasicConv, FAB, OCAB, MFFE, MDFusion, CBAM
from .layers import *

class FeatureBlock(nn.Module):
    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]   # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index
    def __init__(self, channels,windows_size=8, overlap_ratio=0.5):
        super(FeatureBlock, self).__init__()
        self.fab1 = FAB(channels)
        self.fab2 = FAB(channels)
        self.cbam = CBAM(channels)
        self.mffe = MFFE(channels)

    def forward(self, x):
        res = self.fab1(x)
        res = self.fab2(res)
        res = self.cbam(res)
        res = self.mffe(res)
        return x + res


class EBlock(nn.Module):
    def __init__(self, channels):
        super(EBlock, self).__init__()
        self.eb = FeatureBlock(channels, windows_size=8, overlap_ratio=0.5)

    def forward(self, x):
        return self.eb(x)


class DBlock(nn.Module):
    def __init__(self, channels):
        super(DBlock, self).__init__()
        self.db = FeatureBlock(channels, windows_size=8, overlap_ratio=0.5)

    def forward(self, x):
        return self.db(x)
        # return x


class BottleNeck(nn.Module):
    def __init__(self, channels):
        super(BottleNeck, self).__init__()
        self.b = FeatureBlock(channels, windows_size=8, overlap_ratio=0.5)

    def forward(self, x):
        return self.b(x)

class EBlockFlare(nn.Module):
    def __init__(self, channels):
        super(EBlockFlare, self).__init__()
        self.db = MDFusion(channels, channels)

    def forward(self, x):
        return self.db(x)


class ConvS(nn.Module):
    def __init__(self, out_channels):
        super(ConvS, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_channels // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channels // 4, out_channels // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channels // 2, out_channels, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_channels, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


# gaussian transform block, return gaussian pyramid[layer] and laplacian pyramid[layer-1]
def GTB(x, layer=4):
    res_gaussian = []
    res_laplacian = []
    kernel = torch.tensor([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]], dtype=torch.float32, device=x.device) / 256.0
    b, c, h, w = x.shape
    kernel = kernel.view(1, 1, 5, 5).repeat(c, 1, 1, 1)
    current = x
    res_gaussian.append(current)
    for i in range(layer):
        pad = F.pad(current, (2, 2, 2, 2), mode='reflect')
        blurred = F.conv2d(pad, kernel, groups=c)
        blurred = blurred[:, :, ::2, ::2]
        res_gaussian.append(blurred)
        upsampled = F.interpolate(blurred, size=current.shape[2:], mode='bilinear', align_corners=False)
        laplacian = current - upsampled
        res_laplacian.append(laplacian)
        current = blurred
    return res_gaussian, res_laplacian


# pixelshuffle
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.PixelUnshuffle(2)
        self.conv = BasicConv(in_channels*4, out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv = BasicConv(in_channels, out_channels*4, kernel_size=3, stride=1, padding=1)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x


class MyNet2_2_p(nn.Module):
    def __init__(self, base_channels=16, num_block=3, num_bottleneck=2):
        super(MyNet2_2_p, self).__init__()
        self.num_block = num_block
        self.num_bottleneck = num_bottleneck
        self.proj_in = BasicConv(3, base_channels, kernel_size=3, padding=1)
        self.proj_laplacian = nn.ModuleList([BasicConv(3, base_channels * 2 ** (i), kernel_size=3, padding=1) for i in
                               range(num_block)])
        self.ebs = nn.ModuleList([EBlock(base_channels * 2 ** i) for i in range(num_block)])
        self.ebs_flare = nn.ModuleList([EBlockFlare(base_channels * 2 ** (i)) for i in range(num_block)])
        self.bottleneck = nn.ModuleList([BottleNeck(base_channels * 2 ** (num_block)) for _ in range(num_bottleneck)])
        self.dbs_pred = nn.ModuleList([DBlock(base_channels * 2 ** (i+1)) for i in range(num_block)])
        self.out_reduce = nn.ModuleList([BasicConv(base_channels * 2**(i+1),base_channels * 2**i,relu=True, act=nn.LeakyReLU) for i in range(num_block)])

        self.ups = nn.ModuleList([UpSample(base_channels * 2 ** (i+1), base_channels * 2 ** i) for i in range(num_block)])
        self.downs = nn.ModuleList([DownSample(base_channels * 2 ** (i), base_channels * 2 ** (i+1)) for i in range(num_block)])
        self.projout = BasicConv(base_channels, 3, kernel_size=1, padding=0, norm=False, relu=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip = []
        gauss, laplacian = GTB(x, layer=self.num_block)
        res = self.proj_in(x)

        for i in range(0, self.num_block):
            res = self.ebs[i](res)
            skip1 = self.ebs_flare[i](res+self.proj_laplacian[i](laplacian[i]))
            skip.append(skip1)
            res = self.downs[i](skip1)

        for i in range(self.num_bottleneck):
            res = self.bottleneck[i](res)

        for i in range(0, self.num_block):
            res = self.ups[-1-i](res)
            res = torch.cat((res, skip[-1 - i]), dim=1)
            res = self.dbs_pred[-1 - i](res)
            res = self.out_reduce[-1-i](res)
        # res = self.sigmoid(self.projout(res))
        return self.projout(res)
