import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BasicConv, FAB, OCAB, MFFE, MDFusion, CBAM
from .layers import *

class EBlock(nn.Module):
    def __init__(self, out_channel,  num_res=8, data='GTA5'):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel, data) for _ in range(num_res - 1)]
        layers.append(ResBlock(out_channel, out_channel, data, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, data='GTA5'):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel, data) for _ in range(num_res - 1)]
        layers.append(ResBlock(channel, channel, data, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class BottleNeck(nn.Module):
    def __init__(self, channel, num_res=8, data='GTA5'):
        super(BottleNeck, self).__init__()
        layers = [ResBlock(channel, channel, data) for _ in range(num_res - 1)]
        layers.append(ResBlock(channel, channel, data, filter=True))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

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


class MyNet2_2_ConvIR(nn.Module):
    def __init__(self, base_channels=16, num_block=3, num_bottleneck=2):
        super(MyNet2_2_ConvIR, self).__init__()
        self.num_block = num_block
        self.num_bottleneck = num_bottleneck
        self.proj_in = BasicConv(3, base_channels, kernel_size=3, padding=1)
        self.proj_laplacian = nn.ModuleList([BasicConv(3, base_channels * 2 ** (i), kernel_size=3, padding=1) for i in
                               range(num_block)])
        self.ebs = nn.ModuleList([EBlock(base_channels * 2 ** i) for i in range(num_block)])
        self.ebs_flare = nn.ModuleList([EBlockFlare(base_channels * 2 ** (i)) for i in range(num_block)])
        self.bottleneck = nn.ModuleList([BottleNeck(base_channels * 2 ** (num_block)) for _ in range(num_bottleneck)])
        self.dbs_pred = nn.ModuleList([DBlock(base_channels * 2 ** (i+1)) for i in range(num_block)])
        self.out_reduce = nn.ModuleList([BasicConv(base_channels * 2**(i+1),base_channels * 2**i) for i in range(num_block)])

        self.ups = nn.ModuleList([UpSample(base_channels * 2 ** (i+1), base_channels * 2 ** i) for i in range(num_block)])
        self.downs = nn.ModuleList([DownSample(base_channels * 2 ** (i), base_channels * 2 ** (i+1)) for i in range(num_block)])
        self.projout = BasicConv(base_channels, 6, kernel_size=1, padding=0, norm=False, relu=False)
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
        res = self.projout(res)
        pred,flare = torch.chunk(res,2,dim=1)
        return pred+x,flare+x
