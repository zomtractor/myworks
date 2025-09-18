import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BasicConv
from .layers import *


class EBlock(nn.Module):
    def __init__(self, channels):
        super(EBlock, self).__init__()
        layers = [ResBlock(channels, channels, "ITS") for _ in range(8 - 1)]
        layers.append(ResBlock(channels, channels, "ITS", filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlockPred(nn.Module):
    def __init__(self, channels):
        super(DBlockPred, self).__init__()
        layers = [ResBlock(channels, channels, "ITS") for _ in range(8 - 1)]
        layers.append(ResBlock(channels, channels, "ITS", filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
        # return x


class BottleNeck(nn.Module):
    def __init__(self, channels):
        super(BottleNeck, self).__init__()
        self.convMlp = nn.Sequential(
            BasicConv(channels, channels // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(channels // 2, channels, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.convMlp(x)

class DBlockFlare(nn.Module):
    def __init__(self, channels):
        super(DBlockFlare, self).__init__()
        layers = [ResBlock(channels, channels, "ITS") for _ in range(8 - 1)]
        layers.append(ResBlock(channels, channels, "ITS", filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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


# gaussian transform block
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
        blurred = F.conv2d(current, kernel, padding=2, groups=c)
        blurred = blurred[:, :, ::2, ::2]
        res_gaussian.append(blurred)
        upsampled = F.interpolate(blurred, size=current.shape[2:], mode='bilinear', align_corners=False)
        laplacian = current - upsampled
        res_laplacian.append(laplacian)
        current = blurred
    return res_gaussian, res_laplacian


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.down(x)
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class MyNet(nn.Module):
    def __init__(self, base_channels=16, num_block=3, num_bottleneck=2):
        super(MyNet, self).__init__()
        self.num_block = num_block
        self.num_bottleneck = num_bottleneck

        self.proj = nn.ModuleList([BasicConv(3, base_channels, kernel_size=3, padding=1)])
        for i in range(num_block - 1):
            self.proj.append(ConvS(base_channels * 2 ** i))
        self.proj_laplacian = nn.ModuleList([BasicConv(3, base_channels * 2 ** (i + 1), kernel_size=3, padding=1) for i in
                               range(num_block)])
        self.ebs = nn.ModuleList([EBlock(base_channels * 2 ** i) for i in range(num_block)])
        self.bottleneck = nn.ModuleList([BottleNeck(base_channels * 2 ** (num_block - 1)) for _ in range(num_bottleneck)])
        self.dbs_pred = nn.ModuleList([DBlockPred(base_channels * 2 ** (num_block  - i)) for i in range(num_block)])
        self.dbs_flare = nn.ModuleList([DBlockFlare(base_channels * 2 ** (num_block  - i)) for i in range(num_block)])
        self.ups_pred = nn.ModuleList([UpSample(base_channels * 2 ** (num_block - 1), base_channels * 2 ** (num_block - 1))])
        self.ups_flare = nn.ModuleList([UpSample(base_channels * 2 ** (num_block - 1), base_channels * 2 ** (num_block - 1))])
        for i in range(1, num_block):
            self.ups_pred.append(
                UpSample(base_channels * 2 ** (num_block - i + 1), base_channels * 2 ** (num_block - 1 - i)))
            self.ups_flare.append(
                UpSample(base_channels * 2 ** (num_block - i + 1), base_channels * 2 ** (num_block - 1 - i)))
        self.downs = nn.ModuleList([DownSample(base_channels * 2 ** i, base_channels * 2 ** i) for i in range(num_block)])
        self.projout_pred = nn.ModuleList([BasicConv(base_channels * 2 ** (num_block - i), 3, kernel_size=3, padding=1, norm=True) for
                             i in range(num_block)])
        self.projout_flare = nn.ModuleList([BasicConv(base_channels * 2 ** (num_block - i), 3, kernel_size=3, padding=1, norm=True)
                              for i in range(num_block)])

    def forward(self, x):
        skip = []
        gauss, laplacian = GTB(x, layer=3)
        res = self.ebs[0](self.proj[0](gauss[0]))
        skip.append(res)
        for i in range(1, self.num_block):
            res = self.downs[i - 1](res)
            res = torch.cat((res, self.proj[i](gauss[i])), dim=1)
            res = self.ebs[i](res)
            skip.append(res)
        res = self.downs[-1](res)
        for i in range(self.num_bottleneck):
            res = self.bottleneck[i](res)
        res_pred = res
        res_flare = res
        outs_pred = []
        outs_flare = []
        for i in range(self.num_block):
            res_pred = self.ups_pred[i](res_pred)
            res_pred = torch.cat((skip[-1 - i], res_pred), dim=1)
            res_pred = self.dbs_pred[i](res_pred)

            res_flare = self.ups_flare[i](res_flare)
            res_flare = torch.cat((skip[-1 - i], res_flare), dim=1)
            res_flare += self.proj_laplacian[-1 - i](laplacian[-1 - i])
            res_flare = self.dbs_flare[i](res_flare)

            outs_flare.append(self.projout_flare[i](res_flare))

            res_pred = res_pred - res_flare
            outs_pred.append(self.projout_pred[i](res_pred)+gauss[-2 - i])

        return outs_pred, outs_flare

#
# if __name__ == '__main__':
#     model = MyNet(base_channels=16).cuda()
#     x = torch.randn(2, 3, 384, 384).cuda()  # Batch size of 1, 3 channels, 512x512 image
#     pred, flare = model(x)
#     print(pred, flare)  # Should be (1, 3, 512, 512)
