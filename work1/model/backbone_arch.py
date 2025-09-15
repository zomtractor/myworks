import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from model import FAB, CBAM, MFFE


class FeatureBlock(nn.Module):
    def __init__(self, channels):
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


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class WeightedConnect(nn.Module):
    def __init__(self, channels, height, width,requires_grad=True):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(1, channels, height, width))
        if not requires_grad:
            self.weights.requires_grad = False
        self.activation = nn.Sigmoid()
    def forward(self,weighted, x):
        return weighted * self.activation(self.weights)+x

class UBlock(nn.Module):
    def __init__(self, in_channels=3, base_channels=32,in_height=512,in_width=512,weight_connect=True):
        super(UBlock, self).__init__()
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.eb1 = FeatureBlock(base_channels)
        self.encoder1 = DownSample(base_channels, base_channels * 2)
        self.eb2 = FeatureBlock(base_channels * 2)
        self.encoder2 = DownSample(base_channels * 2, base_channels * 4)
        self.eb3 = FeatureBlock(base_channels * 4)
        self.encoder3 = DownSample(base_channels * 4, base_channels * 8)
        self.eb4 = FeatureBlock(base_channels * 8)
        self.encoder4 = DownSample(base_channels * 8, base_channels * 16)
        self.bottleneck = FeatureBlock(base_channels * 16)
        self.decoder4 = UpSample(base_channels * 16, base_channels * 8)
        self.db4 = FeatureBlock(base_channels * 8)
        self.decoder3 = UpSample(base_channels * 8, base_channels * 4)
        self.db3 = FeatureBlock(base_channels * 4)
        self.decoder2 = UpSample(base_channels * 4, base_channels * 2)
        self.db2 = FeatureBlock(base_channels * 2)
        self.decoder1 = UpSample(base_channels * 2, base_channels)
        self.db1 = FeatureBlock(base_channels)

        self.tail = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

        c,h,w=in_channels,in_height,in_width
        self.weight0 = WeightedConnect(c,h,w,requires_grad=weight_connect)
        self.weight1 = WeightedConnect(base_channels,h,w,requires_grad=weight_connect)
        self.weight2 = WeightedConnect(base_channels*2,h//2,w//2,requires_grad=weight_connect)
        self.weight3 = WeightedConnect(base_channels*4,h//4,w//4,requires_grad=weight_connect)
        self.weight4 = WeightedConnect(base_channels*8,h//8,w//8,requires_grad=weight_connect)

    def forward(self, x):
        out = self.head(x)
        v1 = self.eb1(out)
        out = self.encoder1(v1)
        v2 = self.eb2(out)
        out = self.encoder2(v2)
        v3 = self.eb3(out)
        out = self.encoder3(v3)
        v4 = self.eb4(out)
        out = self.encoder4(v4)
        out = self.bottleneck(out)
        out = self.decoder4(out)
        out = self.weight4(v4,out)
        out = self.db4(out)
        out = self.decoder3(out)
        out = self.weight3(v3,out)
        out = self.db3(out)
        out = self.decoder2(out)
        out = self.weight2(v2,out)
        out = self.db2(out)
        out = self.decoder1(out)
        out = self.weight1(v1,out)
        out = self.db1(out)
        out = self.tail(out)
        return self.weight0(x,out)


if __name__ == '__main__':
    model = UBlock(in_channels=3, base_channels=32,in_height=256,in_width=256,weight_connect=True)
    model = model.cuda()
    x = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 512x512 image
    for i in range(100):
        x = x.cuda()
    output = model(x)
    print(output.shape)  # Should be (1, 3, 512, 512)

