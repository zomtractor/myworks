import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from model import MDFusion, LayerNorm, FAB, MFFE, CBAM, BasicConv, OCAB


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
        self.window_size = windows_size
        self.overlap_ratio = overlap_ratio
        self.rpi = self.calculate_rpi_oca()

        self.fab1 = FAB(channels)
        self.fab2 = FAB(channels)
        self.ocab = OCAB(dim=channels,
                         window_size=windows_size,
                         overlap_ratio=overlap_ratio,
                         num_heads=4)
        self.mffe = MFFE(channels)

    def forward(self, x):
        b,c,h,w = x.shape
        res = self.fab1(x)
        res = self.fab2(res)
        # (b,c,h,w)->(b,h*w,c)
        res = res.flatten(2).transpose(1, 2)
        res = self.ocab(res,(h,w),self.rpi)
        res = res.transpose(1, 2).view(b, c, h, w)
        res = self.mffe(res)
        return x + res


class EBlock(nn.Module):
    def __init__(self, channels):
        super(EBlock, self).__init__()
        self.eb = FeatureBlock(channels, windows_size=8, overlap_ratio=0.5)

    def forward(self, x):
        return self.eb(x)


class DBlockPred(nn.Module):
    def __init__(self, channels):
        super(DBlockPred, self).__init__()
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

class DBlockFlare(nn.Module):
    def __init__(self, channels):
        super(DBlockFlare, self).__init__()
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


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels=None, downscale_factor=2):
        super(DownSample, self).__init__()
        if out_channels is None:
            out_channels = in_channels*2
        self.down = nn.PixelUnshuffle(downscale_factor)
        self.conv = nn.Conv2d(in_channels * (downscale_factor ** 2), out_channels, kernel_size=3, stride=1, padding=1,
                              groups=in_channels)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels=None, upscale_factor=2):
        super(UpSample, self).__init__()
        if out_channels is None:
            out_channels = in_channels//2
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1,
                              groups=out_channels)
        self.up = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x

class UBlock(nn.Module):
    def __init__(self, in_channels=3, base_channels=32,in_height=512,in_width=512,weight_connect=True):
        super(UBlock, self).__init__()
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.eb1 = FeatureBlock(base_channels)
        self.down1 = DownSample(base_channels)
        self.eb2 = FeatureBlock(base_channels * 2)
        self.down2 = DownSample(base_channels * 2)
        self.eb3 = FeatureBlock(base_channels * 4)
        self.down3 = DownSample(base_channels * 4)
        self.eb4 = FeatureBlock(base_channels * 8)
        self.down4 = DownSample(base_channels * 8)
        self.bottleneck = FeatureBlock(base_channels * 16)
        self.dbp4 = DBlockFlare(base_channels * 16)
        self.up4 = UpSample(base_channels * 16)
        self.db4 = FeatureBlock(base_channels * 8)
        self.dbp3 = DBlockFlare(base_channels * 8)
        self.up3 = UpSample(base_channels * 8)
        self.db3 = FeatureBlock(base_channels * 4)
        self.dbp2 = DBlockFlare(base_channels * 4)
        self.up2 = UpSample(base_channels * 4)
        self.db2 = FeatureBlock(base_channels * 2)
        self.dbp1 = DBlockFlare(base_channels * 2)
        self.up1 = UpSample(base_channels * 2)
        self.db1 = FeatureBlock(base_channels)

        self.tail = nn.Conv2d(base_channels, in_channels*2, kernel_size=3, padding=1)

    def forward(self, x):
        gauss, laplacian = GTB(x, layer=4)

        out = self.head(x)
        v1 = self.eb1(out)
        out = self.down1(v1)
        v2 = self.eb2(out)
        out = self.down2(v2)
        v3 = self.eb3(out)
        out = self.down3(v3)
        v4 = self.eb4(out)
        out = self.down4(v4)
        out = self.bottleneck(out)
        out = self.up4(out)
        out = self.dbp4(out+laplacian[3])
        out = self.db4(out)
        out = self.up3(out)
        out = self.dbp3(out+laplacian[2])
        out = self.db3(out)
        out = self.up2(out)
        out = self.dbp2(out+laplacian[1])
        out = self.db2(out)
        out = self.up1(out)
        out = self.dbp1(out+laplacian[0])
        out = self.db1(out)
        out = self.tail(out)
        pred,flare = torch.chunk(out, 2, dim=1)
        return x+pred, x+flare


if __name__ == '__main__':
    model = UBlock(in_channels=3, base_channels=32,in_height=256,in_width=256,weight_connect=True)
    model = model.cuda()
    x = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 512x512 image
    for i in range(100):
        x = x.cuda()
    output = model(x)
    print(output.shape)  # Should be (1, 3, 512, 512)

