import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import BasicConv, FAB, OCAB, MFFE, MDFusion, CBAM

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


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = BasicConv(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv = BasicConv(in_channels, out_channels*4, kernel_size=3, stride=1, padding=1)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x

class LightParamPredictor(nn.Module):
    def __init__(self, in_ch, n_lights=3):
        super().__init__()
        self.n = n_lights
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, n_lights * 11)  # 每个光源11个参数
    def forward(self, feat):
        x = self.conv(feat).view(feat.size(0), -1)
        p = self.fc(x).view(feat.size(0), self.n, -1)
        # 参数解码
        x_pos = torch.sigmoid(p[...,0])  # [0,1]
        y_pos = torch.sigmoid(p[...,1])
        a = F.softplus(p[...,2]) + 1e-3
        b = F.softplus(p[...,3]) + 1e-3
        angle = torch.tanh(p[...,4]) * math.pi
        intensity = F.softplus(p[...,5])
        alpha = torch.sigmoid(p[...,6])
        color_r = torch.sigmoid(p[...,7])
        color_g = torch.sigmoid(p[...,8])
        color_b = torch.sigmoid(p[...,9])
        falloff = F.softplus(p[...,10]) + 1e-3  # 衰减控制
        params = torch.stack([x_pos,y_pos,a,b,angle,intensity,alpha,
                              color_r,color_g,color_b,falloff], dim=-1)
        return params

def render_light_batch(params, H, W):
    """ 根据预测参数生成光源图 (B,3,H,W) """
    B, n, _ = params.shape
    device = params.device
    yy, xx = torch.meshgrid(
        torch.linspace(0,1,H,device=device),
        torch.linspace(0,1,W,device=device),
        indexing='ij'
    )
    xx, yy = xx.unsqueeze(0).unsqueeze(0), yy.unsqueeze(0).unsqueeze(0)
    xx = xx.expand(B,n,H,W)
    yy = yy.expand(B,n,H,W)
    x_pos = params[...,0].unsqueeze(-1).unsqueeze(-1)
    y_pos = params[...,1].unsqueeze(-1).unsqueeze(-1)
    a = params[...,2].unsqueeze(-1).unsqueeze(-1)
    b = params[...,3].unsqueeze(-1).unsqueeze(-1)
    angle = params[...,4].unsqueeze(-1).unsqueeze(-1)
    intensity = params[...,5].unsqueeze(-1).unsqueeze(-1)
    alpha = params[...,6].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    color_r = params[...,7].unsqueeze(-1).unsqueeze(-1)
    color_g = params[...,8].unsqueeze(-1).unsqueeze(-1)
    color_b = params[...,9].unsqueeze(-1).unsqueeze(-1)
    falloff = params[...,10].unsqueeze(-1).unsqueeze(-1)

    dx, dy = xx - x_pos, yy - y_pos
    cos, sin = torch.cos(angle), torch.sin(angle)
    X = cos*dx + sin*dy
    Y = -sin*dx + cos*dy
    G = torch.exp(-((X/a)**2 + (Y/b)**2) ** falloff)
    R = intensity * color_r * G
    Gc = intensity * color_g * G
    Bc = intensity * color_b * G
    light_rgb = torch.stack([R,Gc,Bc], dim=-1)  # (B,n,H,W,3)
    light_rgb = (alpha * light_rgb).sum(dim=1)  # (B,H,W,3)
    return light_rgb.permute(0,3,1,2)  # (B,3,H,W)

class LightAdjustModule(nn.Module):
    """ 光源调整模块 LAM """
    def __init__(self, in_ch, n_lights=3):
        super().__init__()
        self.predictor = LightParamPredictor(in_ch, n_lights=n_lights)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_ch + 3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )
    def forward(self, feat, pred_img):
        B, C, H, W = pred_img.shape
        params = self.predictor(feat)
        light = render_light_batch(params, H, W)
        fuse_in = torch.cat([pred_img, feat], dim=1)  # 用原始RGB特征辅助融合
        alpha = self.fuse_conv(fuse_in)
        out = pred_img * (1 - alpha) + light * alpha
        return out, light, alpha, params



class MyNet2_5(nn.Module):
    def __init__(self, base_channels=16, num_block=3, num_bottleneck=2):
        super(MyNet2_5, self).__init__()
        self.num_block = num_block
        self.num_bottleneck = num_bottleneck
        self.proj_in = BasicConv(3, base_channels, kernel_size=3, padding=1)
        self.proj_laplacian = nn.ModuleList([BasicConv(3, base_channels * 2 ** (i), kernel_size=3, padding=1) for i in
                               range(num_block)])
        self.ebs = nn.ModuleList([EBlock(base_channels * 2 ** i) for i in range(num_block)])
        self.ebs_flare = nn.ModuleList([EBlockFlare(base_channels * 2 ** (i)) for i in range(num_block)])
        self.bottleneck = nn.ModuleList([BottleNeck(base_channels * 2 ** (num_block)) for _ in range(num_bottleneck)])
        self.reduce = nn.ModuleList([BasicConv(base_channels * 2**(i+1),base_channels * 2**i,relu=True,act=nn.LeakyReLU) for i in range(num_block)])
        self.dbs_pred = nn.ModuleList([DBlock(base_channels * 2 ** (i)) for i in range(num_block)])

        self.ups = nn.ModuleList([UpSample(base_channels * 2 ** (i+1), base_channels * 2 ** i) for i in range(num_block)])
        self.downs = nn.ModuleList([DownSample(base_channels * 2 ** (i), base_channels * 2 ** (i+1)) for i in range(num_block)])
        self.projout = BasicConv(base_channels, 3, kernel_size=1, padding=0, norm=False, relu=False)
        # self.sigmoid = nn.Sigmoid()
        self.lam = LightAdjustModule(base_channels, n_lights=3)

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
            res = self.reduce[-1-i](res)
            res = self.dbs_pred[-1 - i](res)

        # res = self.sigmoid(self.projout(res))

        pred = self.projout(res)

        # 光源调整模块（高分辨率修正）
        out, light_map, alpha, params = self.lam(res, x+pred)

        return out, light_map, alpha, params

    def getFeatureMaps(self, x):
        features = {}
        with torch.no_grad():
            skip = []
            gauss, laplacian = GTB(x, layer=self.num_block)
            res = self.proj_in(x)
            features['proj_in'] = res

            for i in range(0, self.num_block):
                res = self.ebs[i](res)
                features[f'ebs_{i}'] = res
                skip1 = self.ebs_flare[i](res + self.proj_laplacian[i](laplacian[i]))
                features[f'ebs_flare_{i}'] = skip1
                skip.append(skip1)
                res = self.downs[i](skip1)
                features[f'downs_{i}'] = res

            for i in range(self.num_bottleneck):
                res = self.bottleneck[i](res)
                features[f'bottleneck_{i}'] = res

            for i in range(0, self.num_block):
                res = self.ups[-1 - i](res)
                features[f'ups_{-1 - i}'] = res
                res = torch.cat((res, skip[-1 - i]), dim=1)
                features[f'concat_{-1 - i}'] = res
                res = self.reduce[-1 - i](res)
                features[f'reduce_{-1 - i}'] = res
                res = self.dbs_pred[-1 - i](res)
                features[f'dbs_pred_{-1 - i}'] = res


            features['projout'] = self.projout(res)
        return features
