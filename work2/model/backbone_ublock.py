import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from model import MDFusion, LayerNorm, FAB, MFFE, CBAM


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x2)*x1 + F.gelu(x1)*x2
        x = self.project_out(x)
        return x


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
    def __init__(self, in_channels):
        super(DownSample, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,stride=2,padding=1)

    def forward(self, x):
        x1 = self.down(x)
        x2 = self.conv(x)
        return torch.cat([x1, x2], dim=1)  # Concatenate along channel dimension


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.inconv = nn.ConvTranspose2d(in_channels//2, in_channels//2,
                                            kernel_size=2, stride=2)

    def forward(self, x):
        x1,x2 = torch.chunk(x, 2, dim=1)  # Split into two parts
        x1 = self.up(x1)
        x2 = self.inconv(x2)
        return (x1+x2)/2


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
        self.up4 = UpSample(base_channels * 16)
        self.db4 = FeatureBlock(base_channels * 8)
        self.up3 = UpSample(base_channels * 8)
        self.db3 = FeatureBlock(base_channels * 4)
        self.up2 = UpSample(base_channels * 4)
        self.db2 = FeatureBlock(base_channels * 2)
        self.up1 = UpSample(base_channels * 2)
        self.db1 = FeatureBlock(base_channels)

        self.tail = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        self.betas = nn.Parameter(torch.ones(5), requires_grad=True)  # Learnable parameters for each upsample block

    def forward(self, x):
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
        out = v4+self.betas[0]*self.up4(out)
        out = self.db4(out)
        out = v3+self.betas[1]*self.up3(out)
        out = self.db3(out)
        out = v2+self.betas[2]*self.up2(out)
        out = self.db2(out)
        out = v1+self.betas[3]*self.up1(out)
        out = self.db1(out)
        out = self.tail(out)
        return x+self.betas[4]*out


if __name__ == '__main__':
    model = UBlock(in_channels=3, base_channels=32,in_height=256,in_width=256,weight_connect=True)
    model = model.cuda()
    x = torch.randn(1, 3, 256, 256)  # Batch size of 1, 3 channels, 512x512 image
    for i in range(100):
        x = x.cuda()
    output = model(x)
    print(output.shape)  # Should be (1, 3, 512, 512)

