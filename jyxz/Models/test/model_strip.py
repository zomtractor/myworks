import torch
import torch.nn as nn
import torch.nn.functional as F
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        self.strip = StripBlock(out_channel, 1, 19)
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = self.strip(self.proj(x))  # B H*W C
        return x


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        out = self.conv(x)  # B H*W C
        return out


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        out = self.deconv(x) # B H*W C
        return out


class StripBlock(nn.Module):
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim,dim,kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)     
        self.conv_spatial2 = nn.Conv2d(dim,dim,kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):   
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)

        return x * attn



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, d_model,k1,k2):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripBlock(d_model,k1,k2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
class CustomResidualModule(nn.Module):
    def __init__(self, channels):
        """
        参数:
        - channels: 输入特征图的通道数（C）
        - module_N: 任意子模块，例如卷积、注意力等
        - ffn: 前馈网络（Feed-Forward Network），可为任意处理模块
        """
        super(CustomResidualModule, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.module_N = Attention(channels, 1, 19)
        self.bn2 = nn.BatchNorm2d(channels)
        self.ffn = Mlp(in_features=channels, hidden_features=4*channels, act_layer=nn.GELU, drop=0.)

    def forward(self, f1):
        # f1: [B, C, H, W]
        x = self.bn1(f1)
        x = self.module_N(x)
        f2 = f1 + x  # 残差连接1

        x = self.bn2(f2)
        x = self.ffn(x)
        f3 = f2 + x  # 残差连接2

        return f3
class DoubleConv(nn.Module):
    """两个连续的Conv + BN + ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):

    def rgb_to_y_channel(self, rgb_image):
        """
        输入: rgb_image, 形状为 [B, 3, H, W]，值范围为 [0, 1] 或 [0, 255]
        输出: y_channel, 形状为 [B, 1, H, W]
        """

        # RGB to Y 公式（ITU-R BT.601）
        # Y = 0.299 * R + 0.587 * G + 0.114 * B

        R = rgb_image[:, 0:1, :, :]
        G = rgb_image[:, 1:2, :, :]
        B = rgb_image[:, 2:3, :, :]

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        return Y

    def __init__(self, in_channels, out_channels, base_channels=32):
        super(UNet, self).__init__()
        self.InputProjy = InputProj(in_channel=1, out_channel=base_channels)
        self.InputProj = InputProj(in_channel=3, out_channel=base_channels)
        # 下采样路径
        self.enc1 = CustomResidualModule(base_channels)
        self.enc2 = CustomResidualModule(base_channels*2)
        self.enc3 = CustomResidualModule(base_channels*4)
        self.enc4 = CustomResidualModule(base_channels*8)
        self.down1 = Downsample(base_channels, base_channels*2)
        self.down2 = Downsample(base_channels*2, base_channels*4)
        self.down3 = Downsample(base_channels*4, base_channels*8)
        self.down4 = Downsample(base_channels*8, base_channels*16)


        # bottleneck
        self.bottleneck = CustomResidualModule(base_channels * 16)  # 1024

        # 上采样路径
        self.up4 = Upsample(base_channels * 16, base_channels * 8)
        self.up3 = Upsample(base_channels * 16, base_channels * 4)
        self.up2 = Upsample(base_channels * 8, base_channels * 2)
        self.up1 = Upsample(base_channels * 4, base_channels)
        self.dec4 = CustomResidualModule(base_channels * 16)
        self.dec3 = CustomResidualModule(base_channels * 8)
        self.dec2 = CustomResidualModule(base_channels * 4)
        self.dec1 = CustomResidualModule(base_channels * 2)
        self.output_proj = OutputProj(in_channel=base_channels * 2, out_channel=3)
        self.output_projy = OutputProj(in_channel=base_channels * 2, out_channel=1)

    def forward(self, x):
        # print(x.shape)
        # 编码器部分
        y = self.rgb_to_y_channel(x)
        fy = self.InputProjy(y)
        fi = self.InputProj(x)
        f = fy + fi
        e1 = self.enc1(f)
        down1 = self.down1(e1)
        e2 = self.enc2(down1)
        down2 = self.down2(e2)
        e3 = self.enc3(down2)
        down3 = self.down3(e3)
        e4 = self.enc4(down3)
        down4 = self.down4(e4)
        

        # bottleneck  
        b = self.bottleneck(down4)

        # 解码器部分
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.output_proj(d1) + fi
        out_y = self.output_projy(d1)
        out_y = out_y + y
        out = out + x
        # print(out.shape)
        return out, out_y

if __name__ == '__main__':
    block = UNet(in_channels=3, out_channels=1).to('cuda')
    input = torch.rand(1, 3, 512, 512).to('cuda')
    output = block(input)
    print(output.shape)



