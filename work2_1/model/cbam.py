import torch
import torch.nn as nn


class CAB(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        # 最大池化分支
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))

        out = avg_out + max_out
        out = self.sigmoid(out).unsqueeze(2).unsqueeze(3)

        return x * out.expand_as(x)


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接并卷积
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)

        return x * out


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = CAB(in_channels, reduction_ratio)
        self.spatial_attention = SAB(kernel_size)

    def forward(self, x):
        # 先应用通道注意力，再应用空间注意力
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


if __name__ == '__main__':
    # 示例使用
    x = torch.randn(4, 64, 256, 256)  # batch_size=4, channels=64, height=32, width=32

    # 单独使用通道注意力
    ca = CAB(64)
    out_ca = ca(x)
    print("Channel Attention output shape:", out_ca.shape)

    # 单独使用空间注意力
    sa = SAB()
    out_sa = sa(x)
    print("Spatial Attention output shape:", out_sa.shape)

    # 使用完整的CBAM模块
    cbam = CBAM(64)
    out_cbam = cbam(x)
    print("CBAM output shape:", out_cbam.shape)