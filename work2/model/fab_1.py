import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from model import LayerNorm, CAB, ABTB
from model.drmoe import DrMoE


class FAB(nn.Module):  # Feature Attention Block
    def __init__(self, channels):
        super(FAB, self).__init__()
        self.ln1 = LayerNorm(channels)
        self.ln2 = LayerNorm(channels)
        self.cab = CAB(channels)
        self.abtb = ABTB(channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ff = DrMoE(channels,num_expert_group=2)

    def forward(self, x):
        res = self.ln1(x)
        r1 = self.cab(res)
        r2 = self.abtb(res)
        res = res+r1+r2
        out = self.ln2(res)
        # out,_= self.ff(out)
        out= self.ff(out)
        return res+out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


if __name__ == '__main__':
    # 示例使用
    x = torch.randn(4, 64, 256, 256)  # batch_size=4, channels=64, height=32, width=32
    fab=FAB(64)
    out = fab(x)
    print("FAB output shape:", out.shape)
