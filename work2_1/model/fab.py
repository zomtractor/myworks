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

    def forward(self, x):
        res = self.ln1(x)
        r1 = self.cab(res)
        r2 = self.abtb(res)
        res = res+r1+r2
        out = self.ln2(res)
        # out,_= self.ff(out)
        out= self.ff(out)
        return res+out




if __name__ == '__main__':
    # 示例使用
    x = torch.randn(4, 64, 256, 256)  # batch_size=4, channels=64, height=32, width=32
    fab=FAB(64)
    out = fab(x)
    print("FAB output shape:", out.shape)
