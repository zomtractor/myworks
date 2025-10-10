import torch
import torch.nn as nn
from model import BasicDrConv, BasicConv, WindowAttention, ABTB, LayerNorm


class MDFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ln1 = LayerNorm(in_channels)
        self.ln2 = LayerNorm(in_channels)
        self.d0 = nn.Sequential(
            BasicDrConv(in_channels,in_channels//2),
            BasicConv(in_channels//2,in_channels),
            BasicDrConv(in_channels,in_channels,kernel_length=5)
        )
        self.d1 = nn.Sequential(
            BasicDrConv(in_channels,in_channels//2,direction=1),
            BasicConv(in_channels//2,in_channels),
            BasicDrConv(in_channels,in_channels,direction=1,kernel_length=5)
        )
        self.d2 = nn.Sequential(
            BasicDrConv(in_channels,in_channels//2,direction=2),
            BasicConv(in_channels//2,in_channels),
            BasicDrConv(in_channels,in_channels,direction=2,kernel_length=5)
        )
        self.d3 = nn.Sequential(
            BasicDrConv(in_channels,in_channels//2,direction=3),
            BasicConv(in_channels//2,in_channels),
            BasicDrConv(in_channels,in_channels,direction=3,kernel_length=5)
        )
        self.mixer = nn.Conv2d(in_channels,out_channels,kernel_size=1)

        #todo
        self.attn = ABTB(out_channels)
        self.ff = nn.Sequential(
            BasicConv(out_channels, out_channels // 4, 3, 1, norm=True, relu=True),
            BasicConv(out_channels // 4, out_channels, 3, 1, norm=True, relu=False)
        )


    def forward(self, x):
        inp = self.ln1(x)
        res = self.d0(inp)+self.d1(inp)+self.d2(inp)+self.d3(inp)
        res = self.mixer(res)
        res = self.attn(res)
        res = x+res
        out = self.ln2(res)
        out = self.ff(out)
        return res+out


if __name__ == '__main__':
    # 示例使用
    x = torch.randn(4, 64, 256, 256)  # batch_size=4, channels=64, height=32, width=32
    mdfusion = MDFusion(64, 64)
    out = mdfusion(x)
    print("MDFusion output shape:", out.shape)

