from model import DrConv, BasicDrConv, BasicConv, SwinBlock
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
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
        self.attn = nn.Sequential(
            SwinBlock(dim=in_channels, input_resolution=None, num_heads=4, shift_size=0),
            SwinBlock(dim=in_channels, input_resolution=None, num_heads=4, shift_size=7),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )


    def forward(self, x):
        inp = self.proj(x)
        res = self.d0(inp)+self.d1(inp)+self.d2(inp)+self.d3(inp)
        res = self.mixer(res)
        out = inp*res
        return x+self.attn(out)


if __name__ == '__main__':
    # 示例使用
    x = torch.randn(4, 64, 256, 256)  # batch_size=4, channels=64, height=32, width=32
    mdfusion = MDFusion(64, 64)
    out = mdfusion(x)
    print("MDFusion output shape:", out.shape)

