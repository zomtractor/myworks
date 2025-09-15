
from model import DrConv, SwinBlock, LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c3 = DrConv(in_channels, out_channels, kernel_length=3)
        self.c5 = DrConv(in_channels, out_channels, kernel_length=5)
        self.c7 = DrConv(in_channels, out_channels, kernel_length=7)
        self.activation = nn.GELU()
        self.ln1 = LayerNorm(in_channels)
        self.ln2 = LayerNorm(4*in_channels)
        self.attn = nn.Sequential(
            SwinBlock(dim=4 * in_channels, input_resolution=None, num_heads=4, shift_size=0),
            SwinBlock(dim=4 * in_channels, input_resolution=None, num_heads=4, shift_size=7),
            nn.Conv2d(4 * in_channels, in_channels, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        o11, o21, o31, o41 = self.c3(x)
        o12, o22, o32, o42 = self.c5(x)
        o13, o23, o33, o43 = self.c7(x)
        o1 = torch.concat([o11, o12, o13], dim=1)
        o2 = torch.concat([o21, o22, o23], dim=1)
        o3 = torch.concat([o31, o32, o33], dim=1)
        o4 = torch.concat([o41, o42, o43], dim=1)

        fs1 = o1.view(b, 3, c, h, w).max(dim=1)[0] + o1.view(b, 3, c, h, w).mean(dim=1)
        fs2 = o2.view(b, 3, c, h, w).max(dim=1)[0] + o2.view(b, 3, c, h, w).mean(dim=1)
        fs3 = o3.view(b, 3, c, h, w).max(dim=1)[0] + o3.view(b, 3, c, h, w).mean(dim=1)
        fs4 = o4.view(b, 3, c, h, w).max(dim=1)[0] + o4.view(b, 3, c, h, w).mean(dim=1)

        fs1 = self.activation(self.ln1(fs1/2))
        fs2 = self.activation(self.ln1(fs2/2))
        fs3 = self.activation(self.ln1(fs3/2))
        fs4 = self.activation(self.ln1(fs4/2))

        fs = torch.concat([fs1,fs2,fs3,fs4], dim=1)
        # out = self.ln2(fs)
        out = self.attn(fs)
        return x + out

if __name__ == '__main__':
    # 示例使用
    x = torch.randn(4, 10, 256, 256).cuda()  # batch_size=4, channels=64, height=32, width=32
    mdfusion = MDFusion(64, 64).cuda()
    out = mdfusion(x)
    print("MDFusion output shape:", out.shape)

