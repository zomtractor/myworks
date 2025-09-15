import torch
import torch.nn as nn
from einops import rearrange


class LayerNorm(nn.Module):
    def __init__(self, dim, norm_type='bias_free'):
        super().__init__()
        self.norm_type = norm_type
        self.dim = dim

        # 使用 PyTorch 原生 LayerNorm
        if norm_type == 'bias_free':
            self.ln = nn.LayerNorm(dim, elementwise_affine=True)
            self.ln.bias.data.zero_()  # 强制偏置为 0
        else:
            self.ln = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(self, x):
        # 动态获取 h, w（兼容任意尺寸）
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.ln(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x
