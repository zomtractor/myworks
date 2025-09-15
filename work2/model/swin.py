import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model import LayerNorm


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x: (B*num_windows, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x)  # (B_, N, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B_, heads, N, head_dim)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, heads, N, N)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)  # (B_, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B_, N, C)  # (B_, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio),kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(dim * mlp_ratio), dim,kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        # x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        shortcut = x
        # x = self.norm1(x).view(B, H, W, C)
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        # partition windows
        window_size = self.window_size
        Hp = int((H + window_size - 1) // window_size) * window_size
        Wp = int((W + window_size - 1) // window_size) * window_size
        pad_h = Hp - H
        pad_w = Wp - W
        shifted_x = F.pad(shifted_x, (0,pad_w,0, pad_h))

        x_windows = rearrange(shifted_x, 'b c (h w1) (w w2) -> (b h w) (w1 w2) c',
                              w1=window_size, w2=window_size)

        attn_windows = self.attn(x_windows)  # (num_windows*B, window_size*window_size, C)

        x = rearrange(attn_windows, '(b h w) (w1 w2) c -> b c (h w1) (w w2)',
                      h=Hp // window_size, w=Wp // window_size,
                      w1=window_size, w2=window_size)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        if pad_h > 0 or pad_w > 0:
            x = x[:,:, :H, :W]

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))  # MLP + skip

        # x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        return x
