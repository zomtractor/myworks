from six.moves import xrange
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum
class ChannelContentMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = dim
        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.fc2 =nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
class SpatialAttentionBlock(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim//8, 1),
            act_layer(),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return x * self.spatial_att(x)
class ChannelAttentionBlock(nn.Module):
    def __init__(self, act_layer=nn.GELU):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        size_1 =3
        size_2 =5
        self.channelConv1 = nn.Conv1d(1, 1, size_1, padding=size_1//2)
        self.channelConv2 = nn.Conv1d(1, 1, kernel_size=size_2, padding=size_2//2)
        self.act = act_layer()

    def forward(self, x):
        res = x.clone()
        x = self.avg_pool(self.act(x))
        x = self.channelConv1(x.squeeze(-1).transpose(-1, -2))
        x = self.act(x)
        x = self.channelConv2(x)
        x = x.transpose(-1, -2).unsqueeze(-1)
        return res + x
class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, groups=1, act=True):
        super().__init__()
        if p is None:
            p = (k // 2) * d
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, dilation=d, groups=groups, bias=False)
        # self.bn   = nn.BatchNorm2d(c_out)
        self.bn   = nn.GroupNorm(c_out//4,c_out)
        self.act  = nn.GELU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
        # return self.act(self.conv(x))

class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
class DWTBlock(nn.Module):
    def __init__(self, dim,norm_layer=GroupNorm, length = 19):
        super(DWTBlock, self).__init__()
        self.dwt = WavePool(dim)
        self.idwt = WaveUnpool(dim)
        # self.norm1 = norm_layer(dim)
        # self.norm2 = norm_layer(dim)
        # self.gcb=GlobalContext(dim)
        # self.ffn=FeedForward(dim)
        # self.block1 = BasicBlock(dim)
        # self.block2 = nn.Sequential(
        #     BasicBlock(dim),
        #     BasicBlock(dim)
        # )
        # self.block3=nn.Sequential(
        #     BasicBlock(dim),
        #     BasicBlock(dim),
        #     BasicBlock(dim)
        # )
        self.vertical_block = nn.Sequential(
            VerticalBlock(dim, length),
            VerticalBlock(dim, length),
            VerticalBlock(dim, length)
        )
        self.horizontal_block = nn.Sequential(
            HorizontalBlock(dim, length),
            HorizontalBlock(dim, length),
            HorizontalBlock(dim, length)
        )
        self.dilate_block = nn.Sequential(
            DilatedBlock(dim),
            DilatedBlock(dim),
            DilatedBlock(dim)
        )
        self.basic_block_ll = nn.Sequential(
            BasicBlock(dim),
            BasicBlock(dim),
            BasicBlock(dim)
        )
        self.conv_fuse=nn.Conv2d(4*dim,dim,3,1,1)
        self.spt_att=SpatialAttentionBlock(dim)
        self.cha_att=ChannelAttentionBlock()
        self.conv1=nn.Conv2d(dim,dim,3,1,1)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        LL, LH, HL, HH = self.dwt(x)
        # print("====================")
        # print(x.shape)
        # print(LL.shape)
        # print(LH.shape)
        # print(HL.shape)
        # print(HH.shape)
        # print("====================")
        # print("=========局部信息恢复=========")
        x1 = x
        HL=self.vertical_block(HL)
        LH=self.horizontal_block(LH)
        HH=self.basic_block_ll(HH)
        # print(HL.shape)
        # print(LH.shape)
        # print(HH.shape)
        LL = self.dilate_block(LL)
        x_total=self.idwt(LL,LH,HL,HH) + x1
        x_total=self.cha_att(x_total)
        x_total=self.spt_att(x_total)
        x_out=x_total+x
        # print("=========局部信息恢复结束=========")
        # print("=========信息融合=========")
        x_result=self.conv1(x_out)
        # print("=========信息融合结束=========")
        x_result = x_result.permute(0, 2, 3, 1)
        return x_result
class BasicBlock(nn.Module):
    def __init__(self, dim,norm_layer=GroupNorm):
        super(BasicBlock, self).__init__()
        self.ccmixer1 = ChannelContentMixer(dim)
        self.sab1 = SpatialAttentionBlock(dim)
        self.cab1 = ChannelAttentionBlock()
    def forward(self, x):
        x1=x
        x = self.ccmixer1(x)
        x = self.cab1(x)
        x = self.sab1(x)
        x = x + x1
        return x
class DilatedBlock(nn.Module):
    def __init__(self, c, d=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c//4, kernel_size=1),
            ConvBNAct(c//4, c//4, k=5, d=d),
            ConvBNAct(c//4, c//4, k=3, d=d),
            nn.Conv2d(c//4, c, kernel_size=1),
            ChannelAttentionBlock(),
            SpatialAttentionBlock(c)
        )
    def forward(self, x):
        return x + self.block(x)

class VerticalBlock(nn.Module):
    def __init__(self, c, k=7):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c//4, kernel_size=1),
            nn.GELU(),
            # ConvBNAct(c//4, c//4, k=(3,3), p=(3//2,3//2)),
            ConvBNAct(c//4, c//4, k=(k,1), p=(k//2,0)),
            # ConvBNAct(c//4, c, k=(k,1), p=(k//2,0)),
            nn.Conv2d(c//4, c, kernel_size=1),
            nn.GELU(),
            ChannelAttentionBlock(),
            SpatialAttentionBlock(c)
        )
    def forward(self, x):
        return x + self.block(x)

class HorizontalBlock(nn.Module):
    def __init__(self, c, k=7):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c//4, kernel_size=1),
            nn.GELU(),
            # ConvBNAct(c, c//4, k=(3,3), p=(3//2,3//2)),
            ConvBNAct(c//4, c//4, k=(1,k), p=(0,k//2)),
            # ConvBNAct(c//4, c, k=(1,k), p=(0,k//2)),
            nn.Conv2d(c//4, c, kernel_size=1),
            nn.GELU(),
            ChannelAttentionBlock(),
            SpatialAttentionBlock(c)
        )
    def forward(self, x):
        return x + self.block(x)

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False



    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)


    return LL, LH, HL, HH
class WavePool(nn.Module):#小波变换
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)
class WaveUnpool(nn.Module):#小波逆变换
    def __init__(self, in_channels, option_unpool='sum'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError

class SpatialAttention(nn.Module):
    def __init__(self, hiddem_dim, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.conv2 = nn.Conv2d(hiddem_dim, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        sc_x = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        m = self.conv2(mask)
        attn = x + m
        # x = sc_x * self.sigmoid(attn)
        x = sc_x * self.sigmoid(attn) + x #0201 10 改版 
        return x, m

## Multi-DConv Head Transposed Self-Attention (MDTA)
class MDTAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(MDTAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        b, c, h, w = x.shape
        # print(x.shape)
        qkv = self.qkv(x)
        qkv = self.qkv_dwconv(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

## Multi-DConv Head Transposed Self-Attention (MDTA)
class MDTAttention_mask(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(MDTAttention_mask, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, l, c = x.shape
        h = int(math.sqrt(l))
        w = int(math.sqrt(l))
        x = x.transpose(-1, -2).contiguous().view(b, c, h, w)
        b, c, h, w = x.shape
        # print(x.shape)
        qkv = self.qkv(x)
        qkv = self.qkv_dwconv(qkv)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


class LeFF_M(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = SpatialAttention(hidden_dim)
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = nn.Identity()

    def forward(self, x, mm):
        # bs x hw x c
        bs, hw, c = x.size()
        mbs, mhw, mc = mm.size()
        hh = int(math.sqrt(hw))
        mhh = int(math.sqrt(mhw))

        x = self.linear1(x)
        m = self.linear1(mm)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        m = rearrange(m, ' b (h w) (c) -> b c h w ', h=mhh, w=mhh)
        # bs,hidden_dim,32x32
        x, m = self.dwconv(x, m)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        # print(x.shape)
        x = self.linear2(x)
        x = self.eca(x)

        return x, m

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        # eca 
        if hasattr(self.eca, 'flops'):
            flops += self.eca.flops()
        return flops
class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = nn.Identity()

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        # eca 
        if hasattr(self.eca, 'flops'):
            flops += self.eca.flops()
        return flops


########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H / 2 * W / 2 * self.in_channel * self.out_channel * 4 * 4
        print("Downsample:{%.2f}" % (flops / 1e9))
        return flops


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
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops
class ChannelUpsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ChannelUpsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Input_proj:{%.2f}" % (flops / 1e9))
        return flops


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Output_proj:{%.2f}" % (flops / 1e9))
        return flops

class OutputMaskProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x.sigmoid()
        # return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Output_proj:{%.2f}" % (flops / 1e9))
        return flops

#########################################
########### LeWinTransformer #############
class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8,
                 mlp_ratio=4., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = MDTAttention(dim=dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer)
        self.alpha = nn.Parameter(torch.ones(1))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))


        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        x_windows = window_partition(x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C

        attn_windows = self.attn(x_windows)  # nW*B, win_size*win_size, C

        x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H * W, self.win_size * self.win_size)

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops

class DWTLeWinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8,
                 mlp_ratio=4., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dwt = DWTBlock(dim)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = MDTAttention(dim=dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer)
        self.alpha = nn.Parameter(torch.ones(1))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))


        shortcut = x
        dwt_x = x.view(B, H, W, C)
        dwt_x = self.dwt(dwt_x)
        dwt_x = dwt_x.view(B, H * W, C)
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        x_windows = window_partition(x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C

        attn_windows = self.attn(x_windows)  # nW*B, win_size*win_size, C

        x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.alpha * x + (2 - self.alpha) * dwt_x
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H * W, self.win_size * self.win_size)

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops
class MaskAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8,
                 mlp_ratio=4., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dwt = DWTBlock(dim)
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.win_size:
            self.win_size = min(self.input_resolution)
        assert 0 < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = MDTAttention_mask(dim=dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF_M(dim, mlp_hidden_dim, act_layer=act_layer)
        self.alpha = nn.Parameter(torch.ones(1))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x, mm):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        shortcut = x
        dwt_x = x.view(B, H, W, C)
        dwt_x = self.dwt(dwt_x)
        dwt_x = dwt_x.view(B, H * W, C)
        shortcut_m = mm
        x = self.norm1(x)
        
        # mm = self.norm1(mm)

        attn_x = self.attn(x)  # nW*B, win_size*win_size, C

        x = attn_x.view(-1, self.win_size, self.win_size, C)

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        # print(x.shape)
        # FFN
        xx, m = self.mlp(self.norm2(x), shortcut_m)
        x = x + self.drop_path(xx)
        x = self.alpha * x + (2 - self.alpha) * dwt_x
        return x, m

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H * W, self.win_size * self.win_size)

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops


#########################################
########### Basic layer of Uformer ################
class BasicUformerLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            LeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    mlp_ratio=mlp_ratio,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,
                                    )
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

class DWTBasicUformerLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            DWTLeWinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    mlp_ratio=mlp_ratio,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer,
                                    )
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

class MaskAttentionUformerLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            MaskAttentionTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, win_size=win_size,
                                    mlp_ratio=mlp_ratio,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer)
            for i in range(depth)])


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x, mm):
        # print("mm.shape", mm.shape)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, m = blk(x, mm)
        return x, m

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class Uformer(nn.Module):
    def __init__(self, img_size=256, in_chans=3, dd_in=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4.,
                 drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=dd_in, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=embed_dim, out_channel=in_chans, kernel_size=3, stride=1)
        self.output_proj_mask = OutputMaskProj(in_channel=embed_dim, out_channel=1, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim * 2,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim * 16,
                                      input_resolution=(img_size // (2 ** 4),
                                                        img_size // (2 ** 4)),
                                      depth=depths[4],
                                      num_heads=num_heads[4],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      drop_path=conv_dpr,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint)

        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint)
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim * 2,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint)
        
        # mask decoder
        self.upsample_3 = upsample(embed_dim * 4, embed_dim * 1)
        self.upsample_4 = ChannelUpsample(embed_dim * 2, embed_dim * 1)
        self.mask_decoderlayer1 = DWTBasicUformerLayer(dim=embed_dim * 1,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=1,
                                                num_heads=1,
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint)
        self.mask_decoderlayer2 = DWTBasicUformerLayer(dim=embed_dim * 1,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=1,
                                                num_heads=1,
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint)
        # self.mask_decoderlayer3 = DWTBasicUformerLayer(dim=embed_dim * 2,
        #                                         input_resolution=(img_size,
        #                                                           img_size),
        #                                         depth=1,
        #                                         num_heads=1,
        #                                         win_size=win_size,
        #                                         mlp_ratio=self.mlp_ratio,
        #                                         drop_path=dec_dpr[sum(depths[4:6]):sum(depths[4:7])],
        #                                         norm_layer=norm_layer,
        #                                         use_checkpoint=use_checkpoint)
        # self.mask_decoderlayer4 = DWTBasicUformerLayer(dim=embed_dim * 2,
        #                                         input_resolution=(img_size,
        #                                                           img_size),
        #                                         depth=1,
        #                                         num_heads=1,
        #                                         win_size=win_size,
        #                                         mlp_ratio=self.mlp_ratio,
        #                                         drop_path=dec_dpr[sum(depths[4:6]):sum(depths[4:7])],
        #                                         norm_layer=norm_layer,
        #                                         use_checkpoint=use_checkpoint)
        
        self.mask_guide1 = MaskAttentionUformerLayer(dim=embed_dim * 1,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=1,
                                                num_heads=1,
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint)
        self.mask_guide2 = MaskAttentionUformerLayer(dim=embed_dim * 1,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=1,
                                                num_heads=1,
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint)
        # self.mask_guide3 = MaskAttentionUformerLayer(dim=embed_dim * 2,
        #                                         input_resolution=(img_size,
        #                                                           img_size),
        #                                         depth=1,
        #                                         num_heads=1,
        #                                         win_size=win_size,
        #                                         mlp_ratio=self.mlp_ratio,
        #                                         drop_path=dec_dpr[sum(depths[4:6]):sum(depths[4:7])],
        #                                         norm_layer=norm_layer,
        #                                         use_checkpoint=use_checkpoint)
        # self.mask_guide4 = MaskAttentionUformerLayer(dim=embed_dim * 2,
        #                                         input_resolution=(img_size,
        #                                                           img_size),
        #                                         depth=1,
        #                                         num_heads=1,
        #                                         win_size=win_size,
        #                                         mlp_ratio=self.mlp_ratio,
        #                                         drop_path=dec_dpr[sum(depths[4:6]):sum(depths[4:7])],
        #                                         norm_layer=norm_layer,
        #                                         use_checkpoint=use_checkpoint)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, win_size={self.win_size}"

    def forward(self, x):
        # Input Projection
        y = self.input_proj(x)
        y = self.pos_drop(y)
        # Encoder
        conv0 = self.encoderlayer_0(y)
        # print(conv0.shape)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0)
        # print(conv1.shape)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1)
        # print(conv2.shape)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2)
        # print(conv3.shape)
        pool3 = self.dowsample_3(conv3)
        # Bottleneck
        # print("===============")
        conv4 = self.conv(pool3)
        # print(conv4.shape)
        # print("===============")

        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0)
        # print(deconv0.shape)
        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1)
        # print(deconv1.shape)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2)
        # print("deconv2.shape:", deconv2.shape)
        

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3)
        # print("deconv3.shape:", deconv3.shape)


        deconv4 = self.upsample_4(deconv3)
        ##
        m_feature1 = self.mask_decoderlayer1(deconv4)
        y_feature1, m1 = self.mask_guide1(deconv4, m_feature1)
        m_feature2 = self.mask_decoderlayer2(m_feature1)
        y_feature2, m2 = self.mask_guide2(y_feature1, m_feature2)
        # m_feature3 = self.mask_decoderlayer3(m_feature2)
        # y_feature3, m3 = self.mask_guide3(y_feature2, m_feature3)
        # m_feature4 = self.mask_decoderlayer4(m_feature3)
        # y_feature4, m4 = self.mask_guide4(y_feature3, m_feature4)


        # print(deconv3.shape, m_feature.shape)
        mm = self.output_proj_mask(m_feature2)
        yy = self.output_proj(y_feature2)
        # print(yy.shape, mm.shape, x.shape)
        y = yy * mm + x
        # y=yy+x
        yy = -yy
        # print(m.shape)
        # b, n, c = m_feature.shape
        # b, n, c = m.shape
        # print("model show", m_deconv3.shape)
        # e = int(math.sqrt(n))
        # mask_feature_output = m.transpose(-1, -2).view(b, c, e, e)
        # # Output Projection 
        # y = self.output_proj(deconv3)
        # mm = self.output_proj_mask(m_deconv3)
        # m = mm.sigmoid()
        # # mm = (mm - mm.min()) / (mm.max() - mm.min())
        # b, n, c = m_deconv3.shape
        # # print("model show", m_deconv3.shape)
        # e = int(math.sqrt(n))
        # m_deconv3 = m_deconv3.transpose(-1, -2).view(b, c, e, e)
        # # print(m_deconv3.shape)
        # print("adj model show", m_deconv3.shape)
        # # return x + y if self.dd_in == 3 else y
        # return (y * m + x, mm, m_deconv3) if self.dd_in == 3 else (y * m, mm, m_deconv3)
        return y, mm, m2

    def flops(self):
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso, self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops() + self.dowsample_0.flops(self.reso, self.reso)
        flops += self.encoderlayer_1.flops() + self.dowsample_1.flops(self.reso // 2, self.reso // 2)
        flops += self.encoderlayer_2.flops() + self.dowsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2)
        flops += self.encoderlayer_3.flops() + self.dowsample_3.flops(self.reso // 2 ** 3, self.reso // 2 ** 3)

        # Bottleneck
        flops += self.conv.flops()

        # Decoder
        flops += self.upsample_0.flops(self.reso // 2 ** 4, self.reso // 2 ** 4) + self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso // 2 ** 3, self.reso // 2 ** 3) + self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2) + self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso // 2, self.reso // 2) + self.decoderlayer_3.flops()

        # Output Projection
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops


if __name__ == "__main__":
    input_size = 512
    arch = Uformer
    depths = [1, 1, 4, 8, 4, 8, 4, 1, 1]
    model_restoration = Uformer(img_size=input_size, embed_dim=32, depths=depths,
                                win_size=8, token_projection='linear')
    # print(model_restoration)
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model_restoration, (3, input_size, input_size), as_strings=True,
    #                                             print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print('# model_restoration parameters: %.2f M' % (
    #             sum(param.numel() for param in model_restoration.parameters()) / 1e6))
    # print("number of GFLOPs: %.2f G" % (model_restoration.flops() / 1e9))
    input_ = torch.randn(1, 3, input_size, input_size)
    output_ = model_restoration(input_)
    print(output_[0].shape)
    # print(output_[1].shape)
    # print(output_[2].shape)

