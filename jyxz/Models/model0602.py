import sys
sys.path.append("..")
import torch
import torch.nn as nn
from thop import clever_format
from utils import network_parameters
from einops import rearrange
from torch import einsum
import numpy as np

class GlobalContext(nn.Module):
    def __init__(self, in_channel, act_layer=nn.GELU, params=params):
        super().__init__()
        # bottleneck information
        # "Compete to compute." NeurIPS 2013
        self.compete = params["global_context"]["compete"]
        if self.compete:
            self.fc1 = nn.Linear(in_channel, 2 * in_channel // params["global_context"]["gc_reduction"])
            self.fc2 = nn.Linear(in_channel // params["global_context"]["gc_reduction"], in_channel)
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_channel, in_channel // params["global_context"]["gc_reduction"]),
                act_layer(),
                nn.Linear(in_channel // params["global_context"]["gc_reduction"], in_channel)
            )
        self.weight_gc = params["global_context"]["weighted_gc"]
        if self.weight_gc:
            self.head = params["global_context"]["head"]
            self.scale = (in_channel // self.head) ** -0.5
            self.rescale_weight = nn.Parameter(torch.ones(self.head))
            self.rescale_bias = nn.Parameter(torch.zeros(self.head))
            self.epsilon = 1e-5

    def _get_gc(self, gap): # gap [b,c]
        if self.compete:
            b,c = gap.size()
            gc = self.fc1(gap).reshape([b,2,-1])
            gc, _ = gc.max(dim=1)
            gc = self.fc2(gc)
            return gc
        else:
            return self.fc(gap)


    def forward(self,x):
        if self.weight_gc:
            b,c,w,h = x.size()
            x = rearrange(x,"b c x y -> b c (x y)")
            gap = x.mean(dim=-1, keepdim=True)
            q, g = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h = self.head), [x,gap])  #[b,head, hdim, n]
            sim = einsum('bhdi,bhjd->bhij', q, g.transpose(-1, -2)).squeeze(dim=-1) * self.scale  #[b,head, w*h]
            std, mean = torch.std_mean(sim, dim=[1,2], keepdim=True)
            sim = (sim-mean)/(std+self.epsilon)
            sim = sim * self.rescale_weight.unsqueeze(dim=0).unsqueeze(dim=-1) + self.rescale_bias.unsqueeze(dim=0).unsqueeze(dim=-1)
            sim = sim.reshape(b,self.head,1, w, h) # [b, head, 1, w, h]
            gc = self._get_gc(gap.squeeze(dim=-1)).reshape(b,self.head,-1).unsqueeze(dim=-1).unsqueeze(dim=-1)  # [b, head, hdim, 1, 1]
            gc = rearrange(sim*gc, "b h d x y -> b (h d) x y")  # [b, head, hdim, w, h] - > [b,c,w,h]
        else:
            gc = self._get_gc(x.mean(dim=-1).mean(dim=-1)).unsqueeze(dim=-1).unsqueeze(dim=-1)

        return gc # [b,c,w,h] for weighted or [b,c,1,1]
class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Downsample(nn.Module):
    def __init__(self, kernel_size=4, padding=1, stride=2, in_chans=3, out_chans=32):
        super().__init__()
        self.down =nn.Conv2d(in_chans, in_chans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv=nn.Conv2d(in_chans,out_chans,kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.down(x)
        x=self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self,in_chans=3, out_chans=32):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chans, in_chans, kernel_size=2, stride=2)
        self.conv=nn.Conv2d(in_chans,out_chans,kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.up(x)
        x=self.conv(x)
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


class FeedForward(nn.Module):
    def __init__(self, dim,act_layer=nn.GELU):
        super(FeedForward, self).__init__()
        hidden_features = dim
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2)
        self.act=act_layer()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.project_out(x)
        return x


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

# -------------------------
# 主模块：按你的图拼装
# -------------------------
class WaveletAttentionBlock(nn.Module):
# class MBlock(nn.Module):
    """
    输入:  (B, C, H, W)
    输出:  (B, C, H, W)  （保持通道与分辨率）
    流程概述：
      x --(DWT)--> [LL,LH,HL,HH]
        LL -> DilatedBlock
        LH -> HorizontalBlock
        HL -> VerticalBlock
        HH -> DilatedBlock   # 给 HH 也走一次空洞块，避免闲置
      拼回 -> (IDWT) -> y
      顶部分支：concat([x, y]) -> BN(LN位) -> SelfAttention -> BN -> MLP -> f_top
      右侧注意力：y -> SpatialAttention -> ChannelAttention -> f_att
      最终：concat([f_top, f_att]) -> 1×1 conv -> 与 x 残差相加
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=4):
        super().__init__()
        C = dim

        # 小波
        self.dwt  = WavePool(C)
        self.idwt = WaveUnpool(C)

        # 四个子带的并行处理（这里复用三种卷积块）
        self.proc_LL = DilatedBlock(C)
        self.proc_LH = HorizontalBlock(C)
        self.proc_HL = VerticalBlock(C)
        self.proc_HH = DilatedBlock(C)

        # 变换前后通道对齐（子带拼接后是 4C）
        self.merge_subbands = nn.Conv2d(4*C, 4*C, 1, bias=False)

        # 顶部分支：concat([x, y]) 后通道为 2C
        self.top_norm1 = nn.BatchNorm2d(C)   # 代替图中的 LN
        self.sa        = SelfAttention2D(C, num_heads=num_heads)
        self.top_norm2 = nn.BatchNorm2d(C)
        self.top_mlp   = MLP(C, expansion=mlp_ratio)

        # 右侧注意力
        self.spatial_attn = SpatialAttentionBlock(C)
        self.channel_attn = ChannelAttentionBlock()

        # 融合 + 输出
        self.fuse   = nn.Conv2d(2*C, C, kernel_size=1, bias=False)  # concat(f_top, f_att)
        self.out_bn = nn.BatchNorm2d(C)

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        # ---- DWT -> 子带处理 ----
        print("x.shape", x.shape)
        LL, LH, HL, HH = self.dwt(x)                          # (B, 4C, H/2, W/2)

        print("LH.shape", LH.shape)
        LL = self.proc_LL(LL)
        LH = self.proc_LH(LH)
        HL = self.proc_HL(HL)
        HH = self.proc_HH(HH)
        # print(LH.shape)
        y = self.idwt(LL, LH, HL, HH)                   # (B, C, H, W)
        # print("y.shape", y.shape)
        # ---- 顶部分支 ----
        top1 = x            # C -> 2C
        # top = torch.cat([x, y], dim=1)            # C -> 2C
        top = self.top_norm1(x)
        # print("before sa")
        top2 = self.sa(top) + top1
        # print("after sa")
        top = self.top_norm2(x)
        top = self.top_mlp(top) + top2                   # (B, 2C, H, W)
        # print(top.shape)
        # ---- 右侧注意力 ----
        att = self.channel_attn(y)              # (B, C, H, W)
        att = self.spatial_attn(att) 
        # print(att.shape)

        # ---- 融合 + 输出（再与输入残差相加）----
        z = torch.cat([top, att], dim=1)          # (B, 3C, H, W)
        z = self.out_bn(self.fuse(z))             # -> (B, C, H, W)
        return z + res
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


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., k1=1, k2=19, drop=0.,drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim,k1,k2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


# class OriginMBlock(nn.Module):
class MBlock(nn.Module):
    def __init__(self, dim,norm_layer=GroupNorm, length = 19):
        super(MBlock, self).__init__()
        self.dwt = WavePool(dim)
        self.idwt = WaveUnpool(dim)
        self.strip = Block(dim)
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
        LL, LH, HL, HH = self.dwt(x)
        # print("====================")
        # print(x.shape)
        # print(LL.shape)
        # print(LH.shape)
        # print(HL.shape)
        # print(HH.shape)
        # print("====================")

        # print("=========全局信息恢复=========")
        # x0=x
        # gc=self.norm1(x)
        # gc=self.gcb(gc)
        # gc=gc+x0       
        # gc2=gc
        # gc=self.norm2(gc)
        # gc=self.ffn(gc)
        # gc=gc+gc2
        gc = self.strip(x)
        # print("=========全局信息恢复结束=========")
        # print("=========局部信息恢复=========")
        x1 = x
        HL=self.vertical_block(HL)
        LH=self.horizontal_block(LH)
        HH=self.dilate_block(HH)
        # print(HL.shape)
        # print(LH.shape)
        # print(HH.shape)
        LL = self.basic_block_ll(LL)
        x_total=self.idwt(LL,LH,HL,HH) + x1
        x_total=self.cha_att(x_total)
        x_total=self.spt_att(x_total)
        x_out=x_total+x
        # print("=========局部信息恢复结束=========")
        # print("=========信息融合=========")
        x_result=torch.cat([x_out,gc],dim=1)
        x_result=self.conv1(gc)
        # print("=========信息融合结束=========")
        return x_result

class ConvFusionBlock2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv=nn.Conv2d(2*dim,dim,3,1,1)
    def forward(self, x1, x2):
        x=torch.cat([x1,x2],dim=1)
        x=self.conv(x)
        return x

class MNet(nn.Module):
    def __init__(self, dim=32,num_blocks=[1,1,1,8]):
        super().__init__()
        self.dwt  = WavePool(3)
        self.idwt = WaveUnpool(3)
        self.input_pro= nn.Conv2d(3,dim,3,1,1)
        self.output_pro=nn.Conv2d(dim,3,3,1,1)
        self.fcblock1=nn.Sequential(*[MBlock(dim=dim) for i in range(num_blocks[0])])
        self.fcblock2 = nn.Sequential(*[MBlock(dim=2*dim) for i in range(num_blocks[1])])
        self.fcblock3 = nn.Sequential(*[MBlock(dim=4*dim) for i in range(num_blocks[2])])
        self.fcblock4 = nn.Sequential(*[MBlock(dim=8*dim) for i in range(num_blocks[3])])
        self.downsample1 = Downsample(in_chans=dim, out_chans=2 * dim)
        self.downsample2 = Downsample(in_chans=2*dim, out_chans=4 * dim)
        self.downsample3 = Downsample(in_chans=4*dim, out_chans=8 * dim)
        self.upsample1 = Upsample(in_chans=8 * dim, out_chans=4 * dim)
        self.upsample2 = Upsample(in_chans=4 * dim, out_chans=2 * dim)
        self.upsample3 = Upsample(in_chans=2 * dim, out_chans= dim)
        self.conv1= ConvFusionBlock2(4 * dim)
        self.conv2 = ConvFusionBlock2(2 * dim)
        self.conv3 = ConvFusionBlock2(dim)
        self.fcblock5=nn.Sequential(*[MBlock(dim=4*dim) for i in range(num_blocks[2])])
        self.fcblock6= nn.Sequential(*[MBlock(dim=2*dim) for i in range(num_blocks[1])])
        self.fcblock7= nn.Sequential(*[MBlock(dim=dim) for i in range(num_blocks[0])])
    def forward(self,x):

        temp=x
        LL, LH, HL, HH = self.dwt(x)
        x=self.input_pro(LL)
        fcb1=self.fcblock1(x) #B H*W C
        down1=self.downsample1(fcb1)
        fcb2=self.fcblock2(down1)
        down2=self.downsample2(fcb2)
        fcb3 = self.fcblock3(down2)
        down3 = self.downsample3(fcb3)
        fcb4 = self.fcblock4(down3)
        up1 = self.upsample1(fcb4)
        up1 = self.conv1(up1, fcb3)
        fcb5 = self.fcblock5(up1)
        up2= self.upsample2(fcb5)
        up2= self.conv2(up2, fcb2)
        fcb6 = self.fcblock6(up2)
        up3 = self.upsample3(fcb6)
        up3 = self.conv3(up3, fcb1)
        fcb7 = self.fcblock7(up3)
        y=self.output_pro(fcb7)
        y=self.idwt(y, LH, HL, HH)
        result=y+temp
        return result
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# 基础模块
# -------------------------
class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, groups=1, act=True):
        super().__init__()
        if p is None:
            p = (k // 2) * d
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, dilation=d, groups=groups, bias=False)
        # self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.GELU() if act else nn.Identity()
    def forward(self, x):
        # return self.act(self.bn(self.conv(x)))
        return self.act(self.conv(x))


class MLP(nn.Module):
    # 1×1 -> GELU -> 1×1
    def __init__(self, c, expansion=4):
        super().__init__()
        hidden = c * expansion
        self.fc1 = nn.Conv2d(c, hidden, 1, bias=False)
        self.act1 = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, c, 1, bias=False)
        self.act2 = nn.GELU()
        self.bn  = nn.BatchNorm2d(c)
    def forward(self, x):
        x = self.act2(self.fc2(self.act1(self.fc1(x))))
        return self.bn(x)
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

# -------------------------
# 三种卷积块（对应：空洞/纵向/横向）
# -------------------------
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
            # ConvBNAct(c//4, c//4, k=(3,3), p=(3//2,3//2)),
            ConvBNAct(c//4, c//4, k=(k,1), p=(k//2,0)),
            # ConvBNAct(c//4, c, k=(k,1), p=(k//2,0)),
            nn.Conv2d(c//4, c, kernel_size=1),
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
            # ConvBNAct(c, c//4, k=(3,3), p=(3//2,3//2)),
            ConvBNAct(c//4, c//4, k=(1,k), p=(0,k//2)),
            # ConvBNAct(c//4, c, k=(1,k), p=(0,k//2)),
            nn.Conv2d(c//4, c, kernel_size=1),
            ChannelAttentionBlock(),
            SpatialAttentionBlock(c)
        )
    def forward(self, x):
        return x + self.block(x)

# -------------------------
# 简易 2D 自注意力（MHSA）
# -------------------------
class SelfAttention2D(nn.Module):
    """把 H*W 作为 token，对 C 做注意力。为避免显存爆炸，可在较小分辨率使用。"""
    def __init__(self, c, num_heads=4):
        super().__init__()
        assert c % num_heads == 0
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(c, c*3, 1, bias=False)
        self.proj = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)  # (B, C, H, W)
        q = q.reshape(B, self.num_heads, C//self.num_heads, H*W)      # (B, h, HW, d)
        k = k.reshape(B, self.num_heads, C//self.num_heads, H*W).transpose(-2, -1)                        # (B, h, d, HW)
        v = v.reshape(B, self.num_heads, C//self.num_heads, H*W)      # (B, h, HW, d)
        
        attn = torch.softmax((q @ k) / (C ** 0.5), dim=-1)                              # (B, h, HW, HW)
        out  = attn @ v                                                                 # (B, h, HW, d)
        out  = out.transpose(-2, -1).reshape(B, C, H, W)
        return self.proj(out)


class StripConvBlock(nn.Module):
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv_spatial1 = nn.Conv2d(dim,dim,kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)     
        self.conv_spatial2 = nn.Conv2d(dim,dim,kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)
        self.conv1 = nn.Conv2d(dim * 2, dim, 1)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):   
        res = x
        attn = self.relu0(self.conv0(x)) + res
        attn1 = self.conv_spatial1(attn)
        attn2 = self.conv_spatial2(attn)
        attn3 = torch.cat([attn1, attn2], dim=1)
        attn3 = self.relu1(self.conv1(attn3))

        return x * attn3 + res


# -------------------------
# 简单自测 dwt
# -------------------------
# if __name__ == "__main__":
#     wt = WavePool(64)
#     idwt = WaveUnpool(64)
#     x = torch.randn(2, 64, 128, 128)
#     LL, LH, HL, HH = wt(x)                          # (B, 4C, H/2, W/2)
#     print(LL.shape, LH.shape, HL.shape, HH.shape)
#     y = idwt(LL, LH, HL, HH) 
#     print(x.shape, "->", y.shape)  # 应为 torch.Size([2, 64, 128, 128]) -> 同尺寸

if __name__ == '__main__':
    input = torch.randn(2, 3,256,256).cuda(1)
    model = MNet().cuda(1)
    out = model(input)
    # print(model)
    p_number = network_parameters(model)
    p_number = clever_format([p_number], "%.3f")
    print(">>>> model Param.: ", p_number)
    print(out.shape)


