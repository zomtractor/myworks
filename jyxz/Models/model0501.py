import pywt
import torch
import torch.nn as nn
from thop import clever_format
from einops import rearrange
from torch import einsum
import numpy as np
from pytorch_wavelets import DWTForward
params= {
    "global_context":{
        "weighted_gc": True,
        "gc_reduction": 8,
        "compete": True,
        "head": 8,
    },
    "spatial_mixer":{
        "use_globalcontext":True,
        "useSecondTokenMix": True,
        "mix_size_1": 11,
        "mix_size_2": 11,
        "fc_factor": 8,
        "fc_min_value": 16,
        "useSpatialAtt": False
    },
    "channel_mixer":{
        "useChannelAtt": False,
        "useDWconv":True,
        "DWconv_size":3
    },
    "spatial_att":{
        "kernel_size": 3,
        "dim_reduction":8
    },
    "channel_att":{
        "size_1": 3,
        "size_2": 5,
    }
}


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

class ResBlock(nn.Module):
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
        attn2 = self.relu1(self.conv1(attn3))

        return x * attn2 + res

class HorizontalSpatialAttentionBlock(nn.Module):
    def __init__(self, dim, kernal_size, act_layer=nn.GELU):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(dim, dim, kernal_size=(kernal_size, 1), padding=1, groups=dim),
            nn.Conv2d(dim, dim//8, 1),
            act_layer(),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return x * self.spatial_att(x)

class VerticalSpatialAttentionBlock(nn.Module):
    def __init__(self, dim, kernal_size, act_layer=nn.GELU):
        super().__init__()
        self.spatial_att = nn.Sequential(
            nn.Conv2d(dim, dim, kernal_size=(1, kernal_size), padding=1, groups=dim),
            nn.Conv2d(dim, dim//8, 1),
            act_layer(),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return x * self.spatial_att(x)
        
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

class HorizontalBlock(nn.Module):
    def __init__(self, dim, kernal_size, norm_layer=GroupNorm):
        super(BasicBlock, self).__init__()
        self.ccmixer1 = ChannelContentMixer(dim)
        self.sab1 = HorizontalSpatialAttentionBlock(dim, kernal_size=kernal_size)
        self.cab1 = ChannelAttentionBlock()
    def forward(self, x):
        x1=x
        x = self.ccmixer1(x)
        x = self.cab1(x)
        x = self.sab1(x)
        x = x + x1
        return x

class VerticalBlock(nn.Module):
    def __init__(self, dim, kernal_size, norm_layer=GroupNorm):
        super(BasicBlock, self).__init__()
        self.ccmixer1 = ChannelContentMixer(dim)
        self.sab1 = VerticalSpatialAttentionBlock(dim, kernal_size=kernal_size)
        self.cab1 = ChannelAttentionBlock()
    def forward(self, x):
        x1=x
        x = self.ccmixer1(x)
        x = self.cab1(x)
        x = self.sab1(x)
        x = x + x1
        return x


class MBlock(nn.Module):
    def __init__(self, dim,norm_layer=GroupNorm):
        super(MBlock, self).__init__()
        # self.norm1 = norm_layer(dim)
        # self.norm2 = norm_layer(dim)
        # self.gcb=GlobalContext(dim)
        # self.ffn=FeedForward(dim)
        self.block1 = BasicBlock(dim)
        self.block2 = nn.Sequential(
            BasicBlock(dim),
            BasicBlock(dim)
        )
        self.block3=nn.Sequential(
            BasicBlock(dim),
            BasicBlock(dim),
            BasicBlock(dim)
        )
        self.conv_fuse=nn.Conv2d(4*dim,dim,3,1,1)
        self.spt_att=SpatialAttentionBlock(dim)
        self.cha_att=ChannelAttentionBlock()
        self.conv1=nn.Conv2d(dim,dim,3,1,1)
    def forward(self, x):
        # x0=x
        # gc=self.norm1(x)
        # gc=self.gcb(gc)
        # gc=gc+x0       
        # gc2=gc
        # gc=self.norm2(gc)
        # gc=self.ffn(gc)
        # gc=gc+gc2
        x1 = x
        x2=self.block1(x)
        x3=self.block2(x)
        x4=self.block3(x)
        x_total=torch.cat([x1,x2,x3,x4],dim=1)
        x_total=self.conv_fuse(x_total)
        x_total=self.cha_att(x_total)
        x_total=self.spt_att(x_total)
        x_out=x_total+x
        # x_result=torch.cat([x_out,gc],dim=1)
        x_result=self.conv1(x_out)
        return x_result

class VerticalMBlock(nn.Module):
    def __init__(self, dim, kernal_size, norm_layer=GroupNorm):
        super(VerticalMBlock, self).__init__()
        # self.norm1 = norm_layer(dim)
        # self.norm2 = norm_layer(dim)
        # self.gcb=GlobalContext(dim)
        # self.ffn=FeedForward(dim)
        self.block1 = BasicBlock(dim)
        self.block2 = nn.Sequential(
            BasicBlock(dim),
            VerticalBlock(dim, kernal_size)
        )
        self.block3=nn.Sequential(
            VerticalBlock(dim, kernal_size),
            BasicBlock(dim),
            VerticalBlock(dim, kernal_size),
        )
        self.conv_fuse=nn.Conv2d(4*dim,dim,3,1,1)
        self.spt_att=SpatialAttentionBlock(dim)
        self.cha_att=ChannelAttentionBlock()
        self.conv1=nn.Conv2d(dim,dim,3,1,1)
    def forward(self, x):
        # x0=x
        # gc=self.norm1(x)
        # gc=self.gcb(gc)
        # gc=gc+x0       
        # gc2=gc
        # gc=self.norm2(gc)
        # gc=self.ffn(gc)
        # gc=gc+gc2
        x1 = x
        x2=self.block1(x)
        x3=self.block2(x)
        x4=self.block3(x)
        x_total=torch.cat([x1,x2,x3,x4],dim=1)
        x_total=self.conv_fuse(x_total)
        x_total=self.cha_att(x_total)
        x_total=self.spt_att(x_total)
        x_out=x_total+x
        # x_result=torch.cat([x_out,gc],dim=1)
        x_result=self.conv1(x_out)
        return x_result

class HorizontalMBlock(nn.Module):
    def __init__(self, dim, kernal_size, norm_layer=GroupNorm):
        super(HorizontalMBlock, self).__init__()
        # self.norm1 = norm_layer(dim)
        # self.norm2 = norm_layer(dim)
        # self.gcb=GlobalContext(dim)
        # self.ffn=FeedForward(dim)
        self.block1 = BasicBlock(dim)
        self.block2 = nn.Sequential(
            BasicBlock(dim),
            HorizontalBlock(dim, kernal_size)
        )
        self.block3=nn.Sequential(
            HorizontalBlock(dim, kernal_size),
            BasicBlock(dim),
            HorizontalBlock(dim, kernal_size)
        )
        self.conv_fuse=nn.Conv2d(4*dim,dim,3,1,1)
        self.spt_att=SpatialAttentionBlock(dim)
        self.cha_att=ChannelAttentionBlock()
        self.conv1=nn.Conv2d(dim,dim,3,1,1)
    def forward(self, x):
        # x0=x
        # gc=self.norm1(x)
        # gc=self.gcb(gc)
        # gc=gc+x0       
        # gc2=gc
        # gc=self.norm2(gc)
        # gc=self.ffn(gc)
        # gc=gc+gc2
        x1 = x
        x2=self.block1(x)
        x3=self.block2(x)
        x4=self.block3(x)
        x_total=torch.cat([x1,x2,x3,x4],dim=1)
        x_total=self.conv_fuse(x_total)
        x_total=self.cha_att(x_total)
        x_total=self.spt_att(x_total)
        x_out=x_total+x
        # x_result=torch.cat([x_out,gc],dim=1)
        x_result=self.conv1(x_out)
        return x_result

class ConvFusionBlock2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv=nn.Conv2d(2*dim,dim,3,1,1)
    def forward(self, x1, x2):
        x=torch.cat([x1,x2],dim=1)
        x=self.conv(x)
        return x
def wavelet_transform_batch(img_batch):
    """
    对 B3HW 格式的 RGB 图像批次进行小波变换
    输入:
        img_batch: numpy 数组, shape = (B, 3, H, W)
    输出:
        LL, LH, HL, HH: 每个 shape = (B, 3, H/2, W/2)
    """
    assert img_batch.ndim == 4 and img_batch.shape[1] == 3, "输入必须是 (B,3,H,W)"
    B, C, H, W = img_batch.shape

    # 用于存储结果
    LL_list, LH_list, HL_list, HH_list = [], [], [], []

    for b in range(B):
        img = img_batch[b]  # shape (3, H, W)
        LL_c, LH_c, HL_c, HH_c = [], [], [], []
        for c in range(3):
            channel = img[c].astype(np.float32)
            LL, (LH, HL, HH) = pywt.dwt2(channel, 'db1')
            LL_c.append(LL)
            LH_c.append(LH)
            HL_c.append(HL)
            HH_c.append(HH)
        # 堆叠单张图像的通道结果，shape = (3,H/2,W/2)
        LL_list.append(np.stack(LL_c, axis=0))
        LH_list.append(np.stack(LH_c, axis=0))
        HL_list.append(np.stack(HL_c, axis=0))
        HH_list.append(np.stack(HH_c, axis=0))

    # 堆叠回批次维度，shape = (B,3,H/2,W/2)
    LL_out = np.stack(LL_list, axis=0)
    LH_out = np.stack(LH_list, axis=0)
    HL_out = np.stack(HL_list, axis=0)
    HH_out = np.stack(HH_list, axis=0)

    return LL_out, LH_out, HL_out, HH_out

def torch_wavelets(img_batch):
    dwt = dwt = DWTForward(J=1, wave='haar', mode='zero')
    LL, high = dwt(img_batch)
    LH = high[0][:, :, 0, :, :]  # shape (B, 3, H/2, W/2)
    HL = high[0][:, :, 1, :, :]  # shape (B, 3, H/2, W/2)
    HH = high[0][:, :, 2, :, :]  # shape (B, 3, H/2, W/2)
    return LL, LH, HL, HH
def rgb_to_y(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    将 RGB (B,3,H,W) Tensor 转换为 Y 通道 (B,1,H,W) Tensor
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    输入和输出都保持在相同设备上（GPU/CPU）
    """
    assert rgb_tensor.ndim == 4 and rgb_tensor.shape[1] == 3, "输入必须是 (B,3,H,W)"

    # 拆分通道
    R = rgb_tensor[:, 0:1, :, :]
    G = rgb_tensor[:, 1:2, :, :]
    B = rgb_tensor[:, 2:3, :, :]

    # 根据ITU-R BT.601标准转换为Y通道
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    return Y
class MNet(nn.Module):
    def __init__(self, dim=32,num_blocks=[1,1,1,1]):
        super().__init__()
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
        # 原图小波变换得到 LL HL LH HH
        LL, LH, HL, HH = torch_wavelets(x)
        # 原图y通道提取 得到y
        Y = rgb_to_y(x)
        # Y和LL作为耀斑区域先验
        # LH横向
        # HL纵向
        # HH正常
        # 
        x=self.input_pro(x)
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
        result=y+temp
        return result

if __name__ == '__main__':
    input = torch.randn(2, 3,64,64)
    model = MNet()
    out = model(input)
    wt = torch_wavelets(input)
    y = rgb_to_y(input)
    print(y.shape)
    print(len(wt))
    print(wt[0].shape)
    print(wt[1].shape)
    print(wt[2].shape)
    print(wt[3].shape)
    # print(model)
    # p_number = network_parameters(model)
    # p_number = clever_format([p_number], "%.3f")
    # print(">>>> model Param.: ", p_number)
    # print(out.shape)

