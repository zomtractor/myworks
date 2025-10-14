import torch
import torch.nn as nn
from thop import clever_format
from utils import network_parameters
from einops import rearrange
from torch import einsum
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


class MBlock(nn.Module):
    def __init__(self, dim,norm_layer=GroupNorm):
        super(MBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.gcb=GlobalContext(dim)
        self.ffn=FeedForward(dim)
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
        self.conv1=nn.Conv2d(2*dim,dim,3,1,1)
    def forward(self, x):
        x0=x
        gc=self.norm1(x)
        gc=self.gcb(gc)
        gc=gc+x0       
        gc2=gc
        gc=self.norm2(gc)
        gc=self.ffn(gc)
        gc=gc+gc2
        x1 = x
        x2=self.block1(x)
        x3=self.block2(x)
        x4=self.block3(x)
        x_total=torch.cat([x1,x2,x3,x4],dim=1)
        x_total=self.conv_fuse(x_total)
        x_total=self.cha_att(x_total)
        x_total=self.spt_att(x_total)
        x_out=x_total+x
        x_result=torch.cat([x_out,gc],dim=1)
        x_result=self.conv1(x_result)
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
    # print(model)
    p_number = network_parameters(model)
    p_number = clever_format([p_number], "%.3f")
    print(">>>> model Param.: ", p_number)
    print(out.shape)

