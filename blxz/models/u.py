import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import network_parameters
import os
import matplotlib.pyplot as plt

def visualize_feature_mean(F, folder_name, prefix):
    # 取特征张量的平均值
    F_mean = F[0]  # 在通道维度和批次维度取平均
    # 将特征张量转换为 NumPy 数组
    i=0
    folder_name=folder_name+'/'+prefix
    for f in F_mean:
        F_mean_numpy = f.cpu().numpy()
        # 创建 Matplotlib 图像对象
        plt.imshow(F_mean_numpy, cmap='jet')  # 使用 'viridis' 颜色映射
        plt.axis('off')  # 关闭坐标轴显示
        # 生成图像名
        image_name = str(i)+'.png'
        i=i+1
        # 拼接保存图像的完整路径
        os.makedirs(folder_name, exist_ok=True)  # 如果文件夹不存在，创建文件夹
        save_path = os.path.join(folder_name,image_name)
        # 保存图像到指定路径
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        # 不显示图像
        plt.close()

try:
    from utils.depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
    class DWConv2D(DepthWiseConv2dImplicitGEMM):
        def __init__(self, in_channels, kernel_size, bias=True):
            super().__init__( in_channels, kernel_size, bias)
except:
    
    class DWConv2D(nn.Conv2d):
        def __init__(self, in_channels, kernel_size, bias=True):
            super().__init__(in_channels, in_channels, kernel_size, stride=1,
                             padding= kernel_size//2,  groups=in_channels, bias=bias)


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class IOpro(nn.Module):
    def __init__(self, kernel_size=3,padding=1, stride=1, in_chans=3, out_chans=32):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_channels=out_chans, kernel_size=kernel_size, stride=stride,
                              padding=padding)
    def forward(self, x):
        x = self.proj(x)
        return x


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



class ChannelContentMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = 2*dim
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
            DWConv2D(dim, kernel_size=3),
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



class DyWeightFusionBlock(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(DyWeightFusionBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4

        return x

class FFN(nn.Module):
    def __init__(self, dim):
        super(FFN, self).__init__()

        self.dwconv3x3_1 = nn.Conv2d(dim, dim, 3, padding=3 // 2, groups=dim)
        self.pwconv1x1_1 = nn.Conv2d(dim, dim, 1)
        self.dwconv3x3_2 = nn.Conv2d(2*dim, 2*dim, 3, padding=3 // 2, groups=2*dim)
        self.pwconv1x1_2 = nn.Conv2d(2*dim, 2*dim, 1)
        self.dwconv5x5_1 = nn.Conv2d(dim, dim, 5, padding=5 // 2, groups=dim)
        self.pwconv1x1_3 = nn.Conv2d(dim, dim, 1)
        self.dwconv5x5_2 = nn.Conv2d(2*dim, 2*dim, 5, padding=5 // 2, groups=2*dim)
        self.pwconv1x1_4 = nn.Conv2d(2*dim, 2*dim, 1)
        self.act = nn.GELU()
        self.confusion = nn.Conv2d(dim * 4, dim, 1, padding=0, stride=1)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.act(self.pwconv1x1_1(self.dwconv3x3_1(input_1)))
        output_5_1 = self.act(self.pwconv1x1_3(self.dwconv5x5_1(input_1)))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.act(self.pwconv1x1_2(self.dwconv3x3_2(input_2)))
        output_5_2 = self.act(self.pwconv1x1_4(self.dwconv5x5_2(input_2)))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        return output


class GlobalContentExBlock(nn.Module):
    def __init__(self, dim,norm_layer=GroupNorm):
        super(GlobalContentExBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.ccmixer1 = ChannelContentMixer(dim)
        self.sab1 = SpatialAttentionBlock(dim)
        self.cab1 = ChannelAttentionBlock()
        self.ffn=FFN(dim)

    def forward(self, x):
        x1 = x
        x = self.norm1(x)
        x = self.ccmixer1(x)
        x = self.cab1(x)
        x = self.sab1(x)
        x = x + x1
        x2 = x
        x = self.norm2(x)
        x=self.ffn(x)
        x = x + x2
        return x




class MutilFiledFusionBlock(nn.Module):
    def __init__(self, dim):
        super(MutilFiledFusionBlock, self).__init__()
        self.dial1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 2, dilation=2),
            nn.GELU(),
            ChannelContentMixer(dim)
        )
        self.dial2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 4, dilation=4),
            nn.GELU(),
            ChannelContentMixer(dim)
        )
        self.dial3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 8, dilation=8),
            nn.GELU(),
            ChannelContentMixer(dim)
        )
        self.dwfb = DyWeightFusionBlock(3 * dim)
    def forward(self, x):
        dial1 = self.dial1(x)
        dial2 = self.dial2(x)
        dial3 = self.dial3(x)
        fusion = self.dwfb(dial1, dial2, dial3)
        return x+fusion

class BasicBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gceb1=GlobalContentExBlock(dim)
        self.mffb1=MutilFiledFusionBlock(dim)
    def forward(self, x):
        x=self.gceb1(x)
        x=self.mffb1(x)
        return x

def gauss_kernel(channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel
def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel.to(img.device), groups=img.shape[1])
    return out

class EdgeExtractBlock(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.kernel = gauss_kernel()
        self.input_pro = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        self.conv1 = nn.Conv2d(2, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # get feature map
        filtered = conv_gauss(x, self.kernel)
        high=x-filtered
        filtered = self.input_pro(filtered)
        avg_out = torch.mean(filtered, dim=1, keepdim=True)
        max_out, _ = torch.max(filtered, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return high,y

class BasicBlock2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
    def forward(self, x):
        return x + self.block(x)

class UNet(nn.Module):
    def __init__(self, dim=32):

        super().__init__()
        self.output_pro=IOpro(in_chans=dim,out_chans=3)
        self.fcblock1=BasicBlock2(dim)
        self.fcblock2 = BasicBlock2(2*dim)
        self.fcblock3 = BasicBlock2(4 * dim)
        self.fcblock4 = BasicBlock2(8 * dim)
        self.fcblock4_1 = BasicBlock(8 * dim)
        self.fcblock4_2 = BasicBlock(8 * dim)
        self.fcblock4_3 = BasicBlock(8 * dim)
        self.fcblock4_4 = BasicBlock(8 * dim)
        self.fcblock4_5 = BasicBlock(8 * dim)
        self.fcblock4_6 = BasicBlock(8 * dim)
        self.fcblock4_7 = BasicBlock(8 * dim)
        self.downsample1 = Downsample(in_chans=dim, out_chans=2 * dim)
        self.downsample2 = Downsample(in_chans=2*dim, out_chans=4 * dim)
        self.downsample3 = Downsample(in_chans=4*dim, out_chans=8 * dim)
        self.upsample1 = Upsample(in_chans=8 * dim, out_chans=4 * dim)
        self.upsample2 = Upsample(in_chans=4 * dim, out_chans=2 * dim)
        self.upsample3 = Upsample(in_chans=2 * dim, out_chans= dim)
        self.conv1=nn.Conv2d(8 * dim, 4 * dim,3, 1, 1)
        self.conv2=nn.Conv2d(4 * dim, 2 * dim, 3, 1, 1)
        self.conv3=nn.Conv2d(2 * dim, dim, 3, 1, 1)
        self.fcblock5=BasicBlock2(4 * dim)
        self.fcblock6= BasicBlock2(2*dim)
        self.fcblock7= BasicBlock2(dim)

    def forward(self, x):
        #get feature map
        #encoder
        fcb1=self.fcblock1(x)
        down1=self.downsample1(fcb1)
        fcb2=self.fcblock2(down1)
        down2=self.downsample2(fcb2)
        fcb3 = self.fcblock3(down2)
        down3 = self.downsample3(fcb3)
        fcb4 = self.fcblock4(down3)
        fcb4 = self.fcblock4_1(fcb4)
        fcb4 = self.fcblock4_2(fcb4)
        fcb4 = self.fcblock4_3(fcb4)
        fcb4 = self.fcblock4_4(fcb4)
        fcb4 = self.fcblock4_5(fcb4)
        fcb4 = self.fcblock4_6(fcb4)
        fcb4 = self.fcblock4_7(fcb4)
        #decoder
        up1=self.upsample1(fcb4)
        up1=torch.cat([up1,fcb3],1)
        fcb5=self.fcblock5(self.conv1(up1))
        up2=self.upsample2(fcb5)
        up2=torch.cat([up2,fcb2],1)
        fcb6=self.fcblock6(self.conv2(up2))
        up3 = self.upsample3(fcb6)
        up3 = torch.cat([up3, fcb1], 1)
        fcb7 = self.fcblock7(self.conv3(up3))
        y=fcb7+x
        #get output
        y=self.output_pro(y)
        return y

class UFCNet(nn.Module):
    def __init__(self, dim=32):

        super().__init__()
        self.edgeblock=EdgeExtractBlock()
        self.input_pro = IOpro(in_chans=3, out_chans=dim)
        self.mask_output_pro=nn.Sequential(
            nn.Conv2d(dim,dim,3,1,1),
            nn.GELU(),
            nn.Conv2d(dim,dim,3,1,1),
            nn.Conv2d(dim,1,3,1,1)
        )
        self.conv3x3=nn.Conv2d(3,dim//4,3,1,1)
        self.fcblock1=BasicBlock(dim)
        self.fcblock2 = BasicBlock(2*dim)
        self.fcblock3 = BasicBlock(4 * dim)
        self.fcblock4 = BasicBlock(8 * dim)
        self.fcblock4_1 = BasicBlock(8 * dim)
        self.fcblock4_2 = BasicBlock(8 * dim)
        self.fcblock4_3 = BasicBlock(8 * dim)
        self.fcblock4_4 = BasicBlock(8 * dim)
        self.fcblock4_5 = BasicBlock(8 * dim)
        self.fcblock4_6 = BasicBlock(8 * dim)
        self.fcblock4_7 = BasicBlock(8 * dim)
        self.downsample1 = Downsample(in_chans=dim, out_chans=2 * dim)
        self.downsample2 = Downsample(in_chans=2*dim, out_chans=4 * dim)
        self.downsample3 = Downsample(in_chans=4*dim, out_chans=8 * dim)
        self.upsample1 = Upsample(in_chans=8 * dim, out_chans=4 * dim)
        self.upsample2 = Upsample(in_chans=4 * dim, out_chans=2 * dim)
        self.upsample3 = Upsample(in_chans=2 * dim, out_chans= dim)
        self.conv1=nn.Conv2d(8 * dim, 4 * dim,3, 1, 1)
        self.conv2=nn.Conv2d(4 * dim, 2 * dim, 3, 1, 1)
        self.conv3=nn.Conv2d(2 * dim, dim, 3, 1, 1)
        self.fcblock5=BasicBlock(4 * dim)
        self.fcblock6= BasicBlock(2*dim)
        self.fcblock7= BasicBlock(dim)
        self.unet=UNet()
        self.convdd=nn.Conv2d(dim+dim//4,dim,3,1,1)
    def forward(self, x,name):
        #get feature map

        high,low=self.edgeblock(x)
        # x1=filtered*x+x
        # print(filtered.shape)
        # x1=self.blur_transform(x)
        # x1=
        high=self.conv3x3(high) #b,c,h,w
        x1 = self.input_pro(x)
        #encoder
        fcb1=self.fcblock1(x1)
        down1=self.downsample1(fcb1)
        fcb2=self.fcblock2(down1)
        down2=self.downsample2(fcb2)
        fcb3 = self.fcblock3(down2)
        down3 = self.downsample3(fcb3)
        fcb4 = self.fcblock4(down3)
        fcb4 = self.fcblock4_1(fcb4)
        fcb4 = self.fcblock4_2(fcb4)
        fcb4 = self.fcblock4_3(fcb4)
        fcb4 = self.fcblock4_4(fcb4)
        fcb4 = self.fcblock4_5(fcb4)
        fcb4 = self.fcblock4_6(fcb4)
        fcb4 = self.fcblock4_7(fcb4)
        #decoder
        up1=self.upsample1(fcb4)
        up1=torch.cat([up1,fcb3],1)
        fcb5=self.fcblock5(self.conv1(up1))
        up2=self.upsample2(fcb5)
        up2=torch.cat([up2,fcb2],1)
        fcb6=self.fcblock6(self.conv2(up2))
        up3 = self.upsample3(fcb6)
        up3 = torch.cat([up3, fcb1], 1)
        fcb7 = self.fcblock7(self.conv3(up3))
        out=fcb7 #b,c,h,w
        visualize_feature_mean(out,"out_image",name)
        #get output
        mask=self.mask_output_pro(fcb7)
        resUnet_input=torch.cat([out, high], 1)
        resUnet_input=self.convdd(resUnet_input)
        result=self.unet(resUnet_input)
        return mask,x+result

if __name__ == '__main__':
    input = torch.randn(2, 3,256,256)
    model = UFCNet()
    out = model(input)
    print(model)
    print(out.shape)

