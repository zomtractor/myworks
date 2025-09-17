import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import BasicConv


class EBlock(nn.Module):
    def __init__(self, channels):
        super(EBlock, self).__init__()

    def forward(self, x):
        return x


class DBlockPred(nn.Module):
    def __init__(self, channel):
        super(DBlockPred, self).__init__()

    def forward(self, x):
        return x


class DBlockFlare(nn.Module):
    def __init__(self, channel):
        super(DBlockFlare, self).__init__()

    def forward(self, x):
        return x


class ConvS(nn.Module):
    def __init__(self, out_channels):
        super(ConvS, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_channels // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channels // 4, out_channels // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channels // 2, out_channels, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_channels, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


# gaussian transform block
def GTB(x, layer=4):
    res_gaussian = []
    res_laplacian = []
    kernel = torch.tensor([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]], dtype=torch.float32, device=x.device) / 256.0
    b, c, h, w = x.shape
    kernel = kernel.view(1, 1, 5, 5).repeat(c, 1, 1, 1)
    current = x
    res_gaussian.append(current)
    for i in range(layer):
        blurred = F.conv2d(current, kernel, padding=2, groups=c)
        blurred = blurred[:, :, ::2, ::2]
        res_gaussian.append(blurred)
        upsampled = F.interpolate(blurred, size=current.shape[2:], mode='bilinear', align_corners=False)
        laplacian = current - upsampled
        res_laplacian.append(laplacian)
        current = blurred
    return res_gaussian, res_laplacian

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.down(x)
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)



class MyBlock(nn.Module):
    def __init__(self, base_channels=16,num_block=3, num_bottleneck=2):
        super(MyBlock, self).__init__()
        self.num_block = num_block
        self.num_bottleneck = num_bottleneck

        self.proj = [BasicConv(3, base_channels, kernel_size=3, padding=1)]
        for i in range(num_block - 1):
            self.proj.append(ConvS(base_channels * 2**i))
        self.proj_laplacian = [BasicConv(3, base_channels * 2**i, kernel_size=3, padding=1) for i in range(num_block)]
        self.ebs = [EBlock(base_channels * 2**i) for i in range(num_block)]
        self.bottleneck = [EBlock(base_channels * 2**(num_block-1)) for _ in range(num_bottleneck)]
        self.dbs_pred = [DBlockPred(base_channels * 2**(num_block-1-i)) for i in range(num_block)]
        self.dbs_flare = [DBlockFlare(base_channels * 2**(num_block-1-i)) for i in range(num_block)]
        self.ups_pred = [UpSample(base_channels * 2**i, base_channels * 2**i) for i in range(num_block)]
        self.ups_flare = [UpSample(base_channels * 2**i, base_channels * 2**i) for i in range(num_block)]
        self.downs = [DownSample(base_channels * 2**i, base_channels * 2**i) for i in range(num_block)]
        self.projout_pred = [BasicConv(base_channels, 3, kernel_size=3, padding=1,norm=True) for _ in range(num_block)]
        self.projout_flare = [BasicConv(base_channels, 3, kernel_size=3, padding=1,norm=True) for _ in range(num_block)]

    def forward(self, x):
        skip=[]
        gauss, laplacian = GTB(x, layer=3)
        res = self.ebs[0](self.proj[0](gauss[0]))
        skip.append(res)
        for i in range(1, self.num_block):
            res = torch.cat((res,self.proj[i](gauss[i])),dim=1)
            res = self.downs[i-1](res)
            res = self.ebs[i](self.proj[i](res))
            skip.append(res)
        res = self.downs[-1](res)
        for i in range(self.num_bottleneck):
            res = self.bottleneck[i](res)
        res_pred,res_flare = torch.chunk(res,2,dim=1)
        outs_pred = []
        outs_flare = []



def tensor_to_pil(tensor):
    tensor = tensor.cpu().detach()
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    tensor = torch.clamp(tensor, 0, 1)
    array = (tensor.numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


if __name__ == '__main__':
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision import transforms

    # 设置图像路径
    image_path1 = "E:/dataset/flare7kpp_r_local/input/c0/48.png"
    image_path2 = "E:/dataset/flare7kpp_r_local/input/c0/49.png"

    # 1. 读取图像
    original_image = Image.open(image_path1).convert('RGB')
    original_image2 = Image.open(image_path2).convert('RGB')

    # 2. 将图像转换为PyTorch张量
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])
    img_tensor = torch.stack((transform(original_image), transform(original_image2)))
    g, l = GTB(img_tensor, layer=4)
    g.pop()

    fig, axes = plt.subplots(4, len(g), figsize=(16, 12))

    # 显示高斯金字塔
    for i, img_tensor in enumerate(g):
        img = tensor_to_pil(img_tensor[0])
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Gaussian Level {i}')
        axes[0, i].axis('off')
        img = tensor_to_pil(img_tensor[1])
        axes[2, i].imshow(img)
        axes[2, i].set_title(f'Gaussian Level {i}')
        axes[2, i].axis('off')

    # 显示拉普拉斯金字塔（需要归一化以便可视化）
    for i, img_tensor in enumerate(l):
        img = img_tensor[0]
        # 对拉普拉斯图像进行归一化以便可视化
        laplacian_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = tensor_to_pil(laplacian_norm)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'Laplacian Level {i}')
        axes[1, i].axis('off')

        img = img_tensor[1]
        # 对拉普拉斯图像进行归一化以便可视化
        laplacian_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = tensor_to_pil(laplacian_norm)
        axes[3, i].imshow(img)
        axes[3, i].set_title(f'Laplacian Level {i}')
        axes[3, i].axis('off')
    plt.tight_layout()
    plt.show()
