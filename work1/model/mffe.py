import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class MFFE(nn.Module):  # Multi-Frequency Fusion Enhancement
    def __init__(self, channels):
        super(MFFE, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.fconv1 = nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1)
        self.fconv2 = nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1)
        self.fconv3 = nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        # identity = x
        # FFT变换
        fft = torch.fft.fft2(x, norm='ortho')
        fft = torch.fft.fftshift(fft)
        fftreal = torch.cat((fft.real, fft.imag), dim=1)
        # 全局平均池化(GAP)
        gap1 = F.adaptive_avg_pool2d(fftreal,(x.size(2)//8,x.size(3)//8))
        gap2 = F.adaptive_avg_pool2d(fftreal,(x.size(2)//4,x.size(3)//4))
        gap3 = F.adaptive_avg_pool2d(fftreal,(x.size(2)//2,x.size(3)//2))
        fout1 = self.relu(self.fconv1(gap1))/2
        fout2 = self.relu(self.fconv2(F.interpolate(fout1, scale_factor=2, mode='bilinear')+gap2))/2
        fout3 = self.relu(self.fconv3(F.interpolate(fout2, scale_factor=2, mode='bilinear')+gap3))/2
        fout = F.interpolate(fout1, scale_factor=8, mode='bilinear')+F.interpolate(fout2,scale_factor=4,mode='bilinear')+F.interpolate(fout3,scale_factor=2,mode='bilinear')
        fout = fout/3
        foutreal, foutimag = torch.chunk(fout, 2, dim=1)
        fout = torch.fft.ifft2(torch.fft.ifftshift(torch.complex(foutreal, foutimag)),norm='ortho').real
        # 卷积路径
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = x + identity

        out = torch.cat((x, fout), 1)
        out = self.conv11(out)
        if torch.isnan(out).any():
          print('nan detected!')
        return out
        # return x


import cv2 as cv
from torchvision.transforms import transforms
if __name__ == '__main__':
    input = cv.imread('D:/Program_Files/stable_diffusion_launcher/outputs/txt2img-images/2023-10-29/00000-1800816867.png')
    input = cv.cvtColor(input,cv.COLOR_BGR2RGB)
    x = transforms.ToTensor()(input).unsqueeze(0)
    # 示例使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)  # batch_size=4, channels=64, height=32, width=32
    print(x.shape)
    mfee = MFFE(3)
    mfee=mfee.cuda()
    for i in range(0,10):
        out = mfee(x)
    print(out.shape)
    output = out.cpu().detach().numpy()


