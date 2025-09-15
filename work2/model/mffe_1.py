import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn

from model import ABTB


class MFFE(nn.Module):  # Multi-Frequency Fusion Enhancement
    def __init__(self, channels):
        super(MFFE, self).__init__()
        self.sconv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.sconv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.sactivate = nn.LeakyReLU(0.2, inplace=True)
        self.fconv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.fconv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.factivate = nn.LeakyReLU(0.2, inplace=True)
        self.fattn = ABTB(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        fft = torch.fft.rfft2(x, norm='ortho')
        amp = torch.abs(fft)
        phase = torch.angle(fft)
        amp = self.fconv1(amp)
        amp = self.factivate(amp)
        amp = self.fconv2(amp)
        phase = self.fattn(phase)
        fout = torch.complex(amp*torch.cos(phase), amp*torch.sin(phase))
        fout = torch.fft.irfft2(fout,norm='ortho')
        # 卷积路径
        x = self.sconv1(x)
        x = self.sactivate(x)
        x = self.sconv2(x)
        x = x + identity

        out = torch.cat((x, fout), 1)
        out = self.conv11(out)
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


