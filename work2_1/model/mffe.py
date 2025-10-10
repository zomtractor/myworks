import torch
import torch.nn as nn

from model import ABTB, BasicConv


class MFFE(nn.Module):  # Multi-Frequency Fusion Enhancement
    def __init__(self, channels):
        super(MFFE, self).__init__()
        self.projin = BasicConv(channels, channels,kernel_size=1,stride=1,padding=0,relu=True,norm=False,act=nn.LeakyReLU)
        self.projout = BasicConv(channels, channels,kernel_size=1,stride=1,padding=0,relu=True,norm=False,act=nn.LeakyReLU)

        self.space = nn.Sequential(
            BasicConv(channels, channels//2,kernel_size=3,stride=1,padding=1,groups=channels//2,relu=True,norm=False,act=nn.LeakyReLU),
            BasicConv(channels//2, channels,kernel_size=3,stride=1,padding=1,groups=channels//2,relu=False,norm=False)
        )
        self.amp = nn.Sequential(
            BasicConv(channels, channels//2,kernel_size=3,stride=1,padding=1,relu=True,norm=False,act=nn.LeakyReLU),
            BasicConv(channels//2, channels,kernel_size=3,stride=1,padding=1,relu=False,norm=False)
        )
        self.phase = nn.Sequential(
            BasicConv(channels, channels//2,kernel_size=3,stride=1,padding=1,relu=True,norm=False,act=nn.LeakyReLU),
            BasicConv(channels//2, channels,kernel_size=3,stride=1,padding=1,relu=False,norm=False)
        )
        self.conv11 = BasicConv(channels*2, channels,kernel_size=1,stride=1,padding=0,relu=False,norm=False)

    def forward(self, x):
        space = x+self.space(x)
        x = self.projin(x)
        fft = torch.fft.rfft2(x, norm='ortho')
        amp = torch.abs(fft)
        phase = torch.angle(fft)
        amp = amp+self.amp(amp)
        phase = phase+self.phase(phase)
        freq = torch.complex(amp*torch.cos(phase), amp*torch.sin(phase))
        freq = torch.fft.irfft2(freq,norm='ortho')

        freq = x+self.projout(freq)

        out = torch.cat((space,freq), 1)
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


