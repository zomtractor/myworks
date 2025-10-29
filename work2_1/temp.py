import torch

from model import MyNet2_2, UBlock
from torchsummary import summary

from utils import MinIOHelper
import os

if __name__ == '__main__':
    model = MyNet2_2(base_channels=4).cuda()
    x = torch.randn(1,3,128,128).cuda()
    y=model(x)
    print(model)

