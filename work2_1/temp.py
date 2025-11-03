import torch

from model import MyNet2_3
from torchsummary import summary

from utils import MinIOHelper
import os

if __name__ == '__main__':
    model = MyNet2_3(base_channels=4).cuda()
    x = torch.randn(1,3,256,256).cuda()
    y=model(x)
    print(model)

