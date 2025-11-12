import torch

from model import MyNet2_4

from utils import MinIOHelper
import os

if __name__ == '__main__':
    model = MyNet2_4(base_channels=4).cuda()
    x = torch.randn(1,3,256,256).cuda()
    y=model(x)
    print(model)

