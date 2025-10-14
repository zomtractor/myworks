import torch

from model import MyNet2, UBlock
from torchsummary import summary

if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)  # Batch size of 1, 3 channels, 512x512 image
    model = UBlock(base_channels=4)
    y=model(x)
    print(y.shape)


