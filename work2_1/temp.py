import torch

from model import MyNet, EBlock, DBlockPred

if __name__ == '__main__':
    model = MyNet(16).cuda()
    x = torch.randn(4, 3, 256, 256).cuda()  # Batch size of 1, 3 channels, 512x512 image
    pred,flare= model(x)
    print(pred,flare)  # Should be (1, 3, 512, 512)
