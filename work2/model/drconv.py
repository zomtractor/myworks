import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DrConv(nn.Module):
    def __init__(self, in_channels, out_channels=1, kernel_length=3, direction=0,
                 padding='same', stride=1, bias=True, dilation=1, groups=1):
        super().__init__()
        self.kernel_size = kernel_length
        self.padding = padding
        self.stride = stride
        self.direction = direction
        self.dilation = dilation
        self.groups = groups

        if direction == 1:
            self.w = nn.Parameter(torch.randn(out_channels, in_channels, kernel_length, 1))
        else:
            self.w = nn.Parameter(torch.randn(out_channels, in_channels, 1, kernel_length))

        if bias:
            self.b = nn.Parameter(torch.zeros(out_channels))
        else:
            self.b = None

        # register reshaper as buffer (faster & device-safe)
        if direction == 2:
            self.register_buffer("reshaper", torch.eye(kernel_length))
        elif direction == 3:
            self.register_buffer("reshaper", torch.flip(torch.eye(kernel_length), [-1]))
        else:
            self.reshaper = None

        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x):
        kernel = self.w
        if self.reshaper is not None:
            kernel = kernel * self.reshaper
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding, bias=self.b,dilation=self.dilation,groups=self.groups)


class BasicDrConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length=3, stride=1,padding='same', direction=0, bias=True,dilation=1,groups=1, norm=True, relu=True):
        super().__init__()
        self.conv = DrConv(in_channels, out_channels, kernel_length, direction=direction,padding=padding, stride=stride, bias=bias,dilation=dilation,groups=groups)
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU() if relu else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


if __name__ == '__main__':
    from torch import nn
    from torch.nn import functional as F
    import torch
    import math

    net=nn.Sequential(
        BasicDrConv(in_channels=3, out_channels=8, kernel_length=3, direction=0),
        BasicDrConv(in_channels=8, out_channels=4, kernel_length=5, direction=1),
        BasicDrConv(in_channels=4, out_channels=8, kernel_length=3, direction=2),
        BasicDrConv(in_channels=8, out_channels=3, kernel_length=5, direction=3)
    ).cuda()
    lq = torch.randn(1, 3, 256, 256).cuda()
    gt = torch.randn(1, 3, 256, 256).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters())
    for i in range(1024):
        pred = net(lq)
        optimizer.zero_grad()
        loss = criterion(pred, gt)
        loss.backward()
        print(f'{i}: {loss.item()}')
        optimizer.step()


