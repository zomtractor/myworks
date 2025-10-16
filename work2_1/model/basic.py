import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding=None, bias=True,dilation=1,groups=1, norm=False, relu=True, trans=False,act=nn.GELU):
        super().__init__()
        self.bn=None
        self.act=None
        if padding is None:
            padding = kernel_size // 2
        if trans:
            padding = kernel_size // 2 - 1
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,padding=padding,
                                           bias=bias,dilation=dilation,groups=groups)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,padding=padding, bias=bias,
                                  dilation=dilation,groups=groups)
        if norm:
            self.bn = nn.BatchNorm2d(out_channels)
        if relu:
            self.act = act()

    def forward(self, x):
        res = self.conv(x)
        if self.bn is not None:
            res = self.bn(res)
        if self.act is not None:
            res = self.act(res)
        return res


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        hidden_channels = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
