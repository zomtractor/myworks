import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_length=3, stride=1,padding='same', bias=True,dilation=1,groups=1, norm=True, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_length, stride=stride,padding=padding,bias=bias,dilation=dilation,groups=groups)
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.ReLU() if relu else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


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