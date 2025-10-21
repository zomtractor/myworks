import torch
from torch import nn

from model import BasicConv


class Mock(nn.Module):
    def __init__(self,base_channels):
        super(Mock, self).__init__()
        self.mlp = nn.Sequential(
            BasicConv(3,base_channels,3,1),
            BasicConv(base_channels,64,3,1),
            BasicConv(64,base_channels,3,1),
            BasicConv(base_channels ,3,3,1)
        )

    def forward(self, x):
        x = x+self.mlp(x)
        return x
