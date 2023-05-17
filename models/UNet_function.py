# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@time    : 2020-02-26
@notice  : the funtion of UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv1D(nn.Module):
    """
    (convolution => [BN] => ReLU) * 2
    in order to keep the border pixels, set padding = 1
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down1D(nn.Module):
    """
    2x2 max pooling operation with stride 2 for downsampling + DoubleConv. 
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up1D(nn.Module):
    """
    (upsample => cancate => conv)
    2x2 convolution for upsampling(halving the dimension meanwhile) + 
    concate with the correspondingly feature map from the contracting path + 
    two 3x3 convolutions(followed by relu)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv1D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        x1 = F.pad(x1, [diffY.div_(2, rounding_mode="trunc"), diffY - diffY.div_(2, rounding_mode="trunc")])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU()
        # )
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)