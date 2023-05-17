# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@time    : 2022.11.09
@notice  : UNet for 1D signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .UNet_function import *
from .BasicModule import BasicModule


class UNet1D(BasicModule):
    """
    In order to utilize more spectral info(4 or more bands)
    we can't just use the vgg as encoder part for it's input bands are fixed as 3
    """
    def __init__(self, n_channels=4, n_classes=4, dim=64):
        super(UNet1D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv1D(n_channels, dim) 
        self.down1 = Down1D(dim, dim * 2)
        self.down2 = Down1D(dim * 2, dim * 4)
        self.down3 = Down1D(dim * 4, dim * 8)
        self.down4 = Down1D(dim * 8, dim * 16)
        self.up1 = Up1D(dim * 16, dim * 8)
        self.up2 = Up1D(dim * 8, dim * 4)
        self.up3 = Up1D(dim * 4, dim * 2)
        self.up4 = Up1D(dim * 2, dim)
        self.outc = OutConv1D(dim, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits