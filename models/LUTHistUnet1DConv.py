# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@time    : 2022.11.09
@notice  : LUTHistUnet1DConv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
from .UNet1D import UNet1D
from config import opt


def getDevice():
    if opt.useGpu and torch.cuda.is_available():
        # use the last GPU by default
        deviceId = opt.deviceId if opt.deviceId != None else torch.cuda.device_count() - 1
        # device = f"cuda:{deviceId}"
        device = torch.device(deviceId)
    else:
        device = torch.device("cpu")
    return device


def takeArrayByIndice(paras, indices):
    """
    map image with LUT
    paras: LUT, C * [bmax] or C * [B, bmax]
    indicies: image, [B, C, H, W]
    """
    B = len(indices)
    if paras[0].ndim == 1:
        isParaShareCrossBatch = True
    elif paras[0].ndim == 2:
        isParaShareCrossBatch = False
    else:
        raise ValueError(f"shape illegal")
    
    # C * [bmax]
    if isParaShareCrossBatch is True:
        # para:[B, bmax], indice: [B, H, W]
        mapped = torch.stack([torch.take(para, indices[:, bandIdx]) for bandIdx, para in enumerate(paras)], dim=1)
    # C * [B, bmax]
    else:
        mapped = torch.stack([
            torch.take(
                para, 
                indices[:, bandIdx] + \
                torch.Tensor([batchId * para.shape[1] for batchId in range(B)]).reshape(B, 1, 1).to(indices.device, torch.int64)
            )
            for bandIdx, para in enumerate(paras)
        ], dim=1)
    return mapped


class LUTHistUnet1DConv(BasicModule):
    def __init__(self, inChannel=opt.inputDim, outChannel=opt.inputDim, bandWidth=opt.bandWidth):
        super(LUTHistUnet1DConv, self).__init__()
        self.device = getDevice()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.bandWidth = bandWidth
        self.bandMax = opt.bandMax
        self.unet = UNet1D(4, 4, dim=8)  # 32
        self.relu = nn.ReLU()

        self.conv1 = nn.Sequential(
            nn.Conv2d(inChannel, inChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inChannel),
            nn.ReLU()
        )

    def forward(self, x, histArr):
        """
        img: [B, C, H, W]
        imgMap: [B, C, segNumSum]
        """
        B = x.shape[0]
        
        # 1. create LUT
        # [B, C, 256]
        bandLUTSInit = self.relu(self.unet(histArr)) * 100.0

        # interpolate lookupTable
        # [B, C, 256] => [C, B, 1, 256]
        bandLUTS = bandLUTSInit.permute(1, 0, 2).unsqueeze(dim=2)
        # C * [B, 1, 256] => C * [B, 1, bandmax] => C * [B, bandmax]
        bandLUTS = [F.interpolate(bandLUT, (bmax), mode="linear", align_corners=True).reshape(B, bmax) for bmax, bandLUT in zip(self.bandMax, bandLUTS)]
        
        # 3. clip
        x = torch.stack([torch.clip(band.squeeze(dim=1), 0, bmax - 1).to(torch.int64) for bmax, band in zip(self.bandMax, torch.split(x, 1, dim=1))], dim=1)
        
        # 4. look up
        x = takeArrayByIndice(bandLUTS, x)

        # 5. add spatial details
        # x = self.conv2(x + self.conv1(x))
        x = x + self.conv1(x)

        return x, bandLUTSInit