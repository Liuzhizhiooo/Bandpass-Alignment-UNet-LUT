# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@time    : 2022.11.09
@notice  : caculate the TVLoss of LUTs
"""


import torch
from torch import nn
from config import opt

def expandAndRepeat(x, dim, num):
    repeatDim = [1 for _ in x.shape]
    repeatDim.insert(dim, num)
    return x.unsqueeze(dim).repeat(tuple(repeatDim))


def getDevice():
    if opt.useGpu and torch.cuda.is_available():
        # use the last GPU by default
        deviceId = opt.deviceId if opt.deviceId != None else torch.cuda.device_count() - 1
        # device = f"cuda:{deviceId}"
        device = torch.device(deviceId)
    else:
        device = torch.device("cpu")
    return device


class TVLoss(nn.Module):
    """
    [C, 250] / [B, C, 250]
    """
    def __init__(self, dim=opt.bandWidth):
        super(TVLoss,self).__init__()
        self.device = getDevice()
        self.weight = torch.ones((opt.inputDim, dim-1), dtype=torch.float).to(self.device)
        self.weight[:, (0,dim-2)] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, bandLUTS):
        """
        bandLUTS: [C, 250] / [B, C, 250]
        """
        # [B, C, 249]
        B = len(bandLUTS)
        diff = bandLUTS[..., :-1] - bandLUTS[..., 1:]
        if bandLUTS.ndim == 2:
            weight = self.weight
        elif bandLUTS.ndim == 3:
            weight = expandAndRepeat(self.weight, 0, B)
        else:
            raise ValueError(f"shape illegal")

        tv = torch.mean(torch.mul((diff ** 2), weight))
        mn = torch.mean(self.relu(diff))
        return tv * torch.numel(bandLUTS), mn * torch.numel(bandLUTS)