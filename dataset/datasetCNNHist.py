# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@time    : 2022.10.24
@notice  : dataset
"""

import torch
from torch.utils import data
import numpy as np
from os.path import join, basename
from config import opt
import pickle
import torch.nn.functional as F
from osgeo import gdal, gdalconst, gdal_array


def readTif(fp):
    """
    read tif
    """
    ds = gdal.Open(fp, gdalconst.GA_ReadOnly)
    if ds is None: raise ValueError(f"Open {fp} Failed !!")
    return gdal_array.LoadFile(fp)


class datasetCNNHist(data.Dataset):
    def __init__(self, tileIds, root=opt.dataRoot):
        """
        root: data dir
        tileIds: selected samples
        """
        self.tileIds = tileIds

        with open(tileIds, "r") as f:
            fileList = [x.strip() for x in f.read().split("\n")]

        varList = ["S2", "GF1", "mask"]
        varDirList = ["S2", "GF1", "mask"]
        for idx, var in enumerate(varList):
            setattr(self, var, [join(opt.dataRoot, x.split("/")[0], varDirList[idx], x.split("/")[1]) for x in fileList])

        # training: patch-level hist
        # val/test: tile-level hist
        if "train.txt" in tileIds:
            with open(opt.Patch512S2HistoramDict, "rb") as f:
                self.histDictS2 = pickle.load(f)
                print(f"{tileIds}-patchHist")
        else:
            with open(opt.TileHistDictS2, "rb") as f:
                self.histDictS2 = pickle.load(f)
                print(f"{tileIds}-tileHist")

    def __getitem__(self, index):
        # S2-img: [C, H, W]
        img = readTif(self.S2[index])
        # GF1-label: [C, H, W]
        label = readTif(self.GF1[index])
        # mask-mask: [H, W]
        mask = readTif(self.mask[index]) > 0

        # hist
        # training: patch-level hist
        # val/test: tile-level hist
        imgName = basename(self.S2[index])
        if "train.txt" in self.tileIds:
            histArrS2 = self.histDictS2[imgName]
        else:
            tileDate = imgName.split("-")[0]
            histArrS2 = self.histDictS2[tileDate]
        histArrS2 = F.normalize(torch.from_numpy(histArrS2).to(torch.float32), p=1, dim=1)

        return [
            torch.from_numpy(img * 1.0).to(torch.float32),
            torch.from_numpy(label * 1.0).to(torch.float32),
            torch.from_numpy(mask), histArrS2
        ]

    def __len__(self):
        return len(self.S2)