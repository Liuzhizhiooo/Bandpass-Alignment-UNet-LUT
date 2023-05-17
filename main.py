# -*- coding: utf-8 -*-
"""
@Author  : liuzhizhi
@time    : 2022.02.25
@notice  : train and test model
"""

import os
from os.path import join, exists
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import models
import dataset
from config import opt
from utils import setupSeed, getDevice, getLrSchedule, trainEpoch, testEpoch, drawLoss, drawLr


def train(**kwargs):
    # update settings
    opt.parse(kwargs)
    # set up random seed
    setupSeed(opt.seed)
    outputDir = join(opt.outputDir, opt.tag)

    # 1. prepare dataset
    trainDataset = getattr(dataset, opt.dataset)(opt.trainTileIds)
    valDataset = getattr(dataset, opt.dataset)(opt.valTileIds)
    trainDataloader = DataLoader(trainDataset, opt.batchSize, shuffle=True, pin_memory=True, num_workers=opt.numWorkers)
    valDataloader = DataLoader(valDataset, 1)

    # 2. define model
    device = getDevice()
    model = getattr(models, opt.model)().to(device)

    # output the model structure
    modelOutputPath = join(outputDir, "model.txt")
    modelOutputMode = "a" if exists(modelOutputPath) else "w"
    with open(modelOutputPath, modelOutputMode, encoding="utf-8") as f:
        print(model, file=f)

    # 3. define the loss and optimizer
    criterion = getattr(torch.nn, opt.loss)().to(device)
    TVLoss = models.TVLoss().to(device) if opt.useTVLoss is True else None
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lrMax, weight_decay=opt.weightDecay)

    # 4. difine lr schedule
    lrScheduler = getLrSchedule(optimizer, opt.lrMode)

    # 5.start training
    # model save Dir
    checkpointsDir = join(outputDir, "checkpoints")
    if not exists(outputDir):
        os.makedirs(outputDir, exist_ok=True)
    if not exists(checkpointsDir):
        os.makedirs(checkpointsDir, exist_ok=True)

    # train loss path and val loss path
    trainLossPath, valLossPath = join(outputDir, "trainLoss.txt"), join(outputDir, "valLoss.txt")
    if not exists(trainLossPath):
        with open(trainLossPath, "w", encoding="utf-8") as f:
            f.write("")
    else:
        print("there is already a trainLoss.txt!")
    if not exists(valLossPath):
        with open(valLossPath, "w", encoding="utf-8") as f:
            f.write("")
    else:
        print("there is already a valLoss.txt!")

    epochs = opt.maxEpoch
    trainLossList, valLossList, valMseList = [], [], []
    lrList = []

    with tqdm(total=epochs, unit='epoch', ncols=100, colour="green") as pbar:
        for epoch in range(epochs):
            # 5.1 train models
            # valLoss, valMse = testEpoch(model, device, valDataloader, criterion, epoch+1, 'val')
            trainLoss, trainMse = trainEpoch(model, device, trainDataloader, criterion, optimizer, 1.0 * epoch / epochs, TVLoss)

            # 5.2 update lr
            lrScheduler.step()
            lrList.append(optimizer.param_groups[0]["lr"])

            # 5.3 save the training loss
            trainLossList.append(trainLoss)

            # 5.4 save the model
            if (epoch + 1) % opt.saveFreq == 0:  # epoch >= 49 and
                modelPath = join(checkpointsDir, f"epochs_{epoch+1}.pth")
                model.save(optimizer, modelPath)

            # 5.5 update pbar
            pbar.update(1)
            pbar.set_postfix({'lossEpoch': trainLoss, 'mseEpoch': trainMse})
            with open(join(outputDir, "trainLoss.txt"), "a", encoding="utf-8") as f:
                f.write(f"epoch{epoch+1}: lossEpoch_{trainLoss:.8} mseEpoch_{trainMse:.8}\n")

            # 5.6 validate the model
            if (epoch + 1) % opt.valStep == 0:
                valLoss, valMse = testEpoch(model, device, valDataloader, criterion, epoch + 1, 'val')
                valLossList.append(valLoss)
                valMseList.append(valMse)

                with open(join(outputDir, "valLoss.txt"), "a", encoding="utf-8") as f:
                    f.write(f"epoch{epoch+1}: lossEpoch_{valLoss:.8} mseEpoch_{valMse:.8}\n")

                # 5.7 draw the training loss and validation loss curve
                drawLoss(trainLossList, join(outputDir, "trainLoss.png"))
                drawLoss(valLossList, join(outputDir, "valLoss.png"))
                drawLoss(valMseList, join(outputDir, "valMse.png"), mode="Mse")

                # 5.8 draw lr
                drawLr([lrList], ["lr"], join(outputDir, "lrScheduler.png"))


def test(**kwargs):
    # update settings
    opt.parse(kwargs)  # 根据字典kwargs更新config参数

    # 1. prepare dataset
    testDataset = getattr(dataset, opt.dataset)(opt.testTileIds)
    testDataloader = DataLoader(testDataset, 1)

    # 2. define model
    device = getDevice()
    model = getattr(models, opt.model)().to(device)

    # 3. load Model
    if opt.testModel:
        testModelPath = join(opt.outputDir, opt.tag, "checkpoints", opt.testModel)
        model.load(testModelPath, None)

    # 4. loss
    criterion = getattr(torch.nn, opt.loss)()

    # 5. test
    testEpoch(model, device, testDataloader, criterion)


import fire
if __name__ == "__main__":
    fire.Fire()  # annotate it when debug
    
    # mode1. run cmd
    # Unet1DConv-TVLoss
    # python main.py train --train=True --useTVLoss=True --tag=Unet1DConv-TVLoss
    # python main.py test --testModel=epochs_195.pth --useTVLoss=True --tag=Unet1DConv-TVLoss

    # mode2. debug
    # train(
    #     train=True,
    #     useTVLoss=True,
    #     tag="Unet1DConv-TVLoss"
    # )

    # test(
    #     useTVLoss=True,
    #     tag="Unet1DConv-TVLoss",
    #     testModel="epochs_195.pth",
    # )