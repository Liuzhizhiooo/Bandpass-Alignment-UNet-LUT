import os
import math
import torch
import random
import numpy as np
from tqdm import tqdm 
from os.path import join
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import opt


def setupSeed(seed):
    """
    set the random seed
    """
    if seed != None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f"set seed {seed}")
    else:
        print("do not set seed")


def getDevice():
    """
    get the assigned device
    """
    if opt.useGpu and torch.cuda.is_available():
        # use the last GPU by default
        deviceId = opt.deviceId if opt.deviceId != None else torch.cuda.device_count() - 1
        # device = f"cuda:{deviceId}"
        device = torch.device(deviceId)
    else:
        device = torch.device("cpu")
    print(f"device: {device}")
    return device


def getLrSchedule(optimizer, mode="exp"):
    """
    get the learning rate schedule
    """

    if mode == "const":
        # lrLambda = lambda iter: iter / opt.warmUpEpochs if iter < opt.warmUpEpochs else 1
        lrLambda = lambda iter: 1

    elif mode == "exp":
        lrLambda = lambda iter: 1 if iter < opt.warmUpEpochs else math.pow(
            opt.gamma, iter - opt.warmUpEpochs)

    elif mode == "poly":
        power = 0.9
        lrLambda = lambda iter: (1 - iter / opt.maxEpoch)**power

    else:
        info = f"lr mode '{mode}' illegal!"
        raise ValueError(info)

    lrScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lrLambda)
    return lrScheduler


def updateHistByLUTS(hist, LUTS, bandMax):
    """
    update the LUT by histogram
    hist: [B, C, binNum]
    LUTS: [B, C, binNum]
    """
    batchNum, bandNum, binNum = hist.shape
    updatedHist = torch.zeros_like(hist).to(hist.device)
    batchIdxList = torch.tensor([[x for _ in range(bandNum)] for x in range(batchNum)]).to(hist.device)
    bandIdxList = torch.tensor([[x for x in range(bandNum)] for _ in range(batchNum)]).to(hist.device)
    binRate = expandAndRepeat(torch.tensor([x / binNum for x in bandMax]).to(hist.device), 0, batchNum)
    for idx in range(binNum):
        mappedValue = torch.div(LUTS[:, :, idx], binRate, rounding_mode='floor').to(torch.int64)
        updatedHist[batchIdxList, bandIdxList, mappedValue] += hist[:, :, idx]
    return updatedHist


def drawLoss(loss, savePath, mode="loss"):
    """
    draw loss curve
    """
    fig, ax = plt.subplots()
    ax.set_ylabel(f'epoch {mode}')
    ax.set_xlabel('epoch')
    ax.plot(np.arange(len(loss)) + 1, loss, 'b-')
    if "train" in savePath:
        ax.set_title('train loss curve')
    elif "val" in savePath:
        ax.set_title(f"val {mode} curve")
        ax.set_xticks(np.arange(len(loss)) + 1)
        ax.set_xticklabels((np.arange(len(loss)) + 1) * opt.valStep)
    else:
        pass
    # ax.set_ylim(0, loss[int(len(loss) * 0.1)])
    plt.savefig(savePath, dpi=500, bbox_inches='tight')
    plt.close("all")


def drawLr(LrList, LabelList, savePath):
    """
    draw leaning rate curve
    """
    assert len(LrList) == len(LabelList), f"LrList dismatch LabelList!"
    colorList = ["orange", "g", "b", "c"]  # LrList最多三条
    fig, ax = plt.subplots()
    ax.set_title('Learning rate curve')
    ax.set_ylabel('lr')
    ax.set_xlabel('epoch')
    for idx, lrCurve in enumerate(LrList):
        ax.plot(np.arange(len(lrCurve)) + 1, lrCurve, '-', color=colorList[idx], label=LabelList[idx])
    ax.legend(LabelList, loc=1, fontsize=14)
    plt.savefig(savePath, dpi=500, bbox_inches='tight')
    plt.close("all")


def getParas(yTruth, yPred):
    """
    get the variable for the final accuracy calculation
    the accuracy metrics RMSE, R2 are calculated by segmentation to save computing memory
    yTruth, yPred: [N, 1]
    """
    n1 = yTruth.shape[0]
    mean1 = np.mean(yTruth)
    rmse1 = np.sqrt(np.sum((yTruth - yPred) ** 2) / n1)
    SSE1 = np.sum((mean1 - yTruth) ** 2)
    return np.array([n1, mean1, rmse1, SSE1]).reshape(-1, 1)


def mergeRMSE(paraArr):
    """
    merge the RMSE metric for all segments
    paraArr: 
    [
        [n1, n2, n3], 
        [mean1, mean2, mean3],
        [rmse1, rmse2, rmse3],
        [SSE1, SSE2, SSE3]
    ]
    """
    fenzi = np.sum(paraArr[0] * paraArr[2] ** 2)
    fenmu = np.sum(paraArr[0])
    return np.sqrt(fenzi / fenmu)


def mergeR2(paraArr):
    """
    merge the R2 metric for all segments
    paraArr: 
    [
        [n1, n2, n3], 
        [mean1, mean2, mean3],
        [rmse1, rmse2, rmse3],
        [SE1, SE2, SE3],
        [SSE1, SSE2, SSE3]
    ]
    """
    newMean = np.sum(paraArr[0] * paraArr[1]) / np.sum(paraArr[0])
    fenzi = np.sum(paraArr[0] * paraArr[2] ** 2)
    fenmu = np.sum(paraArr[3]) + np.sum(paraArr[0] * (newMean - paraArr[1]) ** 2) # + np.sum(2 * (newMean - paraArr[1]) * paraArr[4])
    return 1 - (fenzi / fenmu)


def accCal(paraArr, epoch, name):
    """
    calculate accuracy metrics
    rmseParaArr: [B, 5, N]
    r2ParaArr: [B, 2, N]
    """
    outputDir = join(opt.outputDir, opt.tag)
    if name.startswith("test"):
        outputPath = join(outputDir, f"{name}_acc_{opt.testModel}.txt")
    else:
        outputPath = join(outputDir, f"{name}_acc.txt")

    bandNum = len(paraArr)
    RMSE = np.zeros((bandNum, 1))
    R2 = np.zeros((bandNum, 1))
    for idx in range(bandNum):
        RMSE[idx] = mergeRMSE(paraArr[idx]) * 1e6
        R2[idx] = mergeR2(paraArr[idx])

    # 2. all index
    with open(outputPath, 'a', encoding="utf-8") as f:
        if name.startswith("test"):
            f.write(f"test: {opt.testTileIds}\n")
        if epoch:
            f.write(f"[epoch]:{epoch}\n")
        f.write('mse \n')
        np.savetxt(f, np.array(RMSE), fmt='%.8f')
        f.write('\nR2 \n')
        np.savetxt(f, np.array(R2), fmt='%.8f')

    if name.startswith("test"):
        print(name)
        print('mse \n')
        print(RMSE)
        print('R2 \n\n\n')
        print(R2)
    return R2, RMSE, outputDir


def expandAndRepeat(x, dim, num):
    """
    expand array at dim and repeat for num numbers
    """
    repeatDim = [1 for _ in x.shape]
    repeatDim.insert(dim, num)
    return x.unsqueeze(dim).repeat(tuple(repeatDim))


def maskImg(img, mask, permute=True):
    """
    mask image with mask
    [B, C, H , W] => [N, C]
    img: [B, C, H, W]
    mask: [B, H, W]
    """
    # [B, C, H, W] => [C, B, H, W]
    img = img.permute(1, 0, 2, 3)
    # mask, [C, B, H, W] => [C, N]
    img = img[..., mask]
    if permute is True:
        # [C, N] => [N, C]
        img = img.permute(1, 0)
    return img


def calNDVI(arr):
    """
    calculate NDVI
    arr: [C, N] BGRNir
    """
    r, nir = arr[2:4]
    return (nir * 1.0 - r * 1.0) / (nir * 1.0 + r * 1.0)


def trainEpoch(model, device, dataloader, criterion, optimizer, epoch, TVLoss=None):
    """
    train one epoch
    """
    lossEpoch, mseEpoch = 0, 0
    with tqdm(total=len(dataloader), unit='batch', leave=False, ncols=100, colour="blue") as pbar:
        for idx, batchData in enumerate(dataloader):
            optimizer.zero_grad()

            # 1. data load
            imgs, labels, mask = [x.to(device) for x in batchData[:3]]
            S2HistArr = batchData[3].to(device)
            pred = model(imgs, S2HistArr)
            
            if type(pred) is tuple:
                pred, bandLUTS = pred

            # 2. loss
            # 2.1 bandLUTS-TVLoss
            if TVLoss is not None:
                tv, mn = TVLoss(bandLUTS)
            else:
                tv, mn = 0, 0
            
            # 2.2 mse loss
            # mask
            # [B, C, H, W] => [N, C]
            pred, labels = maskImg(pred, mask), maskImg(labels, mask)
            mseloss = criterion(pred, labels)

            # 2.3 loss sum
            loss = mseloss + opt.lambdaSmooth * tv + opt.lambdaMonotonicity * mn  # + opt.lambdaHist * histLoss
            loss.backward()
            optimizer.step()

            # 3. acc calculation
            mse = F.mse_loss(pred, labels)
            mseEpoch += mse.item() / len(dataloader)
            lossEpoch += loss.item()

            pbar.update(1)
            if (idx + 1) % 10 == 0:
                pbar.set_postfix({'loss(batch)': loss.item() / dataloader.batch_size})
    return lossEpoch, mseEpoch


def testEpoch(model, device, dataloader, criterion, epoch=None, name='test'):
    """
    test one epoch
    """
    model.eval()
    lossEpoch, mseEpoch = 0, 0
    yTruthSeg, yPredSeg = [], []
    segNum = 500
    segmentNum = int(np.ceil(len(dataloader) / segNum))
    paraArr = np.zeros((opt.inputDim, 4, segmentNum))
    paraArrNDVI = np.zeros((1, 4, segmentNum))

    with torch.no_grad():
        with tqdm(total=len(dataloader), unit='batch', ncols=100, colour="yellow", leave=False) as pbar:
            for idx, batchData in enumerate(dataloader):
                # 1. data load
                imgs, labels, mask = [x.to(device) for x in batchData[:3]]
                pred = model(imgs, batchData[3].to(device))

                # pred = [pred, bandLUTS]
                if type(pred) is tuple:
                    pred = pred[0]

                # 2. mask
                # [B, C, H, W] => [C, N]
                pred, labels = maskImg(pred, mask, False), maskImg(labels, mask, False)

                # 3. loss calculation (for validation)
                loss = criterion(pred, labels)
                labels = labels.cpu()
                pred = pred.cpu()

                # 4. acc calculation
                mse = F.mse_loss(pred, labels)
                mseEpoch += mse / len(dataloader)
                lossEpoch += loss.item()

                # 5. acc segment
                yTruthSeg.append(labels)
                yPredSeg.append(pred)
                if (idx + 1) % segNum == 0 or (idx + 1) == len(dataloader):
                    yTruthSeg = torch.cat(yTruthSeg, dim=1).numpy()
                    yPredSeg = torch.cat(yPredSeg, dim=1).numpy()
                    segmentIdx = idx // segNum
                    for bandIdx in range(opt.inputDim):
                        paraArr[bandIdx, :, segmentIdx:segmentIdx+1] = getParas(yTruthSeg[bandIdx] / 1e6, yPredSeg[bandIdx] / 1e6)
                    
                    # NDVI
                    paraArrNDVI[0, :, segmentIdx:segmentIdx+1] = getParas(calNDVI(yTruthSeg), calNDVI(yPredSeg))

                    yTruthSeg, yPredSeg = None, None
                    yTruthSeg, yPredSeg = [], []

                pbar.update(1)
                if (idx + 1) % 10 == 0:
                    pbar.set_postfix({'loss(batch)': loss.item() / dataloader.batch_size})

        # acc calculation
        if opt.isAccCal:
            accCal(paraArr, epoch, name)
        
        if opt.train is False:
            # NDVI
            accCal(paraArrNDVI, epoch, name)

    return lossEpoch, mseEpoch


if __name__ == "__main__":
    pass