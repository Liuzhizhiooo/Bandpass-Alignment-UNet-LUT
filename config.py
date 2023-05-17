#coding:utf8
import os
from os.path import join, exists
import warnings
class DefaultConfig(object):
    train = False
    model = 'LUTHistUnet1DConv' # model Name
    tag = model # output tag
    
    dataset = "datasetCNNHist"  # dataset
    # dataRoot = "./dataset/data"
    dataRoot = r"D:\PaperCode\BandAdjust\dataset\data\512"
    tiledDir = "./dataset/divide"
    trainTileIds = None
    valTileIds = None
    testTileIds = None
    outputDir = './outputs'
    Patch512S2HistoramDict = "./dataset/data/Patch512Historam256-S2.pickle"
    TileHistDictS2 = "./dataset/data/tileHistoram256-S2.pickle"  # Nozero

    bandMax = [3000, 3000, 4000, 6500]
    bandWidth = 256
    seed = 2022  # random seed

    inputDim = 4
    loss = "MSELoss"  # MSELoss, L1Loss, CrossEntropyLoss
    useTVLoss = True
    lambdaSmooth = 0.01
    lambdaMonotonicity = 0.01
    isAccCal = True
    testModel = None

    batchSize = 4  # batch size
    useGpu = True  # user GPU or not
    deviceId = None  # None: use the last one by default
    numWorkers = 0
    saveFreq = 5  # model save frequency (unit: epoch)
    valStep = 5  # validate model frequency (unit: epoch)

    maxEpoch = 200
    lrMax = 0.001
    lrMode = "const"
    weightDecay = 1e-4


def parse(self, kwargs):
    '''
    update parameters
    '''
    # 更新参数
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self, k, v)

    self.trainTileIds = join(self.tiledDir, "train.txt")
    self.valTileIds = join(self.tiledDir, "val.txt")
    self.testTileIds = join(self.tiledDir, "test.txt")

    # 参数输出路径
    paraSaveDir = join(self.outputDir, self.tag)
    if not exists(paraSaveDir):
        os.makedirs(paraSaveDir, exist_ok=True)
    paraSavePath = join(paraSaveDir, "hyperParas.txt")
    if self.train:
        with open(paraSavePath, "w") as f:
            f.write("")

    print('user config:')
    tplt = "{0:>20}\t{1:<10}"
    with open(paraSavePath, "a") as f:
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k!="parse":
                value = str(getattr(self, k))
                print(tplt.format(k, value))
                if self.train:
                    f.write(tplt.format(k, value, chr(12288))+"\n")

DefaultConfig.parse = parse
opt = DefaultConfig()
