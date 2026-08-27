'''查看大数据分块信息'''
import os
from os.path import join
import numpy as np

from libs import Util
from libs.cfg import getCfgInstance


'''根据框子信息，计算限制的块Id'''


def RectLimitBatchId(rect, level, batchSize):
    scale = 2 ** level
    rect2 = rect / scale
    rect2 = 1. * rect2 / batchSize
    low = np.int32(rect2[0])
    up = np.ceil(rect2[1]).astype(np.int32)
    return low, up


'''获取大数据格式每级别换算为512x512x512的块数'''


def GetBigBVLevelInfo(logSignal=None):
    cfg = getCfgInstance(True)
    batchSize = cfg.batchSize
    smallSize = cfg.smallBatchSize
    redunSize = cfg.redunSize
    rectLs = np.array(cfg.rectBoxLs, dtype=np.int32)
    sampleXYZ = cfg.sampleXYZ
    savePath = join(cfg.saveRoot, 'TrainDataSetFiter')
    os.makedirs(savePath, exist_ok=True)
    batchSavePath = join(savePath, 'BatchIdLs.txt')
    level = cfg.level
    batchIdLs = set()
    for rect in rectLs:
        low, up = RectLimitBatchId(rect, level, batchSize)
        for nz in range(low[2], up[2], sampleXYZ[2]):
            for ny in range(low[1], up[1], sampleXYZ[1]):
                for nx in range(low[0], up[0], sampleXYZ[0]):
                    batchIdLs.add('%d_%d_%d' % (nx, ny, nz))
    SignBatchNumber = Util.SliceBatch(batchSize, smallSize, redunSize)
    batchIdLs = list(batchIdLs)
    with open(batchSavePath, 'w') as f:
        for item in batchIdLs:
            f.write(item + '\n')
    print('Number of blocks:', len(batchIdLs) * len(SignBatchNumber))
    if logSignal is not None:
        logSignal.emit('Number of blocks: {}\n'.format(len(batchIdLs) * len(SignBatchNumber)))
    return len(batchIdLs) * len(SignBatchNumber)


if __name__ == '__main__':
    GetBigBVLevelInfo()
