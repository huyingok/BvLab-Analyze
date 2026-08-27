'''并行获取灰度直方图'''
import shutil
import time

import numpy as np
import os
from os.path import join
from multiprocessing import Process, Queue

from libs import Util
from BVExample.BVMoudle.BVMoudle import BVReader, getHist
from libs.cfg import getCfgInstance


'''根据框子信息，计算限制的块Id'''


def RectLimitBatchId(rect, level, batchSize):
    scale = 2 ** level
    rect2 = rect / scale
    rect2 = 1. * rect2 / batchSize
    low = np.int32(rect2[0])
    up = np.ceil(rect2[1]).astype(np.int32)
    return low, up


def MaxProject(img, scale):
    imgLs = []
    for z in range(scale[0]):
        for y in range(scale[1]):
            for x in range(scale[2]):
                imgLs.append(img[z::2, y::2, x::2])
    img2 = np.max(imgLs, axis=0)
    return img2


def SignBvToHist(workQue, finishQue, savePath, threId):
    cfg = getCfgInstance(True)
    batchSize = cfg.batchSize
    smallSize = cfg.smallBatchSize
    redunSize = cfg.redunSize
    add = join(cfg.root, str(cfg.level))
    sliceInfo = Util.SliceBatch(batchSize, smallSize, redunSize)
    readObj = BVReader()
    totalHistLs = {}
    while True:
        info = workQue.get(timeout=1)
        if len(info) == 0:
            np.save(join(savePath, '%s.npy' % str(threId).zfill(5)), totalHistLs)
            finishQue.put(-1)
            return
        name, bx, by, bz = info
        sz = bz * batchSize[2]
        img = readObj.readBV(join(add, name), sz, batchSize[2])
        if not np.all(img.shape == batchSize[::-1]):
            img2 = np.zeros(batchSize[::-1], dtype=np.int32)
            img2[:img.shape[0], :img.shape[1], :img.shape[2]] = img
            img = img2
        for sp, ep, nx, ny, nz in sliceInfo:
            smallImg = img[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
            hist = getHist(smallImg)
            hist2 = hist.T[0].astype(np.int32)
            HistSaveName = '%d_%d_%d-%d_%d_%d.npy' % (bx, by, bz, nx, ny, nz)
            totalHistLs[HistSaveName] = hist2
        finishQue.put(name)


'''Bv大数据格式转灰度直方图'''


def BvBigDataToHist(logSignal=None):
    cfg = getCfgInstance(True)
    # root = cfg.root
    txtPath = join(cfg.saveRoot, 'TrainDataSetFiter/BatchIdLs.txt')
    savePath = join(cfg.saveRoot, 'TrainDataSetFiter/HistData')
    # level = cfg.level
    # batchSize = cfg.batchSize
    MNumber = cfg.MNumber
    # assert not os.path.isdir(savePath), '输出文件已存在，请删除！%s' % savePath
    if os.path.isdir(savePath):
        shutil.rmtree(savePath)
    os.makedirs(savePath, exist_ok=True)
    # Parallel
    workQue, finishQue = Queue(), Queue()
    with open(txtPath, 'r') as f:
        nameIdLs = f.read().strip().split('\n')
    workLen = len(nameIdLs)
    for nameId in nameIdLs:
        t = nameId.split('_')
        x, y, z = int(t[0]), int(t[1]), int(t[2])
        workQue.put(['%d_%d.bv' % (x, y), x, y, z])
    for i in range(MNumber):
        workQue.put([])
    ps = []
    for i in range(MNumber):
        ps.append(Process(target=SignBvToHist, args=(workQue, finishQue, savePath, i,)))
        ps[-1].daemon = True
        ps[-1].start()
    sTime = time.time()
    for i in range(workLen):
        info = finishQue.get()
        if i % 50 == 0:
            userTime = time.time() - sTime
            surplusTime = userTime / (i + 1) * (workLen - i - 1)
            logInfo = '[BvBigDataToHist Progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]\n' % (
            (i + 1) / workLen * 100, userTime, surplusTime)
            print('[BvBigDataToHist Progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
            (i + 1) / workLen * 100, userTime, surplusTime))
            if logSignal is not None:
                logSignal.emit(logInfo)
    # 等待线程结束(很快)
    for i in range(MNumber):
        finishQue.get()


if __name__ == '__main__':
    BvBigDataToHist()
