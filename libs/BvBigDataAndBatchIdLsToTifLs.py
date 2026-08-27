'''BV大数据格式和块id列表转tif序列'''
import json
import os
import shutil
import time
from os.path import join
import numpy as np
from multiprocessing import Process, Queue
import tifffile
from libs.cfg import getCfgInstance




def SignBvToTif(add, savePath, batchSize, workQue, finishQue):
    # from bv.BVUtil import readBV, BVImg_to_numpy, freeBV
    from BVExample.BVMoudle.BVMoudle import BVReader
    cfg = getCfgInstance(True)
    zs = batchSize[2]
    batchSize = batchSize[:2]
    smallSize = cfg.smallBatchSize[:2]
    redunSize = cfg.redunSize[:2]
    readObj = BVReader()
    while True:
        work = workQue.get(timeout=1)
        if len(work) == 0:
            finishQue.put(-1)
            return
        x, y, z, nx, ny, nz, cls = work
        sz = z * zs
        img = readObj.readBV(join(add, '%d_%d.bv' % (x, y)), sz, cfg.smallBatchSize[2])
        if img.shape[1] != batchSize[1] or img.shape[2] != batchSize[0]:
            # img2 = np.zeros([img.shape[0], batchSize[1], batchSize[2]], dtype=img.dtype)
            img2 = np.zeros([img.shape[0], batchSize[1], batchSize[0]], dtype=img.dtype)
            # print(img2.shape, img.shape)
            img2[:img.shape[0], :img.shape[1], :img.shape[2]] = img
            img = img2
        sp = (smallSize - redunSize) * [nx, ny]
        ep = np.min([sp + smallSize, batchSize], axis=0)
        sp = np.max([np.min([sp, ep - smallSize], axis=0), [0, 0]], axis=0)
        tarImg = img[:, sp[1]: ep[1], sp[0]: ep[0]]
        s_id = '%s-%d_%d_%d-%d_%d_%d.tif' % (str(cls).zfill(4), x, y, z, nx, ny, nz)
        tifffile.imwrite(join(savePath, s_id), tarImg, compression='lzw')
        finishQue.put(s_id)


def MulBatchLsToTifLs(logSignal=None):
    cfg = getCfgInstance(True)
    root = cfg.root
    saveRoot = join(cfg.saveRoot, 'TrainDataSetFiter/HistAnaly')
    txtPath = join(saveRoot, 'ImgIdLs.json')
    savePath = join(cfg.saveRoot, 'NeedFiterImg')
    levle = cfg.level
    MNumber = cfg.MNumber
    batchSize = cfg.batchSize
    if os.path.isdir(savePath): shutil.rmtree(savePath)
    os.makedirs(savePath, exist_ok=True)
    add = join(root, str(levle))
    # 读取Id Ls
    with open(txtPath, 'r') as f:
        idLs = json.loads(f.read())
    workQue = Queue()
    finishQue = Queue()
    for id in idLs:
        for it in idLs[id]:
            workQue.put(it)
    workLen = workQue.qsize()
    for i in range(MNumber): workQue.put([])
    # SignBvToTif(add, savePath, batchSize, workQue, finishQue)
    ps = []
    for i in range(MNumber):
        ps.append(Process(target=SignBvToTif, args=(add, savePath, batchSize, workQue, finishQue,)))
        ps[-1].daemon = True
        ps[-1].start()
    sTime = time.time()
    for i in range(workLen):
        id = finishQue.get()
        if i % 10 == 0:
            userTime = time.time() - sTime
            surplusTime = userTime / (i + 1) * (workLen - i - 1)
            print('[MulBatchLsToTifLs Progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
            (i + 1) / workLen * 100, userTime, surplusTime))
            logInfo = '[MulBatchLsToTifLs Progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]\n' % (
            (i + 1) / workLen * 100, userTime, surplusTime)
            if logSignal is not None:
                logSignal.emit(logInfo)
    # 等待线程结束(很快)
    for i in range(MNumber):
        finishQue.get()
    return


def SignTest():
    cfg = getCfgInstance(True)
    root = cfg.root
    saveRoot = join(cfg.saveRoot, 'HistAnaly')
    txtPath = join(saveRoot, 'ImgIdLs.json')
    savePath = join(saveRoot, 'TrainImages')
    levle = cfg.level
    MNumber = cfg.MNumber
    batchSize = cfg.batchSize
    os.makedirs(savePath, exist_ok=True)
    add = join(root, str(levle))
    work = Queue()
    work.put([13, 11, 16, 1, 1, 0, 47])
    finishQue = Queue()
    SignBvToTif(add, savePath, batchSize, work, finishQue)


if __name__ == '__main__':
    MulBatchLsToTifLs()
    # SignTest()

'''
07 2_16_0
02
'''
