'''Swc To Mask'''
import os, tifffile
import numpy as np
from os.path import join
# import PreMake.BigSwcUtil
from multiprocessing import Process, Queue

# '''Swc多树拆分'''
def SplitSwcData(swcData):
    indLs = np.where(swcData[:, -1] == -1)[0].tolist() + [swcData.shape[0]]
    swcDataLs = []
    for i in range(len(indLs) - 1):
        data = swcData[indLs[i]: indLs[i + 1]]
        sp = data[0, 0]
        data[:, 0] -= sp - 1
        data[1:, -1] -= sp - 1
        swcDataLs.append(data)
    return swcDataLs

'''获取点云核点云'''
def GetPcKernelPc(pc, kernelArr, imgShape):
    curPc = (pc[:, None] + kernelArr[None]).reshape([-1, 3])
    curPc = np.round(curPc).astype(np.int32)
    curPc[curPc < 0] = 0
    curPc[curPc[:, 2] > imgShape[0] - 1, 2] = imgShape[0] - 1
    curPc[curPc[:, 1] > imgShape[1] - 1, 1] = imgShape[1] - 1
    curPc[curPc[:, 0] > imgShape[2] - 1, 0] = imgShape[2] - 1
    curPc = np.unique(curPc, axis=0)
    return curPc

'''单个Swc转距离场'''
def SignSwcToDF(workQue, finishQue, swcPath, savePath, kernelLen, imgShape, minD, maxD):
    maskMax = -np.log(minD)
    dfImg = np.zeros(imgShape[::-1], dtype=np.float32)
    kernelArr = []
    for z in range(-kernelLen, kernelLen + 1):
        for y in range(-kernelLen, kernelLen + 1):
            for x in range(-kernelLen, kernelLen + 1):
                kernelArr.append([x, y, z])
    kernelArr = np.array(kernelArr, dtype=np.int32)
    while True:
        name = workQue.get()
        if len(name) == 0:
            return
        nameId = os.path.splitext(name)[0]
        swcData = np.loadtxt(join(swcPath, name), ndmin=2)
        # if len(swcData) == 0:
        #     finishQue.put(1)
        #     continue
        swcDataLs = SplitSwcData(swcData)
        dfImg[...] = maxD
        for swcData in swcDataLs:
            for ii, item in enumerate(swcData):
                if item[-1] == -1:
                    continue
                p0 = item[2: 5]
                p1 = swcData[int(item[-1]) - 1, 2: 5]
                v = (p1 - p0).reshape([1, 3])
                if np.linalg.norm(v) < 0.1: continue
                # 插值
                d = np.linalg.norm(p0 - p1)
                if d > kernelLen:
                    d2 = int(d + 1)
                    xLs = (np.arange(1, d2 + 1, 1) / d2).reshape([-1, 1])
                    data = p0 * xLs + p1 * (1 - xLs)
                else:
                    data = np.array([p0, p1])
                curPc = GetPcKernelPc(data, kernelArr, imgShape[::-1])
                proT = (curPc - p0).dot(v.T) / v.dot(v.T)
                proT = np.clip(proT, 0, 1)
                proP = p0 + proT * v
                d = np.linalg.norm(proP - curPc, axis=1)
                dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]] = np.min([dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]], d], axis=0)
        dfImg = -np.log(minD + dfImg / maxD * (1 - minD))
        dfImg2 = (dfImg / maskMax * 255).astype(np.uint8)
        tifffile.imwrite(join(savePath, nameId + '.tif'), dfImg2, compression='lzw')
        finishQue.put(1)

'''ParallelSwc转距离场'''
def MulSwcToDF(swcPath, savePath, imgShape, MNumber=10,logSignal=None):
    imgShape = np.array(imgShape, dtype=np.int32)
    os.makedirs(savePath, exist_ok=True)
    kernelLen = 3
    maxD = ((kernelLen ** 2) * 3) ** 0.5
    minD = 0.0697
    workQue = Queue()
    finishQue = Queue()
    ls = os.listdir(swcPath)
    lsLen = len(ls)
    for name in ls:
        workQue.put(name)
    for i in range(MNumber): workQue.put('')
    ps = []
    for i in range(MNumber):
        ps.append(Process(target=SignSwcToDF, args=(workQue, finishQue, swcPath, savePath, kernelLen, imgShape, minD, maxD,)))
        ps[-1].daemon = True
        ps[-1].start()
    for i in range(lsLen):
        finishQue.get()
        print('%d|%d' % (i, lsLen))
        if logSignal is not None:
            logSignal.emit('%d|%d\n' % (i, lsLen))

if __name__ == '__main__':
    swcPath = r'D:\NeronDataSet\ZjHospital-C2804-DataSet\SegMentDataSet\TrainDataSet\Swc'
    savePath = r'D:\NeronDataSet\ZjHospital-C2804-DataSet\SegMentDataSet\TrainDataSet\mask'
    imgShape = [272, 272, 144]  # xyz
    MNumber = 10
    MulSwcToDF(swcPath, savePath, imgShape, MNumber)
