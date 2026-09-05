# -*- coding: utf-8 -*-
from os.path import join
import os
# os.environ["OMP_NUM_THREADS"] = '24'
# os.environ["OPENBLAS_NUM_THREADS"] = '24'
import numpy as np
from libs import Util
import shutil
import time
from multiprocessing import Process, Queue
from BVExample.BVMoudle import BVReader, getHist
from sklearn.cluster import KMeans
import json
import sys
import traceback
import tifffile
from collections import defaultdict


'''根据框子信息，计算限制的块Id'''


def RectLimitBatchId(rect, level, batchSize):
    scale = 2 ** level
    rect2 = rect / scale
    rect2 = 1. * rect2 / batchSize
    low = np.int32(rect2[0])
    # up = np.ceil(rect2[1]).astype(np.int32)
    # up = np.array(rect2[1]).astype(np.int32)  # 去除边缘块
    up = np.round(rect2[1]).astype(np.int32)
    return low, up


'''获取大数据格式每级别换算为512x512x512的块数'''


def GetBigBVLevelInfo(save_root, batchSize, smallSize, rectLs, redunSize, sampleXYZ, level):
    TrainDataSetFiterPath = join(save_root, "TrainDataSetFiter")
    os.makedirs(TrainDataSetFiterPath, exist_ok=True)
    BatchIdLsPath = join(TrainDataSetFiterPath, "BatchIdLs.txt")

    batchIdLs = set()
    for rect in rectLs:
        low, up = RectLimitBatchId(rect, level, batchSize)
        for nz in range(low[2], up[2], sampleXYZ[2]):
            for ny in range(low[1], up[1], sampleXYZ[1]):
                for nx in range(low[0], up[0], sampleXYZ[0]):
                    batchIdLs.add('%d_%d_%d' % (nx, ny, nz))
    SignBatchNumber = Util.SliceBatch(batchSize, smallSize, redunSize)
    batchIdLs = list(batchIdLs)
    with open(BatchIdLsPath, 'w') as f:
        for item in batchIdLs:
            f.write(item + '\n')
    print('Number of blocks:', len(batchIdLs) * len(SignBatchNumber))
    return len(batchIdLs) * len(SignBatchNumber)


'''直方图降 bin：将一维计数直方图合并到 target_bins 个 bin，减小存储与后续聚类内存/显存占用。'''
HIST_BINS = 0  # 0 表示不降 bin，直接保留原始直方图
# HIST_BINS = 256  # 0 表示不降 bin，直接保留原始直方图


def DownsampleHist(hist, target_bins=HIST_BINS):
    if target_bins <= 0:
        return hist
    L = hist.shape[0]
    if L <= target_bins:
        return hist
    # 用 cumsum 差分快速合并相邻 bin（等宽边界，不整除时由边界取整处理）
    hist = hist.astype(np.int64)
    edges = np.linspace(0, L, target_bins + 1).astype(np.int32)
    c = np.concatenate([[0], np.cumsum(hist)])
    return (c[edges[1:]] - c[edges[:-1]]).astype(np.int32)

'''大块取最小块数'''
MIN_NUMS = 1

def SignBvToHist(workQue, finishQue, errorQue, savePath, threId, bv_root, batchSize, smallSize, redunSize, level):
    add = join(bv_root, str(level))
    sliceInfo = Util.SliceBatch(batchSize, smallSize, redunSize)
    readObj = BVReader()
    totalHistLs = {}
    error_info = []
    while True:
        try:
            info = workQue.get(timeout=1)
            if len(info) == 0:
                np.save(join(savePath, '%s.npy' % str(threId).zfill(5)), totalHistLs)
                finishQue.put(-1)
                return
            name, bx, by, bz = info
            sz = bz * batchSize[2]
            error_info = [join(add, name), int(sz), int(batchSize[2])]
            img = readObj.readBV(join(add, name), int(sz), int(batchSize[2]))
            if not np.all(img.shape == batchSize[::-1]):
                img2 = np.zeros(batchSize[::-1], dtype=img.dtype)
                img2[:img.shape[0], :img.shape[1], :img.shape[2]] = img
                img = img2
            # 计算该大块内所有小块的直方图
            blockHists = []
            for sp, ep, nx, ny, nz in sliceInfo:
                smallImg = img[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
                hist = getHist(smallImg)
                hist2 = hist.T[0].astype(np.int32)
                hist2 = DownsampleHist(hist2)
                blockHists.append([hist2, nx, ny, nz])

            # 大块内部按直方图特征聚类，选出 2 个代表性小块（每个簇选离中心最近的）
            if len(blockHists) > MIN_NUMS:
                arr = np.array([h[0] for h in blockHists], dtype=np.float32)
                arr = np.log(arr + 1)
                cluster = KMeans(n_clusters=MIN_NUMS, random_state=0, n_init=10).fit(arr)
                keep = []
                for c in range(2):
                    idxs = np.where(cluster.labels_ == c)[0]
                    if len(idxs) == 0:
                        # 聚类退化导致某簇为空时，直接取前 2 个小块
                        keep = blockHists[:MIN_NUMS]
                        break
                    cen = cluster.cluster_centers_[c]
                    errs = np.sum(np.abs(arr[idxs] - cen), axis=1)
                    keep.append(blockHists[idxs[np.argmin(errs)]])
                blockHists = keep

            for hist2, nx, ny, nz in blockHists:
                HistSaveName = '%d_%d_%d-%d_%d_%d.npy' % (bx, by, bz, nx, ny, nz)
                totalHistLs[HistSaveName] = hist2
            finishQue.put(1)
        except Exception as e:
            print(error_info)
            if errorQue.qsize() == 0:
                print("\n=== Error message ===")
                print(f"Exception type: {type(e).__name__}")
                print(f"Error message: {e}")
                print("=== Error location ===")
                tb = sys.exc_info()[2]
                for frame in traceback.extract_tb(tb):
                    print(f"  File: {frame.filename}")
                    print(f"  Line number: {frame.lineno}")
                    print(f"  Function: {frame.name}")
                    print(f"  Code: {frame.line}\n")
            errorQue.put(e)


'''Bv大数据格式转灰度直方图'''


def BvBigDataToHist(bv_root, save_root, batchSize, smallSize, redunSize, level, MNumber):
    TrainDataSetFiterPath = join(save_root, "TrainDataSetFiter")
    BatchIdLsPath = join(TrainDataSetFiterPath, "BatchIdLs.txt")
    HistDataPath = join(TrainDataSetFiterPath, "HistData")
    if os.path.isdir(HistDataPath):
        shutil.rmtree(HistDataPath)
    os.makedirs(HistDataPath, exist_ok=True)
    # Parallel
    workQue, errorQue, finishQue = Queue(), Queue(), Queue()
    with open(BatchIdLsPath, 'r') as f:
        nameIdLs = f.read().strip().split('\n')
    workLen = len(nameIdLs)
    print("workLen：", workLen)
    for nameId in nameIdLs:
        t = nameId.split('_')
        x, y, z = int(t[0]), int(t[1]), int(t[2])
        workQue.put(['%d_%d.bv' % (x, y), x, y, z])
    for i in range(MNumber):
        workQue.put([])
    ps = []
    for i in range(MNumber):
        ps.append(Process(target=SignBvToHist, args=(workQue, finishQue, errorQue, HistDataPath, i,
                                                     bv_root, batchSize, smallSize, redunSize, level)))
        ps[-1].daemon = True
        ps[-1].start()
    sTime = time.time()
    for i in range(workLen):
        info = finishQue.get()
        if i % 50 == 0:
            userTime = time.time() - sTime
            surplusTime = userTime / (i + 1) * (workLen - i - 1)
            logInfo = '[Progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]\n' % (
                (i + 1) / workLen * 100, userTime, surplusTime)
            print('[Progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
            (i + 1) / workLen * 100, userTime, surplusTime))
    # 等待线程结束(很快)
    for i in range(MNumber):
        finishQue.get()
    return workLen


'''读取灰度直方图和梯度特征'''


def ReadHistAndGrad(path):
    print('Start reading data')
    ls = os.listdir(path)
    nameIdLs, histLs, gradLs = [], [], []
    for name in ls:
        t_data = np.load(join(path, name), allow_pickle=True).item()
        for key, item in t_data.items():
            nameIdLs.append(os.path.splitext(key)[0])
            histLs.append(item)
    return nameIdLs, histLs


'''直方图聚类'''


def HistKMean(histLs, n_clusters=20, labelAdd='./tmpData/labels.txt', cenAdd='./tmpData/center.txt'):
    print('PCA+Clustering')
    histLs = np.array(histLs)
    histLs[histLs < 0] = 0
    histLs = np.log(histLs + 1)
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(histLs)
    cluster_centers = cluster.cluster_centers_.astype(np.float16)
    labels_ = cluster.labels_.astype(np.int32)
    np.save(labelAdd, labels_)
    np.save(cenAdd, cluster_centers)


'''聚类结果计算图像IdLs'''


def ClusterToImgIdLs(histLs, labels, cluster_centers, ls, n_clusters, getClsNum, imgIdInfoPath):
    clsHistLs = [[] for i in range(n_clusters)]
    clsIdLs = [[] for i in range(n_clusters)]
    for hi, cls in enumerate(labels):
        clsHistLs[cls].append(histLs[hi])
        clsIdLs[cls].append(ls[hi])
    imgInfos = {}
    reKeyInfos = set()
    for cls, (histLs, idLs) in enumerate(zip(clsHistLs, clsIdLs)):
        if len(histLs) == 0:
            continue
        cenHist = cluster_centers[cls]
        errorLs = np.sum(np.abs(histLs - cenHist), axis=1)
        indLs = np.argsort(errorLs)
        space = indLs.size / (getClsNum + 1)
        for t in range(getClsNum):
            id = int(round(space * (t + 1)))
            if id >= len(indLs):
                continue
            tmpId = idLs[indLs[id]]
            tmpId = tmpId.split('-')
            key = tmpId[0]
            if key in imgInfos:
                bx, by, bz = np.array(key.split('_'), dtype=np.int32).tolist()
                bbx, bby, bbz = np.array(tmpId[1].split('_'), dtype=np.int32).tolist()
                reKey = '%d_%d_%d-%d_%d_%d' % (bx, by, bz, bbx, bby, bbz)
                if reKey in reKeyInfos:
                    continue
                imgInfos[key].append([bx, by, bz, bbx, bby, bbz, cls])
            else:
                bx, by, bz = np.array(key.split('_'), dtype=np.int32).tolist()
                bbx, bby, bbz = np.array(tmpId[1].split('_'), dtype=np.int32).tolist()
                reKey = '%d_%d_%d-%d_%d_%d' % (bx, by, bz, bbx, bby, bbz)
                if reKey in reKeyInfos:
                    continue
                imgInfos[key] = [[bx, by, bz, bbx, bby, bbz, cls]]
            reKeyInfos.add(reKey)
    with open(imgIdInfoPath, 'w') as f:
        f.write(json.dumps(imgInfos))
    return imgInfos


def HistFiterBv(HistDataPath, HistAnalyPath, n_clusters, getClsNum):
    TmpDataPath = join(HistAnalyPath, "TmpData")
    ImgIdLsPath = join(HistAnalyPath, "ImgIdLs.json")
    if os.path.isdir(HistAnalyPath):
        shutil.rmtree(HistAnalyPath)
    os.makedirs(TmpDataPath, exist_ok=True)
    cenAdd = join(TmpDataPath, "kkcent.npy")
    labelAdd = join(TmpDataPath, "kklabel.npy")
    # 开始读取数据
    nameIdLs, histLs = ReadHistAndGrad(HistDataPath)
    # PCA+Clustering
    HistKMean(histLs, n_clusters, labelAdd, cenAdd)
    labels = np.load(labelAdd)
    cluster_centers = np.load(cenAdd)
    ClusterToImgIdLs(histLs, labels, cluster_centers, nameIdLs, n_clusters, getClsNum, ImgIdLsPath)
    return ImgIdLsPath


def SignBvToTif(add, savePath, batchSize, workQue, finishQue, errorQue, smallBatchSize, redunSize2):
    zs = batchSize[2]
    batchSize = batchSize[:2]
    smallSize = smallBatchSize[:2]
    redunSize = redunSize2[:2]
    readObj = BVReader()
    while True:
        try:
            work = workQue.get(timeout=1)
            if len(work) == 0:
                finishQue.put(-1)
                return
            x, y, z, nx, ny, nz, cls = work
            sz = z * zs
            img = readObj.readBV(join(add, '%d_%d.bv' % (x, y)), sz, smallBatchSize[2])
            if img.shape[1] != batchSize[1] or img.shape[2] != batchSize[0]:
                img2 = np.zeros([img.shape[0], batchSize[1], batchSize[0]], dtype=img.dtype)
                img2[:img.shape[0], :img.shape[1], :img.shape[2]] = img
                img = img2
            sp = (smallSize - redunSize) * [nx, ny]
            ep = np.min([sp + smallSize, batchSize], axis=0)
            sp = np.max([np.min([sp, ep - smallSize], axis=0), [0, 0]], axis=0)
            tarImg = img[:, sp[1]: ep[1], sp[0]: ep[0]]
            s_id = '%s-%d_%d_%d-%d_%d_%d.tif' % (str(cls).zfill(4), x, y, z, nx, ny, nz)
            if np.max(tarImg) > 0:
                tifffile.imwrite(join(savePath, s_id), tarImg, compression='lzw')
            finishQue.put(1)
        except Exception as e:
            if errorQue.qsize() == 0:
                print("\n=== Error message ===")
                print(f"Exception type: {type(e).__name__}")
                print(f"Error message: {e}")
                print("=== Error location ===")
                tb = sys.exc_info()[2]
                for frame in traceback.extract_tb(tb):
                    print(f"  File: {frame.filename}")
                    print(f"  Line number: {frame.lineno}")
                    print(f"  Function: {frame.name}")
                    print(f"  Code: {frame.line}\n")
            errorQue.put(e)


def MulBatchLsToTifLs(bv_root, save_root, level, MNumber, batchSize, smallSize, redunSize, NeedFiterImgPath):
    TrainDataSetFiterPath = join(save_root, "TrainDataSetFiter")
    HistAnalyPath = join(TrainDataSetFiterPath, "HistAnaly")
    ImgIdLsPath = join(HistAnalyPath, "ImgIdLs.json")
    if os.path.isdir(NeedFiterImgPath):
        shutil.rmtree(NeedFiterImgPath)
    os.makedirs(NeedFiterImgPath, exist_ok=True)
    add = join(bv_root, str(level))
    # 读取Id Ls
    with open(ImgIdLsPath, 'r') as f:
        idLs = json.loads(f.read())

    workQue, finishQue, errorQue = Queue(), Queue(), Queue()
    for id in idLs:
        for it in idLs[id]:
            workQue.put(it)
    workLen = workQue.qsize()
    for i in range(MNumber):
        workQue.put([])
    ps = []
    for i in range(MNumber):
        ps.append(Process(target=SignBvToTif, args=(add, NeedFiterImgPath, batchSize, workQue, finishQue, errorQue,
                                                    smallSize, redunSize)))
        ps[-1].daemon = True
        ps[-1].start()
    sTime = time.time()
    for i in range(workLen):
        id = finishQue.get()
        if i % 10 == 0:
            userTime = time.time() - sTime
            surplusTime = userTime / (i + 1) * (workLen - i - 1)
            print('[Progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
            (i + 1) / workLen * 100, userTime, surplusTime))
            logInfo = '[Progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]\n' % (
                (i + 1) / workLen * 100, userTime, surplusTime)
    # 等待线程结束(很快)
    for i in range(MNumber):
        finishQue.get()

    if os.path.exists(TrainDataSetFiterPath):
        shutil.rmtree(TrainDataSetFiterPath, ignore_errors=True)


def round_up(num):
    if num - int(num) >= 0.5:
        return int(num) + 1
    else:
        return int(num)


'''根据 ImgIdLs.json 提取 tif 图像，写入 savePath（多份可共用一个 savePath 累加）。'''


def ExtractImgsByImgIdLs(ImgIdLsPath, add, savePath, batchSize, smallSize, redunSize, MNumber,
                         progress0=None, logger=None, error0=None):
    with open(ImgIdLsPath, 'r') as f:
        idLs = json.loads(f.read())

    workQue, finishQue, errorQue = Queue(), Queue(), Queue()
    for id in idLs:
        for it in idLs[id]:
            workQue.put(it)

    workLen = workQue.qsize()
    if progress0 is not None:
        progress0.emit(f"Clustering obtained {workLen} small data blocks", 0)
    for i in range(MNumber):
        workQue.put([])
    original_work_len = workLen

    ps = []
    for i in range(MNumber):
        ps.append(Process(target=SignBvToTif, args=(add, savePath, batchSize, workQue, finishQue, errorQue,
                                                    smallSize, redunSize)))
        ps[-1].daemon = True
        ps[-1].start()
    sTime = time.time()
    curMakeSize = 0
    finished_processes = 0

    while True:
        while finishQue.qsize() > 0:
            result = finishQue.get()
            if result == -1:
                finished_processes += 1
            elif result == 1:
                curMakeSize += 1

        if curMakeSize >= original_work_len:
            break

        if errorQue.qsize():
            e = errorQue.get()
            for p in ps:
                p.terminate()
            if logger is not None:
                logger.error("\n=== Error message ===")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Error message: {e}")
                logger.error("=== Error location ===")
                tb = sys.exc_info()[2]
                for frame in traceback.extract_tb(tb):
                    logger.error(f"  File: {frame.filename}")
                    logger.error(f"  Line number: {frame.lineno}")
                    logger.error(f"  Function: {frame.name}")
                    logger.error(f"  Code: {frame.line}\n")
            if error0 is not None:
                error0.emit(str(e))
            raise RuntimeError(str(e))

        if curMakeSize > 0:
            userTime = time.time() - sTime
            surplusTime = userTime / curMakeSize * (original_work_len - curMakeSize)
            logInfo = '[Image extraction progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                curMakeSize / original_work_len * 100, userTime, surplusTime)
            if progress0 is not None:
                progress0.emit(logInfo, curMakeSize)

        time.sleep(0.5)

    for p in ps:
        p.join(timeout=1)
        if p.is_alive():
            p.terminate()

    return workLen


'''
对指定 batch 块列表提取直方图特征，结果追加写入 HistDataPath。
threIdOffset 用于分批处理时保证不同批次进程保存的文件名不冲突。
'''


def ExtractHistFromBatchIds(bv_root, HistDataPath, nameIdLs, batchSize, smallSize, redunSize, level, MNumber,
                            threIdOffset, progress0=None, logger=None, error0=None):
    workQue, errorQue, finishQue = Queue(), Queue(), Queue()
    for nameId in nameIdLs:
        t = nameId.split('_')
        x, y, z = int(t[0]), int(t[1]), int(t[2])
        workQue.put(['%d_%d.bv' % (x, y), x, y, z])
    workLen = len(nameIdLs)
    for i in range(MNumber):
        workQue.put([])
    original_work_len = workLen

    ps = []
    for i in range(MNumber):
        ps.append(Process(target=SignBvToHist, args=(workQue, finishQue, errorQue, HistDataPath, threIdOffset + i,
                                                     bv_root, batchSize, smallSize, redunSize, level)))
        ps[-1].daemon = True
        ps[-1].start()
    sTime = time.time()
    curMakeSize = 0
    finished_processes = 0

    while True:
        while finishQue.qsize() > 0:
            result = finishQue.get()
            if result == -1:
                finished_processes += 1
            elif result == 1:
                curMakeSize += 1

        if curMakeSize >= original_work_len:
            break

        if errorQue.qsize():
            e = errorQue.get()
            for p in ps:
                p.terminate()
            if logger is not None:
                logger.error("\n=== Error message ===")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Error message: {e}")
            if error0 is not None:
                error0.emit(str(e))
            raise RuntimeError(str(e))

        if curMakeSize > 0:
            userTime = time.time() - sTime
            surplusTime = userTime / curMakeSize * (original_work_len - curMakeSize)
            logInfo = '[Feature extraction progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                curMakeSize / original_work_len * 100, userTime, surplusTime)
            if progress0 is not None:
                progress0.emit(logInfo, curMakeSize + 1)

        time.sleep(0.5)

    for p in ps:
        p.join(timeout=1)
        if p.is_alive():
            p.terminate()
    return workLen


def start_bv(divide_results_dir, img_dir, divide_img_dir, makerInfo,
             progress0=None, error0=None, division_ratio=None, logger=None, cfg_level=0, bv_ROI=None):
    dataNums = makerInfo['dataNums']
    MNumber = makerInfo['m_number']
    smallSize = makerInfo['small_size']  # xyz

    bv_root = img_dir
    save_root = divide_results_dir

    if "config.cfg" not in os.listdir(bv_root):
        res = {
            "error": 1,
            "text": ""
        }
        error0.emit("There is no config.cfg file in the input image folder, please select again!")
        return res, ""
    if "seqInfo.txt" not in os.listdir(bv_root):
        res = {
            "error": 1,
            "text": ""
        }
        error0.emit("There is no seqInfo.txt file in the input image folder, please select again!")
        return res, ""

    seqInfo_path = join(bv_root, "seqInfo.txt")
    with open(seqInfo_path, "r") as f:
        lines = f.readlines()
        seqInfo = [int(float(line)) for line in lines]  # xyz

    BigImgSize = np.array(seqInfo, dtype=np.int32)  # xyz
    rectLs = [np.array([np.array([0, 0, 0]), BigImgSize], dtype=np.int32)]

    if bv_ROI is not None:
        try:
            if not bv_ROI.any():  # 全为0， 取最大范围
                pass
            else:
                MinX, MaxX, MinY, MaxY, MinZ, MaxZ = bv_ROI
                print(bv_ROI)
                small_size = [512, 512, 512]
                if (MaxX - MinX < small_size[0]
                        or MaxY - MinY < small_size[1]
                        or MaxZ - MinZ < small_size[2]):
                    res = {
                        "error": 1,
                        "text": ""
                    }
                    error0.emit(
                        f"The minimum size (Max-Min) of the ROI region in BV format cannot be less than [x, y, z]={small_size}. Please reset the ROI range!")
                    return res, ""

                MinX = int(max(MinX, 0))
                MinY = int(max(MinY, 0))
                MinZ = int(max(MinZ, 0))
                MaxX = int(min(MaxX, seqInfo[0]))
                MaxY = int(min(MaxY, seqInfo[1]))
                MaxZ = int(min(MaxZ, seqInfo[2]))

                if (MaxX - MinX < small_size[0]
                        or MaxY - MinY < small_size[1]
                        or MaxZ - MinZ < small_size[2]):
                    res = {
                        "error": 1,
                        "text": ""
                    }
                    error0.emit(
                        f"There is no valid area or it is smaller than [x, y, z]={small_size}, "
                        f"unable to Make. Please reset the ROI range!")
                    return res, ""

                BigImgSize = np.array([MaxX - MinX, MaxY - MinY, MaxZ - MinZ], dtype=np.int32)  # xyz
                rectLs = [np.array([np.array([MinX, MinY, MinZ]),
                                    np.array([MaxX, MaxY, MaxZ])], dtype=np.int32)]
                print(rectLs)
        except Exception as e:
            pass

    redunSize = np.array([32, 32, 32], dtype=np.int32)  # xyz
    sampleXYZ = np.array([1, 1, 1], dtype=np.int32)  # xyz
    batchSize_x = np.min([512, BigImgSize[0]])
    batchSize_y = np.min([512, BigImgSize[1]])
    batchSize_z = smallSize[2]
    batchSize = np.array([batchSize_x, batchSize_y, batchSize_z], dtype=np.int32)  # xyz
    level = cfg_level

    '''查看块信息'''

    TrainDataSetFiterPath = join(save_root, "TrainDataSetFiter")
    os.makedirs(TrainDataSetFiterPath, exist_ok=True)
    BatchIdLsPath = join(TrainDataSetFiterPath, "BatchIdLs.txt")

    batchIdLs = set()
    SignBatchNumber = Util.SliceBatch(batchSize, smallSize, redunSize)
    SignBatchNumber = min(MIN_NUMS, len(SignBatchNumber))  # 每个大块只提取 2 个小块
    for rect in rectLs:
        low, up = RectLimitBatchId(rect, level, batchSize)
        up = up + (up == 0) * 1  # 防止只有一块小于batchSize的数据块丢失
        up_add = up * np.array([round(batchSize_x / smallSize[0]),
                                round(batchSize_y / smallSize[1]), 1])  # 按比例扩充xy方向尺寸
        res = np.ceil(np.array(up_add) / np.array(sampleXYZ))
        res_sum = np.prod(res)
        print(f"Spacing {sampleXYZ}, Total blocks: ", res_sum)
        for nz in range(low[2], up[2], sampleXYZ[2]):
            for ny in range(low[1], up[1], sampleXYZ[1]):
                for nx in range(low[0], up[0], sampleXYZ[0]):
                    batchIdLs.add('%d_%d_%d' % (nx, ny, nz))

    batchIdLs = list(batchIdLs)
    with open(BatchIdLsPath, 'w') as f:
        for item in batchIdLs:
            f.write(item + '\n')
    counts = len(batchIdLs) * SignBatchNumber
    print(f"{len(batchIdLs)} * {SignBatchNumber}")

    '''提取每一块特征'''

    BatchIdLsPath = join(TrainDataSetFiterPath, "BatchIdLs.txt")
    HistDataPath = join(TrainDataSetFiterPath, "HistData")
    if os.path.isdir(HistDataPath):
        shutil.rmtree(HistDataPath)
    os.makedirs(HistDataPath, exist_ok=True)
    progress0.emit(f"Start filtering {counts} large data blocks", 0)
    progress0.emit(f"Loading process", 0)

    # 一次性提取全部 batch 块的特征
    try:
        ExtractHistFromBatchIds(bv_root, HistDataPath, batchIdLs, batchSize, smallSize, redunSize, level, MNumber,
                                0, progress0, logger, error0)
    except RuntimeError:
        res = {
            "error": 1,
            "text": ""
        }
        return res, ""

    # 设置聚类和提取类数量
    Cluster_Nums = dataNums
    getClsNum = 1  # 每类只取 1 个块
    n_clusters = int(Cluster_Nums / getClsNum)
    # n_clusters = int(Cluster_Nums / getClsNum) + 1
    if counts < n_clusters:
        print(f"counts {counts} < n_clusters {n_clusters}")
        n_clusters = counts
        getClsNum = 1
    print(Cluster_Nums, getClsNum, n_clusters)

    '''Clustering'''

    progress0.emit(f"Starting dataset clustering analysis!", 0)
    HistAnalyPath = join(TrainDataSetFiterPath, "HistAnaly")
    ImgIdLsPath = HistFiterBv(HistDataPath, HistAnalyPath, n_clusters, getClsNum)

    '''聚类结果提取图像'''

    add = join(bv_root, str(level))
    if os.path.isdir(divide_img_dir):
        shutil.rmtree(divide_img_dir)
    os.makedirs(divide_img_dir, exist_ok=True)
    try:
        ExtractImgsByImgIdLs(ImgIdLsPath, add, divide_img_dir, batchSize, smallSize, redunSize, MNumber,
                             progress0, logger, error0)
    except RuntimeError:
        res = {
            "error": 1,
            "text": ""
        }
        return res, ""

    # 提取块数
    filter_amount = len([n for n in os.listdir(divide_img_dir) if ".tif" in n])

    progress0.emit(f"Removing blank images, starting dataset allocation for {filter_amount} data blocks", 0)

    division_text = "Dataset allocation:"
    if division_ratio is not None:
        setNameLs = ['train', 'val', 'test']
        spaceLs = [0, int(filter_amount * division_ratio[0]),
                   int(filter_amount * (division_ratio[0] + division_ratio[1])), filter_amount]
        for ii in range(len(setNameLs)):
            division_text += f"{setNameLs[ii]}:{spaceLs[ii + 1] - spaceLs[ii]} blocks, "
    res = {
        "error": 0,
        "text": f"{filter_amount} / {counts}"
    }

    if os.path.exists(TrainDataSetFiterPath):
        shutil.rmtree(TrainDataSetFiterPath, ignore_errors=True)

    return res, division_text

