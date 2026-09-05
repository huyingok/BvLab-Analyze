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
from BVExample.BVMoudle import getHist
from sklearn.cluster import KMeans
import json
import sys
import zarr
import traceback
import tifffile


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


'''获取Zarr大数据格式每级别换算为512x512x512的块数'''


def GetBigZarrLevelInfo(save_root, batchSize, smallSize, rectLs, redunSize, sampleXYZ, level):
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


def SignZarrToHist(workQue, finishQue, errorQue, savePath, threId, zarr_root, batchSize, smallSize, redunSize, level):
    zarr_store = zarr.open(zarr_root, mode='r')
    zarr_data = zarr_store[str(level)]
    sliceInfo = Util.SliceBatch(batchSize, smallSize, redunSize)
    totalHistLs = {}
    error_info = []
    while True:
        try:
            info = workQue.get(timeout=1)
            if len(info) == 0:
                np.save(join(savePath, '%s.npy' % str(threId).zfill(5)), totalHistLs)
                finishQue.put(-1)
                return
            bx, by, bz = info
            x_start = bx * batchSize[0]
            x_end = x_start + batchSize[0]
            y_start = by * batchSize[1]
            y_end = y_start + batchSize[1]
            z_start = bz * batchSize[2]
            z_end = z_start + batchSize[2]
            
            error_info = [f'Block ({bx}, {by}, {bz})', z_start, z_end, y_start, y_end, x_start, x_end]
            
            img = zarr_data[z_start:z_end, y_start:y_end, x_start:x_end]
            
            target_shape = batchSize[::-1]
            if not np.all(img.shape == target_shape):
                img2 = np.zeros(target_shape, dtype=img.dtype)
                img2[:img.shape[0], :img.shape[1], :img.shape[2]] = img
                img = img2
            for sp, ep, nx, ny, nz in sliceInfo:
                smallImg = img[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
                hist = getHist(smallImg)
                hist2 = hist.T[0].astype(np.int32)
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


'''Zarr大数据格式转灰度直方图'''


def ZarrBigDataToHist(zarr_root, save_root, batchSize, smallSize, redunSize, level, MNumber):
    TrainDataSetFiterPath = join(save_root, "TrainDataSetFiter")
    BatchIdLsPath = join(TrainDataSetFiterPath, "BatchIdLs.txt")
    HistDataPath = join(TrainDataSetFiterPath, "HistData")
    if os.path.isdir(HistDataPath):
        shutil.rmtree(HistDataPath)
    os.makedirs(HistDataPath, exist_ok=True)
    workQue, errorQue, finishQue = Queue(), Queue(), Queue()
    with open(BatchIdLsPath, 'r') as f:
        nameIdLs = f.read().strip().split('\n')
    workLen = len(nameIdLs)
    print("workLen：", workLen)
    for nameId in nameIdLs:
        t = nameId.split('_')
        x, y, z = int(t[0]), int(t[1]), int(t[2])
        workQue.put([x, y, z])
    for i in range(MNumber):
        workQue.put([])
    ps = []
    for i in range(MNumber):
        ps.append(Process(target=SignZarrToHist, args=(workQue, finishQue, errorQue, HistDataPath, i,
                                                     zarr_root, batchSize, smallSize, redunSize, level)))
        ps[-1].daemon = True
        ps[-1].start()
    sTime = time.time()
    for i in range(workLen):
        info = finishQue.get()
        if i % 50 == 0:
            userTime = time.time() - sTime
            surplusTime = userTime / (i + 1) * (workLen - i - 1)
            print('[Progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
            (i + 1) / workLen * 100, userTime, surplusTime))
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
        # histLs.append(np.load(join(path, name)))
        # nameIdLs.append(os.path.splitext(name)[0])
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
    # nums_sum = 0
    # limit_num = int(dataNums / n_clusters)
    for cls, (histLs, idLs) in enumerate(zip(clsHistLs, clsIdLs)):
        # nums = 0
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
                # nums += 1
            else:
                bx, by, bz = np.array(key.split('_'), dtype=np.int32).tolist()
                bbx, bby, bbz = np.array(tmpId[1].split('_'), dtype=np.int32).tolist()
                reKey = '%d_%d_%d-%d_%d_%d' % (bx, by, bz, bbx, bby, bbz)
                if reKey in reKeyInfos:
                    continue
                imgInfos[key] = [[bx, by, bz, bbx, bby, bbz, cls]]
                # nums += 1
            reKeyInfos.add(reKey)
        #     if nums > limit_num:
        #         nums_sum += nums
        #         break
        # if not (nums > limit_num):
        #     nums_sum += nums
    with open(imgIdInfoPath, 'w') as f:
        f.write(json.dumps(imgInfos))
    return imgInfos


def HistFiterZarr(save_root, n_clusters, getClsNum):
    TrainDataSetFiterPath = join(save_root, "TrainDataSetFiter")
    HistDataPath = join(TrainDataSetFiterPath, "HistData")
    HistAnalyPath = join(TrainDataSetFiterPath, "HistAnaly")
    TmpDataPath = join(HistAnalyPath, "TmpData")
    ImgIdLsPath = join(HistAnalyPath, "ImgIdLs.json")
    if os.path.isdir(HistAnalyPath):
        shutil.rmtree(HistAnalyPath)
    os.makedirs(TmpDataPath, exist_ok=True)
    cenAdd = join(TmpDataPath, "kkcent.npy")
    labelAdd = join(TmpDataPath, "kklabel.npy")
    nameIdLs, histLs = ReadHistAndGrad(HistDataPath)
    HistKMean(histLs, n_clusters, labelAdd, cenAdd)
    labels = np.load(labelAdd)
    cluster_centers = np.load(cenAdd)
    ClusterToImgIdLs(histLs, labels, cluster_centers, nameIdLs, n_clusters, getClsNum, ImgIdLsPath)
    return ImgIdLsPath


def SignZarrToTif(zarr_root, savePath, batchSize, workQue, finishQue, errorQue, smallBatchSize, redunSize2, level):
    zarr_store = zarr.open(zarr_root, mode='r')
    zarr_data = zarr_store[str(level)]
    
    zs = batchSize[2]
    batchSize_xy = batchSize[:2]
    smallSize = smallBatchSize[:2]
    redunSize = redunSize2[:2]
    while True:
        try:
            work = workQue.get(timeout=1)
            if len(work) == 0:
                finishQue.put(-1)
                return
            x, y, z, nx, ny, nz, cls = work
            
            x_start = x * batchSize[0]
            x_end = x_start + batchSize[0]
            y_start = y * batchSize[1]
            y_end = y_start + batchSize[1]
            z_start = z * zs
            z_end = z_start + smallBatchSize[2]
            
            img = zarr_data[z_start:z_end, y_start:y_end, x_start:x_end]
            
            if img.shape[0] != zs or img.shape[1] != batchSize_xy[1] or img.shape[2] != batchSize_xy[0]:
                img2 = np.zeros([zs, batchSize_xy[1], batchSize_xy[0]], dtype=img.dtype)
                img2[:img.shape[0], :img.shape[1], :img.shape[2]] = img
                img = img2
            sp = (smallSize - redunSize) * [nx, ny]
            ep = np.min([sp + smallSize, batchSize_xy], axis=0)
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


def ZarrBatchLsToTifLs(zarr_root, save_root, level, MNumber, batchSize, smallSize, redunSize, NeedFiterImgPath):
    TrainDataSetFiterPath = join(save_root, "TrainDataSetFiter")
    HistAnalyPath = join(TrainDataSetFiterPath, "HistAnaly")
    ImgIdLsPath = join(HistAnalyPath, "ImgIdLs.json")
    if os.path.isdir(NeedFiterImgPath):
        shutil.rmtree(NeedFiterImgPath)
    os.makedirs(NeedFiterImgPath, exist_ok=True)
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
        ps.append(Process(target=SignZarrToTif, args=(zarr_root, NeedFiterImgPath, batchSize, workQue, finishQue, errorQue,
                                                      smallSize, redunSize, level)))
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
    for i in range(MNumber):
        finishQue.get()

    if os.path.exists(TrainDataSetFiterPath):
        shutil.rmtree(TrainDataSetFiterPath, ignore_errors=True)


def round_up(num):
    if num - int(num) >= 0.5:
        return int(num) + 1
    else:
        return int(num)


def start_zarr(divide_results_dir, img_dir, divide_img_dir, makerInfo,
               progress0=None, error0=None, division_ratio=None, logger=None, cfg_level=0):
    dataNums = makerInfo['dataNums']
    MNumber = makerInfo['m_number']
    smallSize = makerInfo['small_size']  # xyz

    zarr_root = img_dir
    save_root = divide_results_dir

    file_names = [n for n in os.listdir(zarr_root) if "." not in n]

    if ".ome.zarr" not in img_dir.lower():  # .ome.zarr 格式数据
        res = {
            "error": 1,
            "text": ""
        }
        error0.emit("Not an OME-Zarr format data folder, please select again!")
        return res, ""

    if len(file_names) == 0:
        res = {
            "error": 1,
            "text": ""
        }
        error0.emit("No OME-Zarr format files in the image folder, please select again!")
        return res, ""

    zarr_store = zarr.open(zarr_root, mode='r')
    # shapes = zarr_store[f'{cfg_level}'].shape  # (z,y,x)
    shapes = zarr_store['0'].shape  # (z,y,x)
    seqInfo = np.array(shapes, dtype=np.int32)[::-1]  # xyz
    BigImgSize = np.array(seqInfo, dtype=np.int32)  # xyz
    rectLs = [np.array([np.array([0, 0, 0]), BigImgSize], dtype=np.int32)]
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
    MIN_Counts = 20000
    batchIdLs = set()
    SignBatchNumber = Util.SliceBatch(batchSize, smallSize, redunSize)
    SignBatchNumber = len(SignBatchNumber)
    for rect in rectLs:
        low, up = RectLimitBatchId(rect, level, batchSize)
        up = up + (up == 0) * 1  # 防止只有一块小于batchSize的数据块丢失
        up_add = up * np.array([round(batchSize_x / smallSize[0]),
                                round(batchSize_y / smallSize[1]), 1])  # 按比例扩充xy方向尺寸
        res = np.ceil(np.array(up_add) / np.array(sampleXYZ))
        res_sum = np.prod(res)
        if res_sum > MIN_Counts:
            res = {
                "error": 0,
                "text": ""
            }
            return res, "No sampleXYZ"
        BigImgSize_xyz = np.array(BigImgSize / np.min(BigImgSize), dtype=np.int32)
        new_BigImgSize_xyz = BigImgSize_xyz - 1
        sum_num = np.sum(BigImgSize_xyz)
        old_sampleXYZ = sampleXYZ
        print(f"Spacing {sampleXYZ}, Total blocks: ", res_sum)
        i = -1
        while res_sum > MIN_Counts:  # 限制提取特征块数
            i += 1
            if i == 0:
                sampleXYZ = new_BigImgSize_xyz + (BigImgSize_xyz == 1)
            else:
                sampleXYZ = (BigImgSize_xyz / sum_num) * (sum_num + i)
                sampleXYZ = sampleXYZ.astype(np.int32)
                if np.max(np.abs(old_sampleXYZ - sampleXYZ)) == 0:
                    continue
                old_sampleXYZ = sampleXYZ
            if (np.max(np.abs(BigImgSize_xyz - sampleXYZ)) == 0 and i == 0) or (
                    np.max(np.abs(sampleXYZ - np.array([1, 1, 1]))) == 0):
                continue
            res = np.ceil(np.array(up_add) / sampleXYZ)
            res_sum = np.prod(res)
            print(f"After adjusting spacing {sampleXYZ}, total blocks: ", res_sum)
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
    # Parallel
    workQue, finishQue, errorQue = Queue(), Queue(), Queue()
    with open(BatchIdLsPath, 'r') as f:
        nameIdLs = f.read().strip().split('\n')
    workLen = len(nameIdLs)
    # print("workLen：", workLen)
    progress0.emit(f"Start filtering {counts} large data blocks", 0)
    progress0.emit(f"Loading process", 0)
    for nameId in nameIdLs:
        t = nameId.split('_')
        x, y, z = int(t[0]), int(t[1]), int(t[2])
        workQue.put([x, y, z])
    for i in range(MNumber):
        workQue.put([])
    original_work_len = workLen

    ps = []
    for i in range(MNumber):
        ps.append(Process(target=SignZarrToHist, args=(workQue, finishQue, errorQue, HistDataPath, i,
                                                       zarr_root, batchSize, smallSize, redunSize, level)))
        ps[-1].daemon = True
        ps[-1].start()
    sTime = time.time()
    curMakeSize = 0
    finished_processes = 0

    while True:
        # 检查进程是否全部完成
        while finishQue.qsize() > 0:
            result = finishQue.get()
            if result == -1:
                finished_processes += 1
            elif result == 1:
                curMakeSize += 1

        # 所有实际工作任务完成
        if curMakeSize >= original_work_len:
            break

        # 进程报错
        if errorQue.qsize():
            e = errorQue.get()
            for p in ps:
                p.terminate()
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
            res = {
                "error": 1,
                "text": ""
            }
            error0.emit(str(e))
            return res, ""

        # 更新进度
        if curMakeSize > 0:
            userTime = time.time() - sTime
            surplusTime = userTime / curMakeSize * (original_work_len - curMakeSize)
            logInfo = '[Feature extraction progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                curMakeSize / original_work_len * 100, userTime, surplusTime)
            if progress0 is not None:
                progress0.emit(logInfo, curMakeSize + 1)

        time.sleep(0.5)

    # 等待所有进程完成
    for p in ps:
        p.join(timeout=1)  # 最多等待5Second
        if p.is_alive():
            p.terminate()

    for i, p in enumerate(ps):
        if i == 0:
            userTime = time.time() - sTime
            logInfo = '[Feature extraction progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                100, userTime, 0)
            if progress0 is not None:
                progress0.emit(logInfo, i + 1)
        logInfo = '[End process progress %.2f%%]' % (
                (i + 1) / MNumber * 100)
        progress0.emit(logInfo, i)
        time.sleep(0.2)

    # n_clusters = min(dataNums, workLen)
    # getClsNum = 1

    Cluster_Nums = dataNums
    getClsNum = 3
    n_clusters = int(Cluster_Nums / getClsNum) + 1
    if counts < n_clusters:
        print(f"counts {counts} < n_clusters {n_clusters}")
        n_clusters = counts
        getClsNum = 1
    print(Cluster_Nums, getClsNum, n_clusters)

    '''Clustering'''

    progress0.emit(f"Starting dataset clustering analysis!", 0)
    HistFiterZarr(save_root, n_clusters, getClsNum)

    '''聚类结果提取图像'''

    HistAnalyPath = join(TrainDataSetFiterPath, "HistAnaly")
    ImgIdLsPath = join(HistAnalyPath, "ImgIdLs.json")
    if os.path.isdir(divide_img_dir):
        shutil.rmtree(divide_img_dir)
    os.makedirs(divide_img_dir, exist_ok=True)
    # 读取Id Ls
    with open(ImgIdLsPath, 'r') as f:
        idLs = json.loads(f.read())

    workQue, finishQue, errorQue = Queue(), Queue(), Queue()
    for id in idLs:
        for it in idLs[id]:
            workQue.put(it)

    workLen = workQue.qsize()
    progress0.emit(f"Clustering obtained {workLen} small data blocks", 0)
    for i in range(MNumber):
        workQue.put([])
    original_work_len = workLen

    ps = []
    for i in range(MNumber):
        ps.append(Process(target=SignZarrToTif, args=(zarr_root, divide_img_dir, batchSize, workQue, finishQue, errorQue,
                                                      smallSize, redunSize, level)))
        ps[-1].daemon = True
        ps[-1].start()
    sTime = time.time()
    curMakeSize = 0
    finished_processes = 0

    while True:
        # 检查进程是否全部完成
        while finishQue.qsize() > 0:
            result = finishQue.get()
            if result == -1:
                finished_processes += 1
            elif result == 1:
                curMakeSize += 1

        # 所有实际工作任务完成
        if curMakeSize >= original_work_len:
            break

        # 进程报错
        if errorQue.qsize():
            e = errorQue.get()
            for p in ps:
                p.terminate()
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
            res = {
                "error": 1,
                "text": ""
            }
            error0.emit(str(e))
            return res, ""

        # 更新进度
        if curMakeSize > 0:
            userTime = time.time() - sTime
            surplusTime = userTime / curMakeSize * (original_work_len - curMakeSize)
            logInfo = '[Image extraction progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                curMakeSize / original_work_len * 100, userTime, surplusTime)
            if progress0 is not None:
                progress0.emit(logInfo, curMakeSize)

        time.sleep(0.5)

    # 等待所有进程完成
    for p in ps:
        p.join(timeout=1)  # 最多等待5Second
        if p.is_alive():
            p.terminate()

    for i, p in enumerate(ps):
        if i == 0:
            userTime = time.time() - sTime
            logInfo = '[Image extraction progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                100, userTime, 0)
            if progress0 is not None:
                progress0.emit(logInfo, i + 1)
        logInfo = '[End process progress %.2f%%]' % (
                (i + 1) / MNumber * 100)
        progress0.emit(logInfo, i)
        time.sleep(0.2)

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


if __name__ == '__main__':
    zarr_root = r"D:\SY\cell_data\data_zarr"
    # save_root = r"D:\SY\cell_data\data_zarr_save\MakeResults\DivideResults"
    # os.makedirs(save_root, exist_ok=True)
    # divide_img_dir = join(save_root, "images")
    #
    # zarr_store = zarr.open(zarr_root, mode='r')
    # shapes = zarr_store['0'].shape  # (z,y,x)
    # seqInfo = np.array(shapes, dtype=np.int32)[::-1]  # xyz
    # BigImgSize = np.array(seqInfo, dtype=np.int32)  # xyz
    # rectLs = [np.array([np.array([0, 0, 0]), BigImgSize], dtype=np.int32)]
    # redunSize = np.array([0, 0, 0], dtype=np.int32)  # xyz
    # sampleXYZ = np.array([2, 2, 1], dtype=np.int32)  # xyz
    # level = 0
    #
    # smallSize = np.array([272, 272, 144], dtype=np.int32)  # xyz
    # batchSize = np.array([512, 512, 144], dtype=np.int32)  # xyz
    # MNumber = 5
    # dataNums = 100
    # # 查看块信息
    # GetBigZarrLevelInfo(save_root, batchSize, smallSize, rectLs, redunSize, sampleXYZ, level)
    # # 提取每一块特征
    # workLen = ZarrBigDataToHist(zarr_root, save_root, batchSize, smallSize, redunSize, level, MNumber)
    # # 设置聚类和提取类数量
    # n_clusters = min(20, workLen)
    # getClsNum = n_clusters
    # # Clustering
    # HistFiterZarr(save_root, n_clusters, getClsNum, dataNums)
    # # 聚类结果提取图像
    # ZarrBatchLsToTifLs(zarr_root, save_root, level, MNumber, batchSize, smallSize, redunSize, divide_img_dir)
