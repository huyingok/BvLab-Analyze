# -*- coding: utf-8 -*-
import os
from os.path import join
import tifffile
import numpy as np
from BVExample.BVMoudle import getHist
from libs import Util
import shutil
from sklearn.cluster import KMeans
import json
from multiprocessing import Process, Queue
import time
import sys
from pathlib import Path
import traceback


'''读取灰度直方图和梯度特征'''


def ReadHistAndGrad(path):
    # print('开始读取数据')
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
    # print('PCA+Clustering')
    histLs = np.array(histLs)
    print(len(histLs))
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


def kmeans_analysis(saveRoot, path, n_clusters, getClsNum):
    # path D:\python+vtk+Qt\cell\data1\save/TrainDataSetFiter/HistData
    # Clustering
    savePath = join(saveRoot, 'TrainDataSetFiter/HistAnaly')  # D:\BaiduNetdiskDownload\data\mcherry\EGFP\tif_save/TrainDataSetFiter/HistAnaly

    tmpPath = join(savePath, 'TmpData')  # D:\BaiduNetdiskDownload\data\mcherry\EGFP\tif_save/TrainDataSetFiter/HistAnaly/TmpData
    imgIdInfoPath = join(savePath, 'ImgIdLs.json')  # D:\BaiduNetdiskDownload\data\mcherry\EGFP\tif_save/TrainDataSetFiter/HistAnaly/ImgIdLs.json
    if os.path.isdir(savePath):
        shutil.rmtree(savePath)
    os.makedirs(tmpPath, exist_ok=True)

    cenAdd = join(tmpPath, 'kkcent.npy')  # D:\BaiduNetdiskDownload\data\mcherry\EGFP\tif_save/TrainDataSetFiter/HistAnaly/TmpData/kkcent.npy
    labelAdd = join(tmpPath, 'kklabel.npy')  # D:\BaiduNetdiskDownload\data\mcherry\EGFP\tif_save/TrainDataSetFiter/HistAnaly/TmpData/kklabel.npy

    nameIdLs, histLs = ReadHistAndGrad(path)
    HistKMean(histLs, n_clusters, labelAdd, cenAdd)
    labels = np.load(labelAdd)
    cluster_centers = np.load(cenAdd)

    ClusterToImgIdLs(histLs, labels, cluster_centers, nameIdLs, n_clusters, getClsNum, imgIdInfoPath)

    return imgIdInfoPath


def tif_to_hist(imgDir, workQue, finishQue, errorQue, HistData_savePath, threId):
    totalHistLs = {}
    while True:
        try:
            # try:
            info = workQue.get(timeout=1)
            # except Exception as e:
            #     np.save(join(HistData_savePath, '%s.npy' % str(threId).zfill(5)), totalHistLs)
            #     return
            if len(info) == 0:
                np.save(join(HistData_savePath, '%s.npy' % str(threId).zfill(5)), totalHistLs)
                finishQue.put(-1)
                return
            # print(1 / 0)
            imgName = info
            imgPath = join(str(imgDir), imgName)
            img = tifffile.imread(imgPath)
            img = np.array(img, dtype=np.int32)
            # if np.max(img) != 0:
            hist = getHist(img)
            hist2 = hist.T[0].astype(np.int32)
            # print(imgName)
            names = imgName.split('.')[0].split("-")
            HistSaveName = '%s-%s.npy' % (names[0], names[1])
            # print(HistSaveName)
            totalHistLs[HistSaveName] = hist2
            # finishQue.put(imgName)
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


def hist_to_tif(imgDir, cut_swc_dir, imgNames, savePath, divide_swc_dir, workQue, finishQue):
    while True:
        work = workQue.get(timeout=1)
        if len(work) == 0:
            finishQue.put(-1)
            return
        x, y, z, nx, ny, nz, cls = work

        n_stem = "%d_%d_%d-%d_%d_%d" % (x, y, z, nx, ny, nz)
        names = [n for n in imgNames if n_stem in n]
        tif_id = '%s-%d_%d_%d-%d_%d_%d.tif' % (str(cls).zfill(4), x, y, z, nx, ny, nz)
        swc_id = '%s-%d_%d_%d-%d_%d_%d.swc' % (str(cls).zfill(4), x, y, z, nx, ny, nz)

        if len(names):
            img_path = join(str(imgDir), names[0])
            if os.path.exists(img_path):
                new_path = join(savePath, tif_id)
                shutil.copy(img_path, new_path)

                swc_path = join(str(cut_swc_dir), str(Path(names[0]).stem) + ".swc")
                new_swc_path = join(divide_swc_dir, swc_id)
                if os.path.exists(swc_path):
                    shutil.copy(swc_path, new_swc_path)
                else:
                    with open(new_swc_path, "w") as f:
                        pass
        finishQue.put(tif_id)


def start_one(divide_results_dir, cut_img_dir, cut_swc_dir, divide_img_dir, divide_swc_dir, makerInfo,
              progress0=None, error0=None, division_ratio=None, logger=None):
    # n_clusters = 10
    # getClsNum = 5
    # MNumber = 5

    dataNums = makerInfo['dataNums']
    MNumber = makerInfo['m_number']

    """提取每一块特征"""
    Train_savePath = join(divide_results_dir, 'TrainDataSetFiter')

    del_savePath = Train_savePath
    os.makedirs(Train_savePath, exist_ok=True)

    HistData_savePath = join(Train_savePath, 'HistData')
    imgNames = [l for l in os.listdir(cut_img_dir) if ".tif" in l]
    n_step = 2
    while len(imgNames) > 50000:
        imgNames = [imgNames[n] for n in range(0, len(imgNames), n_step)]
        n_step += 1

    if os.path.isdir(HistData_savePath):
        shutil.rmtree(HistData_savePath)
    os.makedirs(HistData_savePath, exist_ok=True)
    # Parallel
    workQue, finishQue, errorQue = Queue(), Queue(), Queue()
    workLen = len(imgNames)
    # 设置聚类和提取类数量
    if dataNums < 500:
        n_clusters = min(50, workLen)
    else:
        n_clusters = min(500, workLen)
    getClsNum = 20

    # n_clusters = min(dataNums, workLen)
    # getClsNum = 1

    # Cluster_Nums = min(dataNums, workLen) * 2
    # getClsNum = 4
    # n_clusters = max(int(Cluster_Nums / getClsNum), 1)

    if progress0 is not None:
        progress0.emit(f"Start filtering {workLen} small data blocks", 0)
        progress0.emit(f"Loading process", 0)
    for imgName in imgNames:
        workQue.put(imgName)

    for i in range(MNumber):
        workQue.put("")
    # workLen += MNumber
    original_work_len = workLen

    ps = []
    for i in range(MNumber):
        ps.append(Process(target=tif_to_hist, args=(cut_img_dir, workQue, finishQue, errorQue,
                                                    HistData_savePath, i,)))
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

    # 等待所有进程完成
    for i, p in enumerate(ps):
        if i == 0:
            userTime = time.time() - sTime
            logInfo = '[Feature extraction progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                100, userTime, 0)
            if progress0 is not None:
                progress0.emit(logInfo, 1)
        logInfo = '[End process progress %.2f%%]' % (
                (i + 1) / MNumber * 100)
        if progress0 is not None:
            progress0.emit(logInfo, i)
        time.sleep(0.2)

    """Clustering"""

    if progress0 is not None:
        progress0.emit(f"Starting dataset clustering analysis!", 0)
    imgIdInfoPath = kmeans_analysis(divide_results_dir, HistData_savePath, n_clusters, getClsNum)

    # 读取Id Ls
    with open(imgIdInfoPath, 'r') as f:
        idLs = json.loads(f.read())

    # 提取块数
    filter_amount = 0
    for i in idLs:
        filter_amount += len(idLs[i])

    if progress0 is not None:
        progress0.emit(f"Clustering obtained {filter_amount} small data blocks", 0)
        # print("Clustering")

    division_text = "Dataset allocation:"
    if division_ratio is not None:
        setNameLs = ['train', 'val', 'test']
        spaceLs = [0, int(filter_amount * division_ratio[0]),
                   int(filter_amount * (division_ratio[0] + division_ratio[1])), filter_amount]
        for ii in range(len(setNameLs)):
            division_text += f"{setNameLs[ii]}:{spaceLs[ii + 1] - spaceLs[ii]} blocks, "
    res = {
        "error": 0,
        "text": f"{filter_amount} / {len(imgNames)}"
    }

    """聚类结果提取图像"""

    savePath = divide_img_dir

    if os.path.isdir(savePath):
        shutil.rmtree(savePath)
    os.makedirs(savePath, exist_ok=True)

    workList = []
    for idv in idLs:
        for it in idLs[idv]:
            workQue.put(it)
            workList.append(it)
    workLen = len(workList)
    sTime = time.time()
    for j, work in enumerate(workList):
        x, y, z, nx, ny, nz, cls = work
        n_stem = "%d_%d_%d-%d_%d_%d" % (x, y, z, nx, ny, nz)
        names = [n for n in imgNames if n_stem in n]
        tif_id = '%s-%d_%d_%d-%d_%d_%d.tif' % (str(cls).zfill(4), x, y, z, nx, ny, nz)
        swc_id = '%s-%d_%d_%d-%d_%d_%d.swc' % (str(cls).zfill(4), x, y, z, nx, ny, nz)

        if len(names):
            img_path = join(str(cut_img_dir), names[0])
            if os.path.exists(img_path):
                new_path = join(savePath, tif_id)
                shutil.copy(img_path, new_path)

                swc_path = join(str(cut_swc_dir), str(Path(names[0]).stem) + ".swc")
                new_swc_path = join(divide_swc_dir, swc_id)
                if os.path.exists(swc_path):
                    shutil.copy(swc_path, new_swc_path)
                else:
                    with open(new_swc_path, "w") as f:
                        pass

        userTime = time.time() - sTime
        surplusTime = userTime / (j + 1) * (workLen - j - 1)
        logInfo = '[Image extraction progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
            (j + 1) / workLen * 100, userTime, surplusTime)
        if progress0 is not None:
            progress0.emit(logInfo, j)

    if os.path.isdir(del_savePath):  # 删除文件夹
        shutil.rmtree(del_savePath, ignore_errors=True)

    return res, division_text

