'''直方图分析'''
import json
import os
import shutil
from os.path import join
import numpy as np
from sklearn.cluster import KMeans
from libs.cfg import getCfgInstance
from matplotlib import pyplot as plt


'''直方图聚类'''
def HistKMean(histLs, n_clusters=20, labelAdd='./tmpData/labels.txt', cenAdd='./tmpData/center.txt', check=False):
    print('PCA+Clustering')
    histLs = np.array(histLs)
    histLs = np.log(histLs + 1)
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(histLs)
    cluster_centers = cluster.cluster_centers_.astype(np.float16)
    labels_ = cluster.labels_.astype(np.int32)
    np.save(labelAdd, labels_)
    np.save(cenAdd, cluster_centers)


'''聚类结果绘图'''
def ClusterViewDraw(histLs, n_clusters, labels, pltSavePath):
    print('Start plotting')
    os.makedirs(pltSavePath, exist_ok=True)
    clsHistLs = [[] for i in range(n_clusters)]
    for hi, cls in enumerate(labels):
        clsHistLs[cls].append(histLs[hi])
    for cls, histLs in enumerate(clsHistLs):
        f = plt.figure()
        for hist in histLs:
            plt.plot(hist)
        # plt.show()
        plt.savefig(join(pltSavePath, str(cls).zfill(5) + '.png'), dpi=600)
        plt.close()


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
        if len(histLs) == 0: continue
        cenHist = cluster_centers[cls]
        errorLs = np.sum(np.abs(histLs - cenHist), axis=1)
        indLs = np.argsort(errorLs)
        space = indLs.size / (getClsNum + 1)
        for t in range(getClsNum):
            id = int(round(space * (t + 1)))
            if id >= len(indLs): continue
            tmpId = idLs[indLs[id]]
            tmpId = tmpId.split('-')
            key = tmpId[0]
            if key in imgInfos:
                bx, by, bz = np.array(key.split('_'), dtype=np.int32).tolist()
                bbx, bby, bbz = np.array(tmpId[1].split('_'), dtype=np.int32).tolist()
                reKey = '%d_%d_%d-%d_%d_%d' % (bx, by, bz, bbx, bby, bbz)
                if reKey in reKeyInfos: continue
                imgInfos[key].append([bx, by, bz, bbx, bby, bbz, cls])
            else:
                bx, by, bz = np.array(key.split('_'), dtype=np.int32).tolist()
                bbx, bby, bbz = np.array(tmpId[1].split('_'), dtype=np.int32).tolist()
                reKey = '%d_%d_%d-%d_%d_%d' % (bx, by, bz, bbx, bby, bbz)
                if reKey in reKeyInfos: continue
                imgInfos[key] = [[bx, by, bz, bbx, bby, bbz, cls]]
            reKeyInfos.add(reKey)
    with open(imgIdInfoPath, 'w') as f:
        f.write(json.dumps(imgInfos))
    return imgInfos


'''读取灰度直方图和梯度特征'''
def ReadHistAndGrad(path):
    print('开始读取数据')
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


'''直方图筛选BV'''
def HistFiterBv(logSignal=None):
    cfg = getCfgInstance(True)
    path = join(cfg.saveRoot, 'TrainDataSetFiter/HistData')
    savePath = join(cfg.saveRoot, 'TrainDataSetFiter/HistAnaly')
    n_clusters = cfg.n_clusters
    getClsNum = cfg.getClsNum
    tmpPath = join(savePath, 'TmpData')
    # pltSavePath = join(savePath, 'Hist_Draw')
    imgIdInfoPath = join(savePath, 'ImgIdLs.json')
    if os.path.isdir(savePath):
        shutil.rmtree(savePath)
    os.makedirs(tmpPath, exist_ok=True)
    cenAdd = join(tmpPath, 'kkcent.npy')
    labelAdd = join(tmpPath, 'kklabel.npy')
    if logSignal is not None:
        logSignal.emit('Start reading data\n')
    nameIdLs, histLs = ReadHistAndGrad(path)
    if logSignal is not None:
        logSignal.emit('PCA+Clustering\n')
    HistKMean(histLs, n_clusters, labelAdd, cenAdd, labelAdd)
    labels = np.load(labelAdd)
    cluster_centers = np.load(cenAdd)
    # ClusterViewDraw(histLs, n_clusters, labels, pltSavePath)
    ClusterToImgIdLs(histLs, labels, cluster_centers, nameIdLs, n_clusters, getClsNum, imgIdInfoPath)


if __name__ == '__main__':
    HistFiterBv()
