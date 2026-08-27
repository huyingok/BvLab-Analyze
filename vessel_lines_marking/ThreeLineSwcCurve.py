# -*- coding: utf-8 -*-
'''3Dswc直线矫正'''
import warnings
import numpy as np
warnings.filterwarnings("ignore")
# warnings.simplefilter('ignore', np.RankWarning)
import tifffile
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

'''获取正方体核'''


def GetKernelArr(kernelSize):
    kernelArr = []
    for nz in range(-kernelSize, kernelSize + 1):
        for ny in range(-kernelSize, kernelSize + 1):
            for nx in range(-kernelSize, kernelSize + 1):
                kernelArr.append([nx, ny, nz])
    kerealArr = np.array(kernelArr)
    return kerealArr


'''种子点获取核点云'''


def GetKernealPc(curP3d, kerealArr, imgSize):
    newP = (curP3d[:, None] + kerealArr[None]).reshape([-1, 3])
    newP[newP < 0] = 0
    newP[newP[:, 0] >= imgSize[0], 0] = imgSize[0] - 1
    newP[newP[:, 1] >= imgSize[1], 1] = imgSize[1] - 1
    newP[newP[:, 2] >= imgSize[2], 2] = imgSize[2] - 1
    newP = np.unique(newP, axis=0)
    return newP


'''三维点云拟合'''


def ThreePcFit(pc1, x, y, z, weight, n):
    try:
        weight = (1. * weight - weight.min()) / (weight.max() - weight.min()) * 0.9 + 0.1
        data = np.array([x, y, z]).T
        newData = pca.fit_transform(data)
        newPc = pca.transform(pc1)

        xx1, yy1 = newData[:, 0], newData[:, 1]
        p_xy = np.polyfit(xx1, yy1, n, w=weight)
        y_out = np.polyval(p_xy, newPc[:, 0])

        xx1, yy1 = newData[:, 0], newData[:, 2]
        p_xy = np.polyfit(xx1, yy1, n, w=weight)
        z_out = np.polyval(p_xy, newPc[:, 0])
        coor = np.array([newPc[:, 0], y_out, z_out]).T
        newXY = pca.inverse_transform(coor)
        return newXY
    except:
        return pc1


'''动态加权'''


def PartWeightImg(swcData, sp, ep, img, newImg):
    batchSize = img.shape[::-1]
    thetaLs = np.arange(0, 2 * np.pi, np.pi / 5.0).reshape([-1, 1])
    for ii in range(sp, ep):
        item = swcData[ii]
        if item[-1] == -1: continue
        p0 = item[2: 5]
        p1 = swcData[int(item[-1]) - 1, 2: 5]
        d = np.linalg.norm(p0 - p1)
        if d > 1:
            d2 = int(d + 1)
            xLs = (np.arange(1, d2 + 1, 1) / d2).reshape([-1, 1])
            data = p0 * xLs + p1 * (1 - xLs)
        else:
            data = [p0, p1]
        partPs = []
        for i in range(len(data) - 1):
            p0, p1 = data[i], data[i + 1]
            m = (p0 + p1) / 2
            v0 = p1 - p0
            v0 /= np.linalg.norm(v0)
            if v0[0] == 0:
                v1 = np.array([0.0, -v0[2], v0[1]], dtype=np.float32)
            elif v0[1] == 0:
                v1 = np.array([-v0[2], 0.0, v0[0]], dtype=np.float32)
            else:
                v1 = np.array([-v0[1], v0[0], 0.0], dtype=np.float32)
            # v1 = np.array([v0[1], -v0[0], 0.0], dtype=np.float32)
            v1 /= np.linalg.norm(v1)
            v2 = np.cross(v0, v1)
            # ps = m + 1 * (v1 * np.cos(thetaLs) + v2 * np.sin(thetaLs))
            ps = np.r_[
                m + 1 * (v1 * np.cos(thetaLs) + v2 * np.sin(thetaLs)),
                m + 1.5 * (v1 * np.cos(thetaLs) + v2 * np.sin(thetaLs)),
                m + 2 * (v1 * np.cos(thetaLs) + v2 * np.sin(thetaLs))
            ]
            ps = np.round(ps)
            ps = ps.astype(np.int32)
            ps[ps < 0] = 0
            ps[ps[:, 0] > batchSize[0] - 1, 0] = batchSize[0] - 1
            ps[ps[:, 1] > batchSize[1] - 1, 1] = batchSize[1] - 1
            ps[ps[:, 2] > batchSize[2] - 1, 2] = batchSize[2] - 1
            partPs += ps.tolist()
        partPs = np.array(partPs)
        w = img[partPs[:, 2], partPs[:, 1], partPs[:, 0]]
        newImg[partPs[:, 2], partPs[:, 1], partPs[:, 0]] = (1. * w - w.min()) / (w.max() - w.min()) * 255


def GetFiberImg(imgPath, swcPath, savePath):
    kernelArr = GetKernelArr(5)
    swcData = np.loadtxt(swcPath, ndmin=2, dtype=np.float32)
    curP = np.round(swcData[:, 2: 5]).astype(np.int32)
    oriImg = tifffile.imread(imgPath)
    imgSize = np.array(oriImg.shape[::-1])
    newP = GetKernealPc(curP, kernelArr, imgSize[::-1])
    img = np.zeros(imgSize, dtype=np.uint16)
    img[newP[:, 2], newP[:, 1], newP[:, 0]] = oriImg[newP[:, 2], newP[:, 1], newP[:, 0]]
    tifffile.imwrite(savePath, img, compression='lzw')


'''纤维点云拟合'''


def ThreeCurveFitPc(img, sp, ep):
    kernelSize = 2
    kerealArr = GetKernelArr(kernelSize)
    imgSize = img.shape[::-1]
    '''矫正'''
    v = ep - sp
    d = np.linalg.norm(v)
    v /= d
    dStep = np.r_[np.arange(0, d, 2), d].reshape([-1, 1])
    pn = dStep.shape[0]
    branch = np.zeros([pn, 7])
    branch[:, 0] = np.arange(1, pn + 1, 1)
    branch[:, -1] = branch[:, 0] - 1
    branch[0, -1] = -1
    branch[:, 2: 5] = sp + dStep * v
    newImg = np.zeros_like(img)
    for i in range(2):
        curP3d = np.round(branch[:, 2: 5]).astype(np.int32)
        newP = GetKernealPc(curP3d, kerealArr, imgSize)
        x, y, z = newP.T
        weight = img[newP[:, 2], newP[:, 1], newP[:, 0]]
        branch[:, 2: 5] = ThreePcFit(curP3d, x, y, z, weight, int(pn / 3))
    '''二次精细拟合，局部动态加权'''
    for i in range(5):
        curP3d = np.round(branch[:, 2: 5]).astype(np.int32)
        newP = GetKernealPc(curP3d, kerealArr, imgSize)
        x, y, z = newP.T
        PartWeightImg(branch, 0, pn, img, newImg)
        weight = newImg[newP[:, 2], newP[:, 1], newP[:, 0]]
        branch[:, 2: 5] = ThreePcFit(curP3d, x, y, z, weight, int(pn / 3))
        newImg[...] = 0
    return branch
    # np.savetxt(savePath, branch)
    # BigSwcUtil.SaveSwcLs(savePath, branchLs)


if __name__ == '__main__':
    imgPath = r'D:\SY\nneural_data\testData\images\0226-28_11_46-1_2_0.tif'
    sp = np.array([38.57, 84.60, 14.93], dtype=np.float32)  # 起点
    ep = np.array([82.48, 73.26, 10.49], dtype=np.float32)  # 终点
    savePath = r'D:\SY\nneural_data\testData\tt\001-res.swc'
    img = tifffile.imread(imgPath)
    branch = ThreeCurveFitPc(img, sp, ep)[1:-1, 2:5]
    np.savetxt(savePath, branch)
