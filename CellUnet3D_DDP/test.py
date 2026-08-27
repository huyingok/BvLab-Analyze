# -*- coding: utf-8 -*-
import time

import numpy as np
import torch
import cc3d
from scipy.spatial import distance
import tifffile
from speed_cc3d import speed_cc3d


# 尝试导入 SpeedCC3D Module
try:
    from SpeedCC3D import SpeedCC3D
    print("SpeedCC3D 模块加载成功")
except ImportError as e:
    print(f"加载 SpeedCC3D 模块失败: {e}")


def test_gpu():
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Selected device:", device)

    if torch.cuda.is_available():
        print("GPU is available.")
    else:
        print("GPU is not available.")

    print(torch.version.cuda)


def mask_to_skeleton_swc(mask, savePath):
    thre = 5000
    mask2 = mask.copy()
    mask2[mask2 <= 103] = 0
    mask2[mask2 > 0] = 255
    labels_out, N = cc3d.connected_components(mask2, connectivity=26, return_N=True, out_dtype=np.uint32)

    flag = 1
    if N != 0:
        start = time.time()
        # out1 = SpeedCC3D.groupTypes(labels_out)
        out1 = speed_cc3d.group_type(labels_out)
        print(time.time() - start)

        # # 计算每个连通域的统计信息
        # stats = cc3d.statistics(labels_out)
        # # 创建一个字典来存储每个连通域的质心
        # out1 = {}
        # center_point = []
        # for point in stats['centroids']:  # 使用 stats['label_ids'] 获取所有连通域的标签
        #     centroid = point[::-1]
        #     tmp = np.array([flag, 0, centroid[0], centroid[1], centroid[2], 0, -1])
        #     center_point.append(tmp)
        #     flag += 1

        # 提取每个连通域的点集
        start = time.time()
        out2 = {}
        for label in np.unique(labels_out):
            if label == 0:
                continue  # 跳过背景（标签为0的部分）
            points = np.argwhere(labels_out == label)
            out2[label] = [point[::-1] for point in points]
        print(time.time() - start)

        center_point = []
        for i, key in enumerate(out1):
            # print('进度：%f' % (i / len(out1) * 100) + '%')
            point = np.array(out1[key], dtype=np.int32)
            if len(point) <= 297 or len(point) >= thre:
                continue
            dist = distance.cdist(point, point)
            dist_max = np.max(dist)
            rhos = []
            for j in range(len(point)):
                rhos.append(mask[point
                [j, 2], point[j, 1], point[j, 0]])
            max_rho = max(rhos)
            # 计算每个点的最小距离
            sigmas = np.zeros(len(point))
            nearest_neighbor = np.zeros(len(point))
            sorted_id = sorted(range(len(rhos)), key=lambda k: rhos[k], reverse=True)
            for j, index in enumerate(sorted_id):
                if j == 0:
                    sigmas[index] = 20
                    continue
                higher_rho_idx = sorted_id[:j]
                sigmas[index] = np.min(dist[index, higher_rho_idx])
                temp = np.argmin(dist[index, higher_rho_idx]).astype(int)
                nearest_neighbor[index] = higher_rho_idx[temp]
            # plt.scatter(rhos, sigmas)
            # plt.show()
            # center_idx = sorted_id[0]
            center_idx = np.where(sigmas > 5)[0]
            for o in range(len(center_idx)):
                p = point[center_idx[o]]
                tmp = np.array([flag, 0, p[0], p[1], p[2], 0, -1])
                center_point.append(tmp)
                flag += 1
        if len(center_point) != 0:
            center_point = np.array(center_point)
            np.savetxt(savePath, center_point)
        # else:
        #     temp_p = np.array([0, 0, 0, 0, 0, 0, -1]).reshape(1, -1)
        #     np.savetxt(savePath, temp_p)


if __name__ == "__main__":
    # D:\LIG-main\torch-2.1.1-cu118-cp38\torch-2.1.1+cu118-cp38-cp38-win_amd64.whl
    # D:\LIG-main\torch-2.1.1-cu118-cp38\torchaudio-2.1.1+cu118-cp38-cp38-win_amd64.whl
    # D:\LIG-main\torch-2.1.1-cu118-cp38\torchvision-0.16.1+cu118-cp38-cp38-win_amd64.whl
    # nvcc --version
    # pip install napari
    # pip install PyQt5
    # Run: napari
    # python -m pip install cellpose[gui]
    # Run: python -m cellpose
    # pip uninstall PyQt5 PyQt5-Qt5 PyQt5-sip

    # test_gpu()

    mask = tifffile.imread(r'D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D_DDP\TrainDataSet\mask_to_swc_test\0013-5_17_18-1_0_0.tif')
    savePath = r'D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D_DDP\TrainDataSet\0013-5_17_18-1_0_0.swc'

    mask_to_skeleton_swc(mask, savePath)
