import os
import time
import tifffile as tiff
import numpy as np


def improved_fast_iterative_method(image_3d, delta_x=1, delta_y=1, delta_z=1, f=1, detal=1):
    """
    :param image_3d: 3D 输入图像，形状为 (nz, ny, nx)，值为 0 或 1
    :param delta_x: x方向的步长
    :param delta_y: y方向的步长
    :param delta_z: z方向的步长
    :param f: 距离因子
    :param detal: 细节因子
    :return: 距离场
    """
    shape_z, shape_y, shape_x = image_3d.shape  # 获取图像的尺寸
    T_new = np.ones((shape_z, shape_y, shape_x)) * image_3d  # 初始化为0的3DArray

    for kk in range(30):
        # print(kk)
        for i in range(1, shape_z - 1):
            for j in range(1, shape_y - 1):
                for k in range(1, shape_x - 1):
                    if image_3d[i, j, k] > 0:
                        min_dz = min(T_new[i-1, j, k], T_new[i+1, j, k])
                        min_dy = min(T_new[i, j-1, k], T_new[i, j+1, k])
                        min_dx = min(T_new[i, j, k-1], T_new[i, j, k+1])

                        # Sort min_dx, min_dy, min_dz
                        a = np.sort([min_dx, min_dy, min_dz])
                        a1, a2, a3 = a[0], a[1], a[2]

                        if np.abs(a1 - a3) < detal:
                            p1 = 2 * np.sum(a)
                            p2 = np.sqrt((4 * (np.sum(a) ** 2)) -
                                         (12 * ((a1 ** 2 + a2 ** 2 + a3 ** 2) - (detal ** 2) / (f ** 2))))
                            T_new[i, j, k] = (p1 + p2) / 6
                        elif np.abs(a1 - a2) < detal:
                            p1 = a1 + a2
                            p2 = np.sqrt((2 * (detal ** 2) / (f ** 2)) - ((a1 - a2) ** 2))
                            T_new[i, j, k] = (p1 + p2) / 2
                        else:
                            T_new[i, j, k] = a1 + detal / f

    # 处理边界点
    T_new[0, :, :] = T_new[1, :, :]  # z=0 的边界
    T_new[-1, :, :] = T_new[-2, :, :]  # z=max 的边界
    T_new[:, 0, :] = T_new[:, 1, :]  # y=0 的边界
    T_new[:, -1, :] = T_new[:, -2, :]  # y=max 的边界
    T_new[:, :, 0] = T_new[:, :, 1]  # x=0 的边界
    T_new[:, :, -1] = T_new[:, :, -2]  # x=max 的边界

    # 应用公式（6）进行最终调整
    TT = np.zeros((shape_z, shape_y, shape_x))
    # for i in range(1, shape_z - 1):
    #     for j in range(1, shape_y - 1):
    #         for k in range(1, shape_x - 1):
    #             if image_3d[i, j, k] > 0:
    #                 phi_zmin = min(T_new[i - 1, j, k], T_new[i + 1, j, k])
    #                 phi_ymin = min(T_new[i, j - 1, k], T_new[i, j + 1, k])
    #                 phi_xmin = min(T_new[i, j, k - 1], T_new[i, j, k + 1])
    #
    #                 term1 = (T_new[i, j, k] - phi_zmin) / delta_z if T_new[i, j, k] > phi_zmin else 0
    #                 term2 = (T_new[i, j, k] - phi_ymin) / delta_y if T_new[i, j, k] > phi_ymin else 0
    #                 term3 = (T_new[i, j, k] - phi_xmin) / delta_x if T_new[i, j, k] > phi_xmin else 0
    #                 denominator = [term1, term2, term3]
    #                 denominator = np.clip(denominator, 0, 10000)
    #                 TT[i, j, k] = np.sum(denominator * denominator)
    #
    # # 处理边界点
    # TT[0, :, :] = TT[1, :, :]  # z=0 的边界
    # TT[-1, :, :] = TT[-2, :, :]  # z=max 的边界
    # TT[:, 0, :] = TT[:, 1, :]  # y=0 的边界
    # TT[:, -1, :] = TT[:, -2, :]  # y=max 的边界
    # TT[:, :, 0] = TT[:, :, 1]  # x=0 的边界
    # TT[:, :, -1] = TT[:, :, -2]  # x=max 的边界

    return T_new, TT


def DistTest(mask):
    # aa = np.zeros((192, 192, 192))
    # aa[:, 100:120, 100:120] = 10
    bb = mask.copy()
    # bb[aa > 5] = 1
    # bb[aa < 5] = 0.0

    nx, ny, nz = mask.shape
    T = np.ones((nx, ny, nz)) * bb

    for kk in range(30):
        print(kk)
        for i in range(nx - 2):
            for j in range(ny - 2):
                for ij in range(nz - 2):
                    if bb[i, j, ij] > 0:
                        ax = min(T[i, j + 1, ij + 1], T[i + 2, j + 1, ij + 1])
                        ay = min(T[i + 1, j, ij], T[i + 1, j + 2, ij])
                        az = min(T[i + 1, j + 1, ij], T[i + 1, j + 1, ij + 2])
                        ss = np.sort([ax, ay, az])
                        if kk == 19 and T[i + 1, j + 1, ij + 1] == 0:
                            print(ss)

                        if np.abs(ss[2] - ss[0]) < 1:
                            T[i + 1, j + 1, ij + 1] = 1 / 6 * (
                                    2 * np.sum(ss) + np.sqrt(4 * np.sum(ss) ** 2 - 12 * (np.sum(ss * ss) - 1)))
                        elif np.abs(ss[1] - ss[0]) < 1:
                            T[i + 1, j + 1, ij + 1] = 1 / 2 * (ss[0] + ss[1] + np.sqrt(2 - (ss[0] - ss[1]) ** 2))
                        else:
                            T[i + 1, j + 1, ij + 1] = ss[0] + 1

    TT = np.zeros((nx, ny, nz))
    for i in range(nx - 2):
        for j in range(ny - 2):
            for ij in range(nz - 2):
                if bb[i, j, ij] > 0:
                    ax = min(T[i, j + 1, ij + 1], T[i + 2, j + 1, ij + 1])
                    ay = min(T[i + 1, j, ij], T[i + 1, j + 2, ij])
                    az = min(T[i + 1, j + 1, ij], T[i + 1, j + 1, ij + 2])

                    ss = [T[i + 1, j + 1, ij + 1] - ax, T[i + 1, j + 1, ij + 1] - ay, T[i + 1, j + 1, ij + 1] - az]
                    ss = np.clip(ss, 0, 10000)
                    TT[i + 1, j + 1, ij + 1] = np.sum(ss * ss)
    return T, TT


if __name__ == '__main__':
    # root = r"D:\xueguan\TrainDateSet3\train_data"
    root = r"D:\xueguan\TrainDateSet2\train"
    # root = r"D:\xueguan\TrainDateSet3\train_data\ttt"
    # root = r"D:\xueguan\TrainDateSet3\train_data\test"
    # root = r"D:\xueguan\TrainDateSet3\train_data\128"
    # root = r"D:\xueguan\TrainDateSet3\train_data\ddd"
    # root = r"D:\xueguan\TrainDateSet3\train_data\128\data"
    # root = r"D:\xueguan\TrainDateSet3\train_data\128\test_dist"
    # mask_dir = os.path.join(root, "mask")
    mask_dir = os.path.join(root, "mySignal_big")
    # mask_dir = os.path.join(root, "masks")
    # mask_dir = os.path.join(root, "predict")
    # mask_dir = os.path.join(root, "centerline")
    # mask_dir = os.path.join(root, "images")
    # save_dir = os.path.join(mask_dir, "dist-30+")
    save_dir = os.path.join(root, "dist-30")
    # save_dir = os.path.join(root, "save")
    os.makedirs(save_dir, exist_ok=True)

    names = [l for l in os.listdir(mask_dir) if ".tif" in l]
    # names = ["mask.tif"]
    # names = ["merge2_2_5_0.tif"]
    for ii, name in enumerate(names):
        s = time.time()
        mask_path = os.path.join(mask_dir, name)
        save_path = os.path.join(save_dir, name)
        mask = tiff.imread(mask_path)
        mask = mask / 255.

        dist, distT = improved_fast_iterative_method(mask)
        # dist, distT = DistTest(mask)
        dist = ((dist - dist.min()) / (dist.max() - dist.min()) * 255).astype(np.uint8)
        # distT = ((distT - distT.min()) / (distT.max() - distT.min()) * 255).astype(np.uint8)
        tiff.imwrite(save_path, dist)
        # tiff.imwrite(save_path.replace(".tif", "_TT.tif"), distT)
        print(ii + 1, name, time.time() - s)
