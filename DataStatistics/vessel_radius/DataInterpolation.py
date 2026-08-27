# -*- coding: utf-8 -*-
import os
import shutil
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# '''Swc多树拆分'''
def SplitSwcData(swcData):
    # 获取所有根节点id为-1的位置序号，并加上最后一个节点位置序号
    indLs = np.where(swcData[:, -1] == -1)[0].tolist() + [swcData.shape[0]]
    swcDataLs = []
    for i in range(len(indLs) - 1):
        data = swcData[indLs[i]: indLs[i + 1]]  # 提取分支树
        sp = data[0, 0]  # 原数据根节点序号
        data[:, 0] -= sp - 1  # 更新分支树所有节点序号
        data[1:, -1] -= sp - 1  # 除第一个节点（根节点），其余节点更新id
        swcDataLs.append(data)
    return swcDataLs


def swc_data_interpolation(swc_path, swc_add_path, kernelLen=2):
    swc_lines_all = []
    is_ok = False
    if os.path.exists(swc_path):
        if os.path.isfile(swc_path):
            if os.path.getsize(swc_path) > 0:
                is_ok = True
    else:
        with open(swc_path, 'w') as f:
            pass
        with open(swc_add_path, "w") as f:
            pass
        return
    if is_ok:
        swcData = np.loadtxt(swc_path, ndmin=2)  # 读取swcFile
        swcDataLs = SplitSwcData(swcData)  # swc多树拆分
        count = 0
        for swcData in swcDataLs:  # 遍历每个分支树
            count_list = []
            inter_add_list = []
            inter_add = 0
            for ii, item in enumerate(swcData):  # 分支树信息
                if item[-1] == -1:  # 父节点
                    swc_lines_all.append(
                        f"{count + 1} {item[1]} "
                        f"{item[2]} {item[3]} {item[4]} "
                        f"{item[5]} -1\n")
                    count_list.append(count)
                    inter_add_list.append(inter_add)
                    count += 1
                    continue
                p0 = item[2: 5]  # 当前位置坐标x，y，z
                index0 = int(item[-1])
                index_id = int(item[0])
                try:
                    index1 = index0 - 1
                    if index0 != index_id - 1:
                        p1 = swcData[index1, 2:5]
                        add_sum = inter_add_list[index1]
                        # print(add_sum, add_sum + index0)
                    else:
                        p1 = swcData[index1, 2:5]  # 前一个位置坐标x，y，z
                except IndexError:
                    print(int(item[-1]) - 1)
                    print(item)
                    continue
                v = (p1 - p0).reshape([1, 3])  # 计算当前节点（p0）与其父节点（p1）之间的向量v， p0 -> p1
                # 检查 v 长度是否小于0.1，若小于则跳过，这种情况下两个节点几乎重合
                if np.linalg.norm(v) < 0.1:  # 求范数，范数不小于0.1，默认参数(矩阵整体元素平方和开根号，不保留矩阵二维特性)
                    if index0 != index_id - 1:
                        swc_lines_all.append(
                            f"{count + 1} {item[1]} "
                            f"{item[2]} {item[3]} {item[4]} "
                            f"{item[5]} {count_list[index0 + add_sum]}\n")
                    else:
                        swc_lines_all.append(
                            f"{count + 1} {item[1]} "
                            f"{item[2]} {item[3]} {item[4]} "
                            f"{item[5]} {count}\n")
                    count_list.append(count)
                    inter_add_list.append(inter_add)
                    count += 1
                    continue
                # 插值
                d = np.linalg.norm(p0 - p1)  # p1 -> p0
                # 如果节点之间的距离大于kernelLen，则在p0和p1之间进行插值，生成一系列中间点data
                if d > kernelLen:  # 节点
                    d2 = int(d + 1)
                    d2 = d2 // 3 + 1  # 减少插值点数，取点数量直接砍到 1/3
                    xLs = (np.arange(1, d2 + 1, 1) / d2).reshape([-1, 1])
                    data = p0 * xLs + p1 * (1 - xLs)
                    for n, kd in enumerate(data):
                        if index0 != index_id - 1 and n == 0:
                            swc_lines_all.append(
                                f"{count + 1} {item[1]} "
                                f"{kd[0]} {kd[1]} {kd[2]} "
                                f"{item[5]} {count_list[index0 + add_sum]}\n")
                        else:
                            swc_lines_all.append(
                                f"{count + 1} {item[1]} "
                                f"{kd[0]} {kd[1]} {kd[2]} "
                                f"{item[5]} {count}\n")
                        count_list.append(count)
                        count += 1
                    inter_add += len(data) - 1
                    inter_add_list.append(inter_add)
                    continue
                # else:  # 否则，直接将p0和p1作为data
                #     data = np.array([p0, p1])
                if index0 != index_id - 1:
                    swc_lines_all.append(
                        f"{count + 1} {item[1]} "
                        f"{item[2]} {item[3]} {item[4]} "
                        f"{item[5]} {count_list[index0 + add_sum]}\n")
                else:
                    swc_lines_all.append(
                        f"{count + 1} {item[1]} "
                        f"{item[2]} {item[3]} {item[4]} "
                        f"{item[5]} {count}\n")
                count_list.append(count)
                inter_add_list.append(inter_add)
                count += 1
    with open(swc_add_path, "w") as f:
        f.writelines(swc_lines_all)


def data_add(swc2_dir, swc2_add_dir, progress_text=None):
    if os.path.exists(swc2_add_dir):
        shutil.rmtree(swc2_add_dir, ignore_errors=True)
    os.makedirs(swc2_add_dir, exist_ok=True)
    names = [l for l in os.listdir(swc2_dir) if '.swc' in l]
    kernelLen = 2
    start_time = time.time()
    for ni, name in enumerate(names):
        swc2_path = os.path.join(swc2_dir, name)
        swc2_add_path = os.path.join(swc2_add_dir, name)
        swc_lines_all = []
        if not os.path.exists(swc2_path):
            with open(swc2_path, 'w') as f:
                pass
        else:
            swcData = np.loadtxt(swc2_path, ndmin=2)  # 读取swcFile
            swcDataLs = SplitSwcData(swcData)  # swc多树拆分
            count = 0
            for swcData in swcDataLs:  # 遍历每个分支树
                count_list = []
                inter_add_list = []
                inter_add = 0
                for ii, item in enumerate(swcData):  # 分支树信息
                    if item[-1] == -1:  # 父节点
                        swc_lines_all.append(
                            f"{count + 1} {item[1]} "
                            f"{item[2]} {item[3]} {item[4]} "
                            f"{item[5]} -1\n")
                        count_list.append(count)
                        inter_add_list.append(inter_add)
                        count += 1
                        continue

                    p0 = item[2: 5]  # 当前位置坐标x，y，z
                    index0 = int(item[-1])
                    index_id = int(item[0])
                    try:
                        index1 = index0 - 1
                        if index0 != index_id - 1:
                            p1 = swcData[index1, 2:5]
                            add_sum = inter_add_list[index1]
                            # print(add_sum, add_sum + index0)
                        else:
                            p1 = swcData[index1, 2:5]  # 前一个位置坐标x，y，z
                    except IndexError:
                        print(int(item[-1]) - 1)
                        print(item)
                        continue
                    v = (p1 - p0).reshape([1, 3])  # 计算当前节点（p0）与其父节点（p1）之间的向量v， p0 -> p1
                    # 检查 v 长度是否小于0.1，若小于则跳过，这种情况下两个节点几乎重合
                    if np.linalg.norm(v) < 0.1:   # 求范数，范数不小于0.1，默认参数(矩阵整体元素平方和开根号，不保留矩阵二维特性)
                        if index0 != index_id - 1:
                            swc_lines_all.append(
                                f"{count + 1} {item[1]} "
                                f"{item[2]} {item[3]} {item[4]} "
                                f"{item[5]} {count_list[index0 + add_sum]}\n")
                        else:
                            swc_lines_all.append(
                                f"{count + 1} {item[1]} "
                                f"{item[2]} {item[3]} {item[4]} "
                                f"{item[5]} {count}\n")
                        count_list.append(count)
                        inter_add_list.append(inter_add)
                        count += 1
                        continue
                    # 插值
                    d = np.linalg.norm(p0 - p1)  # p1 -> p0
                    # 如果节点之间的距离大于kernelLen，则在p0和p1之间进行插值，生成一系列中间点data
                    if d > kernelLen:  # 节点
                        d2 = int(d + 1)
                        xLs = (np.arange(1, d2 + 1, 1) / d2).reshape([-1, 1])
                        data = p0 * xLs + p1 * (1 - xLs)
                        for n, kd in enumerate(data):
                            if index0 != index_id - 1 and n == 0:
                                swc_lines_all.append(
                                    f"{count + 1} {item[1]} "
                                    f"{kd[0]} {kd[1]} {kd[2]} "
                                    f"{item[5]} {count_list[index0 + add_sum]}\n")
                            else:
                                swc_lines_all.append(
                                    f"{count + 1} {item[1]} "
                                    f"{kd[0]} {kd[1]} {kd[2]} "
                                    f"{item[5]} {count}\n")
                            count_list.append(count)
                            count += 1
                        inter_add += len(data) - 1
                        inter_add_list.append(inter_add)
                        continue
                    # else:  # 否则，直接将p0和p1作为data
                    #     data = np.array([p0, p1])
                    if index0 != index_id - 1:
                        swc_lines_all.append(
                            f"{count + 1} {item[1]} "
                            f"{item[2]} {item[3]} {item[4]} "
                            f"{item[5]} {count_list[index0 + add_sum]}\n")
                    else:
                        swc_lines_all.append(
                            f"{count + 1} {item[1]} "
                            f"{item[2]} {item[3]} {item[4]} "
                            f"{item[5]} {count}\n")
                    count_list.append(count)
                    inter_add_list.append(inter_add)
                    count += 1
        with open(swc2_add_path, "w") as f:
            f.writelines(swc_lines_all)

        userTime = time.time() - start_time
        surplusTime = userTime / (ni + 1) * (len(names) - ni - 1)
        if surplusTime >= 0:
            logInfo = '[Interpolation progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                (ni + 1) / len(names) * 100, userTime, surplusTime)
            if progress_text is not None:
                progress_text.emit(logInfo, ni)
            else:
                print(logInfo)


if __name__ == '__main__':

    root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\testData"

    # swc_dir = os.path.join(root, "swc")
    swc_dir = os.path.join(root, "nnUnet_swc")

    # swc_add_dir = os.path.join(root, "swc_i")
    swc_add_dir = os.path.join(root, "nnUnet_swc_i")

    data_add(swc_dir, swc_add_dir)

