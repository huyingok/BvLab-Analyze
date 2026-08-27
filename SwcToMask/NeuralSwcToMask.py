# -*- coding: utf-8 -*-
import time
import sys
import traceback
import numpy as np
from os.path import join
import os
import tifffile
from pathlib import Path
import random
import cv2
from collections import Counter
from multiprocessing import Process, Queue
import warnings
warnings.filterwarnings("ignore")


from PyQt5.QtCore import QThread, pyqtSignal


class NeuralSwcToMaskQThread(QThread):
    finish0 = pyqtSignal()
    warning0 = pyqtSignal(str)
    progress0 = pyqtSignal(str, np.ndarray, np.ndarray)
    progress1 = pyqtSignal(str, int)
    error0 = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(NeuralSwcToMaskQThread, self).__init__()
        self.win = kwargs.get('win')
        self.logger = self.win.logger

    def run(self):
        self.is_keep_on = False
        try:
            if not self.win.train_stop:
                imgPath = self.win.data_train_dict['imgDir']
                swc_dir = self.win.data_train_dict['swcDir']
                self.is_keep_on = self.win.data_train_dict['is_keep_on']

                names = [l for l in os.listdir(imgPath) if ".tif" in l]
                if len(names) < 10:
                    self.warning0.emit("The number of Tif format files in the image folder is less than 10")
                    return
                shapes_list = []

                for name in names[:5]:
                    shapes_list.append(tifffile.imread(join(imgPath, name)).shape)

                # 统计每个尺寸出现的次数
                shape_counts = Counter(shapes_list)
                # 获取众数尺寸
                most_common_shape = shape_counts.most_common(1)[0][0]

                if len(most_common_shape) != 3:
                    self.warning0.emit("The input image data is not in 3D")
                    return

                shapes = most_common_shape[::-1]  # 转为xyz
                self.win.data_train_dict['shapes'] = list(shapes)

                # with open(cell_cfgPath, 'w') as configfile:
                #     cell_cfg.cfg.write(configfile)

                self.rootPath = str(Path(imgPath).parent.absolute())
                mask_save_dir = join(self.rootPath, 'mask')
                # 转换
                # self.swc_to_mask(imgPath, swc_dir, mask_save_dir, names, shapes)
                self.swc_to_mask_sub(imgPath, swc_dir, mask_save_dir, names, shapes)
                # 数据分配
                if not self.win.train_stop:
                    self.data_set_txt_make(names)
            time.sleep(1)
            self.finish0.emit()
        except Exception as e:
            self.error0.emit(str(e))
            self.logger.error("\n=== Error message ===")
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Error message: {e}")
            self.logger.error("=== Error location ===")
            tb = sys.exc_info()[2]
            for frame in traceback.extract_tb(tb):
                self.logger.error(f"  File: {frame.filename}")
                self.logger.error(f"  Line number: {frame.lineno}")
                self.logger.error(f"  Function: {frame.name}")
                self.logger.error(f"  Code: {frame.line}\n")

    def swc_to_mask_sub(self, imgPath, swc_dir, mask_save_dir, names, shapes):
        os.makedirs(swc_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)

        kernelLen = 3
        # kernelLen = 2
        imgShape = np.array(shapes, dtype=np.int32)  # xyz
        maxD = ((kernelLen ** 2) * 3) ** 0.5
        minD = 0.0697

        # lsLen = len(names)
        # if lsLen > 1:
        nums_work = min(int(os.cpu_count() / 2), 6)
        print("nums_work: ", nums_work)

        pathLsQue = Queue()
        finishQue = Queue()
        errorQue = Queue()
        resultQue = Queue()

        for i, name in enumerate(names):
            mask_save_path = join(str(mask_save_dir), name)
            if self.is_keep_on:
                if os.path.exists(mask_save_path):
                    continue
            swc_name = str(Path(name).stem) + ".swc"
            swc_path = join(str(swc_dir), swc_name)
            if not os.path.exists(swc_path):
                with open(swc_path, "w", encoding="utf-8") as swc_f:
                    swc_f.write("")

            img_path = join(str(imgPath), name)

            pathLsQue.put((swc_path, mask_save_path, img_path, kernelLen, imgShape, maxD, minD))

        original_work_len = pathLsQue.qsize()

        if not original_work_len:
            return

        for i in range(nums_work):
            pathLsQue.put(())

        ps = []
        for i in range(nums_work):
            ps.append(Process(target=swc_to_mask_process, args=(pathLsQue, finishQue, errorQue, resultQue)))
            ps[-1].daemon = True
            ps[-1].start()

        start_time = time.time()
        curMakeSize = 0
        finished_processes = 0
        is_stop_count = 0

        while True:
            # 检查进程是否全部完成
            while finishQue.qsize() > 0:
                is_stop_count = 0
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
                return

            # 结束训练
            if self.win.train_stop:
                for p in ps:
                    p.terminate()
                return

            # 更新进度
            if curMakeSize > 0:
                userTime = time.time() - start_time
                surplusTime = userTime / curMakeSize * (original_work_len - curMakeSize)
                text = '[SWC to mask progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                    curMakeSize / original_work_len * 100, userTime, surplusTime)
                add = resultQue.get()
                if len(add) > 0:
                    dfImg2_2d, img_2d = add
                    self.progress0.emit(text, img_2d, dfImg2_2d)

            is_stop_count += 1
            time.sleep(2)

        # 等待所有进程完成
        for i, p in enumerate(ps):
            p.join(timeout=0.5)  # 最多等待5Second
            if p.is_alive():
                if i == 0:
                    userTime = time.time() - start_time
                    text = '[SWC to mask progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        100, userTime, 0)
                    add = resultQue.get()
                    if len(add) > 0:
                        dfImg2_2d, img_2d = add
                        self.progress0.emit(text, img_2d, dfImg2_2d)
                text = '[End process progress %.2f%%]' % ((i + 1) / nums_work * 100)
                self.progress1.emit(text, i)
                p.terminate()

    def swc_to_mask(self, imgPath, swc_dir, mask_save_dir, names, shapes):
        os.makedirs(swc_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)

        kernelLen = 3
        imgShape = np.array(shapes, dtype=np.int32)  # xyz
        maxD = ((kernelLen ** 2) * 3) ** 0.5
        minD = 0.0697
        maskMax = -np.log(minD)
        dfImg = np.zeros(imgShape[::-1], dtype=np.float32)  # zyx
        kernelArr = []
        for z in range(-kernelLen, kernelLen + 1):
            for y in range(-kernelLen, kernelLen + 1):
                for x in range(-kernelLen, kernelLen + 1):
                    kernelArr.append([x, y, z])
        kernelArr = np.array(kernelArr, dtype=np.int32)

        lsLen = len(names)
        start_time = time.time()

        for i, name in enumerate(names):
            if self.win.train_stop:
                break
            mask_save_path = join(str(mask_save_dir), name)
            if self.is_keep_on:
                if os.path.exists(mask_save_path):
                    continue
            swc_name = str(Path(name).stem) + ".swc"
            swc_path = join(str(swc_dir), swc_name)
            if not os.path.exists(swc_path):
                with open(swc_path, "w", encoding="utf-8") as swc_f:
                    swc_f.write("")
            swcData = np.loadtxt(swc_path, ndmin=2)  # 读取骨架文件
            swcDataLs = SplitSwcData(swcData)
            dfImg[...] = maxD
            for swcData in swcDataLs:
                for ii, item in enumerate(swcData):
                    if item[-1] == -1 or len(item) != 7:
                        continue
                    p0 = item[2: 5]
                    p1 = swcData[int(item[-1]) - 1, 2: 5]
                    v = (p1 - p0).reshape([1, 3])
                    if np.linalg.norm(v) < 0.1:
                        continue
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
                    if np.isnan(d).any():
                        print()

                    dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]] = np.min([dfImg[curPc[:, 2], curPc[:, 1],
                    curPc[:, 0]], d], axis=0)

            dfImg = -np.log(minD + dfImg / maxD * (1 - minD))
            dfImg2 = (dfImg / maskMax * 255).astype(np.uint8)
            tifffile.imwrite(mask_save_path, dfImg2, compression='lzw')

            dfImg2_2d = MaxProject(dfImg2, 0)
            img = tifffile.imread(join(str(imgPath), name))
            img_2d = MaxProject(img, 1)

            if (i + 1) % 2 == 0 or i == 0 or i == lsLen - 1:
                userTime = time.time() - start_time
                surplusTime = userTime / (i + 1) * (lsLen - i - 1)
                text = '[SWC to mask progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                    (i + 1) / lsLen * 100, userTime, surplusTime)
                self.progress0.emit(text, img_2d, dfImg2_2d)

    def data_set_txt_make(self, ls):
        """
        数据集分配
        """
        totalName_path = join(self.rootPath, 'totalName.txt')
        if not os.path.exists(totalName_path):
            # 训练集,验证集,测试集比例
            dataSetRadio = [0.7, 0.20, 0.10]
            setNameLs = ['train', 'val', 'test']
            # lsLen = len(ls)
            nameLs = []
            for ii, name in enumerate(ls):
                nameLs.append(name)

            # 写入总数
            lsLen = len(nameLs)

            with open(totalName_path, 'w') as f:
                [f.write('%s\n' % name) for name in nameLs]

            spaceLs = [0, int(lsLen * dataSetRadio[0]), int(lsLen * (dataSetRadio[0] + dataSetRadio[1])),
                       int(lsLen * (dataSetRadio[0] + dataSetRadio[1] + dataSetRadio[2]))]
            random.shuffle(nameLs)
            for ii in range(len(setNameLs)):
                curNameLS = nameLs[spaceLs[ii]: spaceLs[ii + 1]]
                path = join(self.rootPath, setNameLs[ii] + '.txt')
                with open(path, 'w') as f:
                    [f.write('%s\n' % name) for name in curNameLS]


def MaxProject(img, is_enhance):
    img = np.max(img, axis=0)
    img = normalize_and_scale_to_uint8(img)
    if is_enhance:
        img = enhance_contrast_clahe(img)
    return img


def enhance_contrast_clahe(img):
    """
    使用自适应直方图均衡化（CLAHE）增强图像对比度。

    Parameter:
        img (numpy.ndarray): 输入图像。
    返回:
        numpy.ndarray: 对比度增强后的图像。
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return img_clahe


def normalize_and_scale_to_uint8(data):
    """
    将灰度值归一化到0到1，然后缩放到0到255，并转换为uint8类型。

    Parameter:
        data (numpy.ndarray): 输入数据。
    返回:
        numpy.ndarray: 转换后的uint8Data。
    """
    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min)
    scaled_data = (normalized_data * 255).astype(np.uint8)
    return scaled_data


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


def GetPcKernelPc(pc, kernelArr, imgShape):
    curPc = (pc[:, None] + kernelArr[None]).reshape([-1, 3])
    curPc = np.round(curPc).astype(np.int32)
    curPc[curPc < 0] = 0
    curPc[curPc[:, 2] > imgShape[2] - 1, 2] = imgShape[2] - 1
    curPc[curPc[:, 1] > imgShape[1] - 1, 1] = imgShape[1] - 1
    curPc[curPc[:, 0] > imgShape[0] - 1, 0] = imgShape[0] - 1
    curPc = np.unique(curPc, axis=0)
    return curPc


def swc_to_mask_process(pathLsQue, finishQue, errorQue, resultQue):
    """处理单个图像，用于并行处理"""
    while True:
        try:
            add = pathLsQue.get(timeout=5)

            if len(add) == 0:
                finishQue.put(-1)
                return

            swc_path, mask_save_path, img_path, kernelLen, imgShape, maxD, minD = add

            maskMax = -np.log(minD)
            dfImg = np.zeros(imgShape[::-1], dtype=np.float32)  # zyx
            kernelArr = []
            for z in range(-kernelLen, kernelLen + 1):
                for y in range(-kernelLen, kernelLen + 1):
                    for x in range(-kernelLen, kernelLen + 1):
                        kernelArr.append([x, y, z])
            kernelArr = np.array(kernelArr, dtype=np.int32)

            if not os.path.getsize(swc_path):
                swcDataLs = []
            else:
                swcData = np.loadtxt(swc_path, ndmin=2)
                swcDataLs = SplitSwcData(swcData)

            dfImg[...] = maxD
            for swcData in swcDataLs:
                for ii, item in enumerate(swcData):
                    if item[-1] == -1 or len(item) != 7:
                        continue
                    p0 = item[2: 5]
                    p1 = swcData[int(item[-1]) - 1, 2: 5]
                    v = (p1 - p0).reshape([1, 3])
                    if np.linalg.norm(v) < 0.1:
                        continue
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
                    if np.isnan(d).any():
                        print()

                    dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]] = np.min([dfImg[curPc[:, 2], curPc[:, 1],
                    curPc[:, 0]], d], axis=0)

            dfImg = -np.log(minD + dfImg / maxD * (1 - minD))
            dfImg2 = (dfImg / maskMax * 255).astype(np.uint8)
            dfImg2[dfImg2 < 13] = 0
            tifffile.imwrite(mask_save_path, dfImg2, compression='lzw')

            dfImg2_2d = MaxProject(dfImg2, 0)
            img = tifffile.imread(img_path)
            img_2d = MaxProject(img, 1)

            finishQue.put(1)
            resultQue.put((dfImg2_2d, img_2d))
        except Exception as e:
            if errorQue.qsize() == 0:
                finishQue.put(1)
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


def NeuralSwcToDF(swc_dir, mask_save_dir, shapes, kernelLen=3):
    os.makedirs(swc_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    imgShape = np.array(shapes, dtype=np.int32)  # xyz
    maxD = ((kernelLen ** 2) * 3) ** 0.5
    minD = 0.0697
    maskMax = -np.log(minD)
    dfImg = np.zeros(imgShape[::-1], dtype=np.float32)  # zyx
    kernelArr = []
    for z in range(-kernelLen, kernelLen + 1):
        for y in range(-kernelLen, kernelLen + 1):
            for x in range(-kernelLen, kernelLen + 1):
                kernelArr.append([x, y, z])
    kernelArr = np.array(kernelArr, dtype=np.int32)
    ls = [l for l in os.listdir(swc_dir) if ".swc" in l]

    for i, name in enumerate(ls):
        print(i + 1, "|", len(ls), name)
        swc_name = str(Path(name).stem) + ".swc"
        tif_name = str(Path(name).stem) + ".tif"
        mask_save_path = join(str(mask_save_dir), tif_name)
        swc_path = join(str(swc_dir), swc_name)
        if not os.path.exists(swc_path):
            with open(swc_path, "w", encoding="utf-8") as swc_f:
                swc_f.write("")
        swcData = np.loadtxt(swc_path, ndmin=2)  # 读取骨架文件
        swcDataLs = SplitSwcData(swcData)
        dfImg[...] = maxD
        for swcData in swcDataLs:
            for ii, item in enumerate(swcData):
                if item[-1] == -1 or len(item) != 7:
                    continue
                p0 = item[2: 5]
                p1 = swcData[int(item[-1]) - 1, 2: 5]
                v = (p1 - p0).reshape([1, 3])
                if np.linalg.norm(v) < 0.1:
                    continue
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
                if np.isnan(d).any():
                    print()

                dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]] = np.min([dfImg[curPc[:, 2], curPc[:, 1],
                curPc[:, 0]], d], axis=0)

        dfImg = -np.log(minD + dfImg / maxD * (1 - minD))
        dfImg2 = (dfImg / maskMax * 255).astype(np.uint8)
        tifffile.imwrite(mask_save_path, dfImg2, compression='lzw')


def process_single_neural_swc(swc_name, swc_dir, mask_save_dir, imgShape, maxD, minD, maskMax, kernelArr, kernelLen):
    """处理单个神经元 SWC 文件，生成距离场 TIFF"""
    name_id = os.path.splitext(swc_name)[0]
    swc_path = os.path.join(swc_dir, swc_name)
    tif_path = os.path.join(mask_save_dir, name_id + '.tif')

    # 如果 SWC 文件不存在则创建空文件（保持与原逻辑一致）
    if not os.path.exists(swc_path):
        with open(swc_path, "w", encoding="utf-8") as f:
            f.write("")

    # 读取骨架数据
    swcData = np.loadtxt(swc_path, ndmin=2)

    # 分割为独立的神经突段
    swcDataLs = SplitSwcData(swcData)

    # 初始化距离场（注意 imgShape 是 (z, y, x) 顺序）
    df_img = np.full(imgShape, maxD, dtype=np.float32)

    for seg in swcDataLs:
        for item in seg:
            if item[-1] == -1 or len(item) != 7:
                continue
            p0 = item[2:5]  # 当前点坐标 (x, y, z)
            parent_idx = int(item[-1]) - 1
            # 确保父节点索引有效
            if parent_idx < 0 or parent_idx >= len(seg):
                continue
            p1 = seg[parent_idx, 2:5]  # 父节点坐标

            v = p1 - p0
            if np.linalg.norm(v) < 0.1:
                continue

            # 两点间距离
            d_seg = np.linalg.norm(p0 - p1)

            # 插值生成中间点（使线段离散化密度足够）
            if d_seg > kernelLen:
                num_pts = int(d_seg) + 1
                t = np.linspace(0, 1, num_pts, endpoint=True).reshape(-1, 1)
                data = p0 * t + p1 * (1 - t)
            else:
                data = np.array([p0, p1])

            # 获取在图像范围内的核偏移坐标
            cur_pc = GetPcKernelPc(data, kernelArr, imgShape)
            if cur_pc.size == 0:
                continue

            # 计算点到线段的最短距离：投影参数 t_proj
            # 将 cur_pc 投影到线段 (p0 -> p1) 上
            # 向量 v = p1 - p0
            # 对于每个点 q，投影参数 t = (q-p0)·v / (v·v)
            q = cur_pc - p0
            v_dot_v = np.dot(v, v)
            t_proj = np.dot(q, v) / v_dot_v
            t_proj = np.clip(t_proj, 0, 1)
            proj_pt = p0 + t_proj[:, np.newaxis] * v
            dist = np.linalg.norm(cur_pc - proj_pt, axis=1)

            # 更新距离场（取最小值）
            idx_z = cur_pc[:, 2]
            idx_y = cur_pc[:, 1]
            idx_x = cur_pc[:, 0]
            df_img[idx_z, idx_y, idx_x] = np.minimum(df_img[idx_z, idx_y, idx_x], dist)

    # 距离值到最终像素值的转换
    df_img = -np.log(minD + df_img / maxD * (1 - minD))
    df_img2 = (df_img / maskMax * 255).astype(np.uint8)

    tifffile.imwrite(tif_path, df_img2, compression='lzw')
    return swc_name


from multiprocessing import Pool
from functools import partial


def NeuralSwcToDF_parallel(swc_dir, mask_save_dir, shapes=(192, 192, 192), kernelLen=3, n_workers=None):
    """
    多进程版本：并行处理目录下所有 .swc 神经元骨架文件，生成距离场 TIFF。

    Parameters
    ----------
    swc_dir : str
        SWC 文件所在目录
    mask_save_dir : str
        输出 TIFF 的保存目录
    shapes : tuple of int (x, y, z)
        图像的原始空间尺寸（x 宽度, y 高度, z 深度）
    kernelLen : int, default=3
        核半径
    n_workers : int, optional
        并行进程数，默认为 CPU 核心数
    """
    # 确保输入输出目录存在
    os.makedirs(swc_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    # 形状转换：原始 shapes 为 (x, y, z)，内部距离场用 (z, y, x) 顺序
    imgShape = np.array(shapes[::-1], dtype=np.int32)  # 变为 (z, y, x)
    maxD = ((kernelLen ** 2) * 3) ** 0.5
    minD = 0.0697
    maskMax = -np.log(minD)

    # 预计算核偏移列表 (x, y, z)
    kernelArr = []
    for z in range(-kernelLen, kernelLen + 1):
        for y in range(-kernelLen, kernelLen + 1):
            for x in range(-kernelLen, kernelLen + 1):
                kernelArr.append([x, y, z])
    kernelArr = np.array(kernelArr, dtype=np.int32)

    # 收集所有 .swc 文件
    swc_files = [f for f in os.listdir(swc_dir) if f.lower().endswith('.swc')]
    if not swc_files:
        print("未找到任何 .swc 文件。")
        return

    print(f"找到 {len(swc_files)} 个 SWC 文件，使用 {n_workers or os.cpu_count()} 个进程并行处理...")

    # 部分固定参数的函数
    worker_func = partial(process_single_neural_swc,
                          swc_dir=swc_dir,
                          mask_save_dir=mask_save_dir,
                          imgShape=imgShape,
                          maxD=maxD,
                          minD=minD,
                          maskMax=maskMax,
                          kernelArr=kernelArr,
                          kernelLen=kernelLen)

    with Pool(processes=n_workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(worker_func, swc_files), 1):
            print(f"已处理: {i}/{len(swc_files)}")

    print("全部处理完成。")


if __name__ == '__main__':
    # root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_bv_res\FilterResults"
    # root = r"D:\SY\10GTestData\10GVesselTestData\Vessel_bv_res\FilterResults"
    # root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\TrainDataSet\res\FilterResults"
    # root = r"D:\SY\10GTestData\10GVesselTestData\Vessel_DataSet\TrainDataSet\res\FilterResults"
    # root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\TrainDataSet"
    # root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\testData"
    # root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\KS2_TrainSet"
    # root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\testData2"
    # root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\TrainDataSet"
    # root = r"D:\SY\10GTestData\10GVesselTestData\Vessel_DataSet\KS_TrainSet"
    root = r"D:\BaiduNetdiskDownload\Data"
    # root = r"D:\BaiduNetdiskDownload\val"
    # root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\KS_TrainSet\predict_res\PredictResults"
    # root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\TrainDataSet\predict_res\PredictResults"
    # root = r"D:\SY\10GTestData\10GVesselTestData\Vessel_DataSet\testData"
    # root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\KS_TrainSet"
    # root = r"D:\SY\10GTestData\10GVesselTestData\Vessel_DataSet\KS_TrainSet"
    swc_dir = os.path.join(root, "swc")
    # swc_dir = os.path.join(root, "Unet_swc")
    # swc_dir = os.path.join(root, "Predict_label2_swc")
    # swc_dir = os.path.join(root, "Swc")
    # swc_dir = os.path.join(root, "nnUnet_swc")
    # swc_dir = os.path.join(root, "Unet3DV3_swc")
    # swc_dir = os.path.join(root, "Unet_ks_swc")

    # swc_dir = os.path.join(root, "swc2")
    # swc_dir = os.path.join(root, "predict_swc")
    # swc_dir = os.path.join(root, "self-train_predict_swc")
    # swc_dir = os.path.join(root, "KS_predict_swc")

    # mask_save_dir = os.path.join(root, "mask")
    # mask_save_dir = os.path.join(root, "Unet_ks_mask")
    # mask_save_dir = os.path.join(root, "Unet_mask")
    # mask_save_dir = os.path.join(root, "Predict_label2_mask")
    # mask_save_dir = os.path.join(root, "nnUnet_swc_mask")
    # mask_save_dir = os.path.join(root, "Unet3DV3_mask")

    # mask_save_dir = os.path.join(root, "predict_mask")
    # mask_save_dir = os.path.join(root, "predict_swc_mask")
    # mask_save_dir = os.path.join(root, "self-train_mask2")
    # mask_save_dir = os.path.join(root, "swc2_mask")
    # mask_save_dir = os.path.join(root, "KS_mask")
    # mask_save_dir = os.path.join(root, "mask_4")
    # mask_save_dir = os.path.join(root, "mask_5")
    mask_save_dir = os.path.join(root, "mask")

    # shape = [64, 64, 64]  # x,y,z
    # NeuralSwcToDF(swc_dir, mask_save_dir, shape)

    NeuralSwcToDF_parallel(
        swc_dir=swc_dir,
        mask_save_dir=mask_save_dir,
        # shapes=(64, 64, 64),  # (width, height, depth)
        # kernelLen=2,  # 神经
        # kernelLen=3,  # 血管
        # kernelLen=5,  # 血管
        n_workers=int(os.cpu_count() / 2)
    )

