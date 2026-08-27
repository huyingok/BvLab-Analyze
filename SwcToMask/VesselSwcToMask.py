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


class VesselSwcToMaskQThread(QThread):
    finish0 = pyqtSignal()
    warning0 = pyqtSignal(str)
    progress0 = pyqtSignal(str, np.ndarray, np.ndarray)
    progress1 = pyqtSignal(str, int)
    error0 = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(VesselSwcToMaskQThread, self).__init__()
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
            # ps.append(Process(target=swc_to_mask_process, args=(pathLsQue, finishQue, errorQue, resultQue)))
            ps.append(Process(target=swc_to_mask_from_radii_sub, args=(pathLsQue, finishQue, errorQue, resultQue)))
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


def swc_to_mask_from_radii_sub(pathLsQue, finishQue, errorQue, resultQue):
    """处理单个图像，用于并行处理"""
    while True:
        try:
            add = pathLsQue.get(timeout=5)

            if len(add) == 0:
                finishQue.put(-1)
                return

            swc_path, save_path, img_path, kernelLen, imgShape, maxD, minD = add

            if not os.path.getsize(swc_path):
                swcDataLs = []
            else:
                swcData = np.loadtxt(swc_path, ndmin=2)
                swcDataLs = SplitSwcData(swcData)
            dfImg = np.zeros(imgShape[::-1], dtype=np.float32)  # zyx
            dfImg[...] = maxD
            # dfImg2[...] = maxD2
            list_r = [kernelLen]
            img_list = [dfImg]
            maxD_list = [maxD]
            for swcData in swcDataLs:
                for ii, item in enumerate(swcData):
                    if item[-1] == -1:
                        continue
                    p0 = item[2: 5]
                    p1 = swcData[int(item[-1]) - 1, 2: 5]
                    v = (p1 - p0).reshape([1, 3])
                    p_r = int((int(item[5]) + int(swcData[int(item[-1]) - 1, 5])) / 2)
                    # p_r = int(item[5])
                    p_r = max(p_r, 3)  # 默认最小为3
                    # p_r = min(p_r, 3)
                    # p_r = int(item[5])
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

                    kernelArr = []
                    for z in range(-p_r, p_r + 1):
                        for y in range(-p_r, p_r + 1):
                            for x in range(-p_r, p_r + 1):
                                kernelArr.append([x, y, z])
                    kernelArr = np.array(kernelArr, dtype=np.int32)

                    curPc = GetPcKernelPc(data, kernelArr, imgShape[::-1])

                    proT = (curPc - p0).dot(v.T) / v.dot(v.T)
                    proT = np.clip(proT, 0, 1)
                    proP = p0 + proT * v
                    d = np.linalg.norm(proP - curPc, axis=1)
                    if np.isnan(d).any():
                        print()

                    if p_r in list_r:
                        id_ = list_r.index(p_r)
                        Img = img_list[id_]
                        Img[curPc[:, 2], curPc[:, 1], curPc[:, 0]] = np.min([Img[curPc[:, 2], curPc[:, 1], curPc[:, 0]], d],
                                                                            axis=0)
                        img_list[id_] = Img
                    else:
                        list_r.append(p_r)
                        m = ((p_r ** 2) * 3) ** 0.5
                        # m = maxD
                        maxD_list.append(m)
                        Img = np.zeros(imgShape[::-1], dtype=np.float32)
                        Img[...] = m
                        Img[curPc[:, 2], curPc[:, 1], curPc[:, 0]] = np.min([Img[curPc[:, 2], curPc[:, 1], curPc[:, 0]], d],
                                                                            axis=0)
                        img_list.append(Img)

            dfImg_ = np.array([])
            for i, d in enumerate(img_list):
                if i == 0:
                    maxD = maxD_list[i]
                    dfImg_ = -np.log(minD + d / maxD * (1 - minD))
                else:
                    maxD = maxD_list[i]
                    d = -np.log(minD + d / maxD * (1 - minD))
                    # 先压到 float32 0-1
                    if float(dfImg_.max()) != 0:
                        dfImg_ = dfImg_ / dfImg_.max()
                    if float(d.max()) != 0:
                        d = d / d.max()
                    # 融合
                    dfImg_ = np.maximum(dfImg_, d)

            dfImg_add = dfImg_
            if float(dfImg_add.max()) == 0:
                dfImg_add2 = dfImg_add.astype(np.uint8)
            else:
                dfImg_add2 = (dfImg_add / dfImg_add.max() * 255).astype(np.uint8)
            tifffile.imwrite(save_path, dfImg_add2, compression='lzw')

            dfImg2_2d = MaxProject(dfImg_add2, 0)
            img = tifffile.imread(img_path)
            img_2d = MaxProject(img, 1)

            resultQue.put((dfImg2_2d, img_2d))
            finishQue.put(1)
        except Exception as e:
            if errorQue.qsize() == 0:
                finishQue.put(1)
                print("\n=== 错误信息 ===")
                print(f"异常类型: {type(e).__name__}")
                print(f"错误信息: {e}")
                print("=== 错误位置 ===")
                tb = sys.exc_info()[2]
                for frame in traceback.extract_tb(tb):
                    print(f"  文件: {frame.filename}")
                    print(f"  行号: {frame.lineno}")
                    print(f"  函数: {frame.name}")
                    print(f"  代码: {frame.line}\n")
            errorQue.put(e)


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
            # dfImg2[dfImg2 < 23] = 0
            tifffile.imwrite(mask_save_path, dfImg2, compression='lzw')

            dfImg2_2d = MaxProject(dfImg2, 0)
            img = tifffile.imread(img_path)
            img_2d = MaxProject(img, 1)

            resultQue.put((dfImg2_2d, img_2d))
            finishQue.put(1)
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
