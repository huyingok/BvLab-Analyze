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


class CellSwcToMaskQThread(QThread):
    finish0 = pyqtSignal()
    warning0 = pyqtSignal(str)
    progress0 = pyqtSignal(str, np.ndarray, np.ndarray)
    progress1 = pyqtSignal(str, int)
    error0 = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(CellSwcToMaskQThread, self).__init__()
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
                    self.data_set(names)
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
        kernelLen = 10
        imgShape = np.array(shapes, dtype=np.int32)  # xyz
        maxD = ((kernelLen ** 2) * 3) ** 0.5
        minD = 0.0697

        os.makedirs(swc_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)

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
        kernelLen = 10
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

        os.makedirs(swc_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)

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
            swcData = np.loadtxt(swc_path, ndmin=2)
            dfImg[...] = maxD
            for item in swcData:
                if len(item) < 7:
                    continue
                p0 = item[2: 5]
                curPc = GetPcKernelPc(np.array([p0]), kernelArr, imgShape)
                d = np.linalg.norm(p0 - curPc, axis=1)
                dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]] = np.min([dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]], d],
                                                                      axis=0)
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

    def data_set(self, ls):
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


'''获取点云核点云'''


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
                swcData = []
            else:
                swcData = np.loadtxt(swc_path, ndmin=2)

            dfImg[...] = maxD
            for item in swcData:
                if len(item) < 7:
                    continue
                p0 = item[2: 5]
                curPc = GetPcKernelPc(np.array([p0]), kernelArr, imgShape)
                d = np.linalg.norm(p0 - curPc, axis=1)
                dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]] = np.min([dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]], d],
                                                                      axis=0)

            dfImg = -np.log(minD + dfImg / maxD * (1 - minD))
            dfImg2 = (dfImg / maskMax * 255).astype(np.uint8)
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


'''细胞标注转距离场'''


def CellSwcToDF(swc_dir, mask_save_dir, shape, kernelLen=10):
    imgShape = np.array(shape, dtype=np.int32)
    maxD = ((kernelLen ** 2) * 3) ** 0.5
    minD = 0.0697
    maskMax = -np.log(minD)
    dfImg = np.zeros(imgShape[::-1], dtype=np.float32)
    kernelArr = []
    for z in range(-kernelLen, kernelLen + 1):
        for y in range(-kernelLen, kernelLen + 1):
            for x in range(-kernelLen, kernelLen + 1):
                kernelArr.append([x, y, z])
    kernelArr = np.array(kernelArr, dtype=np.int32)
    os.makedirs(mask_save_dir, exist_ok=True)
    ls = [l for l in os.listdir(swc_dir) if ".swc" in l]
    lsLen = len(ls)
    for i, name in enumerate(ls):
        print('%d | %d' % (i + 1, lsLen), str(Path(name).stem))
        nameId = os.path.splitext(name)[0]
        swcData = np.loadtxt(join(swc_dir, name), ndmin=2)
        dfImg[...] = maxD
        for item in swcData:
            p0 = item[2: 5]
            curPc = GetPcKernelPc(np.array([p0]), kernelArr, imgShape)
            d = np.linalg.norm(p0 - curPc, axis=1)
            dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]] = np.min([dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]], d],
                                                                  axis=0)
        dfImg = -np.log(minD + dfImg / maxD * (1 - minD))
        dfImg2 = (dfImg / maskMax * 255).astype(np.uint8)
        tifffile.imwrite(join(mask_save_dir, nameId + '.tif'), dfImg2, compression='lzw')


def process_single_swc(swc_name, swc_dir, mask_save_dir, imgShape, maxD, minD, maskMax, kernelArr):
    """处理单个 SWC 文件，生成对应的距离场 TIFF 文件"""
    name_id = os.path.splitext(swc_name)[0]
    swc_path = os.path.join(swc_dir, swc_name)

    # 读取 SWC 数据（假设格式：id, type, x, y, z, radius, parent）
    swc_data = np.loadtxt(swc_path, ndmin=2)

    # 初始化距离场为最大值
    df_img = np.full(imgShape[::-1], maxD, dtype=np.float32)

    for item in swc_data:
        p0 = item[2:5]  # (x, y, z)
        cur_pc = GetPcKernelPc(np.array([p0]), kernelArr, imgShape)  # (M, 3)
        if cur_pc.size == 0:
            continue
        d = np.linalg.norm(p0 - cur_pc, axis=1)

        # 更新距离场：取已有距离与当前点距离的较小值
        idx_z = cur_pc[:, 2]
        idx_y = cur_pc[:, 1]
        idx_x = cur_pc[:, 0]
        current_vals = df_img[idx_z, idx_y, idx_x]
        df_img[idx_z, idx_y, idx_x] = np.minimum(current_vals, d)

    # 转换距离值为最终输出
    df_img = -np.log(minD + df_img / maxD * (1 - minD))
    df_img2 = (df_img / maskMax * 255).astype(np.uint8)

    # 保存为 TIFF
    out_path = os.path.join(mask_save_dir, name_id + '.tif')

    tifffile.imwrite(out_path, df_img2, compression='lzw')
    return swc_name  # 可选，用于调试


from multiprocessing import Pool
from functools import partial


def CellSwcToDF_parallel(swc_dir, mask_save_dir, shape=(272, 272, 144), kernelLen=10, n_workers=None):
    """
    多进程版本：并行处理目录下所有 .swc 文件，生成距离场 TIFF。

    Parameters
    ----------
    swc_dir : str
        SWC 文件所在目录
    mask_save_dir : str
        输出 TIFF 的保存目录
    shape : tuple of int
        图像的空间形状 (width, height, depth)
    kernelLen : int, default=10
        核半径
    n_workers : int, optional
        并行进程数，默认为 CPU 核心数
    """
    imgShape = np.array(shape, dtype=np.int32)
    maxD = ((kernelLen ** 2) * 3) ** 0.5
    minD = 0.0697
    maskMax = -np.log(minD)

    # 预计算核偏移列表
    kernel_arr = []
    for z in range(-kernelLen, kernelLen + 1):
        for y in range(-kernelLen, kernelLen + 1):
            for x in range(-kernelLen, kernelLen + 1):
                kernel_arr.append([x, y, z])
    kernel_arr = np.array(kernel_arr, dtype=np.int32)

    # 确保输出目录存在
    os.makedirs(mask_save_dir, exist_ok=True)

    # 收集所有 .swc 文件
    swc_files = [f for f in os.listdir(swc_dir) if f.lower().endswith('.swc')]
    if not swc_files:
        print("未找到任何 .swc 文件。")
        return

    print(f"找到 {len(swc_files)} 个 SWC 文件，使用 {n_workers or os.cpu_count()} 个进程并行处理...")

    # 创建部分固定参数的函数
    worker_func = partial(process_single_swc,
                          swc_dir=swc_dir,
                          mask_save_dir=mask_save_dir,
                          imgShape=imgShape,
                          maxD=maxD,
                          minD=minD,
                          maskMax=maskMax,
                          kernelArr=kernel_arr)

    # 使用进程池并行处理
    with Pool(processes=n_workers) as pool:
        # 使用 imap_unordered 可以实时获取完成的任务（如果需要进度提示可改用 tqdm）
        for i, _ in enumerate(pool.imap_unordered(worker_func, swc_files), 1):
            print(f"已处理: {i}/{len(swc_files)}")

    print("全部处理完成。")


'''Mask转Swc'''


def MaskToSwc(mask_dir, swc_save_dir):
    thre = 135
    os.makedirs(swc_save_dir, exist_ok=True)
    ls = [l for l in os.listdir(mask_dir) if ".tif" in l]
    lsLen = len(ls)
    for i, name in enumerate(ls):
        print('%d | %d' % (i + 1, lsLen), str(Path(name).stem))
        print()
        img = tifffile.imread(join(mask_dir, name))
        pc = np.where(img > thre)
        pc = np.array(pc).T
        """ 计算点集的质心 """
        # z_center = np.mean(pc[:, 0])
        # y_center = np.mean(pc[:, 1])
        # x_center = np.mean(pc[:, 2])
        with open(join(swc_save_dir, os.path.splitext(name)[0] + '.swc'), 'w') as f:
            for ii, item in enumerate(pc):
                f.write('%d %d %d %d %d %d -1\n' % (ii + 1, 1, item[2], item[1], item[0], 1))
            # f.write('%d %d %d %d %d %d -1\n' % (i + 1, 1, x_center, y_center, z_center, 1))


if __name__ == '__main__':
    # root = r"D:\SY\10GTestData\10GCellTestData\Cell_bv_res\FilterResults"
    # root = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\TrainDataSet\ins\FilterResults"
    # root = r"D:\SY\10GTestData\KS\Cell\Kennard_Stone_Img_res\PredictResults\data_pro\add_filter"
    root = r"D:\SY\10GTestData\KS\Cell\Kennard_Stone_Img_res\PredictResults\new_pro_data_115"

    # 标准
    # swc_dir = os.path.join(root, "swc_train_res")
    # swc_dir = os.path.join(root, "swc_train_res_cut")
    # mask_save_dir = os.path.join(root, "swc_train_res_mask")
    # swc_dir = os.path.join(root, "swc_std")
    # swc_dir = os.path.join(root, "swc_std_cut")
    # mask_save_dir = os.path.join(root, "swc_std_mask")
    # mask_save_dir = os.path.join(root, "swc_std_cut_mask")
    # 初始
    # swc_dir = os.path.join(root, "origin_predict_swc")
    # swc_dir = os.path.join(root, "origin_predict_swc_cut")
    # mask_save_dir = os.path.join(root, "origin_predict_mask")
    # mask_save_dir = os.path.join(root, "origin_predict_cut_mask")
    # KS
    # swc_dir = os.path.join(root, "KS_predict_swc")
    # swc_dir = os.path.join(root, "KS_predict_swc_cut")
    # mask_save_dir = os.path.join(root, "KS_mask")
    # mask_save_dir = os.path.join(root, "KS_cut_mask")
    # KS Pro
    # swc_dir = os.path.join(root, "KS_pro_predict_swc")
    # swc_dir = os.path.join(root, "KS_pro_predict_swc_cut")
    # mask_save_dir = os.path.join(root, "KS_pro_mask")
    # mask_save_dir = os.path.join(root, "KS_pro_cut_mask")
    # self-train
    # swc_dir = os.path.join(root, "self-train_predict_swc")
    # swc_dir = os.path.join(root, "self-train_predict_swc_cut")
    # mask_save_dir = os.path.join(root, "self-train_mask")
    # mask_save_dir = os.path.join(root, "self-train_cut_mask")

    # swc_dir = os.path.join(root, "nnUNet_swc")
    # swc_dir = os.path.join(root, "nnUNet_swc_cut")
    # swc_dir = os.path.join(root, "imagesTs_results_postprocessing_tif_swc_cut")
    # swc_dir = os.path.join(root, "cut_predict_swc")
    # mask_save_dir = os.path.join(root, "nnUNet_mask")
    # mask_save_dir = os.path.join(root, "nnUNet_cut_mask")
    # mask_save_dir = os.path.join(root, "imagesTs_results_postprocessing_tif_swc_cut_mask")
    # mask_save_dir = os.path.join(root, "cut_predict_mask")

    # swc_dir = os.path.join(root, "swc")
    # swc_dir = r"D:\SY\10GTestData\KS\Cell\Kennard_Stone_Img_res\PredictResults\new_pro_data_114\swc"
    # swc_dir = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\TrainDataSet\swc"
    swc_dir = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\KS_Pro_TrainSet\swc"
    # mask_save_dir = os.path.join(root, "mask")
    # mask_save_dir = r"D:\SY\10GTestData\KS\Cell\Kennard_Stone_Img_res\PredictResults\new_pro_data_114\mask_contrast\std_mask"
    # mask_save_dir = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\TrainDataSet\mask_10"
    mask_save_dir = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\KS_Pro_TrainSet\mask_12"

    # shape = (64, 64, 64)  # x,y,z
    # CellSwcToDF(swc_dir, mask_save_dir, shape, kernelLen=5)

    CellSwcToDF_parallel(
        swc_dir,
        mask_save_dir,
        # shape=(64, 64, 64),
        # kernelLen=5,
        kernelLen=12,
        n_workers=int(os.cpu_count() / 2)  # 可根据需要调整
    )

    # mask_dir = r"D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D-DDP\TrainDataSet\mask"
    # swc_save_dir = r"D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D-DDP\TrainDataSet\mask_to_swc"
    # MaskToSwc(mask_dir, swc_save_dir)


