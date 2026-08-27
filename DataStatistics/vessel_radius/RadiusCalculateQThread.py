# -*- coding: utf-8 -*-
import os
from os.path import join
import time
import sys
import traceback
from pathlib import Path
import tempfile
import tifffile as tiff
import cv2
import shutil
import numpy as np
from multiprocessing import Process, Queue
from PyQt5.QtCore import QThread, pyqtSignal
from scipy.ndimage import distance_transform_edt
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from tqdm import tqdm
from functools import partial
import warnings
# from gpu_device_use import max_safe_workers
# from DataStatistics.vessel_radius.DataInterpolation import swc_data_interpolation
from DataStatistics.vessel_radius.optimized_radius_smooth import radius_smooth_advanced
from DataStatistics.vessel_radius.Tree_connect import SwcConnect, box_filter_asym
warnings.filterwarnings("ignore")


# 常量定义
EPSILON = 1e-6
MAX_STEPS = 100
STEP_SIZE = 1
THRESH = 0.5


class RadiusCalculateQThread(QThread):
    finish0 = pyqtSignal(str)
    progress_text = pyqtSignal(str, int)
    error0 = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(RadiusCalculateQThread, self).__init__()
        self.win = kwargs.get('win')
        self.logger = self.win.logger
        self.sct = SwcConnect()

    def run(self):
        try:
            self.new_text = ""
            analyze_dict = self.win.analyze_dict
            data_type = analyze_dict['data_type']
            if data_type == 'bv':
                resolution_ratio = analyze_dict['resolution_ratio']
                CutWorkFilesDir = analyze_dict['CutWorkFilesDir']
                self.Bv_calculate_radius(CutWorkFilesDir, resolution_ratio)
            elif data_type == 'small':
                img_path = analyze_dict['img_path']
                swc_path = analyze_dict['swc_path']
                save_path = join(Path(img_path).parent, "calculate_radius")
                resolution_ratio = analyze_dict['resolution_ratio']
                self.small_calculate_radius(img_path, swc_path, save_path, resolution_ratio)
            self.progress_text.emit("Radius calculation completed, label file has been replaced", 0)
            self.finish0.emit(self.new_text)
        except Exception as e:
            self.error0.emit(str(e))
            self.error_logger(e)

    """Error message"""

    def error_logger(self, e):
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

    def small_calculate_radius(self, img_dir, swc_dir, save_dir, resolution_ratio):
        img_names = [n for n in os.listdir(img_dir) if ".tif" in n]
        if not len(img_names):
            self.new_text = "The input image folder has no Tif format files"
            return
        swc_tree_dir = join(save_dir, "swc_tree")
        # swc_add_dir = join(save_dir, "swc_interpolation")
        swc_smooth_dir = join(save_dir, "swc_smooth")
        if os.path.exists(swc_smooth_dir):
            if os.path.isdir(swc_smooth_dir):
                shutil.rmtree(swc_smooth_dir)
        os.makedirs(swc_smooth_dir, exist_ok=True)
        swc_length_dir = join(save_dir, "swc_length")
        if os.path.exists(swc_length_dir):
            if os.path.isdir(swc_length_dir):
                shutil.rmtree(swc_length_dir)
        os.makedirs(swc_length_dir, exist_ok=True)
        # 断点连接、插值
        self.swc_tree_connect_sub(img_dir, swc_dir, swc_tree_dir, swc_length_dir,
                                  resolution_ratio=resolution_ratio)
        # 调用优化版本的函数，计算半径，Smooth
        self.progress_text.emit("Starting radius calculation", 0)
        self.calculate_vessel_radius(img_dir, swc_tree_dir, save_dir, swc_smooth_dir,
                                     resolution_ratio=resolution_ratio)  # 计算半径
        # 平滑处理结果替换
        # for si, img_name in enumerate(img_names):
        #     swc_name = Path(img_name).stem + ".swc"
        #     swc_smooth_path = join(swc_smooth_dir, swc_name)
        #     if os.path.getsize(swc_smooth_path):
        #         shutil.copy(swc_smooth_path, swc_dir)

    def Bv_calculate_radius(self, rootDir, resolution_ratio):
        names = [n for n in os.listdir(rootDir) if "." not in n]
        if not len(names):
            self.new_text = f"There is no block data in the {rootDir} folder"
            return
        start_time = time.time()
        for i, name in enumerate(names):
            print(name)
            root = join(rootDir, name)
            img_dir = join(root, "images")
            img_names = [n for n in os.listdir(img_dir) if ".tif" in n]
            if len(img_names):
                swc_dir = join(root, "swc")
                save_dir = join(root, "calculate_radius")
                swc_tree_dir = join(save_dir, "swc_tree")
                # swc_add_dir = join(save_dir, "swc_interpolation")
                swc_smooth_dir = join(save_dir, "swc_smooth")
                if os.path.exists(swc_smooth_dir):
                    if os.path.isdir(swc_smooth_dir):
                        shutil.rmtree(swc_smooth_dir)
                os.makedirs(swc_smooth_dir, exist_ok=True)
                swc_length_dir = join(save_dir, "swc_length")
                if os.path.exists(swc_length_dir):
                    if os.path.isdir(swc_length_dir):
                        shutil.rmtree(swc_length_dir)
                os.makedirs(swc_length_dir, exist_ok=True)
                # 断点连接
                self.swc_tree_connect_sub(img_dir, swc_dir, swc_tree_dir, swc_length_dir,
                                          resolution_ratio=resolution_ratio)
                # 调用优化版本的函数
                self.progress_text.emit("Loading calculation process", 0)
                self.calculate_vessel_radius(img_dir, swc_tree_dir, save_dir, swc_smooth_dir,
                                             resolution_ratio=resolution_ratio)  # 计算半径
                # 平滑处理结果替换
                # for si, img_name in enumerate(img_names):
                #     swc_name = Path(img_name).stem + ".swc"
                #     swc_smooth_path = join(swc_smooth_dir, swc_name)
                #     if os.path.getsize(swc_smooth_path):
                #         shutil.copy(swc_smooth_path, swc_dir)

            userTime = time.time() - start_time
            surplusTime = userTime / (i + 1) * (len(names) - i - 1)
            if surplusTime >= 0:
                logInfo = '[Total progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                    (i + 1) / len(names) * 100, userTime, surplusTime)
                self.progress_text.emit(logInfo, 0)

    def swc_tree_connect_sub(self, img_dir, swc_dir, save_root, swc_length_dir,
                             resolution_ratio=(1, 1, 1)):
        if os.path.exists(save_root):
            shutil.rmtree(save_root)
        os.makedirs(save_root, exist_ok=True)
        # if os.path.exists(swc_add_dir):
        #     shutil.rmtree(swc_add_dir)
        # os.makedirs(swc_add_dir, exist_ok=True)

        img_names = [n for n in os.listdir(img_dir) if ".tif" in n]

        # MNumber_max = max_safe_workers(single_mem_MB=800)['recommended_pool_size']
        # MNumber_min = 1

        MNumber_max = 20
        MNumber_min = 4

        pathLsQue = Queue()
        finishQue = Queue()
        errorQue = Queue()

        for img_name in img_names:
            stem = Path(img_name).stem
            swc_name = stem + ".swc"
            txt_name = stem + ".txt"
            img_path = join(img_dir, img_name)  # 图像路径
            swc_path = join(swc_dir, swc_name)  # swc标签路径
            save_path = join(save_root, swc_name)  # 连接标签路径
            # swc_add_path = join(swc_add_dir, swc_name)  # 插值标签路径
            swc_length_path = join(swc_length_dir, txt_name)  # 长度计算路径
            if os.path.exists(swc_path):
                if os.path.getsize(swc_path) > 0:
                    pathLsQue.put((img_path, swc_path, save_path, swc_length_path))
                else:
                    # with open(save_path, 'w') as f:
                    #     pass
                    safe_write(save_path, "", sync=True)
                    # with open(swc_length_path, 'w') as f:
                    #     f.write(f"{0.0}\n")

        file_count = pathLsQue.qsize()

        MNumber = int(min(os.cpu_count(), file_count) / 2) + 1

        for i in range(MNumber):
            pathLsQue.put(())

        # workLen = pathLsQue.qsize()
        original_work_len = file_count

        self.progress_text.emit("Start connecting", 0)
        ps = []
        for i in range(MNumber):
            ps.append(Process(target=swc_tree_connect_add, args=(pathLsQue, finishQue, errorQue,
                                                                 self.sct, resolution_ratio)))
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

            # 进程卡住检测
            if is_stop_count == 120:  # 大约一分钟无更新
                print("Starting process hang detection")
                # 检测连接没有处理的文件
                names = [Path(n).stem + ".tif" for n in os.listdir(save_root) if ".swc" in n]
                if len(names) == len(img_names):
                    break
                else:
                    add_list = []
                    s_names = set(names)
                    for img_name in img_names:
                        if img_name in s_names:
                            continue
                        else:
                            add_list.append(img_name)
                    if len(add_list) > 0:
                        for img_name in add_list:
                            print(f"{curMakeSize + 1} / {original_work_len}")
                            stem = Path(img_name).stem
                            swc_name = stem + ".swc"
                            txt_name = stem + ".txt"
                            img_path = join(img_dir, img_name)  # 图像路径
                            swc_path = join(swc_dir, swc_name)  # swc标签路径
                            save_path = join(save_root, swc_name)  # 连接标签路径
                            # swc_add_path = join(swc_add_dir, swc_name)  # 插值标签路径
                            swc_length_path = join(swc_length_dir, txt_name)  # 长度计算路径
                            swc_tree_connect(img_path, swc_path, save_path, swc_length_path, self.sct, resolution_ratio)
                            curMakeSize += 1
            # 进程报错
            if errorQue.qsize():
                e = errorQue.get()
                for p in ps:
                    p.terminate()
                self.error_logger(e)
                self.is_error = 1
                self.error0.emit(str(e))
                return

            # 更新进度
            if curMakeSize > 0:
                userTime = time.time() - start_time
                surplusTime = userTime / curMakeSize * (original_work_len - curMakeSize)
                logInfo = f'[Connection progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                    curMakeSize / original_work_len * 100, userTime, surplusTime)
                self.progress_text.emit(logInfo, 1)

            is_stop_count += 1
            time.sleep(0.5)

        # 等待所有进程完成
        for i, p in enumerate(ps):
            p.join(timeout=1)  # 最多等待5Second
            if p.is_alive():
                p.terminate()
            if i == 0:
                userTime = time.time() - start_time
                logInfo = f'[Connection progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                    100, userTime, 0)
                self.progress_text.emit(logInfo, 1)
            logInfo = f'[End process progress %.2f%%]' % ((i + 1) / MNumber * 100)
            self.progress_text.emit(logInfo, i)
            time.sleep(0.2)

    def calculate_vessel_radius(self, img_dir, swc_dir, save_dir, swc_smooth_dir,
                                resolution_ratio=(1, 1, 1), max_workers=None):
        """
        优化版本的血管半径计算函数
        使用并行处理和向量化操作加速
        Parameter:
        img_dir: str 原图目录
        seg_dir: str 分割结果目录
        swc_dir: str 骨架目录
        save_dir: str 保存路径
        resolution_ratio: tuple 分辨率比例
        max_workers: int 最大工作进程数，None表示使用默认值
        """
        # 创建保存目录
        DirectionDiameterDir = join(save_dir, "DirectionDiameter")
        RadiiSwcDir = join(save_dir, "RadiiSwc")
        BranchDir = join(save_dir, "Branch")

        for dir_path in [DirectionDiameterDir, RadiiSwcDir, BranchDir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)

        # 获取图像列表
        img_names = [n for n in os.listdir(img_dir) if ".tif" in n]  # 限制只处理第一张图像
        total_len = len(img_names)
        if total_len == 0:
            print("No image files to process were found.")
            return

        # MNumber_max = max_safe_workers(single_mem_MB=800)['recommended_pool_size']
        # MNumber_min = 1

        MNumber_max = 20
        MNumber_min = 4

        pathLsQue = Queue()
        finishQue = Queue()
        errorQue = Queue()

        # start_time = time.time()
        # 创建部分函数
        process_func = partial(
            process_single_image,
            img_dir=img_dir,
            # seg_dir=seg_dir,
            swc_dir=swc_dir,
            RadiiSwcDir=RadiiSwcDir,
            DirectionDiameterDir=DirectionDiameterDir,
            BranchDir=BranchDir,
            swc_smooth_dir=swc_smooth_dir,
            resolution_ratio=resolution_ratio
        )

        # 并行处理图像
        if total_len > 1 and max_workers != 1:  # 如果只有一张图像，使用串行处理避免进程创建开销
            for img_name in img_names:
                swc_name = Path(img_name).stem + ".swc"
                txt_name = Path(img_name).stem + ".txt"
                img_path = join(str(img_dir), img_name)
                swc_path = join(str(swc_dir), swc_name)
                RadiiSwcPath = join(RadiiSwcDir, swc_name)
                DirectionDiameterPath = join(DirectionDiameterDir, txt_name)
                BranchPath = join(BranchDir, txt_name)
                swc_smooth_path = join(swc_smooth_dir, swc_name)

                if not os.path.exists(swc_path):
                    safe_write(swc_path, "", sync=True)
                    safe_write(RadiiSwcPath, "", sync=True)
                    safe_write(DirectionDiameterPath, "", sync=True)
                    safe_write(BranchPath, "", sync=True)
                    safe_write(swc_smooth_path, "", sync=True)
                    # with open(swc_path, 'w') as f:
                    #     pass
                    # with open(RadiiSwcPath, 'w') as f:
                    #     pass
                    # with open(DirectionDiameterPath, 'w') as f:
                    #     pass
                    # with open(BranchPath, 'w') as f:
                    #     pass
                    # with open(swc_smooth_path, 'w') as f:
                    #     pass
                else:
                    if os.path.getsize(swc_path):
                        # 优化图像读取
                        img = tiff.imread(img_path)
                        if np.max(img) == 0:
                            safe_write(RadiiSwcPath, "", sync=True)
                            safe_write(DirectionDiameterPath, "", sync=True)
                            safe_write(BranchPath, "", sync=True)
                            safe_write(swc_smooth_path, "", sync=True)
                            # with open(RadiiSwcPath, 'w') as f:
                            #     pass
                            # with open(DirectionDiameterPath, 'w') as f:
                            #     pass
                            # with open(BranchPath, 'w') as f:
                            #     pass
                            # with open(swc_smooth_path, 'w') as f:
                            #     pass
                        else:
                            pathLsQue.put((img_path,
                                           swc_path,
                                           RadiiSwcPath,
                                           DirectionDiameterPath,
                                           BranchPath,
                                           swc_smooth_path,
                                           img,
                                           resolution_ratio))
                    else:
                        safe_write(RadiiSwcPath, "", sync=True)
                        safe_write(DirectionDiameterPath, "", sync=True)
                        safe_write(BranchPath, "", sync=True)
                        safe_write(swc_smooth_path, "", sync=True)
                        # with open(RadiiSwcPath, 'w') as f:
                        #     pass
                        # with open(DirectionDiameterPath, 'w') as f:
                        #     pass
                        # with open(BranchPath, 'w') as f:
                        #     pass
                        # with open(swc_smooth_path, 'w') as f:
                        #     pass

            file_count = pathLsQue.qsize()

            MNumber = int(min(os.cpu_count(), file_count) / 2) + 1

            for i in range(MNumber):
                pathLsQue.put(())

            original_work_len = file_count

            ps = []
            for i in range(MNumber):
                ps.append(Process(target=process_single_image_sub, args=(pathLsQue, finishQue, errorQue)))
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

                # 进程卡住检测
                if is_stop_count == 120:  # 大约一分钟无更新
                    print("Starting process hang detection")
                    # 检测平滑结果没有处理的文件
                    names = [Path(n).stem + ".tif" for n in os.listdir(swc_smooth_dir) if ".swc" in n]
                    if len(names) == len(img_names):
                        break
                    else:
                        add_list = []
                        s_names = set(names)
                        for img_name in img_names:
                            if img_name in s_names:
                                continue
                            else:
                                add_list.append(img_name)
                        if len(add_list) > 0:
                            for img_name in add_list:
                                print(f"{curMakeSize + 1} / {original_work_len}")
                                swc_name = Path(img_name).stem + ".swc"
                                txt_name = Path(img_name).stem + ".txt"
                                img_path = join(str(img_dir), img_name)
                                swc_path = join(str(swc_dir), swc_name)
                                RadiiSwcPath = join(RadiiSwcDir, swc_name)
                                DirectionDiameterPath = join(DirectionDiameterDir, txt_name)
                                BranchPath = join(BranchDir, txt_name)
                                swc_smooth_path = join(swc_smooth_dir, swc_name)
                                process_single_image_one(img_path, swc_path, RadiiSwcPath, DirectionDiameterPath,
                                                         BranchPath, swc_smooth_path, resolution_ratio)
                                curMakeSize += 1
                # 进程报错
                if errorQue.qsize():
                    e = errorQue.get()
                    for p in ps:
                        p.terminate()
                    self.error_logger(e)
                    self.is_error = 1
                    self.error0.emit(str(e))
                    return

                # 更新进度
                if curMakeSize > 0:
                    userTime = time.time() - start_time
                    surplusTime = userTime / curMakeSize * (original_work_len - curMakeSize)
                    logInfo = f'[Calculation progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        curMakeSize / original_work_len * 100, userTime, surplusTime)
                    self.progress_text.emit(logInfo, 1)

                is_stop_count += 1
                time.sleep(0.5)

            # 等待所有进程完成
            for i, p in enumerate(ps):
                p.join(timeout=1)  # 最多等待5Second
                if p.is_alive():
                    p.terminate()
                if i == 0:
                    userTime = time.time() - start_time
                    logInfo = f'[Calculation progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        100, userTime, 0)
                    self.progress_text.emit(logInfo, 1)
                logInfo = f'[End process progress %.2f%%]' % ((i + 1) / MNumber * 100)
                self.progress_text.emit(logInfo, i)
                time.sleep(0.2)

            # futs = []
            # with ProcessPoolExecutor(max_workers=max_workers) as exe:
            #     for name in img_names:
            #         futs.append(exe.submit(process_func, name))  # 返回 Future
            #
            #     # 每完成一个 Future 就更新一次进度条
            #     results = []
            #     for f in tqdm(as_completed(futs), total=len(futs)):
            #         results.append(f.result())

            # with ProcessPoolExecutor(max_workers=max_workers) as executor:
            #     results = list(executor.map(process_func, img_names))
            # success_count = sum(results)
            # print(f"成功处理 {success_count}/{total_len} 张图像")
        else:
            # 单图像处理
            for i, img_name in enumerate(img_names):
                st = time.time()
                process_func(img_name)
                print(f"{i + 1}/{total_len}, {img_name}, Processing time: {time.time() - st:.2f} Second")

        # total_time = time.time() - start_time
        # print(f"总处理时间: {total_time:.2f}Second")
        # print("优化版本计算完成")

'''获取点云核点云'''

def GetPcKernelPc(pc, kernelArr, imgShape):
    curPc = (pc[:, None] + kernelArr[None]).reshape([-1, 3])
    curPc = np.round(curPc).astype(np.int32)
    curPc[curPc < 0] = 0
    curPc[curPc[:, 2] > imgShape[0] - 1, 2] = imgShape[0] - 1
    curPc[curPc[:, 1] > imgShape[1] - 1, 1] = imgShape[1] - 1
    curPc[curPc[:, 0] > imgShape[2] - 1, 0] = imgShape[2] - 1
    curPc = np.unique(curPc, axis=0)
    return curPc

def swc_to_mask(shapes, swcPath):
    kernelLen = 1
    # shapes = mask.shape[::-1]  # xyz
    imgShape = np.array(shapes, dtype=np.int32)  # xyz
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

    Data = np.loadtxt(swcPath, ndmin=2)  # 读取骨架文件
    swcDataLs = SplitSwcData(Data)
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
    return dfImg2

def autoAdjustGray(img):
    """优化版本：使用numpy向量化操作优化灰度调整"""
    if img.dtype != np.uint8:
        img = ((1 * img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    tImg = img[img > 0]

    if len(tImg) == 0:
        return img

    # 优化排序和阈值计算
    sorted_signals = np.sort(tImg)[::-1]
    per = 0.098
    threshold_index = int(len(sorted_signals) * per)
    threshold = sorted_signals[min(threshold_index, len(sorted_signals) - 1)]

    grayMax = tImg.max()
    grayMin = cv2.threshold(tImg, tImg.min(), grayMax, cv2.THRESH_OTSU)[0]
    grayMin0 = grayMin

    # 向量化操作
    minInd = tImg < grayMin
    if minInd.sum() / tImg.size > 0.95:
        tImg = tImg[minInd]
        if len(tImg) > 0:
            grayMin = cv2.threshold(tImg, tImg.min(), grayMax, cv2.THRESH_OTSU)[0]

    grayMax = int(min((grayMin + grayMax) / 2, grayMin * 2)) or 1

    # 简化阈值调整逻辑
    th_adjust = grayMin0 - grayMax

    if th_adjust > 9:
        th = grayMax
    elif -20 <= th_adjust <= 9:
        th = threshold + 5
        if th <= 15:
            th += 5
    else:
        th = int((grayMax + grayMin0 + threshold) / 3)

    # 向量化操作应用阈值
    img[img < th] = 0
    img[img > 0] = 255
    return img

def find_edge_point_vectorized(label_image, center, direction, step=STEP_SIZE, max_steps=MAX_STEPS, thresh=THRESH):
    """向量化版本：同时处理多个中心点和方向"""

    # 检查无效方向
    if np.isnan(direction).any():
        return None

    x, y, z = center

    # 预计算所有可能的点
    for i in range(max_steps):
        x_new = x + direction[0] * step * (i + 1)
        y_new = y + direction[1] * step * (i + 1)
        z_new = z + direction[2] * step * (i + 1)

        # 边界检查
        if (z_new < 0 or z_new >= label_image.shape[0] or
                y_new < 0 or y_new >= label_image.shape[1] or
                x_new < 0 or x_new >= label_image.shape[2]):
            return None

        # 获取当前点灰度值
        current_intensity = label_image[int(z_new), int(y_new), int(x_new)]

        # 判断边界条件
        if current_intensity == 0 and i > 0:
            # 回溯到边界内部
            x_final = x + direction[0] * step * (i + 1 - thresh)
            y_final = y + direction[1] * step * (i + 1 - thresh)
            z_final = z + direction[2] * step * (i + 1 - thresh)
            edge_point = np.array([x_final, y_final, z_final])
            return edge_point
    return None


def calculate_distance(center, edge_point, resolution_ratio=(1, 1, 1)):
    """计算血管半径"""
    if edge_point is None:
        return None

    dz = (edge_point[0] - center[0]) * resolution_ratio[2]
    dy = (edge_point[1] - center[1]) * resolution_ratio[1]
    dx = (edge_point[2] - center[2]) * resolution_ratio[0]
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def find_perpendicular_directions(centerline_directions):
    """向量化版本：同时处理多个中心线方向向量"""
    all_perpendicular = []
    small_vector = np.array([EPSILON, EPSILON, EPSILON])

    for v in centerline_directions:
        # 第一个垂直方向
        if abs(v[0]) > abs(v[1]):
            perp1 = np.array([-v[1], v[0], 0])
        else:
            perp1 = np.array([v[2], 0, -v[0]])

        # 处理零向量情况
        if np.linalg.norm(perp1) == 0:
            perp1 = small_vector
        else:
            perp1 = perp1 / np.linalg.norm(perp1)

        # 第二个垂直方向（叉乘）
        perp2 = np.cross(v, perp1)
        if np.linalg.norm(perp2) == 0:
            perp2 = small_vector
        else:
            perp2 = perp2 / np.linalg.norm(perp2)

        # 四个方向：正负两个方向
        all_perpendicular.append([perp1, -perp1, perp2, -perp2])

    return all_perpendicular

def process_single_swc(swcData, edt, resolution_ratio, sums):
    """处理单个SWCData，用于并行处理"""
    swc_lines = []
    dir_lines = []
    branch_lines = []
    add_sums = 0

    if len(swcData) <= 1:
        return swc_lines, dir_lines, branch_lines, add_sums

    # Preprocessing：计算所有方向向量
    center_points = swcData[:, 2:5]
    parent_indices = swcData[:, -1]

    # 计算父节点索引（向量化）
    pre_indices = np.where(parent_indices == -1, 1, parent_indices - 1).astype(np.int64)
    pre_points = swcData[pre_indices, 2:5]
    directions = (pre_points - center_points).reshape(-1, 3)

    # 批量计算垂直方向
    all_perpendicular = find_perpendicular_directions(directions)

    # 处理每个点
    for ii, item in enumerate(swcData):
        p0 = item[2:5]
        v = directions[ii]
        perpendicular_directions = all_perpendicular[ii]
        v0 = perpendicular_directions[0]
        v1 = perpendicular_directions[2]

        if item[-1] == -1:
            pre_index = 1
        else:
            pre_index = int(item[-1]) - 1

        # 计算分支点
        if item[-1] + 1 != item[0] and item[-1] != -1:
            b_v1 = p0 - pre_points[ii]
            # 找到正确的子节点
            item2 = swcData[pre_index + 1]
            # child_indices = np.where(swcData[:, -1] == item[0])[0]
            # if len(child_indices) > 0:
            if int(item2[-1]) + 1 == int(item2[0]):
                # p2 = swcData[child_indices[0], 2:5]
                p2 = item2[2:5]
                b_v2 = p2 - pre_points[ii]
                branch_lines.append(
                    f"{pre_points[ii][0]} {pre_points[ii][1]} {pre_points[ii][2]} "
                    f"{b_v1[0]:.6f} {b_v1[1]:.6f} {b_v1[2]:.6f} "
                    f"{b_v2[0]:.6f} {b_v2[1]:.6f} {b_v2[2]:.6f}\n"
                )

        # 计算半径
        radii = []
        for direction in perpendicular_directions:
            edge_point = find_edge_point_vectorized(edt, p0, direction)
            if edge_point is not None:
                radius = calculate_distance(p0, edge_point, resolution_ratio)
                # 结合EDT距离进行平均
                current_edt = edt[min(int(p0[2]), edt.shape[0] - 1),
                                  min(int(p0[1]), edt.shape[1] - 1),
                                  min(int(p0[0]), edt.shape[2] - 1),]
                radius = (radius + current_edt) / 2
                radii.append(radius)
            else:
                current_edt = edt[min(int(p0[2]), edt.shape[0] - 1),
                                  min(int(p0[1]), edt.shape[1] - 1),
                                  min(int(p0[0]), edt.shape[2] - 1),]
                radii.append(max(current_edt, 0.1))

        # 计算平均半径
        if np.sum(radii) > 0:
            avg_radii = [np.mean(radii[i:i + 2]) for i in range(0, len(radii), 2)]
            rp1 = avg_radii[0]
            rp2 = avg_radii[1]

            # 异常值处理
            if abs(rp1 - rp2) > 5:
                final_avg_radius = min(rp1, rp2)
            else:
                final_avg_radius = np.mean(avg_radii)
        else:
            final_avg_radius = 1

        final_avg_radius = max(final_avg_radius, 1)

        # 生成输出行
        if int(item[6]) == -1:
            swc_lines.append(
                f"{item[0] + sums} {item[1]} {item[2]} {item[3]} {item[4]} {final_avg_radius:.3f} {item[6]}\n"
            )
            dir_lines.append(
                f"{item[0] + sums} {item[2]} {item[3]} {item[4]} "
                f"{v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f} "
                f"{v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f} {item[6]}\n"
            )
        else:
            swc_lines.append(
                f"{item[0] + sums} {item[1]} {item[2]} {item[3]} {item[4]} {final_avg_radius:.3f} {item[6] + sums}\n"
            )
            dir_lines.append(
                f"{item[0] + sums} {item[2]} {item[3]} {item[4]} "
                f"{v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f} "
                f"{v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f} {item[6] + sums}\n"
            )

        add_sums += 1

    return swc_lines, dir_lines, branch_lines, add_sums


def SplitSwcData(swcData):
    """优化版本：使用向量化操作分割SWCData"""
    indLs = np.where(swcData[:, -1] == -1)[0].tolist() + [swcData.shape[0]]
    swcDataLs = []

    for i in range(len(indLs) - 1):
        data = swcData[indLs[i]: indLs[i + 1]]
        sp = data[0, 0]
        data[:, 0] -= sp - 1
        data[1:, -1] -= sp - 1
        swcDataLs.append(data)

    return swcDataLs


def calculate_radii(swc_path, RadiiSwcPath, DirectionDiameterPath, BranchPath, edt, resolution_ratio):
    """优化版本：批量写入文件，减少I/O操作"""
    if not os.path.exists(swc_path):
        return

    swcData = np.loadtxt(swc_path, ndmin=2)
    swcDataLs = SplitSwcData(swcData)

    # 批量收集所有输出行
    swc_lines_all = []
    dir_lines_all = []
    branch_lines_all = []
    sums = 0

    for swcData in swcDataLs:
        swc_lines, dir_lines, branch_lines, add_sums = process_single_swc(
            swcData, edt, resolution_ratio, sums
        )
        swc_lines_all.extend(swc_lines)
        dir_lines_all.extend(dir_lines)
        branch_lines_all.extend(branch_lines)
        sums += add_sums

    safe_write(RadiiSwcPath, ''.join(swc_lines_all), sync=True)
    safe_write(DirectionDiameterPath, ''.join(dir_lines_all), sync=True)
    safe_write(BranchPath, ''.join(branch_lines_all), sync=True)


def process_single_image_one(img_path, swc_path, RadiiSwcPath, DirectionDiameterPath, BranchPath, swc_smooth_path,
                             resolution_ratio):
    """处理单个图像，用于并行处理"""
    if not os.path.exists(swc_path):
        safe_write(swc_path, "", sync=True)
        safe_write(RadiiSwcPath, "", sync=True)
        safe_write(DirectionDiameterPath, "", sync=True)
        safe_write(BranchPath, "", sync=True)
        safe_write(swc_smooth_path, "", sync=True)
        # with open(swc_path, 'w') as f:
        #     pass
        # with open(RadiiSwcPath, 'w') as f:
        #     pass
        # with open(DirectionDiameterPath, 'w') as f:
        #     pass
        # with open(BranchPath, 'w') as f:
        #     pass
        # with open(swc_smooth_path, 'w') as f:
        #     pass
    else:
        if os.path.getsize(swc_path):
            # 优化图像读取
            img = tiff.imread(img_path)
            if np.max(img) == 0:
                safe_write(RadiiSwcPath, "", sync=True)
                safe_write(DirectionDiameterPath, "", sync=True)
                safe_write(BranchPath, "", sync=True)
                safe_write(swc_smooth_path, "", sync=True)
                # with open(RadiiSwcPath, 'w') as f:
                #     pass
                # with open(DirectionDiameterPath, 'w') as f:
                #     pass
                # with open(BranchPath, 'w') as f:
                #     pass
                # with open(swc_smooth_path, 'w') as f:
                #     pass
                return True

            img = ((1 * img - img.min()) / (max(1e-5, img.max() - img.min())) * 255).astype(np.uint8)

            # 标签处理
            shapes = img.shape[::-1]
            mask = swc_to_mask(shapes, swc_path)
            mask = (mask > 0).astype(np.uint8) * 255

            # 优化分割结果处理
            # seg = tiff.imread(seg_path)
            # seg = (seg > 0).astype(np.uint8) * 255

            # 优化信号提取
            signal = autoAdjustGray(img.copy())

            # 先压到 float32 0-1
            signal = signal / signal.max()
            mask = mask / mask.max()
            # 融合
            mask = np.maximum(signal, mask)
            # 归一化到uint8
            mask = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(np.uint8)

            signal = (mask > 0).astype(bool)

            # CalculateEDT
            edt = distance_transform_edt(signal)

            # 计算半径
            calculate_radii(swc_path, RadiiSwcPath, DirectionDiameterPath, BranchPath, edt, resolution_ratio)

            # 平滑处理
            radius_smooth_advanced(RadiiSwcPath, swc_smooth_path, limit_r=0.5)
        else:
            safe_write(RadiiSwcPath, "", sync=True)
            safe_write(DirectionDiameterPath, "", sync=True)
            safe_write(BranchPath, "", sync=True)
            safe_write(swc_smooth_path, "", sync=True)
            # with open(RadiiSwcPath, 'w') as f:
            #     pass
            # with open(DirectionDiameterPath, 'w') as f:
            #     pass
            # with open(BranchPath, 'w') as f:
            #     pass
            # with open(swc_smooth_path, 'w') as f:
            #     pass

def process_single_image(img_name, img_dir, swc_dir, RadiiSwcDir, DirectionDiameterDir, BranchDir, swc_smooth_dir,
                         resolution_ratio):
    """处理单个图像，用于并行处理"""
    try:
        swc_name = Path(img_name).stem + ".swc"
        txt_name = Path(img_name).stem + ".txt"

        img_path = join(str(img_dir), img_name)
        # seg_path = join(str(seg_dir), img_name)
        swc_path = join(str(swc_dir), swc_name)

        RadiiSwcPath = join(RadiiSwcDir, swc_name)
        DirectionDiameterPath = join(DirectionDiameterDir, txt_name)
        BranchPath = join(BranchDir, txt_name)

        swc_smooth_path = join(swc_smooth_dir, swc_name)

        if not os.path.exists(swc_path):
            safe_write(swc_path, "", sync=True)
            safe_write(RadiiSwcPath, "", sync=True)
            safe_write(DirectionDiameterPath, "", sync=True)
            safe_write(BranchPath, "", sync=True)
            safe_write(swc_smooth_path, "", sync=True)
            # with open(swc_path, 'w') as f:
            #     pass
            # with open(RadiiSwcPath, 'w') as f:
            #     pass
            # with open(DirectionDiameterPath, 'w') as f:
            #     pass
            # with open(BranchPath, 'w') as f:
            #     pass
            # with open(swc_smooth_path, 'w') as f:
            #     pass
        else:
            if os.path.getsize(swc_path):
                # 优化图像读取
                img = tiff.imread(img_path)
                if np.max(img) == 0:
                    safe_write(RadiiSwcPath, "", sync=True)
                    safe_write(DirectionDiameterPath, "", sync=True)
                    safe_write(BranchPath, "", sync=True)
                    safe_write(swc_smooth_path, "", sync=True)
                    # with open(RadiiSwcPath, 'w') as f:
                    #     pass
                    # with open(DirectionDiameterPath, 'w') as f:
                    #     pass
                    # with open(BranchPath, 'w') as f:
                    #     pass
                    # with open(swc_smooth_path, 'w') as f:
                    #     pass
                    return True

                img = ((1 * img - img.min()) / (max(1e-5, img.max() - img.min())) * 255).astype(np.uint8)

                # 标签处理
                shapes = img.shape[::-1]
                mask = swc_to_mask(shapes, swc_path)
                mask = (mask > 0).astype(np.uint8) * 255

                # 优化分割结果处理
                # seg = tiff.imread(seg_path)
                # seg = (seg > 0).astype(np.uint8) * 255

                # 优化信号提取
                signal = autoAdjustGray(img.copy())

                # 先压到 float32 0-1
                signal = signal / signal.max()
                mask = mask / mask.max()
                # 融合
                mask = np.maximum(signal, mask)
                # 归一化到uint8
                mask = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(np.uint8)

                signal = (mask > 0).astype(bool)

                # CalculateEDT
                edt = distance_transform_edt(signal)

                # 计算半径
                calculate_radii(swc_path, RadiiSwcPath, DirectionDiameterPath, BranchPath, edt, resolution_ratio)

                # 平滑处理
                radius_smooth_advanced(RadiiSwcPath, swc_smooth_path, limit_r=0.5)
            else:
                safe_write(RadiiSwcPath, "", sync=True)
                safe_write(DirectionDiameterPath, "", sync=True)
                safe_write(BranchPath, "", sync=True)
                safe_write(swc_smooth_path, "", sync=True)
                # with open(RadiiSwcPath, 'w') as f:
                #     pass
                # with open(DirectionDiameterPath, 'w') as f:
                #     pass
                # with open(BranchPath, 'w') as f:
                #     pass
                # with open(swc_smooth_path, 'w') as f:
                #     pass
    except Exception as e:
        print(img_dir, img_name)
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
    return True

def process_single_image_sub(pathLsQue, finishQue, errorQue):
    """处理单个图像，用于并行处理"""
    while True:
        try:
            add = pathLsQue.get(timeout=5)

            if len(add) == 0:
                finishQue.put(-1)
                return

            (img_path,
             swc_path,
             RadiiSwcPath,
             DirectionDiameterPath,
             BranchPath,
             swc_smooth_path,
             img,
             resolution_ratio) = add

            if img.dtype != np.uint8:
                img = ((1 * img - img.min()) / (max(1e-5, img.max() - img.min())) * 255).astype(np.uint8)

            # 标签处理
            shapes = img.shape[::-1]
            mask = swc_to_mask(shapes, swc_path)
            mask = (mask > 0).astype(np.uint8) * 255
            # 优化分割结果处理
            # seg = tiff.imread(seg_path)
            # seg = (seg > 0).astype(np.uint8) * 255
            # 优化信号提取
            signal = autoAdjustGray(img.copy())
            # 先压到 float32 0-1
            signal = signal / signal.max()
            mask = mask / mask.max()
            # 融合
            mask = np.maximum(signal, mask)
            # 归一化到uint8
            mask = ((mask - mask.min()) / (mask.max() - mask.min()) * 255).astype(np.uint8)
            signal = (mask > 0).astype(bool)
            # CalculateEDT
            edt = distance_transform_edt(signal)
            # 计算半径
            # print(swc_path)
            calculate_radii(swc_path, RadiiSwcPath, DirectionDiameterPath, BranchPath, edt, resolution_ratio)
            # 平滑处理
            radius_smooth_advanced(RadiiSwcPath, swc_smooth_path, limit_r=0.5)
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


def swc_tree_connect(img_path, swc_path, save_path, swc_length_path, sct, resolution_ratio):
    is_ok = False
    if os.path.exists(swc_path):
        if os.path.isfile(swc_path):
            if os.path.getsize(swc_path):
                is_ok = True

    if not is_ok:
        safe_write(save_path, "", sync=True)
        return

    # 存在标签
    img, shapes = sct.read_img(img_path)  # 读图
    if np.max(img) == 0:
        safe_write(save_path, "", sync=True)
        return

    swc_data = sct.read_swc(swc_path)  # 读swcTag
    swcDataLs = sct.collect_swc_tree(swc_data)  # Group

    new_points_dict = {}  # 新点集
    new_endpoints_dict = {}  # 新末端点
    tuple_points_all = []  # 线段集
    # 收集
    for ii, swcData in enumerate(swcDataLs):
        new_points, new_endpoints, tuple_points = sct.check_swc_endpoint(swcData, min_s=(2, 2, 2))
        # 去除长度较小的分支
        if len(tuple_points) == 0:
            continue
        if len(swcData) <= 10:
            v_len = total_length_vox(tuple_points, (1, 1, 1))
            if v_len <= 3:
                continue

        new_points_dict[str(ii)] = new_points
        new_endpoints_dict[str(ii)] = new_endpoints
        tuple_points_all += tuple_points
    # 分支树连接关系
    tree_connect_set = set()
    # Connect
    for name, new_endpoints in new_endpoints_dict.items():
        np_check_points = None
        for n, new_points in new_points_dict.items():
            if not len(new_points):
                continue
            if name != n:
                if np_check_points is None:
                    np_check_points = new_points
                else:
                    if len(np_check_points):
                        np_check_points = np.concatenate([np_check_points, new_points], axis=0)

        if np_check_points is None or not len(np_check_points):
            continue

        for point_list in new_endpoints:
            point0 = np.array(point_list[0])
            point = np.array(point_list[-1])

            if tuple(point) in tree_connect_set:
                continue

            axis_u = point - point0

            # mask = box_filter(np_check_points, point, axis_u, 40, 4, 4)
            mask = box_filter_asym(np_check_points, point, axis_u, 6, 3, len_v=6, len_w=6)

            # print('落入长方体的点数:', mask.sum())
            selected = np_check_points[mask]  # 即为所求

            # 3. 最近点
            if selected.size:  # 框内非空
                dist2 = np.sum((selected - point) ** 2, axis=1)  # 平方距离
                nearest = selected[dist2.argmin()]  # 形状 (3,)
                min_dist = np.sqrt(dist2.min())  # 实际距离
            else:
                nearest = None
                min_dist = np.inf

            if nearest is not None:
                if min_dist > 0:
                    tuple_points_all.append([tuple(nearest), tuple(point)])
                    tree_connect_set.add(tuple(nearest))
                    # tuple_points_all.append([tuple(nearest), tuple(point + axis_u * 0.1)])
                    # tuple_points_all.append([tuple(nearest), tuple(point)])
                    # 更新树点集
                    new_points_dict[name] = np.concatenate([new_points_dict[name],
                                                            np.array([nearest])], axis=0)
    # 计算总长度
    lengths = 0.0
    if len(tuple_points_all):
        lengths = total_length_vox(tuple_points_all, resolution_ratio)

    # with open(swc_length_path, "w") as f:
    #     f.write(f'{lengths}\n')
    safe_write(swc_length_path, f'{lengths}\n', sync=True)
    # 重组
    sct.swc_data_combine(tuple_points_all, save_path)
    # 插值
    # swc_data_interpolation(save_path, swc_add_path)


def swc_tree_connect_add(pathLsQue, finishQue, errorQue, sct, resolution_ratio):
    while True:
        try:
            add = pathLsQue.get(timeout=5)
            if len(add) == 0:
                finishQue.put(-1)
                return
            img_path, swc_path, save_path, swc_length_path = add

            is_ok = False
            if os.path.exists(swc_path):
                if os.path.isfile(swc_path):
                    if os.path.getsize(swc_path):
                        is_ok = True
            if not is_ok:
                safe_write(save_path, "", sync=True)
                # with open(save_path, 'w') as f:
                #     pass
                finishQue.put(1)
                continue

            # 存在标签
            img, shapes = sct.read_img(img_path)  # 读图
            if np.max(img) == 0:
                safe_write(save_path, "", sync=True)
                # with open(save_path, 'w') as f:
                #     pass
                finishQue.put(1)
                continue

            swc_data = sct.read_swc(swc_path)  # 读swcTag
            swcDataLs = sct.collect_swc_tree(swc_data)  # Group

            new_points_dict = {}  # 新点集
            new_endpoints_dict = {}  # 新末端点
            tuple_points_all = []  # 线段集
            # 收集
            for ii, swcData in enumerate(swcDataLs):
                new_points, new_endpoints, tuple_points = sct.check_swc_endpoint(swcData, min_s=(2, 2, 2))
                # 去除长度较小的分支
                if len(tuple_points) == 0:
                    continue
                if len(swcData) <= 10:
                    v_len = total_length_vox(tuple_points, (1, 1, 1))
                    if v_len <= 3:
                        continue

                new_points_dict[str(ii)] = new_points
                new_endpoints_dict[str(ii)] = new_endpoints
                tuple_points_all += tuple_points
            # 分支树连接关系
            tree_connect_set = set()
            # Connect
            for name, new_endpoints in new_endpoints_dict.items():
                np_check_points = None
                for n, new_points in new_points_dict.items():
                    if not len(new_points):
                        continue
                    if name != n:
                        if np_check_points is None:
                            np_check_points = new_points
                        else:
                            if len(np_check_points):
                                np_check_points = np.concatenate([np_check_points, new_points], axis=0)

                if np_check_points is None or not len(np_check_points):
                    continue

                for point_list in new_endpoints:
                    point0 = np.array(point_list[0])
                    point = np.array(point_list[-1])

                    if tuple(point) in tree_connect_set:
                        continue

                    axis_u = point - point0

                    # mask = box_filter(np_check_points, point, axis_u, 40, 4, 4)
                    mask = box_filter_asym(np_check_points, point, axis_u, 6, 3, len_v=6, len_w=6)

                    # print('落入长方体的点数:', mask.sum())
                    selected = np_check_points[mask]  # 即为所求

                    # 3. 最近点
                    if selected.size:  # 框内非空
                        dist2 = np.sum((selected - point) ** 2, axis=1)  # 平方距离
                        nearest = selected[dist2.argmin()]  # 形状 (3,)
                        min_dist = np.sqrt(dist2.min())  # 实际距离
                    else:
                        nearest = None
                        min_dist = np.inf

                    if nearest is not None:
                        if min_dist > 0:
                            tuple_points_all.append([tuple(nearest), tuple(point)])
                            tree_connect_set.add(tuple(nearest))
                            # tuple_points_all.append([tuple(nearest), tuple(point + axis_u * 0.1)])
                            # tuple_points_all.append([tuple(nearest), tuple(point)])
                            # 更新树点集
                            new_points_dict[name] = np.concatenate([new_points_dict[name],
                                                                    np.array([nearest])], axis=0)
            # 计算总长度
            lengths = 0.0
            if len(tuple_points_all):
                lengths = total_length_vox(tuple_points_all, resolution_ratio)

            # with open(swc_length_path, "w") as f:
            #     f.write(f'{lengths}\n')
            safe_write(swc_length_path, f'{lengths}\n', sync=True)
            # 重组
            sct.swc_data_combine(tuple_points_all, save_path)
            # 插值
            # swc_data_interpolation(save_path, swc_add_path)
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


def total_length_vox(segments, voxel_size):
    """
    segments: (N,2,3) 的整数体素坐标
    voxel_size: (3,) Array，Sequential [vx, vy, vz]
    """
    # 例：x,y,z Resolution 0.5×0.5×1.0 µm
    # vox = np.array([0.5, 0.5, 1.0])
    # lines = [[(x0,y0,z0), (x1,y1,z1)], [(x2,y2,z2), (x3,y3,z3)], ...]
    # print(total_length_vox(lines, vox))
    segs = np.asarray(segments, dtype=float)
    delta = (segs[:, 1] - segs[:, 0]) * np.array(voxel_size)  # 先减再乘，广播
    return np.sqrt(np.einsum('ij,ij->i', delta, delta)).sum()


def safe_write(path: str | os.PathLike, data: bytes | str,
               *, encoding='utf-8', sync=True) -> None:
    """
    原子写文件：
      - 文件已存在则抛 FileExistsError
      - 写中途崩溃不会留下半写文件
      - 可选落盘
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)          # 确保目录存在

    # 1. 先写同目录临时文件（同设备保证 rename 原子）
    with tempfile.NamedTemporaryFile(mode='wb' if isinstance(data, bytes) else 'w',
                                     dir=p.parent, delete=False,
                                     encoding=encoding if isinstance(data, str) else None) as tmp:
        tmp.write(data)
        tmp.flush()                    # 刷到内核缓冲区
        if sync:                       # 真正落盘
            os.fsync(tmp.fileno())
        tmp_name = tmp.name            # 记住临时文件路径

    # 2. 原子改名：要么全新出现，要么抛 FileExistsError
    try:
        os.replace(tmp_name, p)        # 3.3+ 保证原子
    except BaseException:              # 任何失败都清理临时文件
        os.unlink(tmp_name)
        raise


if __name__ == "__main__":
    swcPath = r"D:\BaiduNetdiskDownload\question\0023_00_01\0023_00_01.swc"
    shapes = [192, 192, 192]
    swc_to_mask(shapes, swcPath)
