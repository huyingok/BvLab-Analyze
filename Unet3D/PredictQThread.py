# -*- coding: utf-8 -*-
import os
import time
import sys
import traceback
from pathlib import Path
import tifffile
from tifffile import TiffFile
import cv2
# 必须在任何 import torch 之前执行
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import shutil
import torch
from os.path import join
import numpy as np
import random
import zarr
from scipy import ndimage
# from .DataLoader import GetMultiTypeMemoryDataSetAndCropQxz2
# from torch.utils.data import DataLoader
# from .models.model import LoadModel
import subprocess
from multiprocessing import Process, Queue
from PyQt5.QtCore import QThread, pyqtSignal
from Unet3D.ModelPredictPy import ModelPredictClass, modelCfg
from BVExample.BVMoudle import BVReader
import json
from config import exe_cfg
import gpu_device_use
from EasyTracing.EasyTracing import voxel_scoping_centerline_tracing
from EasyTracing.Skeleton_3D_to_swc import VascularSkeletonExtractor
from DataStatistics.vessel_radius.segment_to_swc_optimized import segments_to_swc


def mask_label(pred_mask, th=103, nums=100):
    pred_mask[pred_mask < th] = 0
    # pred_mask[pred_mask > 0] = 255
    # 假设 pred_mask 是模型输出的二值掩膜 (H, W, D)
    # 1. 去除小于 threshold 个像素的孤立小区域（去除零散杂质）
    labeled, num_features = ndimage.label(pred_mask)
    sizes = np.bincount(labeled.ravel())
    # 保留最大的连通域，或者保留大于 500 体素的区域
    for i in range(1, num_features + 1):
        if sizes[i] < nums:  # 根据你目标器官的大小调整这个阈值
            pred_mask[labeled == i] = 0
    # 2. 填充内部空洞（可选）
    # 使用 ndimage.binary_fill_holes(pred_mask)
    return pred_mask


def SignMaskToGMM(workQue, finishQue, errorQue, savePath, gpuId, exePath, processDir, proceLs):
    while True:
        try:
            add = workQue.get(timeout=2)  # 设置超时避免阻塞
            if len(add) == 0:
                return
            name = os.path.basename(add)
            if name in proceLs:
                continue
            print('%s %s %s %d' % (exePath, add, savePath, gpuId))
            p = subprocess.Popen('%s %s %s %d' % (exePath, add, savePath, gpuId))
            p.wait()
            with open(join(processDir, name), 'w') as f:
                f.write('ok')
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


def QueGmmToSwc(nameLsQue, finishQue, errorQue, exePath, processDir, proceLs):
    while True:
        try:
            add = nameLsQue.get(timeout=1)
            if len(add) == 0:
                return
            name = os.path.basename(add)
            if name in proceLs:
                continue
            print(add)
            p = subprocess.Popen('%s %s' % (exePath, add))
            p.wait()
            with open(join(processDir, name), 'w') as f:
                f.write('ok')
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


class NeuralDataPredictQThread(QThread):
    finish0 = pyqtSignal(str)
    progress0 = pyqtSignal(str, np.ndarray, np.ndarray)
    progress_text = pyqtSignal(str, int)
    error0 = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(NeuralDataPredictQThread, self).__init__()
        self.win = kwargs.get('win')
        self.source = kwargs.get('source')
        self.smallSize = [192, 192, 192]
        self.min_size = 200**3
        self.small_img = []
        self.is_error = 0
        self.is_keep_on = 0  # 默认重新预测
        self.logger = self.win.logger
        self.base_path = exe_cfg.cfg['ExePath']['base_path']
        self.gsExePath = join(self.base_path, 'Gaussian_without_ui_pred.exe')
        self.gsExePath = os.path.abspath(self.gsExePath)
        self.normalExePath = join(self.base_path, 'ellipsoid_connect_pred.exe')
        self.normalExePath = os.path.abspath(self.normalExePath)
        # self.r = np.array([4, 4, 2])  # xyz
        self.r = np.array([10, 10, 10])  # xyz
        self.fieldLen = 16
        self.rSp = 2  # 边缘不要部分
        self.readObj = BVReader()  # 读取bv格式对象
        self.min_seeds = 300  # 最小种子数
        self.min_shapes_step = int(2 ** (len(modelCfg['f_maps']) - 1))
        # [z,y,x]
        self.min_shapes = [self.min_shapes_step,
                           2 * self.min_shapes_step,
                           2 * self.min_shapes_step]
        self.del_seg = []
        self.swc_min_size = 250

    def run(self):
        try:
            self.small_img = []
            self.del_seg = []
            imageDir = saveDir = ""
            self.is_error = 0
            self.is_keep_on = 0  # 默认重新预测
            cfg_level = 0
            model_path = ""
            bv_ROI = []
            self.new_text = ""
            if self.source == "make":
                making_arguments = self.win.making_arguments
                self.smallSize = making_arguments['make_arguments']['small_size']  # xyz
                smallSize = np.array(self.smallSize, dtype=np.float64)
                if smallSize[0] * smallSize[1] * smallSize[2] > self.min_size:
                    self.smallSize = [192, 192, 192]
                elif all(x >= y for x, y in zip(self.smallSize, smallSize)):
                    self.smallSize = smallSize
                model_path = making_arguments['model_path']  # 模型路径
                imageDir = making_arguments['divide_img_dir']
                saveDir = making_arguments['divide_swc_dir']
            elif self.source == "pred":
                self.is_keep_on = self.win.data_predict_dict.get("is_keep_on", self.is_keep_on)
                cfg_level = self.win.data_predict_dict.get('cfg_level', cfg_level)
                self.smallSize = self.win.data_predict_dict['small_size']  # xyz
                model_path = self.win.data_predict_dict['modelPath']  # 模型路径
                imageDir = self.win.data_predict_dict['imageDir']
                saveDir = self.win.data_predict_dict['saveDir']
                bv_ROI = self.win.data_predict_dict.get("bv_ROI", [])
                os.makedirs(saveDir, exist_ok=True)
                saveDir = join(saveDir, "PredictResults")
                if not self.is_keep_on:  # 重新预测
                    if os.path.isdir(saveDir):
                        shutil.rmtree(saveDir, ignore_errors=True)
            if imageDir and saveDir and model_path:
                os.makedirs(saveDir, exist_ok=True)
                if ".ome.zarr" in imageDir.lower():  # .ome.zarr 格式数据
                    imageDir0 = join(imageDir, f"{cfg_level}")  # .ome.zarr 图像路径
                    if not os.path.exists(imageDir0):
                        self.error0.emit("Downsampling level of 'OME-Zarr' is too high, cannot predict!")
                        return
                    file_names = [n for n in os.listdir(imageDir) if "." not in n]
                    if len(file_names) == 0:
                        self.error0.emit("No OME-Zarr format files in the image folder, cannot predict!")
                        return
                    else:
                        if self.source == "pred":
                            self.NeuralDataZarrImgPredict(imageDir, model_path, saveDir, cfg_level)
                else:
                    filenames = os.listdir(imageDir)
                    if "config.cfg" in filenames:
                        imageDir0 = join(imageDir, f"{cfg_level}")  # bv图像路径
                        if not os.path.exists(imageDir0):
                            self.error0.emit("Downsampling level of 'BV' is too high, cannot predict!")
                            return
                        file_names = [n for n in os.listdir(imageDir0) if ".bv" in n]
                        if len(file_names) == 0:
                            self.error0.emit("No BV format files in the image folder, cannot predict!")
                            return
                        else:
                            if self.source == "pred":
                                self.NeuralDataCfgImgPredict(imageDir, model_path, saveDir, cfg_level, bv_ROI=bv_ROI)
                    else:
                        file_names = [n for n in os.listdir(imageDir) if ".tif" in n]
                        if len(file_names) == 0:
                            self.error0.emit("No TIF format files in the image folder, cannot predict!")
                            return
                        else:
                            if self.source == "pred":
                                self.NeuralDataBigImgPredict(imageDir, model_path, saveDir)
                                if len(self.small_img):  # 没有小图
                                    self.NeuralDataImgBatchPredict(imageDir, model_path, saveDir)
                            if self.source == "make":
                                img_path = join(imageDir, file_names[0])
                                img = tifffile.imread(img_path)
                                if (img.shape[0] > self.smallSize[2] or
                                        img.shape[1] > self.smallSize[1] or img.shape[2] > self.smallSize[1]):
                                    self.NeuralDataBigImgPredict(imageDir, model_path, saveDir)
                                else:
                                    self.NeuralDataImgBatchPredict(imageDir, model_path, saveDir)
            time.sleep(2)
            torch.cuda.empty_cache()
            print("Clear memory")
            if not self.is_error:
                self.finish0.emit(self.new_text)
        except Exception as e:
            torch.cuda.empty_cache()
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

    """最小数据尺寸补充"""

    def pad_to_min_size_tail(self, img_zyx):
        """
        img_zyx: 3-D numpy array, order (z,y,x)
        返回: (补后的数组, is_small标志)
        """
        min_shape = np.array(self.min_shapes)
        curr_shape = np.array(img_zyx.shape)
        # 1. 先补到偶数
        deficit_even = curr_shape & 1  # 奇数维补1，偶数维补0
        curr_even = curr_shape + deficit_even
        # 2. 再补到最小尺寸
        deficit_min = np.maximum(min_shape - curr_even, 0)
        deficit_total = deficit_even + deficit_min
        if np.all(deficit_total == 0):
            return img_zyx, False
        paddings = tuple((0, d) for d in deficit_total)  # 只后补
        img_padded = np.pad(img_zyx, paddings, mode='constant', constant_values=0)
        return img_padded, True

    """神经OME-Zarr格式预测"""

    def NeuralDataZarrImgPredict(self, zarrPath, modelPath, saveDir, cfg_level):
        self.progress_text.emit("Neural OME-Zarr format data prediction!", 0)
        gpu_id = gpu_device_use.get_gpu_utilization()
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = torch.device("cuda", gpu_id)
        else:
            self.new_text = "No available GPU"
            return

        zarr_store = zarr.open(zarrPath, mode='r')

        shapes = zarr_store[f'{cfg_level}'].shape  # (z,y,x)
        seqInfo = np.array(shapes, dtype=np.int32)[::-1]  # xyz
        sx = sy = 512
        sz = seqInfo[2]
        print(sx, sy, sz)
        bvSmallSize = np.array([sx, sy, sz])
        BvSliceNumber = np.ceil((seqInfo - bvSmallSize) / (bvSmallSize - self.r)).astype(np.int32) + 1

        self.smallSize = [512, 512, 512]
        imgSize = np.array(self.smallSize)  # xyz
        model = ModelPredictClass(modelPath, device=device, fieldLen=self.fieldLen)

        file_stems = [f"{nx}_{ny}" for ny in range(BvSliceNumber[1]) for nx in range(BvSliceNumber[0])]
        lsLen = len(file_stems)
        shapes_list = [
            [nx * (sx - self.r[0]),  # start_x
             min(nx * (sx - self.r[0]) + sx, seqInfo[0]),  # end_x
             ny * (sy - self.r[1]),  # start_y
             min(ny * (sy - self.r[1]) + sy, seqInfo[1]),  # end_y
             0,  # start_z
             sz]  # end_z
            for ny in range(BvSliceNumber[1])
            for nx in range(BvSliceNumber[0])
        ]

        CutWorkFilesDir = join(saveDir, "CutWorkFiles")
        if not self.is_keep_on:  # 重新预测
            if os.path.exists(CutWorkFilesDir):
                shutil.rmtree(CutWorkFilesDir, ignore_errors=True)
        os.makedirs(CutWorkFilesDir, exist_ok=True)

        cut_infos_path = join(CutWorkFilesDir, "cut_infos.json")
        if self.is_keep_on:
            if os.path.exists(cut_infos_path):
                self.new_text = "Data has already been fully predicted"
                return

        cut_infos = {}
        cut_infos["dataType"] = "OME-Zarr"
        cut_infos["CutWorkFilesDir"] = CutWorkFilesDir
        cut_infos["OME-ZarrSliceNumberXYZ"] = list(BvSliceNumber)
        cut_infos["OME-ZarrRedunXYZ"] = list(self.r)
        cut_infos["OME-ZarrBigSizeXYZ"] = list(seqInfo)
        cut_infos["OME-ZarrSmallSizeXYZ"] = list(bvSmallSize)
        cut_infos["cfg_level"] = int(cfg_level)

        total_time = time.time()
        keep_count = 0

        for i, file_stem in enumerate(file_stems):
            self.del_seg = []
            cut_info_dir = join(CutWorkFilesDir, file_stem)
            os.makedirs(cut_info_dir, exist_ok=True)
            cut_info_path = join(cut_info_dir, f"{file_stem}.json")

            total_progress = (i + 1) / lsLen * 100

            if self.is_keep_on:
                if os.path.exists(cut_info_path):
                    text = '[Total progress %.2f%%]' % (total_progress)
                    self.progress_text.emit(text, 0)
                    keep_count += 1
                    time.sleep(0.001)
                    continue

            sl = shapes_list[i]

            bigImg = zarr_store[f'{cfg_level}'][sl[4]:sl[5], sl[2]:sl[3], sl[0]:sl[1]]  # 读roi

            cut_info = {}
            bigSize = np.array(bigImg.shape)[::-1]  # xyz
            # 补充图像不足区块，加大尺寸
            bigSize = [max(bigSize[0], imgSize[0]), max(bigSize[1], imgSize[1]), max(bigSize[2], imgSize[2])]
            bigSize = np.array(bigSize)  # xyz
            # # 边缘图像补充，防止边缘重复预测部分过多
            # for bi in range(len(bigSize)):
            #     if 0 < bigSize[bi] - imgSize[bi] < int(imgSize[bi] / 2):
            #         bigSize[bi] += int(imgSize[bi] / 2)
            # 小尺寸，xyz块数列表
            sliceNumber = np.ceil((bigSize - imgSize) / (imgSize - self.r)).astype(np.int32) + 1
            sliceLen = 1
            for s in sliceNumber:
                sliceLen = s * sliceLen

            cut_info["bigSizeXYZ"] = list(bigSize)
            cut_info["imgSizeXYZ"] = list(imgSize)
            cut_info["rXYZ"] = list(self.r)
            cut_info["sliceNumberXYZ"] = list(sliceNumber)

            cut_img_dir = join(cut_info_dir, "images")
            cut_seg_dir = join(cut_info_dir, "segment")
            cut_swc_dir = join(cut_info_dir, "swc")
            cut_info["cut_img_dir"] = cut_img_dir
            cut_info["cut_seg_dir"] = cut_seg_dir
            cut_info["cut_swc_dir"] = cut_swc_dir
            os.makedirs(cut_img_dir, exist_ok=True)
            os.makedirs(cut_seg_dir, exist_ok=True)
            os.makedirs(cut_swc_dir, exist_ok=True)

            progress_num = 0
            start_time = time.time()
            with torch.no_grad():
                for nz in range(sliceNumber[2]):
                    for ny in range(sliceNumber[1]):
                        for nx in range(sliceNumber[0]):
                            img_name = f"{str(nz).zfill(4)}_{str(ny).zfill(2)}_{str(nx).zfill(2)}.tif"
                            swc_name = f"{str(nz).zfill(4)}_{str(ny).zfill(2)}_{str(nx).zfill(2)}.swc"
                            cut_img_path = join(cut_img_dir, img_name)
                            cut_seg_path = join(cut_seg_dir, img_name)
                            cut_swc_path = join(cut_swc_dir, swc_name)

                            if progress_num == 0:
                                text = '[Total progress %.2f%%] [Prediction progress %.2f%%]' % (
                                    total_progress, 0)
                                self.progress_text.emit(text, progress_num)
                            else:
                                userTime = time.time() - start_time
                                surplusTime = userTime / (progress_num + 1) * (sliceLen - progress_num - 1)
                                text = '[Total progress %.2f%%] [Prediction progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                                    total_progress, ((progress_num + 1) / sliceLen * 100), userTime, surplusTime)
                                self.progress_text.emit(text, progress_num)

                            if self.is_keep_on:
                                if os.path.exists(cut_swc_path):  # 存在swcFile，且大小不为0，已经完成追踪
                                    if os.path.getsize(cut_swc_path) > self.swc_min_size:
                                        self.del_seg.append(img_name)  # 添加图像名字到去除列表
                                        progress_num += 1
                                        continue

                            with open(cut_swc_path, "w") as swc_file:  # 创建swcFile
                                pass
                            swc_file.close()

                            if self.is_keep_on:
                                if os.path.exists(cut_seg_path):  # 存在分割文件，已经完成分割
                                    progress_num += 1
                                    continue

                            step = imgSize - self.r
                            sp = step * [nx, ny, nz]
                            ep = sp + imgSize

                            for dim in range(3):
                                if ep[dim] > bigSize[dim]:
                                    sp[dim] = max(0, bigSize[dim] - imgSize[dim])
                                    ep[dim] = bigSize[dim]

                            img = bigImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
                            img_shapes = img.shape

                            tifffile.imwrite(cut_img_path, img, compression="lzw")

                            if not img.any():
                                tifffile.imwrite(cut_seg_path, img, compression="lzw")
                                progress_num += 1
                                self.del_seg.append(img_name)  # 添加图像名字到去除列表
                                continue

                            img, is_small = self.pad_to_min_size_tail(img)  # cfg
                            # Predict
                            # seg = model(img)
                            seg = self.big_image_seg(img, model)

                            if is_small:
                                seg = seg[:img_shapes[0], :img_shapes[1], :img_shapes[2]]
                            seg = zero_border(seg, self.rSp)

                            seg = mask_label(seg)
                            # seg[seg < 103] = 0
                            tifffile.imwrite(cut_seg_path, seg, compression="lzw")
                            progress_num += 1
                            torch.cuda.empty_cache()
            self.progress_text.emit("[Prediction finished]", 0)
            # Tracking
            self.progress_text.emit("[Tracking started]", 1)
            # all_swc_files_none = self.BigImageCutPredict(cut_seg_dir, cut_swc_dir, total_progress)
            # if all_swc_files_none:
            self.VoxelScopingCenterlineTracing(cut_seg_dir, cut_swc_dir, total_progress)

            userTime = time.time() - total_time
            surplusTime = userTime / (i - keep_count + 1) * (lsLen - i - 1)
            text = '[Total progress %.2f%%] [Time elapsed %ds] [Estimated total remaining time %ds]' % (
            total_progress, userTime, surplusTime)
            self.progress_text.emit(text, 0)
            # 随机预测图展示
            image_i = random.randint(0, int(sliceLen / 2))
            if sliceLen > image_i:
                image_name = os.listdir(cut_img_dir)[image_i]
                image = tifffile.imread(join(cut_img_dir, image_name))
                mask = tifffile.imread(join(cut_seg_dir, image_name))
                if image.any():
                    image = image[:192, :192, :192]
                    mask = mask[:192, :192, :192]
                    image_2d = self.MaxProject(image, 1)
                    mask_2d = self.MaxProject(mask, 0)
                    self.progress0.emit("", image_2d, mask_2d)
            if sliceLen > 4:
                image_j = random.randint(int(sliceLen / 2) + 1, sliceLen)
                if sliceLen > image_j and image_j > 0:
                    image_name = os.listdir(cut_img_dir)[image_j]
                    image = tifffile.imread(join(cut_img_dir, image_name))
                    mask = tifffile.imread(join(cut_seg_dir, image_name))
                    if image.any():
                        image = image[:192, :192, :192]
                        mask = mask[:192, :192, :192]
                        image_2d = self.MaxProject(image, 1)
                        mask_2d = self.MaxProject(mask, 0)
                        self.progress0.emit("", image_2d, mask_2d)
            # 更新配置文件
            cut_info["dataType"] = "TIF"
            cut_info = convert_to_builtin_types(cut_info)
            with open(cut_info_path, 'w') as f:
                f.write(json.dumps(cut_info, indent=4))

        del model
        del device

        cut_infos = convert_to_builtin_types(cut_infos)
        with open(cut_infos_path, 'w') as f:
            f.write(json.dumps(cut_infos, indent=4))

        self.win.cut_infos_path = cut_infos_path

    """神经大数据格式预测"""

    def NeuralDataCfgImgPredict(self, bvDir, modelPath, saveDir, cfg_level, bv_ROI):
        """
        bvDir: 大数据文件夹
        model_path: 模型路径
        saveDir: 保存路径
        cfg_level: 等级
        """
        self.progress_text.emit("Neural BV format data prediction!", 0)
        gpu_id = gpu_device_use.get_gpu_utilization()
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = torch.device("cuda", gpu_id)
        else:
            self.new_text = "No available GPU"
            return

        config_path = join(bvDir, "config.cfg")
        if not os.path.exists(config_path):
            self.new_text = "No available config.cfg file"
            return
        seqInfo_path = join(bvDir, "seqInfo.txt")
        with open(seqInfo_path, "r") as f:
            lines = f.readlines()
            seqInfo = [int(float(line)) for line in lines]  # xyz

        if len(bv_ROI):
            MinX, MaxX, MinY, MaxY, MinZ, MaxZ = bv_ROI
            MinX = max(MinX, 0)
            MinY = max(MinY, 0)
            MinZ = max(MinZ, 0)
            MaxX = min(MaxX, seqInfo[0])
            MaxY = min(MaxY, seqInfo[1])
            MaxZ = min(MaxZ, seqInfo[2])
            min_size = [64, 64, 64]
            if (MaxX - MinX < min_size[0]
                    or MaxY - MinY < min_size[1]
                    or MaxZ - MinZ < min_size[2]):
                # There is no valid area or it is smaller than [], unable to predict. Please reset the ROI range
                self.new_text = (f"There is no valid area or it is smaller than [x, y, z]={min_size}, "
                                 f"unable to predict. Please reset the ROI range!")
                return
        else:
            MinX = MinY = MinZ = 0
            MaxX, MaxY, MaxZ = seqInfo

        seqInfo = [MaxX - MinX, MaxY - MinY, MaxZ - MinZ]
        seqInfo = np.array(seqInfo, dtype=np.int32)  # xyz
        sx = sy = 512 * (2 ** cfg_level)
        sz = MaxZ - MinZ
        print(sx, sy, sz)
        bvSmallSize = np.array([sx, sy, sz])
        BvSliceNumber = np.ceil((seqInfo - bvSmallSize) / (bvSmallSize - self.r)).astype(np.int32) + 1

        self.smallSize = [512, 512, 512]
        imgSize = np.array(self.smallSize)  # xyz
        model = ModelPredictClass(modelPath, device=device, fieldLen=self.fieldLen)

        file_stems = [f"{nx}_{ny}" for ny in range(BvSliceNumber[1]) for nx in range(BvSliceNumber[0])]
        lsLen = len(file_stems)
        shapes_list = [
            [nx * (sx - self.r[0]) + MinX,  # start_x
             min(nx * (sx - self.r[0]) + MinX + sx, MaxX),  # end_x
             ny * (sy - self.r[1]) + MinY,  # start_y
             min(ny * (sy - self.r[1]) + MinY + sy, MaxY),  # end_y
             MinZ,  # start_z
             MaxZ]  # end_z
            for ny in range(BvSliceNumber[1])
            for nx in range(BvSliceNumber[0])
        ]

        CutWorkFilesDir = join(saveDir, "CutWorkFiles")
        if not self.is_keep_on:  # 重新预测
            if os.path.exists(CutWorkFilesDir):
                shutil.rmtree(CutWorkFilesDir, ignore_errors=True)
        os.makedirs(CutWorkFilesDir, exist_ok=True)

        cut_infos_path = join(CutWorkFilesDir, "cut_infos.json")
        if self.is_keep_on:
            if os.path.exists(cut_infos_path):
                self.new_text = "Data has already been fully predicted"
                return

        cut_infos = {}
        cut_infos["dataType"] = "BV"
        cut_infos["CutWorkFilesDir"] = CutWorkFilesDir
        cut_infos["BvSliceNumberXYZ"] = list(BvSliceNumber)
        cut_infos["BvRedunXYZ"] = list(self.r)
        cut_infos["BvBigSizeXYZ"] = list(seqInfo)
        cut_infos["BvSmallSizeXYZ"] = list(bvSmallSize)
        cut_infos["cfg_level"] = int(cfg_level)
        cut_infos["ROI"] = [MinX, MaxX, MinY, MaxY, MinZ, MaxZ]

        total_time = time.time()
        keep_count = 0

        for i, file_stem in enumerate(file_stems):
            self.del_seg = []
            cut_info_dir = join(CutWorkFilesDir, file_stem)
            os.makedirs(cut_info_dir, exist_ok=True)
            cut_info_path = join(cut_info_dir, f"{file_stem}.json")

            total_progress = (i + 1) / lsLen * 100

            if self.is_keep_on:
                if os.path.exists(cut_info_path):
                    text = '[Total progress %.2f%%]' % (total_progress)
                    self.progress_text.emit(text, 0)
                    keep_count += 1
                    time.sleep(0.001)
                    continue

            sl = shapes_list[i]
            print(sl)
            bigImg = self.readObj.loadROI(config_path, int(cfg_level), sl[0], sl[1], sl[2], sl[3], sl[4], sl[5])  # 读roi

            cut_info = {}
            bigSize = np.array(bigImg.shape)[::-1]  # xyz
            # 补充图像不足区块，加大尺寸
            bigSize = [max(bigSize[0], imgSize[0]), max(bigSize[1], imgSize[1]), max(bigSize[2], imgSize[2])]
            bigSize = np.array(bigSize)  # xyz
            # # 边缘图像补充，防止边缘重复预测部分过多
            # for bi in range(len(bigSize)):
            #     if 0 < bigSize[bi] - imgSize[bi] < int(imgSize[bi] / 2):
            #         bigSize[bi] += int(imgSize[bi] / 2)
            # 小尺寸，xyz块数列表
            sliceNumber = np.ceil((bigSize - imgSize) / (imgSize - self.r)).astype(np.int32) + 1
            sliceLen = 1
            for s in sliceNumber:
                sliceLen = s * sliceLen

            cut_info["bigSizeXYZ"] = list(bigSize)
            cut_info["imgSizeXYZ"] = list(imgSize)
            cut_info["rXYZ"] = list(self.r)
            cut_info["sliceNumberXYZ"] = list(sliceNumber)

            cut_img_dir = join(cut_info_dir, "images")
            cut_seg_dir = join(cut_info_dir, "segment")
            cut_swc_dir = join(cut_info_dir, "swc")
            cut_info["cut_img_dir"] = cut_img_dir
            cut_info["cut_seg_dir"] = cut_seg_dir
            cut_info["cut_swc_dir"] = cut_swc_dir
            os.makedirs(cut_img_dir, exist_ok=True)
            os.makedirs(cut_seg_dir, exist_ok=True)
            os.makedirs(cut_swc_dir, exist_ok=True)

            progress_num = 0
            start_time = time.time()
            with torch.no_grad():
                for nz in range(sliceNumber[2]):
                    for ny in range(sliceNumber[1]):
                        for nx in range(sliceNumber[0]):
                            img_name = f"{str(nz).zfill(4)}_{str(ny).zfill(2)}_{str(nx).zfill(2)}.tif"
                            swc_name = f"{str(nz).zfill(4)}_{str(ny).zfill(2)}_{str(nx).zfill(2)}.swc"
                            cut_img_path = join(cut_img_dir, img_name)
                            cut_seg_path = join(cut_seg_dir, img_name)
                            cut_swc_path = join(cut_swc_dir, swc_name)

                            if progress_num == 0:
                                text = '[Total progress %.2f%%] [Prediction progress %.2f%%]' % (
                                    total_progress, 0)
                                self.progress_text.emit(text, progress_num)
                            else:
                                userTime = time.time() - start_time
                                surplusTime = userTime / (progress_num + 1) * (sliceLen - progress_num - 1)
                                text = '[Total progress %.2f%%] [Prediction progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                                    total_progress, ((progress_num + 1) / sliceLen * 100), userTime, surplusTime)
                                self.progress_text.emit(text, progress_num)

                            if self.is_keep_on:
                                if os.path.exists(cut_swc_path):  # 存在swcFile，且大小不为0，已经完成追踪
                                    if os.path.getsize(cut_swc_path) > self.swc_min_size:
                                        self.del_seg.append(img_name)  # 添加图像名字到去除列表
                                        progress_num += 1
                                        continue

                            with open(cut_swc_path, "w") as swc_file:  # 创建swcFile
                                pass
                            swc_file.close()

                            if self.is_keep_on:
                                if os.path.exists(cut_seg_path):  # 存在分割文件，已经完成分割
                                    progress_num += 1
                                    continue

                            step = imgSize - self.r
                            sp = step * [nx, ny, nz]
                            ep = sp + imgSize

                            for dim in range(3):
                                if ep[dim] > bigSize[dim]:
                                    sp[dim] = max(0, bigSize[dim] - imgSize[dim])
                                    ep[dim] = bigSize[dim]

                            img = bigImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
                            img_shapes = img.shape

                            tifffile.imwrite(cut_img_path, img, compression="lzw")

                            if not img.any():
                                tifffile.imwrite(cut_seg_path, img, compression="lzw")
                                progress_num += 1
                                self.del_seg.append(img_name)  # 添加图像名字到去除列表
                                continue

                            img, is_small = self.pad_to_min_size_tail(img)  # cfg
                            # Predict
                            # seg = model(img)
                            seg = self.big_image_seg(img, model)
                            if is_small:
                                seg = seg[:img_shapes[0], :img_shapes[1], :img_shapes[2]]
                            seg = zero_border(seg, self.rSp)
                            seg = mask_label(seg)
                            # seg[seg < 103] = 0
                            tifffile.imwrite(cut_seg_path, seg, compression="lzw")
                            progress_num += 1
                            torch.cuda.empty_cache()
            self.progress_text.emit("[Prediction finished]", 0)
            # Tracking
            self.progress_text.emit("[Tracking started]", 1)
            # all_swc_files_none = self.BigImageCutPredict(cut_seg_dir, cut_swc_dir, total_progress)
            # if all_swc_files_none:
            self.VoxelScopingCenterlineTracing(cut_seg_dir, cut_swc_dir, total_progress)

            userTime = time.time() - total_time
            surplusTime = userTime / (i - keep_count + 1) * (lsLen - i - 1)
            text = '[Total progress %.2f%%] [Time elapsed %ds] [Estimated total remaining time %ds]' % (total_progress, userTime, surplusTime)
            self.progress_text.emit(text, 0)
            # 随机预测图展示
            image_i = random.randint(0, int(sliceLen / 2))
            if sliceLen > image_i:
                image_name = os.listdir(cut_img_dir)[image_i]
                image = tifffile.imread(join(cut_img_dir, image_name))
                mask = tifffile.imread(join(cut_seg_dir, image_name))
                if image.any():
                    image = image[:192, :192, :192]
                    mask = mask[:192, :192, :192]
                    image_2d = self.MaxProject(image, 1)
                    mask_2d = self.MaxProject(mask, 0)
                    self.progress0.emit("", image_2d, mask_2d)
            if sliceLen > 4:
                image_j = random.randint(int(sliceLen / 2) + 1, sliceLen)
                if sliceLen > image_j and image_j > 0:
                    image_name = os.listdir(cut_img_dir)[image_j]
                    image = tifffile.imread(join(cut_img_dir, image_name))
                    mask = tifffile.imread(join(cut_seg_dir, image_name))
                    if image.any():
                        image = image[:192, :192, :192]
                        mask = mask[:192, :192, :192]
                        image_2d = self.MaxProject(image, 1)
                        mask_2d = self.MaxProject(mask, 0)
                        self.progress0.emit("", image_2d, mask_2d)
            # 更新配置文件
            cut_info["dataType"] = "TIF"
            cut_info = convert_to_builtin_types(cut_info)
            with open(cut_info_path, 'w') as f:
                f.write(json.dumps(cut_info, indent=4))

        del model
        del device

        cut_infos = convert_to_builtin_types(cut_infos)
        with open(cut_infos_path, 'w') as f:
            f.write(json.dumps(cut_infos, indent=4))

        self.win.cut_infos_path = cut_infos_path

    def big_image_seg(self, oriImg, model):
        oriImg_shapes = oriImg.shape
        min_x = min(oriImg_shapes[2], 192)
        min_y = min(oriImg_shapes[1], 192)
        min_z = min(oriImg_shapes[0], 192)
        imgSize = np.array([min_x, min_y, min_z], dtype=np.int32)
        r = np.array([min(min_x, 32), min(min_y, 32), min(min_z, 32)], dtype=np.int32)  # 冗余
        rSp = np.array([min(min_x, 16), min(min_y, 16), min(min_z, 16)], dtype=np.int32)  # 边缘不要部分
        bigImgSize = np.array(list(oriImg_shapes)[::-1], dtype=np.int32)
        maskBigImg = np.zeros(bigImgSize[::-1], dtype=np.uint8)
        sliceNumber = np.ceil((bigImgSize - imgSize) / (imgSize - r)).astype(np.int32) + 1
        for nz in range(sliceNumber[2]):
            for ny in range(sliceNumber[1]):
                for nx in range(sliceNumber[0]):
                    sp = (imgSize - r) * [nx, ny, nz]
                    ep = np.min([sp + imgSize, bigImgSize], axis=0)
                    sp = np.min([sp, ep - imgSize], axis=0)
                    img = oriImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
                    mask = model(img)
                    rsp2 = rSp * np.sign([nx, ny, nz])
                    sp += rsp2
                    maskBigImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]] = mask[rsp2[2]:, rsp2[1]:, rsp2[0]:]
        return maskBigImg

    """设置骨架工作文件"""

    def set_skeleton_files(self, SkeletonWorkFilesDir):
        MaskWorkSplitPath = join(SkeletonWorkFilesDir, 'MaskWorkSplit')  # 高斯任务分配
        ResGMMPath = join(SkeletonWorkFilesDir, 'ResGMM')  # 距离变换结果
        ResGMMNewProcessPath = join(SkeletonWorkFilesDir, 'ResGMMNewProcess')  # 高斯完成
        GMMWorkSplitPath = join(SkeletonWorkFilesDir, 'GMMWorkSplit')  # 归一化任务分配
        ResSwcPath = join(SkeletonWorkFilesDir, 'ResSwc')  # 追踪结果
        ResSwcGMMNewProcessPath = join(SkeletonWorkFilesDir, 'ResSwcGMMNewProcess')  # 归一化完成
        paths = [MaskWorkSplitPath, ResGMMPath, ResGMMNewProcessPath, GMMWorkSplitPath,
                 ResSwcGMMNewProcessPath, ResSwcPath]
        return paths

    """神经元大图预测"""

    def NeuralDataBigImgPredict(self, imageDir, modelPath, saveDir):
        self.progress_text.emit("Neuron large image prediction!", 0)

        gpu_id = gpu_device_use.get_gpu_utilization()
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = torch.device("cuda", gpu_id)
        else:
            self.new_text = "No available GPU"
            return

        # imgSize = np.array(self.smallSize, dtype=np.int32)  # xyz
        imgSize = np.array([512, 512, 512], dtype=np.int32)  # xyz
        model = ModelPredictClass(modelPath, fieldLen=self.fieldLen, device=device)

        file_names = [n for n in os.listdir(imageDir) if ".tif" in n]
        lsLen = len(file_names)

        no_3d_count = 0

        CutWorkFilesDir = join(saveDir, "CutWorkFiles")
        if not self.is_keep_on:  # 重新预测
            if os.path.exists(CutWorkFilesDir):
                shutil.rmtree(CutWorkFilesDir, ignore_errors=True)
        os.makedirs(CutWorkFilesDir, exist_ok=True)

        total_time = time.time()
        keep_count = 0

        if self.source == "make":
            swc_splice_saveDir = saveDir
        else:
            swc_splice_saveDir = join(saveDir, "Swc")
        os.makedirs(swc_splice_saveDir, exist_ok=True)

        for i, file_name in enumerate(file_names):
            self.del_seg = []
            img_path = join(imageDir, file_name)  # 图像路径
            file_stem = str(Path(file_name).stem)
            cut_info_dir = join(CutWorkFilesDir, file_stem)  # 大图工作目录
            cut_info_path = join(cut_info_dir, f"{file_stem}.json")

            if self.is_keep_on:
                if os.path.exists(cut_info_path):  # 存在配置文件，已经完成预测
                    text = '[Total progress %.2f%%]' % ((i + 1) / lsLen * 100)
                    self.progress_text.emit(text, i)
                    keep_count += 1
                    continue

            # ***************************** 判断图像尺寸 ********************************
            # 获取图像尺寸
            shapes = get_1ch_shape(img_path)
            bigSize = np.array(shapes)  # zyx

            # 最小尺寸为1，则是二维图像
            if np.min(shapes) == 1 or shapes is None:
                no_3d_count += 1
                text = '[Total progress %.2f%%]' % ((i + 1) / lsLen * 100)
                self.progress_text.emit(text, i)
                keep_count += 1
                continue

            # 小块图像
            if np.all(np.array(bigSize)[::-1] <= np.array(self.smallSize)) and self.source == "pred":
                self.small_img.append(file_name)
                text = '[Total progress %.2f%%]' % ((i + 1) / lsLen * 100)
                self.progress_text.emit(text, i)
                keep_count += 1
                continue

            bigImg = tifffile.imread(img_path)  # 读取图像

            if not bigImg.any():
                no_3d_count += 1
                text = '[Total progress %.2f%%]' % ((i + 1) / lsLen * 100)
                self.progress_text.emit(text, i)
                keep_count += 1
                continue

            os.makedirs(cut_info_dir, exist_ok=True)

            cut_info = {}
            bigSize = bigSize[::-1]  # 图像尺寸xyz
            # 补充图像不足区块，加大尺寸
            bigSize = [max(bigSize[0], imgSize[0]),
                       max(bigSize[1], imgSize[1]),
                       max(bigSize[2], imgSize[2])]
            bigSize = np.array(bigSize)  # xyz
            # # 边缘图像补充，防止边缘重复预测部分过多
            # for bi in range(len(bigSize)):
            #     if 0 < bigSize[bi] - imgSize[bi] < int(imgSize[bi] / 2):
            #         bigSize[bi] += int(imgSize[bi] / 2)
            # print(bigSize)
            # 小尺寸，xyz块数列表
            sliceNumber = np.ceil((bigSize - imgSize) / (imgSize - self.r)).astype(np.int32) + 1
            sliceLen = 1
            for s in sliceNumber:  # 获取总块数
                sliceLen = s * sliceLen

            cut_info["bigSizeXYZ"] = list(bigSize)
            cut_info["imgSizeXYZ"] = list(imgSize)
            cut_info["rXYZ"] = list(self.r)
            cut_info["sliceNumberXYZ"] = list(sliceNumber)

            cut_img_dir = join(cut_info_dir, "images")
            cut_seg_dir = join(cut_info_dir, "segment")
            cut_swc_dir = join(cut_info_dir, "swc")
            cut_info["cut_img_dir"] = cut_img_dir
            cut_info["cut_seg_dir"] = cut_seg_dir
            cut_info["cut_swc_dir"] = cut_swc_dir
            os.makedirs(cut_img_dir, exist_ok=True)
            os.makedirs(cut_seg_dir, exist_ok=True)
            os.makedirs(cut_swc_dir, exist_ok=True)

            progress_num = 0
            total_progress = (i + 1) / lsLen * 100
            start_time2 = time.time()
            # 对每块进行预测
            with torch.no_grad():
                for nz in range(sliceNumber[2]):
                    for ny in range(sliceNumber[1]):
                        for nx in range(sliceNumber[0]):
                            img_name = f"{str(nz).zfill(4)}_{str(ny).zfill(2)}_{str(nx).zfill(2)}.tif"
                            swc_name = f"{str(nz).zfill(4)}_{str(ny).zfill(2)}_{str(nx).zfill(2)}.swc"
                            cut_img_path = join(cut_img_dir, img_name)
                            cut_seg_path = join(cut_seg_dir, img_name)
                            cut_swc_path = join(cut_swc_dir, swc_name)

                            userTime = time.time() - start_time2
                            surplusTime = userTime / (progress_num + 1) * (sliceLen - progress_num - 1)
                            text = '[Total progress %.2f%%] [Prediction progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                                total_progress, ((progress_num + 1) / sliceLen * 100), userTime, surplusTime)
                            self.progress_text.emit(text, progress_num)

                            if self.is_keep_on:
                                if os.path.exists(cut_swc_path):  # 存在swcFile，且大小不为0，已经完成追踪
                                    if os.path.getsize(cut_swc_path) > self.swc_min_size:
                                        self.del_seg.append(img_name)  # 添加图像名字到去除列表
                                        progress_num += 1
                                        continue

                            with open(cut_swc_path, "w") as swc_file:  # 创建swcFile
                                pass
                            swc_file.close()

                            if self.is_keep_on:
                                if os.path.exists(cut_seg_path):  # 存在分割文件，已经完成分割
                                    progress_num += 1
                                    continue

                            step = imgSize - self.r
                            sp = step * [nx, ny, nz]
                            ep = sp + imgSize

                            for dim in range(3):
                                if ep[dim] > bigSize[dim]:
                                    sp[dim] = max(0, bigSize[dim] - imgSize[dim])
                                    ep[dim] = bigSize[dim]

                            img = bigImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]]
                            img_shapes = img.shape

                            tifffile.imwrite(cut_img_path, img, compression="lzw")

                            if not img.any():
                                tifffile.imwrite(cut_seg_path, img, compression="lzw")
                                progress_num += 1
                                self.del_seg.append(img_name)  # 添加图像名字到去除列表
                                continue

                            img, is_small = self.pad_to_min_size_tail(img)  # big
                            # Predict
                            # seg = model(img)
                            seg = self.big_image_seg(img, model)
                            if is_small:
                                seg = seg[:img_shapes[0], :img_shapes[1], :img_shapes[2]]
                            seg = zero_border(seg, self.rSp)
                            seg = mask_label(seg)
                            # seg[seg < 103] = 0

                            tifffile.imwrite(cut_seg_path, seg, compression="lzw")
                            progress_num += 1
                            torch.cuda.empty_cache()
            self.progress_text.emit("[Prediction finished]", 0)
            # Tracking
            self.progress_text.emit("[Tracking started]", 1)
            # all_swc_files_none = self.BigImageCutPredict(cut_seg_dir, cut_swc_dir, total_progress)
            # if all_swc_files_none:
            self.VoxelScopingCenterlineTracing(cut_seg_dir, cut_swc_dir, total_progress)

            userTime = time.time() - total_time
            surplusTime = userTime / (i - keep_count + 1) * (lsLen - i - 1)
            text = '[Total progress %.2f%%] [Time elapsed %ds] [Estimated total remaining time %ds]' % (total_progress, userTime, surplusTime)
            self.progress_text.emit(text, 0)
            # 随机预测图展示
            image_i = random.randint(0, int(sliceLen / 2))
            if sliceLen > image_i:
                image_name = os.listdir(cut_img_dir)[image_i]
                image = tifffile.imread(join(cut_img_dir, image_name))
                mask = tifffile.imread(join(cut_seg_dir, image_name))
                if image.any():
                    image = image[:192, :192, :192]
                    mask = mask[:192, :192, :192]
                    image_2d = self.MaxProject(image, 1)
                    mask_2d = self.MaxProject(mask, 0)
                    self.progress0.emit("", image_2d, mask_2d)
            if sliceLen > 4:
                image_j = random.randint(int(sliceLen / 2) + 1, sliceLen - 1)
                if sliceLen > image_j and image_j > 0:
                    image_name = os.listdir(cut_img_dir)[image_j]
                    image = tifffile.imread(join(cut_img_dir, image_name))
                    mask = tifffile.imread(join(cut_seg_dir, image_name))
                    if image.any():
                        image = image[:192, :192, :192]
                        mask = mask[:192, :192, :192]
                        image_2d = self.MaxProject(image, 1)
                        mask_2d = self.MaxProject(mask, 0)
                        self.progress0.emit("", image_2d, mask_2d)
            # 更新配置文件
            cut_info['dataType'] = "TIF"
            cut_info = convert_to_builtin_types(cut_info)

            # file_stem, cut_info, swc_splice_saveDir
            bigSizeXYZ = cut_info["bigSizeXYZ"]
            imgSizeXYZ = cut_info["imgSizeXYZ"]
            rXYZ = cut_info["rXYZ"]
            sliceNumberXYZ = cut_info["sliceNumberXYZ"]
            cut_swc_dir = cut_info["cut_swc_dir"]
            savePath = join(swc_splice_saveDir, f"{file_stem}.swc")

            workLen = 1
            for s in sliceNumberXYZ:
                workLen *= s

            self.SwcSplice(bigSize=bigSizeXYZ,
                           imgSize=imgSizeXYZ,
                           r=rXYZ,
                           sliceNumber=sliceNumberXYZ,
                           workLen=workLen,
                           savePath=savePath,
                           cut_swc_dir=cut_swc_dir,
                           total_progress=total_progress,
                           i=0)

            with open(cut_info_path, 'w') as f:
                f.write(json.dumps(cut_info, indent=4))

            self.win.cut_infos_path = cut_info_path

        del model
        del device
        torch.cuda.empty_cache()

        if self.source == "make":
            if os.path.isdir(CutWorkFilesDir):  # 如果路径存在并且是文件夹
                shutil.rmtree(CutWorkFilesDir)

        if no_3d_count == lsLen:
            self.is_error = 1
            self.error0.emit(f"{imageDir} folder has no 3D TIF images or they are blank, cannot predict!")

    """神经元小图批量预测"""

    def NeuralDataImgBatchPredict(self, imageDir, modelPath, saveDir):
        self.progress_text.emit("Neuron small image prediction!", 0)
        time.sleep(2)

        gpu_id = gpu_device_use.get_gpu_utilization()
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = torch.device("cuda", gpu_id)
        else:
            self.new_text = "No available GPU"
            return

        nameLs = [l for l in self.small_img if ".tif" in l]
        if len(nameLs) == 0:
            file_names = [l for l in os.listdir(imageDir) if ".tif" in l]
        else:
            file_names = nameLs

        lsLen = len(file_names)
        no_3d_count = 0
        if self.source == "pred":
            saveDir = join(saveDir, "Swc")
            os.makedirs(saveDir, exist_ok=True)
        if not lsLen:  # 如果不存在图像
            if os.path.isdir(saveDir):
                shutil.rmtree(saveDir)
            os.makedirs(saveDir, exist_ok=True)

        # 重新生成PredictWorkFiles
        # PredictWorkFiles——seg
        # PredictWorkFiles——PredictWorkFiles

        PredictWorkFilesDir = join(str(Path(saveDir).parent), "PredictWorkFiles")
        if not self.is_keep_on:  # 重新预测
            if os.path.isdir(PredictWorkFilesDir):
                shutil.rmtree(PredictWorkFilesDir)
        os.makedirs(PredictWorkFilesDir, exist_ok=True)

        seg_saveDir = join(PredictWorkFilesDir, 'seg')
        os.makedirs(seg_saveDir, exist_ok=True)

        # 加载网络
        model = ModelPredictClass(modelPath, fieldLen=self.fieldLen, device=device)

        start_time = time.time()
        self.del_seg = []

        # 保存信息
        for kk, name in enumerate(file_names):
            img_path = join(str(imageDir), name)  # 图像路径
            file_stem = str(Path(name).stem)
            savePath = join(saveDir, f"{file_stem}.swc")
            seg_savePath = join(seg_saveDir, name)

            if self.is_keep_on:
                # 补充：继续预测判断文件之前是否预测完成，无需追踪
                if os.path.exists(savePath):
                    if os.path.getsize(savePath) > self.swc_min_size:
                        # 需要删除分割文件夹中的分割结果，防止重新追踪
                        if os.path.exists(seg_savePath):
                            self.del_seg.append(name)  # 添加图像名字到去除列表
                        text = '[Prediction progress %.2f%%]' % ((kk + 1) / lsLen * 100)
                        self.progress_text.emit(text, kk)
                        continue

            with open(savePath, "w") as swc_file:  # 创建swcFile
                pass
            swc_file.close()

            if self.is_keep_on:
                # 补充：继续预测判断文件之前是否分割完成，任然需要追踪
                if os.path.exists(seg_savePath):
                    text = '[Prediction progress %.2f%%]' % ((kk + 1) / lsLen * 100)
                    self.progress_text.emit(text, kk)
                    continue

            img = tifffile.imread(img_path)  # 读取图像
            if len(img.shape) != 3 or not img.any():
                no_3d_count += 1
                text = '[Prediction progress %.2f%%]' % ((kk + 1) / lsLen * 100)
                self.progress_text.emit(text, kk)
                continue
            img_shapes = img.shape
            # 数据补充，防止模型分割数据尺寸问题
            img, is_small = self.pad_to_min_size_tail(img)  # small

            seg = model(img)
            seg = mask_label(seg)
            # seg[seg < 103] = 0
            if len(seg[seg > 0]) >= self.min_seeds:
                if is_small:
                    seg = seg[:img_shapes[0], :img_shapes[1], :img_shapes[2]]
                tifffile.imwrite(seg_savePath, seg, compression="lzw")

            userTime = time.time() - start_time
            surplusTime = userTime / (kk + 1) * (lsLen - kk - 1)
            text = '[Prediction progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                (kk + 1) / lsLen * 100, userTime, surplusTime)
            self.progress_text.emit(text, kk)
            torch.cuda.empty_cache()

        if no_3d_count == lsLen:
            self.is_error = 1
            self.error0.emit(f"{imageDir} folder has no 3D TIF images or they are blank, cannot predict!")
            if self.source == "make":
                if os.path.isdir(PredictWorkFilesDir):
                    shutil.rmtree(PredictWorkFilesDir)
            return
        self.progress_text.emit("[Prediction finished]", 0)
        # Tracking
        self.progress_text.emit("[Tracking started]", 1)
        # all_swc_files_none = self.BigImageCutPredict(seg_saveDir, saveDir, 100)
        # if all_swc_files_none:
        self.VoxelScopingCenterlineTracing(seg_saveDir, saveDir, 100)

        userTime = time.time() - start_time
        surplusTime = 0
        text = '[Total progress %.2f%%] [Time elapsed %ds] [Estimated total remaining time %ds]' % (100, userTime, surplusTime)
        self.progress_text.emit(text, 0)

        # 随机预测图展示
        image_i = random.randint(0, int(lsLen / 2))
        if lsLen > image_i:
            image_name = os.listdir(imageDir)[image_i]
            swc_path = join(saveDir, str(Path(image_name).stem) + ".swc")
            if os.path.exists(swc_path):
                image = tifffile.imread(join(imageDir, image_name))
                if image.any():
                    mask = self.swc_to_mask(image.shape[::-1], swc_path)
                    image_2d = self.MaxProject(image, 1)
                    mask_2d = self.MaxProject(mask, 0)
                    self.progress0.emit("", image_2d, mask_2d)
        if lsLen > 4:
            image_j = random.randint(int(lsLen / 2) + 1, lsLen - 1)
            if lsLen > image_j and image_j > 0:
                image_name = os.listdir(imageDir)[image_j]
                swc_path = join(saveDir, str(Path(image_name).stem) + ".swc")
                if os.path.exists(swc_path):
                    image = tifffile.imread(join(imageDir, image_name))
                    if image.any():
                        mask = self.swc_to_mask(image.shape[::-1], swc_path)
                        image_2d = self.MaxProject(image, 1)
                        mask_2d = self.MaxProject(mask, 0)
                        self.progress0.emit("", image_2d, mask_2d)

        del model
        del device

        if self.source == "make":
            if os.path.isdir(PredictWorkFilesDir):
                shutil.rmtree(PredictWorkFilesDir)

    def BigImageCutPredict(self, cut_seg_dir, cut_swc_dir, total_progress):
        # 大图分割结果，各方向切块数xyz，切块总数，切块大小xyz，大图大小xyz，大图swc保存文件夹
        CutWorkFilesDir = Path(cut_seg_dir).parent

        PredictWorkFilesDir = join(CutWorkFilesDir, "PredictWorkFiles")
        if os.path.isdir(PredictWorkFilesDir):
            shutil.rmtree(PredictWorkFilesDir)
        os.makedirs(PredictWorkFilesDir, exist_ok=True)

        SkeletonWorkFilesDir = join(PredictWorkFilesDir, 'SkeletonWorkFiles')  # 追踪处理文件夹
        os.makedirs(SkeletonWorkFilesDir, exist_ok=True)
        SkeletonDir = join(SkeletonWorkFilesDir, 'ResSwc', 'Skeleton')  # 自动生成
        paths = self.set_skeleton_files(SkeletonWorkFilesDir)

        # 大图切块追踪
        self.mask_batch_to_skeleton_swc(cut_swc_dir, cut_seg_dir, paths, SkeletonDir, total_progress)

        try:
            if os.path.isdir(PredictWorkFilesDir):
                shutil.rmtree(PredictWorkFilesDir)
        except Exception as e:
            print()

        all_swc_files_none = True

        for name in [n for n in os.listdir(cut_swc_dir) if ".swc" in n]:
            swc_path = join(cut_swc_dir, name)
            if os.path.isfile(swc_path):
                if os.path.getsize(swc_path) > 0:
                    all_swc_files_none = False
                    break

        return all_swc_files_none

    """快速追踪"""

    def VoxelScopingCenterlineTracing(self, cut_predict_dir, cut_swc_dir, total_progress):
        names = [n for n in os.listdir(cut_predict_dir) if ".tif" in n and n not in self.del_seg]
        namesLen = len(names)
        print(namesLen)
        use_skeletonize = True
        prune = False
        if namesLen:
            workQue = Queue()
            finishQue = Queue()
            errorQue = Queue()
            # 如果没有指定工作进程数，则使用CPU核心数
            num_workers = int(min(os.cpu_count(), len(names)) / 2) + 1
            print(num_workers)
            # root = str(Path(cut_predict_dir).parent)
            # signal_op_dir = os.path.join(root, "signal_op")
            # os.makedirs(signal_op_dir, exist_ok=True)
            signal_op_dir = None

            extractor = VascularSkeletonExtractor(
                vessel_scale=1.0,
                threshold=0.1,
                min_vessel_length=2
            )  # swc转化

            for i, name in enumerate(names):
                cut_seg_path = join(cut_predict_dir, name)
                cut_swc_path = join(cut_swc_dir, Path(name).stem + ".swc")

                workQue.put((cut_seg_path, cut_swc_path, extractor, signal_op_dir, use_skeletonize, prune))

            original_work_len = workQue.qsize()

            for i in range(num_workers):
                workQue.put(())

            ps = []
            for i in range(num_workers):
                ps.append(
                    Process(target=voxel_scoping_centerline_tracing_process, args=(workQue, finishQue, errorQue,)))
                ps[-1].daemon = True
                ps[-1].start()

            start_time = time.time()
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
                    self.error_logger(e)
                    self.is_error = 1
                    self.error0.emit(str(e))
                    return

                # 更新进度
                if curMakeSize > 0:
                    userTime = time.time() - start_time
                    surplusTime = userTime / curMakeSize * (original_work_len - curMakeSize)
                    logInfo = f'[Total progress %.2f%%] [Fast tracking progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        total_progress, curMakeSize / original_work_len * 100, userTime, surplusTime)
                    self.progress_text.emit(logInfo, curMakeSize)

                time.sleep(0.5)

            # 等待所有进程完成
            for p in ps:
                p.join(timeout=1)  # 最多等待5Second
                if p.is_alive():
                    p.terminate()

            for i in range(num_workers):
                if i == 0:
                    userTime = time.time() - start_time
                    logInfo = f'[Total progress %.2f%%] [Fast tracking progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        total_progress, 100, userTime, 0)
                    self.progress_text.emit(logInfo, 1)
                logInfo = f'[End process progress %.2f%%]' % ((i + 1) / num_workers * 100)
                self.progress_text.emit(logInfo, i)
                time.sleep(0.2)

            if signal_op_dir is not None:
                if os.path.exists(signal_op_dir):
                    shutil.rmtree(signal_op_dir, ignore_errors=True)

    def MaxProject(self, img, is_enhance):
        img = np.max(img, axis=0)
        img = self.normalize_and_scale_to_uint8(img)
        if is_enhance:
            img = self.enhance_contrast_histogram_equalization(img)
        return img

    def enhance_contrast_histogram_equalization(self, img):
        """
        使用直方图均衡化增强图像对比度。

        Parameter:
            img (numpy.ndarray): 输入图像。
        返回:
            numpy.ndarray: 对比度增强后的图像。
        """
        # 应用直方图均衡化
        img_eq = cv2.equalizeHist(img)
        return img_eq

    def normalize_and_scale_to_uint8(self, data):
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

    def del_file(self, path_data):
        for i in os.listdir(path_data):  # os.listdir(path_data)  # 返回一个列表，里面是当前目录下面的所有东西的相对路径
            file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
            if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
                os.remove(file_data)
            else:
                self.del_file(file_data)

    """批量追踪"""

    def mask_batch_to_skeleton_swc(self, cut_swc_dir, seg_saveDir, paths, SkeletonDir, total_progress):
        (MaskWorkSplitPath, ResGMMPath, ResGMMNewProcessPath,
         GMMWorkSplitPath, ResSwcGMMNewProcessPath, ResSwcPath) = paths

        # 清空处理文件夹
        for path in paths:
            if os.path.exists(path):
                self.del_file(path)
            os.makedirs(path, exist_ok=True)

        gs_Number = 1
        normal_Number = 4
        ls = [n for n in os.listdir(seg_saveDir) if ".tif" in n and n not in self.del_seg]
        print("*" * 100, len(ls))
        for name in ls.copy():
            img = tifffile.imread(join(seg_saveDir, name))
            if len(img[img > 0]) < self.min_seeds:
                ls.remove(name)
        print("*" * 100, len(ls))
        if not len(ls):
            return

        self.cut_swc_dir = cut_swc_dir
        print("Distance transform initialization")
        # print(ls)
        ls = self.GMMDataSplit(gs_Number, seg_saveDir, ls, MaskWorkSplitPath)

        print("Gaussian distance transform")
        self.MulMaskToGMMBatch(gs_Number, self.gsExePath, MaskWorkSplitPath, ResGMMPath, ResGMMNewProcessPath, ls,
                               total_progress=total_progress)

        print("Normalization initialization")
        self.GMMResDataSplit(normal_Number, ResGMMPath, GMMWorkSplitPath, ResSwcPath)

        print("Max normalization")
        self.MulGmmToSwcBatch(normal_Number, self.normalExePath, GMMWorkSplitPath, ResSwcGMMNewProcessPath, ls,
                              total_progress=total_progress, SkeletonDir=SkeletonDir)

        if os.path.exists(SkeletonDir):
            names = [l for l in os.listdir(SkeletonDir) if ".swc" in l and len(l) > 4]
            if len(names):
                for name_ in names:
                    cut_swc_path = join(cut_swc_dir, name_)
                    Skeleton_path = join(SkeletonDir, name_)
                    # print("Skeleton_path: ", os.path.getsize(Skeleton_path))
                    shutil.copy(Skeleton_path, cut_swc_path)

    '''GMM数据任务分配'''

    def GMMDataSplit(self, MNumber, segDir, ls, MaskWorkSplitPath):
        # self.total_len = len(ls)
        # ls = [l for l in ls if '.tif' in l and ("09" in l.split("_")[-1] or "08" in l.split("_")[-1])]
        ls = sorted(ls, key=lambda x: os.path.getsize(join(segDir, x)))
        mulInfo = [[] for i in range(MNumber)]
        for i, name in enumerate(ls):
            mulInfo[i % MNumber].append(name)
        for i, info in enumerate(mulInfo):
            with open(join(MaskWorkSplitPath, str(i).zfill(5) + '.txt'), 'w') as f:
                for name in info:
                    f.write(join(segDir, name).replace('\\', '/') + '\n')
        return ls

    def MulMaskToGMMBatch(self, MNumber, gsExePath, MaskWorkSplitPath, ResGMMPath, ResGMMNewProcessPath, ls,
                          total_progress=None):
        workLen = len(ls)
        names = os.listdir(MaskWorkSplitPath)
        nameLen = len(names)
        proceLs = set(os.listdir(ResGMMNewProcessPath))
        workQue = Queue()
        finishQue = Queue()
        errorQue = Queue()
        lines = ls
        for name in names:
            path = str(Path(join(MaskWorkSplitPath, name)))
            workQue.put(path)
        for i in range(MNumber):
            workQue.put("")
        ps = []
        for i in range(MNumber):
            ps.append(Process(target=SignMaskToGMM, args=(workQue, finishQue, errorQue, ResGMMPath, i, gsExePath,
                                                          ResGMMNewProcessPath, proceLs, )))
            ps[-1].daemon = True
            ps[-1].start()

        start_time = time.time()
        ii = -1
        is_over = 0
        pro_c = 0
        while True:
            if pro_c == 0:
                logInfo = '[Total progress %.2f%%] [Distance transform progress %.2f%%]' % (
                    total_progress, 0)
                self.progress_text.emit(logInfo, 1)
            # 进程报错
            if errorQue.qsize():
                e = errorQue.get()
                for p in ps:
                    p.terminate()
                self.error_logger(e)
                self.is_error = 1
                self.error0.emit(str(e))
                return
            if os.path.exists(ResGMMPath):
                wi = 0
                for i, line in enumerate(lines):
                    path = Path(join(ResGMMPath, str(Path(line).stem) + "_GMMResult.swc"))
                    if os.path.exists(path):
                        if os.path.getsize(path) > 0:
                            # print(path)
                            wi += 1
                            if line == lines[-1]:
                                is_over = 1
                                break
                wi -= 1
                if is_over:
                    wi = workLen - 1
                    userTime = time.time() - start_time
                    surplusTime = userTime / (wi + 1) * (workLen - wi - 1)
                    logInfo = '[Total progress %.2f%%] [Distance transform progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        total_progress, (wi + 1) / workLen * 100, userTime, surplusTime)
                    self.progress_text.emit(logInfo, wi + pro_c)
                    break
                if wi > ii:
                    ii = wi
                if wi >= 0:
                    userTime = time.time() - start_time
                    surplusTime = userTime / (wi + 1) * (workLen - wi - 1)
                    logInfo = '[Total progress %.2f%%] [Distance transform progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        total_progress, (wi + 1) / workLen * 100, userTime, surplusTime)
                    self.progress_text.emit(logInfo, wi + pro_c)
            if finishQue.qsize() == nameLen:
                wi = workLen - 1
                userTime = time.time() - start_time
                surplusTime = userTime / (wi + 1) * (workLen - wi - 1)
                if surplusTime >= 0:
                    logInfo = '[Total progress %.2f%%] [Tracking progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        total_progress, (wi + 1) / workLen * 100, userTime, surplusTime)
                    self.progress_text.emit(logInfo, wi + pro_c)
                break
            pro_c = 1
            time.sleep(1)

        for i in range(MNumber):
            if i == 0:
                logInfo = '[End process progress %.2f%%]' % ((i + 1) / MNumber * 100)
                self.progress_text.emit(logInfo, i)
                ps[i].join()
            else:
                ps[i].join()
                logInfo = '[End process progress %.2f%%]' % ((i + 1) / MNumber * 100)
                self.progress_text.emit(logInfo, i)
        print('Finish!')

    '''GMM数据任务分配'''

    def GMMResDataSplit(self, MNumber, ResGMMPath, GMMWorkSplitPath, ResSwcPath):
        ls = os.listdir(ResGMMPath)
        ls = sorted(ls, key=lambda x: os.path.getsize(join(ResGMMPath, x)))
        mulInfo = [[] for i in range(MNumber)]
        for i, name in enumerate(ls):
            mulInfo[i % MNumber].append(name)
        for i, info in enumerate(mulInfo):
            with open(join(GMMWorkSplitPath, str(i).zfill(5) + '.txt'), 'w') as f:
                f.write(ResGMMPath.replace('\\', '/') + '/\n')
                f.write(ResSwcPath.replace('\\', '/') + '/\n')
                [f.write(name + '\n') for name in info]

    def MulGmmToSwcBatch(self, MNumber, normalExePath, GMMWorkSplitPath, ResSwcGMMNewProcessPath, ls,
                         total_progress=None, SkeletonDir=None):
        workLen = len(ls)
        nameLsQue = Queue()
        finishQue = Queue()
        errorQue = Queue()
        names = os.listdir(GMMWorkSplitPath)
        nameLen = len(names)
        proceLs = set(os.listdir(ResSwcGMMNewProcessPath))
        for name in names:
            # nameLsQue.put(join(GMMWorkSplitPath, name).replace('\\', '/'))
            nameLsQue.put(str(Path(join(GMMWorkSplitPath, name))))
        # [nameLsQue.put(join(GMMWorkSplitPath, name).replace('\\', '/')) for name in names]
        for i in range(MNumber):
            nameLsQue.put("")
        ps = []
        for i in range(MNumber):
            ps.append(Process(target=QueGmmToSwc, args=(nameLsQue, finishQue, errorQue, normalExePath,
                                                        ResSwcGMMNewProcessPath, proceLs, )))
            ps[-1].daemon = True
            ps[-1].start()
        start_time = time.time()
        ii = -1
        size_dict = {}
        pro_c = 0
        while True:
            if pro_c == 0:
                logInfo = '[Total progress %.2f%%] [Tracking progress %.2f%%]' % (
                    total_progress, 0)
                self.progress_text.emit(logInfo, 0)
            if errorQue.qsize():
                e = errorQue.get()
                for p in ps:
                    p.terminate()
                self.error_logger(e)
                self.is_error = 1
                self.error0.emit(str(e))
                return
            if os.path.exists(SkeletonDir):
                names = [l for l in os.listdir(SkeletonDir) if ".swc" in l and len(l) > 4]
                wi = 0
                for name in names:
                    cut_swc_path = Path(join(self.cut_swc_dir, name))
                    path = Path(join(SkeletonDir, name))
                    if os.path.exists(path):
                        file_size = os.path.getsize(path)
                        if file_size:
                            wi += 1
                            if size_dict.get(name, None) is None:
                                size_dict[name] = file_size
                            else:
                                if size_dict[name] > 0:
                                    if file_size == size_dict[name]:
                                        shutil.copy(path, cut_swc_path)
                                    else:
                                        size_dict[name] = file_size
                wi -= 1
                if wi > ii:
                    ii = wi
                if wi >= 0:
                    userTime = time.time() - start_time
                    surplusTime = userTime / (wi + 1) * (workLen - wi - 1)
                    if surplusTime >= 0:
                        logInfo = '[Total progress %.2f%%] [Tracking progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                            total_progress, (wi + 1) / workLen * 100, userTime, surplusTime)
                        self.progress_text.emit(logInfo, wi + pro_c)

            if finishQue.qsize() == nameLen:
                wi = workLen - 1
                userTime = time.time() - start_time
                surplusTime = userTime / (wi + 1) * (workLen - wi - 1)
                if surplusTime >= 0:
                    logInfo = '[Total progress %.2f%%] [Tracking progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        total_progress, (wi + 1) / workLen * 100, userTime, surplusTime)
                    self.progress_text.emit(logInfo, wi + pro_c)
                break
            pro_c = 1
            time.sleep(2)

        for i in range(MNumber):
            if i == 0:
                logInfo = '[End process progress %.2f%%]' % ((i + 1) / MNumber * 100)
                self.progress_text.emit(logInfo, i)
                ps[i].join()
            else:
                ps[i].join()
                logInfo = '[End process progress %.2f%%]' % ((i + 1) / MNumber * 100)
                self.progress_text.emit(logInfo, i)
        print('Finish!')

    def swc_to_mask(self, shapes, swcPath):
        kernelLen = 2
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

        swcData = np.loadtxt(swcPath, ndmin=2)  # 读取骨架文件
        swcDataLs = self.SplitSwcData(swcData)
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
                curPc = self.GetPcKernelPc(data, kernelArr, imgShape[::-1])

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

    def SplitSwcData(self, swcData):
        indLs = np.where(swcData[:, -1] == -1)[0].tolist() + [swcData.shape[0]]
        swcDataLs = []
        for i in range(len(indLs) - 1):
            data = swcData[indLs[i]: indLs[i + 1]]
            sp = data[0, 0]
            data[:, 0] -= sp - 1
            data[1:, -1] -= sp - 1
            swcDataLs.append(data)
        return swcDataLs

    '''获取点云核点云'''

    def GetPcKernelPc(self, pc, kernelArr, imgShape):
        curPc = (pc[:, None] + kernelArr[None]).reshape([-1, 3])
        curPc = np.round(curPc).astype(np.int32)
        curPc[curPc < 0] = 0
        curPc[curPc[:, 2] > imgShape[0] - 1, 2] = imgShape[0] - 1
        curPc[curPc[:, 1] > imgShape[1] - 1, 1] = imgShape[1] - 1
        curPc[curPc[:, 0] > imgShape[2] - 1, 0] = imgShape[2] - 1
        curPc = np.unique(curPc, axis=0)
        return curPc

    '''预测结果拼接'''

    def SwcSplice(self,
                  bigSize=None,
                  imgSize=None,
                  r=None,
                  sliceNumber=None,
                  workLen=None,
                  savePath=None,
                  cut_swc_dir=None,
                  total_progress=None,
                  i=None):
        swc_lines_all = []

        block_size_x, block_size_y, block_size_z = imgSize
        br_x = block_size_x - r[0]
        br_y = block_size_y - r[1]
        br_z = block_size_z - r[2]

        count = 0
        swc_add = 0
        swc_add_ = 0
        sTime = time.time()

        # 遍历所有图像并拼接到结果数组中
        for nz in range(sliceNumber[2]):  # ProcessZ轴方向
            for ny in range(sliceNumber[1]):  # ProcessY轴方向
                for nx in range(sliceNumber[0]):  # ProcessX轴方向
                    swc_name = f"{str(nz).zfill(4)}_{str(ny).zfill(2)}_{str(nx).zfill(2)}.swc"

                    swc_path = join(cut_swc_dir, swc_name)
                    # 判断 swc_path 是否存在
                    if not os.path.isfile(swc_path):
                        print(f"{swc_path} does not exist")
                        userTime = time.time() - sTime
                        surplusTime = userTime / (count + 1) * (workLen - count - 1)
                        logInfo = '[Total progress %.2f%%] [Splicing progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                            total_progress, (count + 1) / workLen * 100, userTime, surplusTime)
                        self.progress_text.emit(logInfo, max(i, count))
                        count += 1
                        continue

                    # 判断 swc_path 是否为空
                    if not os.path.getsize(swc_path):
                        userTime = time.time() - sTime
                        surplusTime = userTime / (count + 1) * (workLen - count - 1)
                        logInfo = '[Total progress %.2f%%] [Splicing progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                            total_progress, (count + 1) / workLen * 100, userTime, surplusTime)
                        self.progress_text.emit(logInfo, max(i, count))
                        count += 1
                        continue

                    is_surpass = False
                    block_size_x, block_size_y, block_size_z = imgSize

                    # 计算块的起始索引（边缘块从末尾向前取 block_size）
                    start_z = min(nz * br_z, bigSize[2] - block_size_z)
                    start_y = min(ny * br_y, bigSize[1] - block_size_y)
                    start_x = min(nx * br_x, bigSize[0] - block_size_x)

                    # 计算结束索引
                    end_z = start_z + block_size_z
                    end_y = start_y + block_size_y
                    end_x = start_x + block_size_x

                    # 大图尺寸不足
                    if bigSize[2] - block_size_z < 0:
                        start_z = 0
                        end_z = bigSize[2]
                        block_size_z = end_z
                        is_surpass = True
                    # 大图边缘切块尺寸不足
                    elif start_z == bigSize[2] - block_size_z and nz * br_z != bigSize[2] - block_size_z:  # 边缘冗余
                        # 倒数第二块的结束索引减冗余，作为最后一块的起始索引，避免过多追踪结果重合
                        start_z = (nz - 1) * br_z + block_size_z - r[2]
                        end_z = bigSize[2]
                        is_surpass = True

                    if bigSize[1] - block_size_y < 0:  # 大图边缘切块尺寸不足，大图尺寸不足
                        start_y = 0
                        end_y = bigSize[1]
                        block_size_y = end_y
                        is_surpass = True
                    elif start_y == bigSize[1] - block_size_y and ny * br_y != bigSize[1] - block_size_y:  # 边缘冗余
                        start_y = (ny - 1) * br_y + block_size_y - r[1]
                        end_y = bigSize[1]
                        is_surpass = True

                    if bigSize[0] - block_size_x < 0:  # 大图不足填充生成的小图
                        start_x = 0
                        end_x = bigSize[0]
                        block_size_x = end_x
                        is_surpass = True
                    elif start_x == bigSize[0] - block_size_x and nx * br_x != bigSize[0] - block_size_x:  # 边缘冗余
                        start_x = (nx - 1) * br_x + block_size_x - r[0]
                        end_x = bigSize[0]
                        is_surpass = True

                    if is_surpass:  # 超过原图
                        swcData = np.loadtxt(swc_path, ndmin=2)
                        if len(swcData):
                            if len(swcData[0]) == 7:
                                swcDataLs = self.SplitSwcData(swcData)
                                dx = end_x - start_x
                                dy = end_y - start_y
                                dz = end_z - start_z
                                xmin, xmax, ymin, ymax, zmin, zmax = (block_size_x - dx, block_size_x,
                                                                      block_size_y - dy, block_size_y,
                                                                      block_size_z - dz, block_size_z)
                                xmax -= 2
                                ymax -= 2
                                zmax -= 2
                                cut_center = np.array([xmax + xmin, ymax + ymin, zmax + zmin]) / 2  # Center
                                cut_v = np.array([xmax - xmin, ymax - ymin, zmax - zmin]) / 2  # 方向向量
                                cut_r = np.linalg.norm(cut_v)

                                line_list = []
                                r_dict = {}

                                for swcData in swcDataLs:
                                    x0 = swcData[..., 2]
                                    y0 = swcData[..., 3]
                                    z0 = swcData[..., 4]

                                    x0_min = np.min(x0)
                                    x0_max = np.max(x0)
                                    y0_min = np.min(y0)
                                    y0_max = np.max(y0)
                                    z0_min = np.min(z0)
                                    z0_max = np.max(z0)

                                    center = np.array([x0_max + x0_min, y0_max + y0_min, z0_max + z0_min]) / 2  # Center
                                    v = np.array([x0_max - x0_min, y0_max - y0_min, z0_max - z0_min]) / 2  # 方向向量
                                    cr = np.linalg.norm(v)
                                    dist = np.linalg.norm(cut_center - center)  # 中心间距
                                    if dist < cr + cut_r:
                                        for ii, item in enumerate(swcData):
                                            # Radius
                                            r_dict[tuple(item[2: 5])] = item[5]

                                            if int(item[-1]) <= -1:
                                                continue
                                            p0 = item[2: 5]
                                            index0 = int(float(item[-1]))
                                            index1 = index0 - 1
                                            p1 = swcData[index1, 2: 5]

                                            is_p0 = 0
                                            is_p1 = 0
                                            for vi in range(3):
                                                if np.abs(cut_center[vi] - p0[vi]) >= np.abs(cut_v[vi]):
                                                    is_p0 += 1
                                                    break
                                                if np.abs(cut_center[vi] - p1[vi]) >= np.abs(cut_v[vi]):
                                                    is_p1 += 1
                                                    break
                                            if not is_p0 and not is_p1:
                                                line_list.append([tuple(p1), tuple(p0)])

                                if len(line_list):
                                    swc_result = segments_to_swc(line_list)
                                    del line_list
                                else:
                                    swc_result = []

                                for d in swc_result:
                                    key = d[2: 5]
                                    rad = r_dict.get(tuple(key)) if r_dict.get(tuple(key)) else 0
                                    value = int(d[-1])
                                    if value == -1:
                                        line = (f"{swc_add + 1} "
                                                f"{0} "
                                                f"{(key[0] - xmin + start_x):.3f} "
                                                f"{(key[1] - ymin + start_y):.3f} "
                                                f"{(key[2] - zmin + start_z):.3f} "
                                                f"{rad} "
                                                f"{value}\n")
                                    else:
                                        line = (f"{swc_add + 1} "
                                                f"{0} "
                                                f"{(key[0] - xmin + start_x):.3f} "
                                                f"{(key[1] - ymin + start_y):.3f} "
                                                f"{(key[2] - zmin + start_z):.3f} "
                                                f"{rad} "
                                                f"{value + swc_add_}\n")
                                    swc_lines_all.append(line)
                                    swc_add += 1
                    else:
                        swcData = np.loadtxt(swc_path, ndmin=2)
                        if len(swcData):
                            if len(swcData[0]) == 7:
                                swcDataLs = self.SplitSwcData(swcData)
                                for swcData in swcDataLs:
                                    for ii, item in enumerate(swcData):
                                        if int(item[-1]) < -1:
                                            continue
                                        if item[0] - 1 != item[-1]:
                                            if int(item[-1]) == -1:
                                                swc_lines_all.append(
                                                    f"{swc_add + 1} "
                                                    f"{item[1]} "
                                                    f"{(item[2] + start_x):.3f} "
                                                    f"{(item[3] + start_y):.3f} "
                                                    f"{(item[4] + start_z):.3f} "
                                                    f"{item[5]} -1\n")
                                            else:
                                                swc_lines_all.append(
                                                    f"{swc_add + 1} "
                                                    f"{item[1]} "
                                                    f"{(item[2] + start_x):.3f} "
                                                    f"{(item[3] + start_y):.3f} "
                                                    f"{(item[4] + start_z):.3f} "
                                                    f"{item[5]} "
                                                    f"{item[6] + swc_add_}\n")
                                        else:
                                            swc_lines_all.append(
                                                f"{swc_add + 1} "
                                                f"{item[1]} "
                                                f"{(item[2] + start_x):.3f} "
                                                f"{(item[3] + start_y):.3f} "
                                                f"{(item[4] + start_z):.3f} "
                                                f"{item[5]} "
                                                f"{swc_add}\n")
                                        swc_add += 1
                                    swc_add_ += len(swcData)

                    swc_add_ = swc_add

                    if (count + 1) % 1 == 0:
                        userTime = time.time() - sTime
                        surplusTime = userTime / (count + 1) * (workLen - count - 1)
                        logInfo = '[Total progress %.2f%%] [Splicing progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                            total_progress, (count + 1) / workLen * 100, userTime, surplusTime)
                        self.progress_text.emit(logInfo, max(i, count))
                    count += 1

        # 一次性写入文件
        with open(savePath, "w") as f0:
            f0.writelines(swc_lines_all)

        del swc_lines_all


def convert_to_builtin_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_builtin_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_types(item) for item in obj]
    else:
        return obj


def get_1ch_shape(tif_path):
    with TiffFile(tif_path) as tif:
        # 1) 必须是单通道
        # if any(p.samplesperpixel != 1 for p in tif.pages):
        #     return None
        # 2) 所有页尺寸一致检查（Optional）
        page0 = tif.pages[0]  # 第一页/第一帧
        # shape_2d = (page0.imagelength, page0.imagewidth)
        # if any((p.imagelength, p.imagewidth) != shape_2d for p in tif.pages):
        #     return None
        if page0.samplesperpixel != 1:
            return None
        # 单通道 shape：(高, 宽) 或 (Z, 高, 宽)
        z_size = len(tif.pages)  # 整个深度
        shape = (z_size, page0.imagelength, page0.imagewidth)
        return shape


def voxel_scoping_centerline_tracing_process(workQue, finishQue, errorQue):
    while True:
        try:
            work = workQue.get(timeout=5)
            if len(work) == 0:
                finishQue.put(-1)
                return
            img_path, save_path, extractor, signal_op_dir, use_skeletonize, prune = work
            voxel_scoping_centerline_tracing(img_path, save_path, extractor,
                                             signal_op_dir=signal_op_dir,
                                             skeletonize=use_skeletonize,
                                             prune=prune)
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


def zero_border(seg: np.ndarray, r: int) -> np.ndarray:
    """
    将三维数组 seg 的边缘（厚度为 r）置零，原地修改。

    参数：
        seg : 三维 numpy 数组
        r   : 边界厚度（非负整数）
    """
    if r <= 0:
        return seg  # 无需操作

    shape = seg.shape
    # 依次处理三个轴（x, y, z）
    for axis in range(3):
        dim = shape[axis]
        # 构造当前轴的切片索引
        idx = [slice(None)] * 3

        # 前边缘（索引 0 到 r-1）
        idx[axis] = slice(0, r)
        seg[tuple(idx)] = 0

        # 后边缘（索引 dim-r 到 dim-1）
        # 当 r > dim 时，dim-r 为负，切片自动转为 0:dim，即整个轴清零
        idx[axis] = slice(dim - r, dim)
        seg[tuple(idx)] = 0
    return seg
