# -*- coding: utf-8 -*-
import os
import time
import sys
import traceback
from pathlib import Path
import tifffile
from tifffile import TiffFile
import cv2
import random
# 必须在任何 import torch 之前执行
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import shutil
import torch
from os.path import join
import numpy as np
import json
import zarr
# from .DataLoader import GetMultiTypeMemoryDataSetAndCropQxz2
# from torch.utils.data import DataLoader
# from .models.model import LoadModel
from .ModelPredictPy import ModelPredictClass, modelCfg
import cc3d
from speed_cc3d import speed_cc3d
from scipy.spatial import distance
from BVExample.BVMoudle import BVReader
from multiprocessing import Process, Queue
from PyQt5.QtCore import QThread, pyqtSignal
from typing import Tuple


import gpu_device_use


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


class CellDataPredictQThread(QThread):
    finish0 = pyqtSignal(str)
    progress0 = pyqtSignal(str, np.ndarray, np.ndarray)
    progress_text = pyqtSignal(str, int)
    error0 = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(CellDataPredictQThread, self).__init__()
        self.win = kwargs.get('win')
        self.source = kwargs.get('source')
        self.smallSize = [272, 272, 144]
        self.min_size = 300**3
        self.small_img = []
        self.is_error = 0
        self.is_keep_on = 0  # 默认重新预测
        self.logger = self.win.logger
        self.r = np.array([32, 32, 20])  # xyz
        # self.r = np.array([1, 1, 1])  # xyz
        self.fieldLen = 16
        self.readObj = BVReader()  # 读取bv格式对象
        self.min_shapes_step = int(2 ** (len(modelCfg['f_maps']) - 1))
        # [z,y,x]
        self.min_shapes = [self.min_shapes_step,
                           self.min_shapes_step,
                           self.min_shapes_step]
        self.del_seg = []

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
                    self.smallSize = [272, 272, 144]
                elif all(x >= y for x, y in zip(self.smallSize, smallSize)):
                    self.smallSize = smallSize
                model_path = making_arguments['model_path']  # 模型路径
                imageDir = making_arguments['divide_img_dir']
                saveDir = making_arguments['divide_swc_dir']
            elif self.source == "pred":
                self.is_keep_on = self.win.data_predict_dict.get("is_keep_on", self.is_keep_on)
                cfg_level = self.win.data_predict_dict['cfg_level']
                self.smallSize = self.win.data_predict_dict['small_size']  # xyz
                model_path = self.win.data_predict_dict['modelPath']  # 模型路径
                imageDir = self.win.data_predict_dict['imageDir']
                saveDir = self.win.data_predict_dict['saveDir']
                bv_ROI = self.win.data_predict_dict.get("bv_ROI", [])
                os.makedirs(saveDir, exist_ok=True)
                saveDir = join(saveDir, "PredictResults")
                if not self.is_keep_on:  # 重新预测
                    if os.path.isdir(saveDir):
                        shutil.rmtree(saveDir)
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
                            self.CellDataZarrImgPredict(imageDir, model_path, saveDir, cfg_level)
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
                                self.CellDataCfgImgPredict(imageDir, model_path, saveDir, cfg_level, bv_ROI=bv_ROI)
                    else:
                        file_names = [n for n in os.listdir(imageDir) if ".tif" in n]
                        if len(file_names) == 0:
                            self.error0.emit("No TIF format files in the image folder, cannot predict!")
                            return
                        else:
                            if self.source == "pred":
                                self.CellDataBigImgPredict(imageDir, model_path, saveDir)
                                if len(self.small_img):
                                    self.CellDataImgBatchPredict(imageDir, model_path, saveDir)
                            if self.source == "make":
                                img_path = join(imageDir, file_names[0])
                                img = tifffile.imread(img_path)
                                if (img.shape[0] > self.smallSize[2] or
                                        img.shape[1] > self.smallSize[1] or img.shape[2] > self.smallSize[1]):
                                    self.CellDataBigImgPredict(imageDir, model_path, saveDir)
                                else:
                                    self.CellDataImgBatchPredict(imageDir, model_path, saveDir)
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
        self.logger.error("=== Wrong Position ===")
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

    """细胞OME-Zarr格式预测"""

    def CellDataZarrImgPredict(self, zarrPath, modelPath, saveDir, cfg_level):
        self.progress_text.emit("Predict Cell OME-Zarr format data!", 0)

        gpu_id = gpu_device_use.get_gpu_utilization()
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = torch.device("cuda", gpu_id)
        else:
            self.new_text = "No GPU available"
            return

        zarr_store = zarr.open(zarrPath, mode='r')

        shapes = zarr_store[f'{cfg_level}'].shape  # (z,y,x)
        seqInfo = np.array(shapes, dtype=np.int32)[::-1]  # xyz
        sx = sy = 512
        sz = seqInfo[2]
        print(sx, sy, sz)
        bvSmallSize = np.array([sx, sy, sz])
        BvSliceNumber = np.ceil((seqInfo - bvSmallSize) / (bvSmallSize - self.r)).astype(np.int32) + 1
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
                shutil.rmtree(CutWorkFilesDir)
        os.makedirs(CutWorkFilesDir, exist_ok=True)

        cut_infos_path = join(CutWorkFilesDir, "cut_infos.json")
        if self.is_keep_on:
            if os.path.exists(cut_infos_path):
                self.new_text = "All data has been predicted"
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
                    text = '[Overall progress %.2f%%]' % (total_progress)
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
                                text = '[Overall progress %.2f%%] [Prediction progress %.2f%%]' % (
                                    total_progress, 0)
                                self.progress_text.emit(text, progress_num)
                            else:
                                userTime = time.time() - start_time
                                surplusTime = userTime / (progress_num + 1) * (sliceLen - progress_num - 1)
                                text = '[Overall progress %.2f%%] [Prediction progress %.2f%%] [Elapsed time %ds] [Estimated remaining time %ds]' % (
                                    total_progress, ((progress_num + 1) / sliceLen * 100), userTime, surplusTime)
                                self.progress_text.emit(text, progress_num)

                            # if self.is_keep_on:
                            #     if os.path.exists(cut_swc_path):  # 存在swcFile，且大小不为0，已经完成追踪
                            #         if os.path.getsize(cut_swc_path) > 0:
                            #             self.del_seg.append(img_name)  # 添加图像名字到去除列表
                            #             progress_num += 1
                            #             continue

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
                                self.del_seg.append(img_name)
                                continue

                            img, is_small = self.pad_to_min_size_tail(img)  # cfg
                            # Predict
                            seg = model(img)
                            seg[seg < 103] = 0
                            if is_small:
                                seg = seg[:img_shapes[0], :img_shapes[1], :img_shapes[2]]
                            tifffile.imwrite(cut_seg_path, seg, compression="lzw")
                            progress_num += 1
                            torch.cuda.empty_cache()
            # Tracking
            self.progress_text.emit("[Start positioning]", 0)
            self.BigImageCutPredict(cut_seg_dir, cut_swc_dir, total_progress)

            userTime = time.time() - total_time
            surplusTime = userTime / (i - keep_count + 1) * (lsLen - i - 1)
            text = '[Overall progress %.2f%%] [Elapsed time %ds] [Estimated total remaining time %ds]' % (
            total_progress, userTime, surplusTime)
            self.progress_text.emit(text, 0)
            # 随机预测图展示
            image_i = random.randint(0, int(sliceLen / 2))
            if sliceLen > image_i:
                image_name = os.listdir(cut_img_dir)[image_i]
                image = tifffile.imread(join(cut_img_dir, image_name))
                mask = tifffile.imread(join(cut_seg_dir, image_name))
                if image.any():
                    image = image[:144, :272, :272]
                    mask = mask[:144, :272, :272]
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
                        image = image[:144, :272, :272]
                        mask = mask[:144, :272, :272]
                        image_2d = self.MaxProject(image, 1)
                        mask_2d = self.MaxProject(mask, 0)
                        self.progress0.emit("", image_2d, mask_2d)

            cut_info['dataType'] = "TIF"
            cut_info = convert_to_builtin_types(cut_info)
            with open(cut_info_path, 'w') as f:
                f.write(json.dumps(cut_info, indent=4))

        del model
        del device
        torch.cuda.empty_cache()

        cut_infos = convert_to_builtin_types(cut_infos)
        with open(cut_infos_path, 'w') as f:
            f.write(json.dumps(cut_infos, indent=4))

        self.win.cut_infos_path = cut_infos_path

    """细胞大数据格式预测"""

    def CellDataCfgImgPredict(self, bvDir, modelPath, saveDir, cfg_level, bv_ROI):
        self.progress_text.emit("Predict Cell BV format data!", 0)

        gpu_id = gpu_device_use.get_gpu_utilization()
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = torch.device("cuda", gpu_id)
        else:
            self.new_text = "No GPU available"
            return

        config_path = join(bvDir, "config.cfg")
        if not os.path.exists(config_path):
            self.new_text = "No config.cfg file available"
            return
        seqInfo_path = join(bvDir, "seqInfo.txt")
        with open(seqInfo_path, "r") as f:
            lines = f.readlines()
            seqInfo = [int(float(line)) for line in lines]  # xyz

        if len(bv_ROI):
            MinX, MaxX, MinY, MaxY, MinZ, MaxZ = bv_ROI
            MinX = int(max(MinX, 0))
            MinY = int(max(MinY, 0))
            MinZ = int(max(MinZ, 0))
            MaxX = int(min(MaxX, seqInfo[0]))
            MaxY = int(min(MaxY, seqInfo[1]))
            MaxZ = int(min(MaxZ, seqInfo[2]))
            min_size = [64, 64, 64]
            if (MaxX - MinX < min_size[0]
                    or MaxY - MinY < min_size[1]
                    or MaxZ - MinZ < min_size[2]):
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
                shutil.rmtree(CutWorkFilesDir)
        os.makedirs(CutWorkFilesDir, exist_ok=True)

        cut_infos_path = join(CutWorkFilesDir, "cut_infos.json")
        if self.is_keep_on:
            if os.path.exists(cut_infos_path):
                self.new_text = "All data has been predicted"
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
                    text = '[Overall progress %.2f%%]' % (total_progress)
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
                                text = '[Overall progress %.2f%%] [Prediction progress %.2f%%]' % (
                                    total_progress, 0)
                                self.progress_text.emit(text, progress_num)
                            else:
                                userTime = time.time() - start_time
                                surplusTime = userTime / (progress_num + 1) * (sliceLen - progress_num - 1)
                                text = '[Overall progress %.2f%%] [Prediction progress %.2f%%] [Elapsed time %ds] [Estimated remaining time %ds]' % (
                                    total_progress, ((progress_num + 1) / sliceLen * 100), userTime, surplusTime)
                                self.progress_text.emit(text, progress_num)

                            # if self.is_keep_on:
                            #     if os.path.exists(cut_swc_path):  # 存在swcFile，且大小不为0，已经完成追踪
                            #         if os.path.getsize(cut_swc_path) > 0:
                            #             self.del_seg.append(img_name)  # 添加图像名字到去除列表
                            #             progress_num += 1
                            #             continue

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
                                self.del_seg.append(img_name)
                                continue

                            img, is_small = self.pad_to_min_size_tail(img)  # cfg
                            # Predict
                            seg = model(img)
                            seg[seg < 103] = 0
                            if is_small:
                                seg = seg[:img_shapes[0], :img_shapes[1], :img_shapes[2]]
                            tifffile.imwrite(cut_seg_path, seg, compression="lzw")
                            progress_num += 1
                            torch.cuda.empty_cache()
            # Tracking
            self.progress_text.emit("[Start positioning]", 0)
            self.BigImageCutPredict(cut_seg_dir, cut_swc_dir, total_progress)

            userTime = time.time() - total_time
            surplusTime = userTime / (i - keep_count + 1) * (lsLen - i - 1)
            text = '[Overall progress %.2f%%] [Elapsed time %ds] [Estimated total remaining time %ds]' % (total_progress, userTime, surplusTime)
            self.progress_text.emit(text, 0)
            # 随机预测图展示
            image_i = random.randint(0, int(sliceLen / 2))
            if sliceLen > image_i:
                image_name = os.listdir(cut_img_dir)[image_i]
                image = tifffile.imread(join(cut_img_dir, image_name))
                mask = tifffile.imread(join(cut_seg_dir, image_name))
                if image.any():
                    image = image[:144, :272, :272]
                    mask = mask[:144, :272, :272]
                    image_2d = self.MaxProject(image, 1)
                    mask_2d = self.MaxProject(mask, 1)
                    self.progress0.emit("", image_2d, mask_2d)
            if sliceLen > 4:
                image_j = random.randint(int(sliceLen / 2) + 1, sliceLen - 1)
                if sliceLen > image_j and image_j > 0:
                    image_name = os.listdir(cut_img_dir)[image_j]
                    image = tifffile.imread(join(cut_img_dir, image_name))
                    mask = tifffile.imread(join(cut_seg_dir, image_name))
                    if image.any():
                        image = image[:144, :272, :272]
                        mask = mask[:144, :272, :272]
                        image_2d = self.MaxProject(image, 1)
                        mask_2d = self.MaxProject(mask, 1)
                        self.progress0.emit("", image_2d, mask_2d)

            cut_info['dataType'] = "TIF"
            cut_info = convert_to_builtin_types(cut_info)
            with open(cut_info_path, 'w') as f:
                f.write(json.dumps(cut_info, indent=4))

        del model
        del device
        torch.cuda.empty_cache()

        cut_infos = convert_to_builtin_types(cut_infos)
        with open(cut_infos_path, 'w') as f:
            f.write(json.dumps(cut_infos, indent=4))

        self.win.cut_infos_path = cut_infos_path


    """细胞大图预测"""

    def CellDataBigImgPredict(self, imageDir, modelPath, saveDir):
        self.progress_text.emit("Large cell image prediction!", 0)

        gpu_id = gpu_device_use.get_gpu_utilization()
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = torch.device("cuda", gpu_id)
        else:
            self.new_text = "No GPU available"
            return

        imgSize = np.array(self.smallSize, dtype=np.int32)  # xyz
        model = ModelPredictClass(modelPath, device=device, fieldLen=self.fieldLen)

        file_names = [n for n in os.listdir(imageDir) if ".tif" in n]
        lsLen = len(file_names)

        no_3d_count = 0

        CutWorkFilesDir = join(saveDir, "CutWorkFiles")
        if not self.is_keep_on:  # 重新预测
            if os.path.exists(CutWorkFilesDir):
                shutil.rmtree(CutWorkFilesDir)
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
            img_path = join(imageDir, file_name)
            file_stem = str(Path(file_name).stem)
            cut_info_dir = join(CutWorkFilesDir, file_stem)
            cut_info_path = join(cut_info_dir, f"{file_stem}.json")

            if self.is_keep_on:
                if os.path.exists(cut_info_path):  # 存在配置文件，已经完成预测
                    text = '[Overall progress %.2f%%]' % ((i + 1) / lsLen * 100)
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
                text = '[Overall progress %.2f%%]' % ((i + 1) / lsLen * 100)
                self.progress_text.emit(text, i)
                keep_count += 1
                continue

            # 小块图像
            if np.all(np.array(bigSize)[::-1] <= np.array(self.smallSize)) and self.source == "pred":
                self.small_img.append(file_name)
                text = '[Overall progress %.2f%%]' % ((i + 1) / lsLen * 100)
                self.progress_text.emit(text, i)
                keep_count += 1
                continue

            bigImg = tifffile.imread(img_path)  # 读取图像

            if not bigImg.any():
                no_3d_count += 1
                text = '[Overall progress %.2f%%]' % ((i + 1) / lsLen * 100)
                self.progress_text.emit(text, i)
                keep_count += 1
                continue

            os.makedirs(cut_info_dir, exist_ok=True)

            cut_info = {}
            bigSize = bigSize[::-1]  # xyz
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
            total_progress = (i + 1) / lsLen * 100
            start_time2 = time.time()
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
                            text = '[Overall progress %.2f%%] [Prediction progress %.2f%%] [Elapsed time %ds] [Estimated remaining time %ds]' % (
                                total_progress, ((progress_num + 1) / sliceLen * 100), userTime, surplusTime)
                            self.progress_text.emit(text, progress_num)

                            # if self.is_keep_on:
                            #     if os.path.exists(cut_swc_path):  # 存在swcFile，且大小不为0，已经完成追踪
                            #         if os.path.getsize(cut_swc_path) > 0:
                            #             self.del_seg.append(img_name)  # 添加图像名字到去除列表
                            #             progress_num += 1
                            #             continue

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
                                self.del_seg.append(img_name)
                                continue

                            img, is_small = self.pad_to_min_size_tail(img)  # big
                            # Predict
                            # seg = model(new_img)
                            seg = model(img)
                            seg[seg < 103] = 0
                            if is_small:
                                seg = seg[:img_shapes[0], :img_shapes[1], :img_shapes[2]]
                            tifffile.imwrite(cut_seg_path, seg, compression="lzw")
                            progress_num += 1
                            torch.cuda.empty_cache()
            # Tracking
            self.progress_text.emit("[Tracking start]", 0)
            self.BigImageCutPredict(cut_seg_dir, cut_swc_dir, total_progress)

            userTime = time.time() - total_time
            surplusTime = userTime / (i - keep_count + 1) * (lsLen - i - 1)
            text = '[Total progress %.2f%%] [Elapsed time %ds] [Estimated remaining time %ds]' % (total_progress, userTime, surplusTime)
            self.progress_text.emit(text, 0)
            # 随机预测图展示
            image_i = random.randint(0, int(sliceLen / 2))
            if sliceLen > image_i:
                image_name = os.listdir(cut_img_dir)[image_i]
                image = tifffile.imread(join(cut_img_dir, image_name))
                mask = tifffile.imread(join(cut_seg_dir, image_name))
                if image.any():
                    image = image[:144, :272, :272]
                    mask = mask[:144, :272, :272]
                    image_2d = self.MaxProject(image, 1)
                    mask_2d = self.MaxProject(mask, 1)
                    self.progress0.emit("", image_2d, mask_2d)
            if sliceLen > 4:
                image_j = random.randint(int(sliceLen / 2) + 1, sliceLen - 1)
                if sliceLen > image_j and image_j > 0:
                    image_name = os.listdir(cut_img_dir)[image_j]
                    image = tifffile.imread(join(cut_img_dir, image_name))
                    mask = tifffile.imread(join(cut_seg_dir, image_name))
                    if image.any():
                        image = image[:144, :272, :272]
                        mask = mask[:144, :272, :272]
                        image_2d = self.MaxProject(image, 1)
                        mask_2d = self.MaxProject(mask, 1)
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
            self.error0.emit(f"No 3D Tif format images or blank images found in folder {imageDir}, cannot predict!")

    """细胞小图批量预测"""

    def CellDataImgBatchPredict(self, imageDir, modelPath, saveDir):
        self.progress_text.emit("Small cell image prediction!", 0)
        time.sleep(2)

        gpu_id = gpu_device_use.get_gpu_utilization()
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            device = torch.device("cuda", gpu_id)
        else:
            self.new_text = "No GPU available"
            return

        nameLs = [n for n in self.small_img if ".tif" in n]
        if len(nameLs) == 0:
            file_names = [n for n in os.listdir(imageDir) if ".tif" in n]
        else:
            file_names = nameLs

        lsLen = len(file_names)

        no_3d_count = 0
        if self.source == "pred":
            saveDir = join(saveDir, "Swc")
            os.makedirs(saveDir, exist_ok=True)
        if not lsLen:
            if os.path.isdir(saveDir):
                shutil.rmtree(saveDir)
            os.makedirs(saveDir, exist_ok=True)

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

            # if self.is_keep_on:  # 继续预测
            #     # 补充：继续预测判断文件之前是否预测完成，无需追踪
            #     if os.path.exists(savePath):
            #         if os.path.getsize(savePath) > 0:
            #             # 需要删除分割文件夹中的分割结果，防止重新追踪
            #             if os.path.exists(seg_savePath):
            #                 self.del_seg.append(name)
            #             text = '[预测进度 %.2f%%]' % ((kk + 1) / lsLen * 100)
            #             self.progress_text.emit(text, kk)
            #             continue

            with open(savePath, "w") as swc_file:  # 创建swcFile
                pass
            swc_file.close()

            if self.is_keep_on:  # 继续预测
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
            seg[seg < 103] = 0
            if np.any(seg):
                if is_small:
                    seg = seg[:img_shapes[0], :img_shapes[1], :img_shapes[2]]
                tifffile.imwrite(seg_savePath, seg, compression="lzw")

            userTime = time.time() - start_time
            surplusTime = userTime / (kk + 1) * (lsLen - kk - 1)
            text = '[Prediction progress %.2f%%] [Elapsed time %ds] [Estimated remaining time %ds]' % (
                (kk + 1) / lsLen * 100, userTime, surplusTime)
            self.progress_text.emit(text, kk)
            torch.cuda.empty_cache()

        if no_3d_count == lsLen:
            self.is_error = 1
            self.error0.emit(f"No 3D Tif format images or blank images found in folder {imageDir}, cannot predict!")
            if self.source == "make":
                if os.path.isdir(PredictWorkFilesDir):
                    shutil.rmtree(PredictWorkFilesDir)
            return

        # Tracking
        self.progress_text.emit("[Tracking start]", 0)
        self.BigImageCutPredict(seg_saveDir, saveDir, 100)

        userTime = time.time() - start_time
        surplusTime = 0
        text = '[Total progress %.2f%%] [Elapsed time %ds] [Estimated remaining time %ds]' % (100, userTime, surplusTime)
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
                    mask_2d = self.MaxProject(mask, 1)
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
                        mask_2d = self.MaxProject(mask, 1)
                        self.progress0.emit("", image_2d, mask_2d)

        del model
        del device

        if self.source == "make":
            if os.path.isdir(PredictWorkFilesDir):
                shutil.rmtree(PredictWorkFilesDir)

    def BigImageCutPredict(self, cut_seg_dir, cut_swc_dir, total_progress):
        names = [n for n in os.listdir(cut_seg_dir) if ".tif" in n and n not in self.del_seg]
        if len(names):
            # 大图分割结果，各方向切块数xyz，切块总数，切块大小xyz，大图大小xyz，大图swc保存文件夹
            # 大图切块追踪
            InfoQueue = Queue()
            workLen = 0
            for img_name in names:
                swc_name = str(Path(img_name).stem) + ".swc"
                cut_seg_path = Path(join(cut_seg_dir, img_name))
                cut_swc_path = Path(join(cut_swc_dir, swc_name))
                seg = tifffile.imread(cut_seg_path)
                if seg.any():
                    InfoQueue.put((cut_swc_path, cut_seg_path))
                    workLen += 1
            # 多进程处理转化
            self.mask_to_skeleton_swc_process(names, InfoQueue=InfoQueue, total_progress=total_progress, workLen=workLen)

    def mask_to_skeleton_swc_process(self, names, InfoQueue=None, total_progress=0, workLen=0):
        if InfoQueue is None:
            InfoQueue = Queue()
        finishQue = Queue()
        errorQue = Queue()
        # 如果没有指定工作进程数，则使用CPU核心数
        MNumber = int(min(os.cpu_count(), len(names)) / 2) + 1
        print(MNumber)
        original_work_len = workLen

        for i in range(MNumber):
            InfoQueue.put(())

        ps = []
        for i in range(MNumber):
            ps.append(Process(target=get_bvInfoQue, args=(InfoQueue, finishQue, errorQue)))
            ps[-1].daemon = True
            ps[-1].start()

        sTime = time.time()
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
                userTime = time.time() - sTime
                surplusTime = userTime / (curMakeSize + 1) * (original_work_len - curMakeSize)
                logInfo = '[Total progress %.2f%%] [Localization progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                    total_progress, curMakeSize / original_work_len * 100, userTime, surplusTime)
                self.progress_text.emit(logInfo, curMakeSize)
            time.sleep(0.5)

        # 等待所有进程完成
        for p in ps:
            p.join(timeout=1)  # 最多等待5Second
            if p.is_alive():
                p.terminate()

        # 等待线程结束(很快)
        for i in range(MNumber):
            if i == 0:
                userTime = time.time() - sTime
                logInfo = '[Total progress %.2f%%] [Localization progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                    total_progress, 100, userTime, 0)
                self.progress_text.emit(logInfo, i + 1)
            logInfo = '[End process progress %.2f%%]' % (
                    (i + 1) / MNumber * 100)
            self.progress_text.emit(logInfo, i)
            time.sleep(0.2)

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

    def mask_to_skeleton_swc(self, mask, savePath, is_small):
        with open(savePath, "w") as f:
            f.write("")

        if not mask.any():  # 空白图像
            return

        thre = 5000
        mask2 = mask.copy()
        mask2[mask2 <= 103] = 0
        mask2[mask2 > 0] = 255
        labels_out, N = cc3d.connected_components(mask2, connectivity=26, return_N=True, out_dtype=np.uint32)

        flag = 1
        if N != 0:
            out1 = speed_cc3d.group_type(labels_out)
            out1Ls = len(out1)
            out1Ls_step = 1
            # out1Ls_step = int(out1Ls / 5) + 1
            center_point = []
            for i, key in enumerate(out1):
                if i % out1Ls_step == 0 and is_small:
                    self.progress_text.emit(f"[Localization progress {((i + 1) / out1Ls * 100):.2f}%]", i)
                # print('进度：%f' % (i / len(out1) * 100) + '%')
                point = np.array(out1[key], dtype=np.int32)
                if len(point) <= 297 or len(point) >= thre:
                    continue
                dist = distance.cdist(point, point)
                dist_max = np.max(dist)
                rhos = []
                for j in range(len(point)):
                    rhos.append(mask[point[j, 2], point[j, 1], point[j, 0]])
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

    def swc_to_mask(self, shapes, swc_path):
        kernelLen = 5
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

        swcData = np.loadtxt(swc_path, ndmin=2)
        dfImg[...] = maxD
        for item in swcData:
            if len(item) < 7:
                continue
            p0 = item[2: 5]
            curPc = self.GetPcKernelPc(np.array([p0]), kernelArr, imgShape)
            d = np.linalg.norm(p0 - curPc, axis=1)
            dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]] = np.min([dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]], d],
                                                                  axis=0)
        dfImg = -np.log(minD + dfImg / maxD * (1 - minD))
        dfImg2 = (dfImg / maskMax * 255).astype(np.uint8)
        return dfImg2

    def GetPcKernelPc(self, pc, kernelArr, imgShape):
        curPc = (pc[:, None] + kernelArr[None]).reshape([-1, 3])
        curPc = np.round(curPc).astype(np.int32)
        curPc[curPc < 0] = 0
        curPc[curPc[:, 2] > imgShape[2] - 1, 2] = imgShape[2] - 1
        curPc[curPc[:, 1] > imgShape[1] - 1, 1] = imgShape[1] - 1
        curPc[curPc[:, 0] > imgShape[0] - 1, 0] = imgShape[0] - 1
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
                        logInfo = '[Total progress %.2f%%] [Stitching progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                            total_progress, (count + 1) / workLen * 100, userTime, surplusTime)
                        self.progress_text.emit(logInfo, max(i, count))
                        count += 1
                        continue

                    # 判断 swc_path 是否为空
                    if not os.path.getsize(swc_path):
                        userTime = time.time() - sTime
                        surplusTime = userTime / (count + 1) * (workLen - count - 1)
                        logInfo = '[Total progress %.2f%%] [Stitching progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
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
                                swcDataLs = SplitSwcData(swcData)  # 点集数组
                                dx = end_x - start_x
                                dy = end_y - start_y
                                dz = end_z - start_z
                                xmin, xmax, ymin, ymax, zmin, zmax = (block_size_x - dx, block_size_x,
                                                                      block_size_y - dy, block_size_y,
                                                                      block_size_z - dz, block_size_z)
                                xmax -= 2
                                ymax -= 2
                                zmax -= 2
                                for swcData in swcDataLs:  # 遍历各点坐标
                                    # 判断是否在框内
                                    for ii, item in enumerate(swcData):
                                        if len(item) < 7:
                                            continue
                                        x0 = item[2]
                                        y0 = item[3]
                                        z0 = item[4]
                                        if xmin <= x0 <= xmax and ymin <= y0 <= ymax and zmin <= z0 <= zmax:
                                            line = (f"{swc_add + 1} "
                                                    f"{0} "
                                                    f"{(x0 - xmin + start_x):.3f} "
                                                    f"{(y0 - ymin + start_y):.3f} "
                                                    f"{(z0 - zmin + start_z):.3f} "
                                                    f"{0} "
                                                    f"{-1}\n")
                                            swc_lines_all.append(line)
                                            swc_add += 1
                    else:
                        swcData = np.loadtxt(swc_path, ndmin=2)
                        if len(swcData):
                            if len(swcData[0]) == 7:
                                swcDataLs = SplitSwcData(swcData)
                                for swcData in swcDataLs:
                                    for ii, item in enumerate(swcData):
                                        if len(item) < 7:
                                            continue
                                        x0 = item[2]
                                        y0 = item[3]
                                        z0 = item[4]
                                        if x0 > r[0] and y0 > r[1] and z0 > r[2]:
                                            line = (f"{swc_add + 1} "
                                                    f"{0} "
                                                    f"{(x0 + start_x):.3f} "
                                                    f"{(y0 + start_y):.3f} "
                                                    f"{(z0 + start_z):.3f} "
                                                    f"{0} "
                                                    f"{-1}\n")
                                            swc_lines_all.append(line)
                                            swc_add += 1

                    if (count + 1) % 1 == 0:
                        userTime = time.time() - sTime
                        surplusTime = userTime / (count + 1) * (workLen - count - 1)
                        logInfo = '[Total progress %.2f%%] [Stitching progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                            total_progress, (count + 1) / workLen * 100, userTime, surplusTime)
                        self.progress_text.emit(logInfo, max(i, count))
                    count += 1

        # 一次性写入文件
        with open(savePath, "w") as f0:
            f0.writelines(swc_lines_all)

        del swc_lines_all

    @classmethod
    def is_tif_empty(cls, tif_path: str, fast_mode: bool = True) -> Tuple[bool, str]:
        """
        🏆 Ultimate empty detection function combining scheme 1 + scheme 2
        4-level judgment strategy: file size → sampling detection → sampling verification → memmap final confirmation

        Args:
            tif_path: Path to the tif file
            fast_mode: Fast mode, returns immediately if any non-empty sample is found

        Returns:
            (is_empty, judgment reason)
        """
        # ---------- Level 1: File size pre‑judgment (zero cost, 0ms)
        try:
            file_size = os.path.getsize(tif_path)
            if file_size < 1024 * 50:  # < 50KB, consider empty
                return True, f"File size is only {file_size} bytes, deemed empty"
        except:
            return True, "File cannot be accessed"

        # ---------- Level 2: Key layer sampling detection (5-50ms, intercepts 99% of empty files)
        try:
            with tifffile.TiffFile(tif_path) as tif:
                total_pages = len(tif.pages)
                if total_pages == 0:
                    return True, "TIF has no valid data pages"

                # Sampling strategy: first layer + middle layer + last layer
                sample_indices = list({0, total_pages // 2, total_pages - 1})
                for idx in sample_indices:
                    if idx >= total_pages:
                        continue
                    layer = tif.pages[idx].asarray()
                    if layer.max() > 0:
                        if fast_mode:
                            return False, f"Non‑zero pixel found in layer {idx}, terminating detection"

                # ---------- Level 3: Expanded sampling verification (for suspicious files)
                verify_count = min(total_pages // 10, 20)
                step = max(1, total_pages // verify_count)
                for i in range(verify_count):
                    idx = i * step
                    layer = tif.pages[idx].asarray()
                    if layer.max() > 0:
                        if fast_mode:
                            return False, f"Non‑zero pixel found in layer {idx}, terminating detection"
        except Exception as e:
            pass

        # ---------- Level 4: memmap full confirmation (1-3s, only <1% of files reach this step)
        try:
            with tifffile.TiffFile(tif_path, mode='r+') as tif:
                img = tif.asarray(out='memmap')
                max_val = img.max()
                if max_val == 0:
                    return True, f"Full scan maximum value is {max_val}, confirmed empty data"
                else:
                    return False, f"Non‑zero pixel found in full scan, maximum value = {max_val}"
        except Exception as e:
            return True, "Read failed, deemed empty"


def get_bvInfoQue(InfoQueue, finishQue, errorQue):
    while True:
        try:
            task = InfoQueue.get(timeout=2)  # 设置超时避免阻塞
            if len(task) == 0:
                finishQue.put(-1)
                return
            cut_swc_path, cut_seg_path = task
            seg = tifffile.imread(cut_seg_path)
            mask_to_skeleton_swc(seg, cut_swc_path)
            # mask_2d = swc_to_mask(seg, savePath)
            finishQue.put(1)
            # print(bvInfoPath)
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


def mask_to_skeleton_swc(mask, savePath, th=123):
    with open(savePath, "w") as f:
        f.write("")
    thre = 5000
    mask2 = mask.copy()
    mask2[mask2 <= th] = 0
    mask2[mask2 > 0] = 255
    labels_out, N = cc3d.connected_components(mask2, connectivity=26, return_N=True, out_dtype=np.uint32)

    flag = 1
    if N != 0:
        out1 = speed_cc3d.group_type(labels_out)
        # out1Ls = len(out1)
        # out1Ls_step = 1
        center_point = []
        for i, key in enumerate(out1):
            # if i % out1Ls_step == 0:
            #     ProcessQueue.put((f"{((i + 1) / out1Ls):.2f}%", i))
            # print('进度：%f' % (i / len(out1) * 100) + '%')
            point = np.array(out1[key], dtype=np.int32)
            # if len(point) <= 297 or len(point) >= thre:
            #     continue
            dist = distance.cdist(point, point)
            dist_max = np.max(dist)
            rhos = []
            for j in range(len(point)):
                rhos.append(mask[point[j, 2], point[j, 1], point[j, 0]])
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

        # 去除边界
        swc_list = []
        c = 1
        for data in center_point:
            x, y, z = data[2: 5]
            if is_inside(x, y, z, shapes=mask.shape):
                swc_list.append([c, 0, x, y, z, 0, -1])
                c += 1
        if len(swc_list) != 0:
            swc_list = np.array(swc_list)
            np.savetxt(savePath, swc_list, fmt='%d %d %.2f %.2f %.2f %.4f %d')

        # if len(center_point) != 0:
        #     center_point = np.array(center_point)
        #     np.savetxt(savePath, center_point)

        # else:
        #     temp_p = np.array([0, 0, 0, 0, 0, 0, -1]).reshape(1, -1)
        #     np.savetxt(savePath, temp_p)


def is_inside(nx, ny, nz, rd_x=4, rd_y=4, rd_z=2, shapes=(144, 272, 272)):
    """判断坐标是否在内部有效范围内（0-based 索引）"""
    mz, my, mx = shapes
    return (rd_x <= nx <= mx - rd_x and
            rd_y <= ny <= my - rd_y and
            rd_z <= nz <= mz - rd_z)


def swc_to_mask(seg2, swc_path):
    # seg2: # zyx
    shapes = np.array(seg2.shape)[::-1]  # xyz

    kernelLen = 5
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
    dfImg2_2d = MaxProject(dfImg2)
    return dfImg2_2d


def GetPcKernelPc(pc, kernelArr, imgShape):
    curPc = (pc[:, None] + kernelArr[None]).reshape([-1, 3])
    curPc = np.round(curPc).astype(np.int32)
    curPc[curPc < 0] = 0
    curPc[curPc[:, 2] > imgShape[2] - 1, 2] = imgShape[2] - 1
    curPc[curPc[:, 1] > imgShape[1] - 1, 1] = imgShape[1] - 1
    curPc[curPc[:, 0] > imgShape[0] - 1, 0] = imgShape[0] - 1
    curPc = np.unique(curPc, axis=0)
    return curPc


def MaxProject(img, is_enhance=0):
    img = np.max(img, axis=0)
    img = normalize_and_scale_to_uint8(img)
    if is_enhance:
        img = enhance_contrast_histogram_equalization(img)
    return img


def enhance_contrast_histogram_equalization(img):
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


if __name__ == '__main__':
    # root = r"D:\SY\SAM2\data\cell_data\images"
    # dir_2d = r"D:\SY\SAM2\data\cell_data\images_enhance_2d"
    # os.makedirs(dir_2d, exist_ok=True)
    # names = os.listdir(root)
    # for name in names:
    #     tif_path = os.path.join(root, name)
    #     img = tifffile.imread(tif_path)
    #     img_2d = MaxProject(img, is_enhance=1)
    #
    #     img_2d_path = os.path.join(dir_2d, name)
    #     tifffile.imwrite(img_2d_path, img_2d)

    root = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\TrainDataSet\nnUnet_mask_cr_dist"
    swc_dir = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\TrainDataSet\nnUnet_mask_cr_dist_swc"
    os.makedirs(swc_dir, exist_ok=True)
    names = os.listdir(root)
    for name in names:
        path = os.path.join(root, name)
        mask = tifffile.imread(path)
        swc_path = os.path.join(swc_dir, name.split('.')[0] + ".swc")
        # Unet: 123, nnUnet: 33
        mask_to_skeleton_swc(mask, swc_path, th=123)
