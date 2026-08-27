# -*- coding: utf-8 -*-
import os
import sys
import time
import traceback
import json
import shutil
from multiprocessing import Process, Queue
import numpy as np
from os.path import join
from pathlib import Path
import tifffile
from PyQt5.QtCore import QThread, pyqtSignal
from DataStatistics.vessel_radius.segment_to_swc_optimized import segments_to_swc


def SplitSwcData(swcData):
    swcData = np.atleast_2d(swcData)
    if swcData.size == 0 or swcData.shape[1] < 7:
        return []
    indLs = np.where(swcData[:, -1] == -1)[0].tolist() + [swcData.shape[0]]
    swcDataLs = []
    for i in range(len(indLs) - 1):
        data = swcData[indLs[i]: indLs[i + 1]]
        sp = data[0, 0]
        data[:, 0] -= sp - 1
        data[1:, -1] -= sp - 1
        swcDataLs.append(data)
    return swcDataLs


"""Line/Neural"""


class ImgSpliceQThread(QThread):
    finish0 = pyqtSignal(str)
    progress0 = pyqtSignal(str, int)
    error0 = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(ImgSpliceQThread, self).__init__()
        self.win = kwargs.get('win')
        self.logger = self.win.logger

    def run(self):
        try:
            self.is_error = 0
            self.new_text = "Splicing completed"
            cut_infos = self.win.img_splice_dict["cut_infos"]
            cfg_path = self.win.img_splice_dict["cfg_path"]
            save_dir = self.win.img_splice_dict["save_dir"]
            dataType = cut_infos["dataType"]
            save_dir = join(save_dir, "SpliceResults")
            os.makedirs(save_dir, exist_ok=True)
            if dataType == "BV":
                CutWorkFilesDir = cut_infos["CutWorkFilesDir"]
                BvSliceNumberXYZ = cut_infos["BvSliceNumberXYZ"]
                BvRedunXYZ = cut_infos.get("BvRedunXYZ", [0, 0, 0])
                BvBigSizeXYZ = cut_infos["BvBigSizeXYZ"]
                BvSmallSizeXYZ = cut_infos["BvSmallSizeXYZ"]
                ROI = cut_infos.get("ROI", None)
                cfg_level = cut_infos.get("cfg_level", 0)
                self.bv_swc_splice_sub(CutWorkFilesDir=CutWorkFilesDir,
                                       BvSliceNumberXYZ=BvSliceNumberXYZ,
                                       BvRedunXYZ=BvRedunXYZ,
                                       BvBigSizeXYZ=BvBigSizeXYZ,
                                       BvSmallSizeXYZ=BvSmallSizeXYZ,
                                       save_dir=save_dir,
                                       cfg_level=cfg_level,
                                       ROI=ROI)
            if dataType == "OME-Zarr":
                CutWorkFilesDir = cut_infos["CutWorkFilesDir"]
                BvSliceNumberXYZ = cut_infos["OME-ZarrSliceNumberXYZ"]
                BvRedunXYZ = cut_infos.get("OME-ZarrRedunXYZ", [0, 0, 0])
                BvBigSizeXYZ = cut_infos["OME-ZarrBigSizeXYZ"]
                BvSmallSizeXYZ = cut_infos["OME-ZarrSmallSizeXYZ"]
                cfg_level = cut_infos.get("cfg_level", 0)
                self.bv_swc_splice_sub(CutWorkFilesDir=CutWorkFilesDir,
                                       BvSliceNumberXYZ=BvSliceNumberXYZ,
                                       BvRedunXYZ=BvRedunXYZ,
                                       BvBigSizeXYZ=BvBigSizeXYZ,
                                       BvSmallSizeXYZ=BvSmallSizeXYZ,
                                       save_dir=save_dir,
                                       cfg_level=cfg_level,
                                       dataType="OME-Zarr")
            if dataType == "TIF":
                bigSizeXYZ = cut_infos["bigSizeXYZ"]
                imgSizeXYZ = cut_infos["imgSizeXYZ"]
                rXYZ = cut_infos["rXYZ"]
                sliceNumberXYZ = cut_infos["sliceNumberXYZ"]
                cut_img_dir = cut_infos["cut_img_dir"]
                cut_swc_dir = cut_infos["cut_swc_dir"]
                cut_seg_dir = cut_infos.get("cut_seg_dir", None)

                stem = str(Path(cfg_path).stem)
                swc_save_dir = join(save_dir, "swc")
                img_save_dir = join(save_dir, "images")
                seg_save_dir = join(save_dir, "segmentations")
                os.makedirs(swc_save_dir, exist_ok=True)
                os.makedirs(img_save_dir, exist_ok=True)
                os.makedirs(seg_save_dir, exist_ok=True)
                save_dict = {
                    "swc_save_path": join(swc_save_dir, f"{stem}.swc"),
                    "img_save_path": join(img_save_dir, f"{stem}.tif"),
                    "seg_save_path": join(seg_save_dir, f"{stem}.tif")
                }

                workLen = 1
                for s in sliceNumberXYZ:
                    workLen *= s
                self.ImgSplice(bigSize=bigSizeXYZ,
                               imgSize=imgSizeXYZ,
                               r=rXYZ,
                               sliceNumber=sliceNumberXYZ,
                               workLen=workLen,
                               save_dict=save_dict,
                               cut_swc_dir=cut_swc_dir,
                               cut_img_dir=cut_img_dir,
                               cut_seg_dir=cut_seg_dir,
                               total_progress=100,
                               i=0)
            time.sleep(1)
            if not self.is_error:
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

    def ImgSplice(self,
                  bigSize=None,
                  imgSize=None,
                  r=None,
                  sliceNumber=None,
                  workLen=None,
                  save_dict=None,
                  cut_swc_dir=None,
                  cut_img_dir=None,
                  cut_seg_dir=None,
                  total_progress=100,
                  i=0):
        block_size_x, block_size_y, block_size_z = imgSize
        br_x = block_size_x - r[0]
        br_y = block_size_y - r[1]
        br_z = block_size_z - r[2]

        big_img = None
        big_array = None
        img_save_path = save_dict.get("img_save_path")
        swc_save_path = save_dict.get("swc_save_path")
        seg_save_path = save_dict.get("seg_save_path")

        Max_x, Max_y, Max_z = bigSize

        count = 0
        swc_add = 0
        swc_add_ = 0
        sTime = time.time()
        swc_lines_all = []

        # Iterate over all images and splice into result array
        for nz in range(sliceNumber[2]):  # Process Z axis
            for ny in range(sliceNumber[1]):  # Process Y axis
                for nx in range(sliceNumber[0]):  # Process X axis
                    swc_name = f"{str(nz).zfill(4)}_{str(ny).zfill(2)}_{str(nx).zfill(2)}.swc"
                    swc_path = join(cut_swc_dir, swc_name)

                    if cut_seg_dir is not None:
                        tif_name = f"{str(nz).zfill(4)}_{str(ny).zfill(2)}_{str(nx).zfill(2)}.tif"
                        img_path = join(cut_img_dir, tif_name)
                        seg_path = join(cut_seg_dir, tif_name)

                    # Check if swc_path exists
                    if not os.path.isfile(swc_path):
                        print(f"{swc_path} does not exist")
                        userTime = time.time() - sTime
                        surplusTime = userTime / (count + 1) * (workLen - count - 1)
                        logInfo = '[Total progress %.2f%%] [Splicing progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                            total_progress, (count + 1) / workLen * 100, userTime, surplusTime)
                        self.progress0.emit(logInfo, max(i, count))
                        count += 1
                        continue

                    is_surpass = False
                    block_size_x, block_size_y, block_size_z = imgSize

                    if cut_seg_dir is not None:
                        # 读取小块的 TIF 分割结果
                        try:
                            block_data = tifffile.imread(seg_path)
                            img = tifffile.imread(img_path)

                            if big_array is None:
                                # 预分配大图数组，初始化为 0 (背景)
                                big_array = np.zeros((bigSize[2], bigSize[1], bigSize[0]), dtype=np.uint8)
                                big_img = np.zeros((bigSize[2], bigSize[1], bigSize[0]), dtype=img.dtype)

                            block_shapes = block_data.shape
                            block_size_x = min(block_size_x, block_shapes[2])
                            block_size_y = min(block_size_y, block_shapes[1])
                            block_size_z = min(block_size_z, block_shapes[0])
                        except Exception:
                            print(f"Failed to read {seg_path}, skipping.")
                            continue

                    # Calculate start index of the block (edge blocks take block_size from the end)
                    start_z = min(nz * br_z, bigSize[2] - block_size_z)
                    start_y = min(ny * br_y, bigSize[1] - block_size_y)
                    start_x = min(nx * br_x, bigSize[0] - block_size_x)

                    # Calculate end index
                    end_z = start_z + block_size_z
                    end_y = start_y + block_size_y
                    end_x = start_x + block_size_x

                    # Large image size insufficient
                    if bigSize[2] - block_size_z < 0:
                        start_z = 0
                        end_z = bigSize[2]
                        block_size_z = end_z
                        is_surpass = True
                    # Edge block size insufficient
                    elif start_z == bigSize[2] - block_size_z and nz * br_z != bigSize[2] - block_size_z:  # Edge redundancy
                        # Subtract redundancy from the end index of the second last block as the start index of the last block to avoid overlapping tracking results
                        start_z = (nz - 1) * br_z + block_size_z - r[2]
                        end_z = bigSize[2]
                        is_surpass = True

                    if bigSize[1] - block_size_y < 0:  # Large image edge block size insufficient, large image size insufficient
                        start_y = 0
                        end_y = bigSize[1]
                        block_size_y = end_y
                        is_surpass = True
                    elif start_y == bigSize[1] - block_size_y and ny * br_y != bigSize[1] - block_size_y:  # Edge redundancy
                        start_y = (ny - 1) * br_y + block_size_y - r[1]
                        end_y = bigSize[1]
                        is_surpass = True

                    if bigSize[0] - block_size_x < 0:  # Large image insufficient to fill the generated small image
                        start_x = 0
                        end_x = bigSize[0]
                        block_size_x = end_x
                        is_surpass = True
                    elif start_x == bigSize[0] - block_size_x and nx * br_x != bigSize[0] - block_size_x:  # Edge redundancy
                        start_x = (nx - 1) * br_x + block_size_x - r[0]
                        end_x = bigSize[0]
                        is_surpass = True

                    if is_surpass:  # Exceeds original image
                        dx = end_x - start_x
                        dy = end_y - start_y
                        dz = end_z - start_z
                        xmin, xmax, ymin, ymax, zmin, zmax = (block_size_x - dx, block_size_x,
                                                              block_size_y - dy, block_size_y,
                                                              block_size_z - dz, block_size_z)
                        if cut_seg_dir is not None:
                            # 将小块数据写入大图对应位置
                            big_array[start_z:end_z, start_y:end_y, start_x:end_x] = block_data[
                                                                                     zmin: zmax,
                                                                                     ymin: ymax,
                                                                                     xmin: xmax]
                            big_img[start_z:end_z, start_y:end_y, start_x:end_x] = img[
                                                                                   zmin: zmax,
                                                                                   ymin: ymax,
                                                                                   xmin: xmax]
                        if os.path.getsize(swc_path):
                            swcData = np.loadtxt(swc_path, ndmin=2)
                            if len(swcData):
                                if len(swcData[0]) == 7:
                                    swcDataLs = SplitSwcData(swcData)

                                    xmax -= 2
                                    ymax -= 2
                                    zmax -= 2
                                    cut_center = np.array([xmax + xmin, ymax + ymin, zmax + zmin]) / 2  # Center
                                    cut_v = np.array([xmax - xmin, ymax - ymin, zmax - zmin]) / 2  # Direction vector
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
                                        v = np.array([x0_max - x0_min, y0_max - y0_min, z0_max - z0_min]) / 2  # Direction vector
                                        cr = np.linalg.norm(v)
                                        dist = np.linalg.norm(cut_center - center)  # Distance between centers
                                        if dist < cr + cut_r:
                                            for ii, item in enumerate(swcData):
                                                # Radius
                                                r_dict[tuple(item[2: 5])] = item[5]

                                                if item[-1] == -1:
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
                        if os.path.getsize(swc_path):
                            swcData = np.loadtxt(swc_path, ndmin=2)
                            if len(swcData):
                                if len(swcData[0]) == 7:
                                    swcDataLs = SplitSwcData(swcData)
                                    for swcData in swcDataLs:
                                        for ii, item in enumerate(swcData):
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
                        if cut_seg_dir is not None:
                            # 将小块数据写入大图对应位置
                            big_array[start_z:end_z, start_y:end_y, start_x:end_x] = block_data
                            big_img[start_z:end_z, start_y:end_y, start_x:end_x] = img
                    Max_x, Max_y, Max_z = end_x, end_y, end_z
                    swc_add_ = swc_add

                    if (count + 1) % 1 == 0:
                        userTime = time.time() - sTime
                        surplusTime = userTime / (count + 1) * (workLen - count - 1)
                        logInfo = '[Total progress %.2f%%] [Splicing progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                            total_progress, (count + 1) / workLen * 100, userTime, surplusTime)
                        self.progress0.emit(logInfo, max(i, count))
                    count += 1

        with open(swc_save_path, "w") as f0:
            f0.writelines(swc_lines_all)

        del swc_lines_all

        if cut_seg_dir is not None:
            big_array = big_array[:Max_z, :Max_y, :Max_x]
            big_img = big_img[:Max_z, :Max_y, :Max_x]
            # 一次性写入大图
            tifffile.imwrite(seg_save_path, big_array, compression="lzw")
            del big_array
            tifffile.imwrite(img_save_path, big_img, compression="lzw")
            del big_img

    """Whole image splicing"""

    def ImgSplice_Whole(self,
                        bigSize=None,
                        imgSize=None,
                        r=None,
                        sliceNumber=None,
                        bv_npy_path=None,
                        swc_add=None,
                        swc_add_=None,
                        lv=(1, 1, 1),
                        ROI=(0, 0, 0)):
        MinX, MinY, MinZ = ROI
        block_size_x, block_size_y, block_size_z = imgSize
        br_x = block_size_x - r[0]
        br_y = block_size_y - r[1]
        br_z = block_size_z - r[2]

        nx, ny, nz = sliceNumber

        # is_surpass = False

        # Calculate start index of the block (edge blocks take block_size from the end)
        start_z = min(nz * br_z, bigSize[2] - block_size_z)
        start_y = min(ny * br_y, bigSize[1] - block_size_y)
        start_x = min(nx * br_x, bigSize[0] - block_size_x)
        # Calculate end index
        # end_z = start_z + block_size_z
        # end_y = start_y + block_size_y
        # end_x = start_x + block_size_x

        if bigSize[2] - block_size_z < 0:  # Large image edge block size insufficient, large image size insufficient
            start_z = 0
        elif start_z == bigSize[2] - block_size_z and nz * br_z != bigSize[2] - block_size_z:  # Edge redundancy
            # Subtract redundancy from the end index of the second last block as the start index of the last block to avoid overlapping tracking results
            start_z = (nz - 1) * br_z + block_size_z - r[2]
            # end_z = bigSize[2]

        if bigSize[1] - block_size_y < 0:  # Large image insufficient to fill the generated small image
            start_y = 0
        elif start_y == bigSize[1] - block_size_y and ny * br_y != bigSize[1] - block_size_y:  # Edge redundancy
            start_y = (ny - 1) * br_y + block_size_y - r[1]
            # end_y = bigSize[1]

        if bigSize[0] - block_size_x < 0:  # Large image insufficient to fill the generated small image
            start_x = 0
        elif start_x == bigSize[0] - block_size_x and nx * br_x != bigSize[0] - block_size_x:  # Edge redundancy
            start_x = (nx - 1) * br_x + block_size_x - r[0]
            # end_x = bigSize[0]

        npy_data = np.load(bv_npy_path, mmap_mode='r')  # millisecond level
        # 注意：原来用 np.loadtxt(swc_path, ndmin=2) 时有 ndmin=2 参数保护，所以不会有这个问题，但换成npy加载后漏掉了这个保护。
        swcData = np.atleast_2d(np.array(npy_data))
        # swcData = np.loadtxt(bv_swc_path, ndmin=2)
        if len(swcData):
            if len(swcData[0]) == 7:
                swcDataLs = SplitSwcData(swcData)
                for swcData in swcDataLs:
                    for ii, item in enumerate(swcData):
                        if item[0] - 1 != item[-1]:
                            if int(item[-1]) == -1:
                                self.swcLines.append(
                                    f"{swc_add + 1} "
                                    f"{item[1]} "
                                    f"{(item[2] * lv[0] + start_x + MinX):.3f} "
                                    f"{(item[3] * lv[1] + start_y + MinY):.3f} "
                                    f"{(item[4] * lv[2] + start_z + MinZ):.3f} "
                                    f"{item[5]} -1\n")
                            else:
                                self.swcLines.append(
                                    f"{swc_add + 1} "
                                    f"{item[1]} "
                                    f"{(item[2] * lv[0] + start_x + MinX):.3f} "
                                    f"{(item[3] * lv[1] + start_y + MinY):.3f} "
                                    f"{(item[4] * lv[2] + start_z + MinZ):.3f} "
                                    f"{item[5]} "
                                    f"{item[6] + swc_add_}\n")
                        else:
                            self.swcLines.append(
                                f"{swc_add + 1} "
                                f"{item[1]} "
                                f"{(item[2] * lv[0] + start_x + MinX):.3f} "
                                f"{(item[3] * lv[1] + start_y + MinY):.3f} "
                                f"{(item[4] * lv[2] + start_z + MinZ):.3f} "
                                f"{item[5]} "
                                f"{swc_add}\n")
                        swc_add += 1
                    swc_add_ += len(swcData)

        swc_add_ = swc_add
        return swc_add, swc_add_

    def bv_swc_splice_sub(self,
                          CutWorkFilesDir,
                          BvSliceNumberXYZ,
                          BvRedunXYZ,
                          BvBigSizeXYZ,
                          BvSmallSizeXYZ,
                          save_dir,
                          cfg_level,
                          dataType=None,
                          ROI=None):
        lv = 2 ** cfg_level
        print(f"Downsampling ratio: {lv:.3f}x")
        if dataType:
            lv = [lv, lv, 1]
        else:
            lv = [lv, lv, lv]
            
        if ROI is not None:
            ROI = [ROI[0], ROI[2], ROI[4]]
        else:
            ROI = [0, 0, 0]

        splice_work_dir = join(save_dir, "splice_work_dir")
        if os.path.exists(splice_work_dir):
            shutil.rmtree(splice_work_dir, ignore_errors=True)
        os.makedirs(splice_work_dir, exist_ok=True)
        npy_work_dir = join(save_dir, "npy_work_dir")
        if os.path.exists(npy_work_dir):
            shutil.rmtree(npy_work_dir, ignore_errors=True)
        os.makedirs(npy_work_dir, exist_ok=True)

        workLsQue = Queue()
        finishQue = Queue()
        errorQue = Queue()

        for nx in range(BvSliceNumberXYZ[0]):
            for ny in range(BvSliceNumberXYZ[1]):
                for nz in range(BvSliceNumberXYZ[2]):
                    stem = f"{nx}_{ny}"
                    bv_swc_name = stem + ".swc"
                    bv_npy_name = stem + ".npy"
                    bv_cut_info_name = stem + ".json"
                    bv_cut_info_path = join(CutWorkFilesDir, stem, bv_cut_info_name)
                    bv_swc_path = join(splice_work_dir, bv_swc_name)
                    bv_npy_path = join(npy_work_dir, bv_npy_name)

                    if not os.path.exists(bv_cut_info_path):
                        print(f"{bv_cut_info_path} Path does not exist!")
                        self.is_error = 1
                        self.error0.emit(f"{bv_cut_info_path} Path does not exist!")
                        return

                    workLsQue.put((bv_cut_info_path,
                                   bv_swc_path,
                                   bv_npy_path))

        file_count = workLsQue.qsize()

        MNumber = int(min(os.cpu_count(), file_count) / 2) + 1

        for i in range(MNumber):
            workLsQue.put(())

        # workLen = workLsQue.qsize()
        original_work_len = file_count

        self.progress0.emit("Start splicing", 0)
        ps = []
        for i in range(MNumber):
            ps.append(Process(target=bv_swc_splice, args=(workLsQue, finishQue, errorQue)))
            ps[-1].daemon = True
            ps[-1].start()

        start_time = time.time()
        curMakeSize = 0
        finished_processes = 0
        is_stop_count = 0

        while True:
            # Check if all processes have finished
            while finishQue.qsize() > 0:
                result = finishQue.get()
                if result == -1:
                    finished_processes += 1
                elif result == 1:
                    curMakeSize += 1

            # All actual work tasks completed
            if curMakeSize >= original_work_len:
                break

            # Process hang detection
            if is_stop_count == 120:  # Approximately one minute without update
                print("Starting process hang detection")
                # Detect files not processed in the connection directory
                files_names = []
                for nx in range(BvSliceNumberXYZ[0]):
                    for ny in range(BvSliceNumberXYZ[1]):
                        for nz in range(BvSliceNumberXYZ[2]):
                            stem = f"{nx}_{ny}"
                            files_names.append(stem)
                names = [Path(n).stem for n in os.listdir(splice_work_dir) if ".swc" in n]
                if len(names) == len(files_names):
                    break
                else:
                    add_list = []
                    s_names = set(names)
                    for files_name in files_names:
                        if files_name in s_names:
                            continue
                        else:
                            add_list.append(files_name)
                    if len(add_list) > 0:
                        for files_name in add_list:
                            print(f"{curMakeSize + 1} / {original_work_len}")
                            stem = files_name
                            bv_swc_name = stem + ".swc"
                            bv_npy_name = stem + ".npy"
                            bv_cut_info_name = stem + ".json"
                            bv_CutWorkFilesDir = join(CutWorkFilesDir, stem)
                            bv_cut_info_path = join(bv_CutWorkFilesDir, bv_cut_info_name)
                            bv_swc_path = join(splice_work_dir, bv_swc_name)
                            bv_npy_path = join(npy_work_dir, bv_npy_name)

                            with open(bv_cut_info_path) as f:
                                bv_cut_info = json.loads(f.read())

                            bigSizeXYZ = bv_cut_info["bigSizeXYZ"]
                            imgSizeXYZ = bv_cut_info["imgSizeXYZ"]
                            rXYZ = bv_cut_info["rXYZ"]
                            sliceNumberXYZ = bv_cut_info["sliceNumberXYZ"]
                            cut_swc_dir = bv_cut_info["cut_swc_dir"]

                            swc_splice_process(bigSize=bigSizeXYZ,
                                               imgSize=imgSizeXYZ,
                                               r=rXYZ,
                                               sliceNumber=sliceNumberXYZ,
                                               savePath=bv_swc_path,
                                               npy_path=bv_npy_path,
                                               cut_swc_dir=cut_swc_dir)

                            curMakeSize += 1

            # Process error
            if errorQue.qsize():
                e = errorQue.get()
                for p in ps:
                    p.terminate()
                self.error_logger(e)
                self.is_error = 1
                self.error0.emit(str(e))
                return

            # Update progress
            if curMakeSize > 0:
                userTime = time.time() - start_time
                surplusTime = userTime / curMakeSize * (original_work_len - curMakeSize)
                logInfo = f'[Splicing progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                    curMakeSize / original_work_len * 100, userTime, surplusTime)
                self.progress0.emit(logInfo, 1)

            is_stop_count += 1
            time.sleep(0.5)

        # Wait for all processes to finish
        for p in ps:
            p.join(timeout=1)  # Wait at most 1 second
            if p.is_alive():
                p.terminate()

        userTime = time.time() - start_time
        logInfo = f'[Splicing progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
            100, userTime, 0)
        self.progress0.emit(logInfo, 1)

        time.sleep(2)
        save_path = join(save_dir, "splice.swc")

        i = 0
        swc_add = 0
        swc_add_ = 0

        self.swcLines = []

        workLen = 1
        for s in BvSliceNumberXYZ:
            workLen *= s

        start_time = time.time()
        for nx in range(BvSliceNumberXYZ[0]):
            for ny in range(BvSliceNumberXYZ[1]):
                for nz in range(BvSliceNumberXYZ[2]):
                    stem = f"{nx}_{ny}"
                    bv_npy_name = stem + ".npy"

                    bv_npy_path = join(npy_work_dir, bv_npy_name)

                    swc_add, swc_add_ = self.ImgSplice_Whole(bigSize=BvBigSizeXYZ,
                                                             imgSize=BvSmallSizeXYZ,
                                                             r=BvRedunXYZ,
                                                             sliceNumber=[nx, ny, nz],
                                                             bv_npy_path=bv_npy_path,
                                                             swc_add=swc_add,
                                                             swc_add_=swc_add_,
                                                             lv=lv,
                                                             ROI=ROI)

                    userTime = time.time() - start_time
                    surplusTime = userTime / (i + 1) * (workLen - i - 1)
                    logInfo = f'[End process progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        (i + 1) / workLen * 100, userTime, surplusTime)
                    self.progress0.emit(logInfo, i)
                    i += 1

        self.progress0.emit("Writing data to file", 0)

        # Write to file at once
        with open(save_path, "w") as f0:
            f0.writelines(self.swcLines)

        del self.swcLines

        if os.path.exists(npy_work_dir):
            shutil.rmtree(npy_work_dir, ignore_errors=True)


def bv_swc_splice(workLsQue, finishQue, errorQue):
    while True:
        try:
            add = workLsQue.get(timeout=2)
            if len(add) == 0:
                finishQue.put(-1)
                return

            (bv_cut_info_path,
             bv_swc_path,
             bv_npy_path) = add

            with open(bv_cut_info_path) as f:
                bv_cut_info = json.loads(f.read())

            bigSizeXYZ = bv_cut_info["bigSizeXYZ"]
            imgSizeXYZ = bv_cut_info["imgSizeXYZ"]
            rXYZ = bv_cut_info["rXYZ"]
            sliceNumberXYZ = bv_cut_info["sliceNumberXYZ"]
            cut_swc_dir = bv_cut_info["cut_swc_dir"]

            swc_splice_process(bigSize=bigSizeXYZ,
                               imgSize=imgSizeXYZ,
                               r=rXYZ,
                               sliceNumber=sliceNumberXYZ,
                               savePath=bv_swc_path,
                               npy_path=bv_npy_path,
                               cut_swc_dir=cut_swc_dir)
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


def swc_splice_process(bigSize=None,
                       imgSize=None,
                       r=None,
                       sliceNumber=None,
                       savePath=None,
                       npy_path=None,
                       cut_swc_dir=None):
    swc_lines_all = []
    npy_lines_all = []

    block_size_x, block_size_y, block_size_z = imgSize
    br_x = block_size_x - r[0]
    br_y = block_size_y - r[1]
    br_z = block_size_z - r[2]

    swc_add = 0
    swc_add_ = 0

    # Iterate over all images and splice into result array
    for nz in range(sliceNumber[2]):  # Process Z axis
        for ny in range(sliceNumber[1]):  # Process Y axis
            for nx in range(sliceNumber[0]):  # Process X axis
                swc_name = f"{str(nz).zfill(4)}_{str(ny).zfill(2)}_{str(nx).zfill(2)}.swc"
                swc_path = join(cut_swc_dir, swc_name)

                # If swc_path does not exist
                if not os.path.isfile(swc_path):
                    print(f"{swc_path} does not exist")
                    continue

                # If swc_path is empty
                if not os.path.getsize(swc_path):
                    continue

                is_surpass = False
                block_size_x, block_size_y, block_size_z = imgSize

                # Calculate start index of the block (edge blocks take block_size from the end)
                start_z = min(nz * br_z, bigSize[2] - block_size_z)
                start_y = min(ny * br_y, bigSize[1] - block_size_y)
                start_x = min(nx * br_x, bigSize[0] - block_size_x)

                # Calculate end index
                end_z = start_z + block_size_z
                end_y = start_y + block_size_y
                end_x = start_x + block_size_x

                # Large image size insufficient
                if bigSize[2] - block_size_z < 0:
                    start_z = 0
                    end_z = bigSize[2]
                    block_size_z = end_z
                    is_surpass = True
                # Edge block size insufficient
                elif start_z == bigSize[2] - block_size_z and nz * br_z != bigSize[2] - block_size_z:  # Edge redundancy
                    # Subtract redundancy from the end index of the second last block as the start index of the last block to avoid overlapping tracking results
                    start_z = (nz - 1) * br_z + block_size_z - r[2]
                    end_z = bigSize[2]
                    is_surpass = True

                if bigSize[1] - block_size_y < 0:  # Large image edge block size insufficient, large image size insufficient
                    start_y = 0
                    end_y = bigSize[1]
                    block_size_y = end_y
                    is_surpass = True
                elif start_y == bigSize[1] - block_size_y and ny * br_y != bigSize[1] - block_size_y:  # Edge redundancy
                    start_y = (ny - 1) * br_y + block_size_y - r[1]
                    end_y = bigSize[1]
                    is_surpass = True

                if bigSize[0] - block_size_x < 0:  # Large image insufficient to fill the generated small image
                    start_x = 0
                    end_x = bigSize[0]
                    block_size_x = end_x
                    is_surpass = True
                elif start_x == bigSize[0] - block_size_x and nx * br_x != bigSize[0] - block_size_x:  # Edge redundancy
                    start_x = (nx - 1) * br_x + block_size_x - r[0]
                    end_x = bigSize[0]
                    is_surpass = True

                if is_surpass:  # Exceeds original image
                    swcData = np.loadtxt(swc_path, ndmin=2)
                    if len(swcData):
                        if len(swcData[0]) == 7:
                            swcDataLs = SplitSwcData(swcData)
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
                            cut_v = np.array([xmax - xmin, ymax - ymin, zmax - zmin]) / 2  # Direction vector
                            cut_r = np.linalg.norm(cut_v)

                            line_list = []
                            r_dict = {}

                            for swcData in swcDataLs:
                                if len(swcData) < 30:
                                    continue
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
                                v = np.array([x0_max - x0_min, y0_max - y0_min, z0_max - z0_min]) / 2  # Direction vector
                                cr = np.linalg.norm(v)
                                dist = np.linalg.norm(cut_center - center)  # Distance between centers
                                if dist < cr + cut_r:
                                    for ii, item in enumerate(swcData):
                                        # Radius
                                        r_dict[tuple(item[2: 5])] = item[5]
                                        if item[-1] == -1:
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
                                    np_line = [swc_add + 1,
                                               0,
                                               key[0] - xmin + start_x,
                                               key[1] - ymin + start_y,
                                               key[2] - zmin + start_z,
                                               rad,
                                               value
                                               ]
                                    line = (f"{swc_add + 1} "
                                            f"{0} "
                                            f"{(key[0] - xmin + start_x):.3f} "
                                            f"{(key[1] - ymin + start_y):.3f} "
                                            f"{(key[2] - zmin + start_z):.3f} "
                                            f"{rad} "
                                            f"{value}\n")
                                else:
                                    np_line = [swc_add + 1,
                                               0,
                                               key[0] - xmin + start_x,
                                               key[1] - ymin + start_y,
                                               key[2] - zmin + start_z,
                                               rad,
                                               value + swc_add_
                                               ]
                                    line = (f"{swc_add + 1} "
                                            f"{0} "
                                            f"{(key[0] - xmin + start_x):.3f} "
                                            f"{(key[1] - ymin + start_y):.3f} "
                                            f"{(key[2] - zmin + start_z):.3f} "
                                            f"{rad} "
                                            f"{value + swc_add_}\n")
                                swc_lines_all.append(line)
                                npy_lines_all.append(np.array(np_line))
                                swc_add += 1
                else:
                    swcData = np.loadtxt(swc_path, ndmin=2)
                    if len(swcData):
                        if len(swcData[0]) == 7:
                            swcDataLs = SplitSwcData(swcData)
                            for swcData in swcDataLs:
                                for ii, item in enumerate(swcData):
                                    if item[0] - 1 != item[-1]:
                                        if int(item[-1]) == -1:
                                            np_line = [swc_add + 1,
                                                       item[1],
                                                       item[2] + start_x,
                                                       item[3] + start_y,
                                                       item[4] + start_z,
                                                       item[5],
                                                       -1
                                                       ]
                                            line = (f"{swc_add + 1} "
                                                    f"{item[1]} "
                                                    f"{(item[2] + start_x):.3f} "
                                                    f"{(item[3] + start_y):.3f} "
                                                    f"{(item[4] + start_z):.3f} "
                                                    f"{item[5]} -1\n")
                                        else:
                                            np_line = [swc_add + 1,
                                                       item[1],
                                                       item[2] + start_x,
                                                       item[3] + start_y,
                                                       item[4] + start_z,
                                                       item[5],
                                                       item[6] + swc_add_
                                                       ]
                                            line = (f"{swc_add + 1} "
                                                    f"{item[1]} "
                                                    f"{(item[2] + start_x):.3f} "
                                                    f"{(item[3] + start_y):.3f} "
                                                    f"{(item[4] + start_z):.3f} "
                                                    f"{item[5]} "
                                                    f"{item[6] + swc_add_}\n")
                                    else:
                                        np_line = [swc_add + 1,
                                                   item[1],
                                                   item[2] + start_x,
                                                   item[3] + start_y,
                                                   item[4] + start_z,
                                                   item[5],
                                                   swc_add
                                                   ]
                                        line = (f"{swc_add + 1} "
                                                f"{item[1]} "
                                                f"{(item[2] + start_x):.3f} "
                                                f"{(item[3] + start_y):.3f} "
                                                f"{(item[4] + start_z):.3f} "
                                                f"{item[5]} "
                                                f"{swc_add}\n")
                                    swc_lines_all.append(line)
                                    npy_lines_all.append(np.array(np_line))
                                    swc_add += 1
                                swc_add_ += len(swcData)

                swc_add_ = swc_add

    with open(savePath, "w") as f0:
        f0.writelines(swc_lines_all)

    np.save(npy_path, np.array(npy_lines_all))

    del swc_lines_all
    del npy_lines_all
