# -*- coding: utf-8 -*-
import shutil
import numpy as np
import tifffile as tiff
from os.path import join
import sys
import traceback
import os
import time
from BVExample.BVMoudle import BVReader
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal


class CellImageCutThread(QThread):
    finish0 = pyqtSignal(str, int)
    progress0 = pyqtSignal(str, int)
    error0 = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(CellImageCutThread, self).__init__()
        self.win = kwargs.get('win')
        self.logger = self.win.logger
        self.readObj = BVReader()  # Object for reading BV format

    def run(self):
        try:
            making_arguments = self.win.making_arguments
            img_dir = making_arguments['img_dir']
            swc_dir = making_arguments['swc_dir']
            results_dir = making_arguments['results_dir']
            select_input_index = making_arguments['select_input_index']
            cut_results_dir = making_arguments['cut_results_dir']
            cut_img_dir = making_arguments['cut_img_dir']
            cut_swc_dir = making_arguments['cut_swc_dir']
            small_size = making_arguments['make_arguments']['small_size']  # xyz
            cut_redun_size = making_arguments['make_arguments']['cut_redun_size']  # xyz

            # Update results folder
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir, ignore_errors=True)
            os.makedirs(results_dir, exist_ok=True)

            is_error, cut_image_count = self.image_cut(img_dir, swc_dir, select_input_index, small_size, cut_redun_size,
                                                       cut_results_dir, cut_img_dir, cut_swc_dir)
            time.sleep(1)
            if is_error:
                return

            self.finish0.emit(f"Cutting finished!", cut_image_count)
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

    def image_cut(self, img_dir, swc_dir, select_input_index, small_size, cut_redun_size,
                  cut_results_dir, cut_img_dir, cut_swc_dir):
        is_error = False
        cut_image_count = 0
        # Image names
        img_names = [n for n in os.listdir(img_dir) if ".tif" in n or ".bv" in n]
        if len(img_names) == 0:
            self.error0.emit(f"{img_dir} has no 3D TIF or BV format files!")
            return True, cut_image_count
        if select_input_index == 1:
            if os.path.exists(swc_dir):
                swc_names = [n for n in os.listdir(swc_dir) if ".swc" in n]
                if len(swc_names) == 0:
                    self.error0.emit(f"{swc_dir} folder has no SWC format files!")
                    return True, cut_image_count
            os.makedirs(swc_dir, exist_ok=True)
        # Calculate related parameters
        r = np.array(cut_redun_size)  # Offset xyz
        small_size = np.array(small_size)  # Small image size xyz
        # Create folders
        os.makedirs(cut_results_dir, exist_ok=True)
        os.makedirs(cut_img_dir, exist_ok=True)
        os.makedirs(cut_swc_dir, exist_ok=True)

        if select_input_index == 0:  # Unlabeled dataset
            is_error, cut_image_count = self.image_only_cut(img_dir, cut_img_dir, img_names, r, small_size)

        elif select_input_index == 1:  # Labeled dataset
            is_error, cut_image_count = self.image_swc_cut(img_dir, swc_dir, cut_img_dir, cut_swc_dir, img_names,
                                                           r, small_size)
        elif select_input_index == 2:  # Large dataset format
            is_error = False

        return is_error, cut_image_count

    """Cut images only"""

    def image_only_cut(self, img_dir, cut_img_dir, img_names, r, small_size):
        start_time = time.time()
        total_len = len(img_names)
        image_count = 0
        cut_image_count = 0
        count = 0
        all_len = 0
        no_3d_count = 0
        for i, img_name in enumerate(img_names):
            img_path = join(str(img_dir), img_name)
            try:
                if str(Path(img_name).suffix) == ".bv":
                    img = self.readObj.readBV(img_path)
                else:
                    img = tiff.imread(img_path)  # Read image
                if len(img.shape) != 3:
                    no_3d_count += 1

                    userTime = time.time() - start_time
                    surplusTime = userTime / (count + 1) * (total_len - count - 1)
                    logInfo2 = '[Total cut progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        (i + 1) / total_len * 100, userTime, surplusTime)
                    self.progress0.emit(logInfo2, count)
                    count += 1
                    continue
            except Exception as e:
                continue
            # Determine image dimensions
            if len(img.shape) > 2:  # 3D image
                image_count += 1
                big_size = np.array(img.shape)[::-1]  # Big image size xyz
                if all(x >= y for x, y in zip(small_size, big_size)):
                    if img.any():
                        new_img_name = f"{str(i).zfill(4)}_{0}_{0}-{0}_{0}_{0}.tif"
                        cut_img_path = join(cut_img_dir, new_img_name)
                        shutil.copyfile(img_path, cut_img_path)
                    cut_image_count += 1

                    userTime = time.time() - start_time
                    surplusTime = userTime / (count + 1) * (total_len - count - 1)
                    logInfo2 = '[Total cut progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        (i + 1) / total_len * 100, userTime, surplusTime)
                    self.progress0.emit(logInfo2, count)
                    count += 1
                    continue
                # Calculate number of blocks xyz
                # sliceNumber = np.ceil((big_size - small_size) / (small_size - r)).astype(np.int32) + 1
                sliceNumber = np.array((big_size - small_size) / (small_size - r)).astype(np.int32) + 1
                # Iterate over each block
                lsLen = sliceNumber[0] * sliceNumber[1] * sliceNumber[2]
                all_len += lsLen
                ci = 0
                for nx in range(sliceNumber[0]):
                    for ny in range(sliceNumber[1]):
                        for nz in range(sliceNumber[2]):
                            # New file name
                            new_img_name = f"{str(i).zfill(4)}_{0}_{0}-{nx}_{ny}_{nz}.tif"
                            cut_img_path = join(cut_img_dir, new_img_name)
                            # Calculate start position of the current block
                            start_x = nx * (small_size[0] - r[0])
                            start_y = ny * (small_size[1] - r[1])
                            start_z = nz * (small_size[2] - r[2])
                            # Create a new array initialized to 0
                            new_img = np.zeros(small_size[::-1], dtype=img.dtype)
                            # Calculate range of current block in the big image
                            end_x = min(start_x + small_size[0], big_size[0])
                            end_y = min(start_y + small_size[1], big_size[1])
                            end_z = min(start_z + small_size[2], big_size[2])
                            # Calculate range of current block in the new array
                            new_end_x = end_x - start_x
                            new_end_y = end_y - start_y
                            new_end_z = end_z - start_z
                            # Copy the current block from big image to new array
                            new_img[:new_end_z, :new_end_y, :new_end_x] = img[start_z:end_z, start_y:end_y,
                                                                              start_x:end_x]
                            if np.max(new_img) != 0:
                                # Save new array as a new TIFF file
                                tiff.imwrite(cut_img_path, new_img, compression="lzw")
                                cut_image_count += 1

                            if lsLen > 100:
                                userTime = time.time() - start_time
                                surplusTime = userTime / (count + 1) * (lsLen * total_len - count - 1)
                                logInfo2 = '[Total cut progress %.2f%%] [Image cut progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                                    (i + 1) / total_len * 100, (ci + 1) / lsLen * 100, userTime, surplusTime)
                                self.progress0.emit(logInfo2, count)
                                count += 1
                                ci += 1
                            else:
                                userTime = time.time() - start_time
                                surplusTime = userTime / (count + 1) * (lsLen * total_len - count - 1)
                                logInfo2 = '[Total cut progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                                    (i + 1) / total_len * 100, userTime, surplusTime)
                                self.progress0.emit(logInfo2, count)
                                count += 1

                # userTime = time.time() - start_time
                # surplusTime = userTime / (i + 1) * (total_len - i - 1)
                # logInfo = '[Read progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                #     (i + 1) / total_len * 100, userTime, surplusTime)
                # self.progress0.emit(logInfo, i)
        self.progress0.emit(f"Read 3D TIF images total: {image_count} blocks", 0)
        self.progress0.emit(f"Cut 3D TIF images total: {cut_image_count} blocks", 0)

        if total_len == no_3d_count:
            self.error0.emit(f"{img_dir} folder has no 3D TIF images, cannot cut!")
            return True, cut_image_count

        return False, cut_image_count

    def image_swc_cut(self, img_dir, swc_dir, cut_img_dir, cut_swc_dir, img_names, r, small_size):
        start_time = time.time()
        total_len = len(img_names)
        image_count = 0
        cut_image_count = 0
        no_3d_count = 0
        no_swc_count = 0
        for i, img_name in enumerate(img_names):
            img_path = join(str(img_dir), img_name)
            swc_path = join(str(swc_dir), str(Path(img_name).stem) + ".swc")
            # print(swc_path)
            swc_size = 0
            if os.path.isfile(swc_path):
                if os.path.getsize(swc_path):
                    swc_size = 1

            try:
                if str(Path(img_name).suffix) == ".bv":
                    img = self.readObj.readBV(img_path)
                else:
                    img = tiff.imread(img_path)  # Read image
                if len(img.shape) != 3:
                    no_3d_count += 1

                    userTime = time.time() - start_time
                    surplusTime = userTime / (i + 1) * (total_len - i - 1)
                    logInfo = '[Total cut progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        (i + 1) / total_len * 100, userTime, surplusTime)
                    self.progress0.emit(logInfo, i)
                    continue
            except Exception as e:
                print(e)
                continue

            image_count += 1

            # if swc_size == 0 and np.max(img) == 0:
            #     no_swc_count += 1
            #     userTime = time.time() - start_time
            #     surplusTime = userTime / (i + 1) * (total_len - i - 1)
            #     logInfo = '[Total cut progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
            #         (i + 1) / total_len * 100, userTime, surplusTime)
            #     self.progress0.emit(logInfo, i)
            #     continue

            big_size = np.array(img.shape)[::-1]  # Big image size xyz
            # If input image size is less than or equal to the minimum size
            if all(x >= y for x, y in zip(small_size, big_size)):
                if img.any():
                    new_img_name = f"{str(i).zfill(4)}_{0}_{0}-{0}_{0}_{0}.tif"
                    new_swc_name = f"{str(i).zfill(4)}_{0}_{0}-{0}_{0}_{0}.swc"
                    cut_img_path = join(cut_img_dir, new_img_name)
                    cut_swc_path = join(cut_swc_dir, new_swc_name)
                    shutil.copyfile(img_path, cut_img_path)
                    shutil.copyfile(swc_path, cut_swc_path)
                cut_image_count += 1
            else:
                # Calculate number of blocks xyz
                # sliceNumber = np.ceil((big_size - small_size) / (small_size - r)).astype(np.int32) + 1
                sliceNumber = np.array((big_size - small_size) / (small_size - r)).astype(np.int32) + 1
                # Iterate over each block
                sliceLen = np.prod(sliceNumber)
                count = 0
                st = time.time()
                for nx in range(sliceNumber[0]):
                    for ny in range(sliceNumber[1]):
                        for nz in range(sliceNumber[2]):
                            # New file name
                            new_img_name = f"{str(i).zfill(4)}_{0}_{0}-{nx}_{ny}_{nz}.tif"
                            new_swc_name = f"{str(i).zfill(4)}_{0}_{0}-{nx}_{ny}_{nz}.swc"
                            cut_img_path = join(cut_img_dir, new_img_name)
                            cut_swc_path = join(cut_swc_dir, new_swc_name)

                            # Calculate start position of the current block
                            start_x = nx * (small_size[0] - r[0])
                            start_y = ny * (small_size[1] - r[1])
                            start_z = nz * (small_size[2] - r[2])
                            # Create a new array initialized to 0
                            new_img = np.zeros(small_size[::-1], dtype=img.dtype)
                            # Calculate range of current block in the big image
                            end_x = min(start_x + small_size[0], big_size[0])
                            end_y = min(start_y + small_size[1], big_size[1])
                            end_z = min(start_z + small_size[2], big_size[2])
                            # Calculate range of current block in the new array
                            new_end_x = end_x - start_x
                            new_end_y = end_y - start_y
                            new_end_z = end_z - start_z
                            # Copy the current block from big image to new array
                            new_img[:new_end_z, :new_end_y, :new_end_x] = img[start_z:end_z,
                                                                              start_y:end_y,
                                                                              start_x:end_x]
                            if np.max(new_img) != 0:
                                # Save new array as a new TIFF file
                                tiff.imwrite(cut_img_path, new_img, compression="lzw")

                                if swc_size:  # Label file is not empty
                                    swcData = np.loadtxt(swc_path, ndmin=2)
                                    swcDataLs = self.SplitSwcData(swcData)  # List of point sets
                                    swc_add = 0
                                    with open(cut_swc_path, "w") as f:
                                        for swcData in swcDataLs:
                                            for ii, item in enumerate(swcData):
                                                if len(item) < 7:
                                                    continue
                                                x0 = item[2]
                                                y0 = item[3]
                                                z0 = item[4]
                                                if (start_x <= x0 < end_x and start_y <= y0 < end_y and
                                                        start_z <= z0 < end_z):
                                                    line = (f"{swc_add + 1} {0} {(x0 - start_x):.3f} "
                                                            f"{(y0 - start_y):.3f} {(z0 - start_z):.3f} {0} {-1}\n")
                                                    f.write(line)
                                                    swc_add += 1
                                    f.close()
                                else:
                                    with open(cut_swc_path, "w") as f:
                                        f.write("")
                                    f.close()
                                cut_image_count += 1

                            userTime = time.time() - st
                            surplusTime = userTime / (count + 1) * (sliceLen - count - 1)
                            logInfo = '[Cut progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                                (count + 1) / sliceLen * 100, userTime, surplusTime)
                            self.progress0.emit(logInfo, count)
                            count += 1

            userTime = time.time() - start_time
            surplusTime = userTime / (i + 1) * (total_len - i - 1)
            logInfo = '[Total cut progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                (i + 1) / total_len * 100, userTime, surplusTime)
            self.progress0.emit(logInfo, i)

        self.progress0.emit(f"Read 3D TIF images total: {image_count} blocks", 0)
        self.progress0.emit(f"Cut 3D TIF images total: {cut_image_count} blocks", 0)

        if total_len == no_3d_count:
            self.error0.emit(f"{img_dir} folder has no 3D TIF images, cannot cut!")
            return True, cut_image_count

        if total_len == no_swc_count:
            self.error0.emit(f"{img_dir} folder has no corresponding SWC files or they are empty, please select unlabeled dataset!")
            return True, cut_image_count

        return False, cut_image_count

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


if __name__ == "__main__":
    cell_c = CellImageCutThread()

    img_dir = r"D:\SY\10GTestData\KS\Cell\Kennard_Stone_Img_res\PredictResults\data_pro\add_filter\images"
    swc_dir = r"D:\SY\10GTestData\KS\Cell\Kennard_Stone_Img_res\PredictResults\data_pro\add_filter\swc"
    cut_img_dir = r"D:\SY\10GTestData\KS\Cell\Kennard_Stone_Img_res\PredictResults\data_pro\add_filter\cut_images"
    cut_swc_dir = r"D:\SY\10GTestData\KS\Cell\Kennard_Stone_Img_res\PredictResults\data_pro\add_filter\cut_swc"
    os.makedirs(cut_img_dir, exist_ok=True)
    os.makedirs(cut_swc_dir, exist_ok=True)

    img_names = [n for n in os.listdir(img_dir) if ".tif" in n]

    r = np.array([16, 16, 16])
    small_size = np.array([64, 64, 64])

    cell_c.image_swc_cut(img_dir, swc_dir, cut_img_dir, cut_swc_dir, img_names, r, small_size)
