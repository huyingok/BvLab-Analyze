# -*- coding: utf-8 -*-
from PyQt5.QtCore import QThread, pyqtSignal
from pathlib import Path
import os
from os.path import join
import numpy as np
import tifffile as tiff
import sys
import traceback
import cv2
import time
import random
import warnings
warnings.filterwarnings("ignore")


class MaskMake(QThread):
    finish0 = pyqtSignal(str)
    warning0 = pyqtSignal(str)
    mask_make_progress0 = pyqtSignal(str, np.ndarray, np.ndarray)
    error0 = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(MaskMake, self).__init__()
        self.win = kwargs.get('win')

    def run(self):
        try:
            # self.data_train_dict = {
            #     'imgDir': imgDir,
            #     'swcDir': swcDir,
            #     'learning_rate': learning_rate,
            #     'weight_decay': weight_decay,
            #     'epochs': epochs,
            #     'batch_size': batch_size,
            #     'shapes': None,
            # }

            images_dir = self.win.data_train_dict['imgDir']
            swc_dir = self.win.data_train_dict['swcDir']

            image_names = [l for l in os.listdir(images_dir) if '.tif' in l]
            if len(image_names) == 0:
                self.warning0.emit("The image folder has no Tif format files")
                return

            img0 = tiff.imread(join(images_dir, image_names[0]))
            shapes = img0.shape[::-1]  # convert to xyz
            self.win.data_train_dict['shapes'] = list(shapes)  # update size

            self.rootPath = str(Path(images_dir).parent.absolute())
            mask_save_dir = join(self.rootPath, 'mask')

            # Conversion
            self.swc_to_mask(images_dir, swc_dir, mask_save_dir, image_names, shapes)
            # Data allocation
            if not self.win.train_is_stop:
                self.data_set(image_names)

            self.finish0.emit("Label conversion finished!")
        except Exception as e:
            self.error0.emit(str(e))
            print("=== Error message ===")
            print(f"Exception type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("\n=== Error location ===")
            tb = sys.exc_info()[2]
            for frame in traceback.extract_tb(tb):
                print(f"  File: {frame.filename}")
                print(f"  Line number: {frame.lineno}")
                print(f"  Function: {frame.name}")
                print(f"  Code: {frame.line}\n")

    def swc_to_mask(self, images_dir, swc_dir, mask_dir, image_names, shapes):
        if len(image_names):
            kernelLen = 3  # Radius
            total_len = len(image_names)  # Number of images
            start_time = time.time()
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
            for ni, image_name in enumerate(image_names):
                if self.win.train_is_stop:
                    break
                swcPath = os.path.join(str(swc_dir), str(Path(image_name).stem) + ".swc")  # Skeleton file
                maskPath = os.path.join(str(mask_dir), image_name)  # Save label path
                if not os.path.exists(swcPath):
                    with open(swcPath, 'w', encoding="utf-8") as f:
                        pass
                swcData = np.loadtxt(swcPath, ndmin=2)  # Read skeleton file
                swcDataLs = self.SplitSwcData(swcData)
                dfImg[...] = maxD
                for swcData in swcDataLs:
                    if self.win.train_is_stop:
                        break
                    for ii, item in enumerate(swcData):
                        if self.win.train_is_stop:
                            break
                        if item[-1] == -1:
                            continue
                        p0 = item[2: 5]
                        p1 = swcData[int(item[-1]) - 1, 2: 5]
                        v = (p1 - p0).reshape([1, 3])
                        if np.linalg.norm(v) < 0.1:
                            continue
                        # Interpolation
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
                # dfImg2[dfImg2 < 10] = 0
                # dfImg = ((dfImg - dfImg.min()) / (dfImg.max() - dfImg.min()) * 255).astype(np.uint8)
                tiff.imwrite(maskPath, dfImg2, compression='lzw')

                dfImg2_2d = self.MaxProject(dfImg2, 0)
                img = tiff.imread(join(str(images_dir), image_name))
                img_2d = self.MaxProject(img, 1)

                if (ni + 1) % 2 == 0 or ni == 0 or ni == total_len - 1:
                    userTime = time.time() - start_time
                    surplusTime = userTime / (ni + 1) * (total_len - ni - 1)
                    text = '[SWC to mask progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        (ni + 1) / total_len * 100, userTime, surplusTime)
                    self.mask_make_progress0.emit(text, img_2d, dfImg2_2d)

    def data_set(self, ls):
        """
        Dataset allocation
        """
        totalName_path = join(self.rootPath, 'totalName.txt')
        if not os.path.exists(totalName_path):
            # Training set, validation set, test set ratios
            dataSetRadio = [0.7, 0.20, 0.10]
            setNameLs = ['train', 'val', 'test']
            # lsLen = len(ls)
            nameLs = []
            for ii, name in enumerate(ls):
                nameLs.append(name)

            # Write total list
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

    def MaxProject(self, img, is_enhance):
        img = np.max(img, axis=0)
        img = self.normalize_and_scale_to_uint8(img)
        if is_enhance:
            img = self.enhance_contrast_clahe(img)
        return img

    def enhance_contrast_clahe(self, img):
        """
        Enhance image contrast using Adaptive Histogram Equalization (CLAHE).

        Parameters:
            img (numpy.ndarray): Input image.
        Returns:
            numpy.ndarray: Contrast-enhanced image.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
        return img_clahe

    def normalize_and_scale_to_uint8(self, data):
        """
        Normalize grayscale values to 0-1, then scale to 0-255 and convert to uint8.

        Parameters:
            data (numpy.ndarray): Input data.
        Returns:
            numpy.ndarray: Converted uint8 data.
        """
        data_min = np.min(data)
        data_max = np.max(data)
        normalized_data = (data - data_min) / (data_max - data_min)
        scaled_data = (normalized_data * 255).astype(np.uint8)
        return scaled_data

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

    '''Get point cloud kernel point cloud'''

    def GetPcKernelPc(self, pc, kernelArr, imgShape):
        curPc = (pc[:, None] + kernelArr[None]).reshape([-1, 3])
        curPc = np.round(curPc).astype(np.int32)
        curPc[curPc < 0] = 0
        curPc[curPc[:, 2] > imgShape[0] - 1, 2] = imgShape[0] - 1
        curPc[curPc[:, 1] > imgShape[1] - 1, 1] = imgShape[1] - 1
        curPc[curPc[:, 0] > imgShape[2] - 1, 0] = imgShape[2] - 1
        curPc = np.unique(curPc, axis=0)
        return curPc
