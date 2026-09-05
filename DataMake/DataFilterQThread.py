# -*- coding: utf-8 -*-
import shutil
import time
from os.path import join
from PyQt5.QtCore import QThread, pyqtSignal
import sys
import traceback
from pathlib import Path
import os
import numpy as np
import random
import tifffile as tiff
from DataMake import DataMaker, DataMakerOfBV, DataMakerOfZarr, DataMakerOfBV_4Parts, DataMakerOfZarr_4Parts


class DataFilterQThread(QThread):
    finish0 = pyqtSignal(str)
    progress0 = pyqtSignal(str, int)
    error0 = pyqtSignal(str)
    make_finish = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super(DataFilterQThread, self).__init__()
        self.win = kwargs.get('win')
        self.DataMaker = DataMaker
        self.DataMakerOfBV = DataMakerOfBV
        self.DataMakerOfBV_4Parts = DataMakerOfBV_4Parts
        self.DataMakerOfZarr = DataMakerOfZarr
        self.DataMakerOfZarr_4Parts = DataMakerOfZarr_4Parts
        self.logger = self.win.logger

    def run(self):
        try:
            making_arguments = self.win.making_arguments
            select_input_index = making_arguments['select_input_index']

            img_dir = making_arguments['img_dir']
            cut_results_dir = making_arguments['cut_results_dir']
            cut_img_dir = making_arguments['cut_img_dir']
            cut_swc_dir = making_arguments['cut_swc_dir']
            divide_results_dir = making_arguments['divide_results_dir']
            divide_img_dir = making_arguments['divide_img_dir']
            divide_swc_dir = making_arguments['divide_swc_dir']

            make_arguments = making_arguments['make_arguments']
            cfg_level = make_arguments['cfg_level']
            division_ratio = make_arguments['division_ratio']
            division_ratio_2 = np.array(division_ratio) / np.sum(division_ratio)
            bv_ROI = making_arguments.get("bv_ROI", None)
            if bv_ROI is not None:
                bv_ROI = np.array(bv_ROI, dtype=np.int32)
            os.makedirs(cut_swc_dir, exist_ok=True)

            if os.path.exists(divide_results_dir):
                # 清空文件夹
                shutil.rmtree(divide_results_dir, ignore_errors=True)
            # 重新生成
            os.makedirs(divide_results_dir, exist_ok=True)
            os.makedirs(divide_img_dir, exist_ok=True)
            os.makedirs(divide_swc_dir, exist_ok=True)

            if select_input_index == 2:
                res, division_text = self.DataMakerOfBV.start_bv(divide_results_dir, img_dir, divide_img_dir,
                                                                 make_arguments, progress0=self.progress0,
                                                                 error0=self.error0, division_ratio=division_ratio_2,
                                                                 logger=self.logger, cfg_level=cfg_level, bv_ROI=bv_ROI)
                if division_text == "No sampleXYZ":
                    print("No sampleXYZ!")
                    res, division_text = self.DataMakerOfBV_4Parts.start_bv(divide_results_dir, img_dir, divide_img_dir,
                                                                            make_arguments, progress0=self.progress0,
                                                                            error0=self.error0,
                                                                            division_ratio=division_ratio_2,
                                                                            logger=self.logger, cfg_level=cfg_level,
                                                                            bv_ROI=bv_ROI)
            elif select_input_index == 3:
                res, division_text = self.DataMakerOfZarr.start_zarr(divide_results_dir, img_dir, divide_img_dir,
                                                                     make_arguments, progress0=self.progress0,
                                                                     error0=self.error0,
                                                                     division_ratio=division_ratio_2,
                                                                     logger=self.logger, cfg_level=cfg_level)
                if division_text == "No sampleXYZ":
                    print("No sampleXYZ!")
                    res, division_text = self.DataMakerOfZarr_4Parts.start_zarr(divide_results_dir, img_dir,
                                                                                divide_img_dir, make_arguments,
                                                                                progress0=self.progress0,
                                                                                error0=self.error0,
                                                                                division_ratio=division_ratio_2,
                                                                                logger=self.logger, cfg_level=cfg_level)
            else:
                res, division_text = self.DataMaker.start_one(divide_results_dir, cut_img_dir, cut_swc_dir,
                                                              divide_img_dir, divide_swc_dir, make_arguments,
                                                              progress0=self.progress0, error0=self.error0,
                                                              division_ratio=division_ratio_2, logger=self.logger)
            time.sleep(1)
            is_err = res['error']  # 为真报错
            text = res['text']
            if is_err:
                return
            if os.path.exists(cut_results_dir):
                shutil.rmtree(cut_results_dir, ignore_errors=True)
            self.progress0.emit(division_text, 0)
            self.finish0.emit(text)
            # 数据集分配
            self.data_set_txt_make(divide_results_dir, divide_img_dir, division_ratio_2)
            self.make_finish.emit()
            # 筛选数据集
            # files_list = self.select_files(allImgDir, counts)
            # 文件转移
            # self.files_transfer(imgDir, files_list)
        except Exception as e:
            self.error_logger(e)
            self.error0.emit(str(e))

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

    def select_files(self, folder_path, counts):
        # 获取所有三维.tifFile
        tif_files = self.get_tif_files(folder_path)  # 完整路径列表
        total_files = len(tif_files)

        counts = counts if total_files > counts else total_files

        # 获取每个文件的灰度值
        grayscale_files = [(file, self.get_grayscale(file)) for file in tif_files]  # 根据灰度排序

        # 按灰度值从大到小排序
        grayscale_files.sort(key=lambda x: x[1], reverse=True)

        # 动态计算每部分的数量
        part1_count = int(total_files * 0.7)
        part2_count = int(total_files * 0.2)
        # 分成三部分---完整路径
        part1 = grayscale_files[:part1_count]
        part2 = grayscale_files[part1_count:part1_count + part2_count]
        part3 = grayscale_files[part1_count + part2_count:]
        # 三部分分别筛选的文件数
        part1_counts = int(counts * 0.7)
        part2_counts = int(counts * 0.2)
        part3_counts = counts - (part1_counts + part2_counts)
        # 确保每部分选取的文件数不超过该部分的文件数
        part1_sample_count = min(part1_counts, len(part1))
        part2_sample_count = min(part2_counts, len(part2))
        part3_sample_count = min(part3_counts, len(part3))
        # 随机选取文件
        selected_files = random.sample(part1, part1_sample_count) + \
                         random.sample(part2, part2_sample_count) + \
                         random.sample(part3, part3_sample_count)
        # 提取文件路径
        selected_file_paths = [file for file, _ in selected_files]
        return selected_file_paths

    def get_tif_files(self, folder_path):
        """
        获取文件夹中所有.tifFile
        """
        tif_files = []
        for file in os.listdir(folder_path):
            if file.endswith('.tif'):
                file_path = join(folder_path, file)
                tif_files.append(file_path)
        return tif_files

    def get_grayscale(self, file_path):
        """
        获取单个.tif文件的灰度值
        """
        img = tiff.imread(file_path)
        # 假设灰度值为所有像素值的平均值
        grayscale = np.mean(img)
        return grayscale

    def files_transfer(self, imgDir, files_list):
        """
        文件转移
        :param imgDir:
        :param files_list:
        :return:
        """
        for file_name in files_list:
            name = str(Path(file_name).name)
            new_path = join(imgDir, name)
            shutil.copy(file_name, new_path)

    def data_set_txt_make(self, divide_results_dir, divide_img_dir, divisionRatio):
        """
        数据集分配
        :param divide_results_dir:
        :param divide_img_dir:
        :param divisionRatio:
        :return:
        """
        # 训练集,验证集,测试集比例
        dataSetRadio = divisionRatio
        setNameLs = ['train', 'val', 'test']
        ls = [l for l in os.listdir(divide_img_dir) if ".tif" in l]

        test_dir = os.path.join(divide_results_dir, "testData")
        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)
        os.makedirs(test_dir, exist_ok=True)
        test_images_dir = os.path.join(test_dir, "images")
        os.makedirs(test_images_dir, exist_ok=True)
        # lsLen = len(ls)
        nameLs = []
        for ii, name in enumerate(ls):
            nameLs.append(name)
        # 写入总数
        lsLen = len(nameLs)
        spaceLs = [0, int(lsLen * dataSetRadio[0]), int(lsLen * (dataSetRadio[0] + dataSetRadio[1])), int(lsLen)]
        random.shuffle(nameLs)
        for ii in range(len(setNameLs)):
            curNameLS = nameLs[spaceLs[ii]: spaceLs[ii + 1]]
            path = join(divide_results_dir, setNameLs[ii] + '.txt')
            with open(path, 'w') as f:
                [f.write('%s\n' % name) for name in curNameLS]
                if setNameLs[ii] == "test":
                    [shutil.copy(os.path.join(divide_img_dir, name), test_images_dir) for name in curNameLS]

        totalName_path = join(divide_results_dir, 'totalName.txt')
        with open(totalName_path, 'w') as f:
            [f.write('%s\n' % name) for name in nameLs]


'''数据集制作Txt2'''


def DataSetTxtMake(root_dir):
    # 训练集,验证集,测试集比例
    dataSetRadio = [0.7, 0.20, 0.10]
    setNameLs = ['train', 'val', 'test']
    makePath = join(root_dir, 'mask')
    ls = [l for l in os.listdir(makePath) if ".tif" in l]
    lsLen = len(ls)
    nameLs = []
    for ii, name in enumerate(ls):
        # img = tifffile.imread(join(makePath, name))
        # if img.sum() > 800:
        nameLs.append(name)
        print('%d | %d' % (ii + 1, lsLen), name)
        print()

    # 写入总数
    lsLen = len(nameLs)
    with open(join(root_dir, 'totalName.txt'), 'w') as f:
        [f.write('%s\n' % name) for name in nameLs]

    print("%s ———— %d" % (join(root_dir, 'totalName.txt'), lsLen))
    # with open(join(path, 'totalName.txt'), 'r') as f:
    #     nameLs = f.read().strip().split('\n')
    # lsLen = len(nameLs)

    spaceLs = [0, int(lsLen * dataSetRadio[0]), int(lsLen * (dataSetRadio[0] + dataSetRadio[1])),
               int(lsLen * (dataSetRadio[0] + dataSetRadio[1] + dataSetRadio[2]))]
    random.shuffle(nameLs)
    for ii in range(len(setNameLs)):
        curNameLS = nameLs[spaceLs[ii]: spaceLs[ii + 1]]
        with open(join(root_dir, setNameLs[ii] + '.txt'), 'w') as f:
            [f.write('%s\n' % name) for name in curNameLS]

        print("%s ———— %s" % (join(root_dir, setNameLs[ii] + '.txt'), f"{len(curNameLS)} / {lsLen}"))


if __name__ == '__main__':
    # root_dir = r"D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D-DDP\TrainDataSet"
    # DataSetTxtMake(root_dir)
    print()

