# -*- coding: utf-8 -*-
import os
import time
import json
import numpy as np
from os.path import join
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from DataStatistics.vessel_radius.segment_to_swc_optimized import segments_to_swc


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


class ImgSpliceQThread(QThread):
    def __init__(self, *args, **kwargs):
        super(ImgSpliceQThread, self).__init__()
        self.img_splice_dict = kwargs.get('img_splice_dict')

    def run(self):
        cut_infos = self.img_splice_dict["cut_infos"]
        cfg_path = self.img_splice_dict["cfg_path"]
        save_dir = self.img_splice_dict["save_dir"]
        dataType = cut_infos["dataType"]
        save_dir = join(save_dir, "SpliceResults")
        # if os.path.exists(save_dir):
        #     shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir, exist_ok=True)
        if dataType == "BV":
            CutWorkFilesDir = cut_infos["CutWorkFilesDir"]
            BvSliceNumberXYZ = cut_infos["BvSliceNumberXYZ"]
            BvRedunXYZ = cut_infos.get("BvRedunXYZ", [0, 0, 0])
            BvBigSizeXYZ = cut_infos["BvBigSizeXYZ"]
            BvSmallSizeXYZ = cut_infos["BvSmallSizeXYZ"]

            cfg_level = cut_infos.get("cfg_level", 0)

            sliceLen = 1
            for slice in BvSliceNumberXYZ:
                sliceLen *= slice

            self.BigImgSplice(CutWorkFilesDir, BvSliceNumberXYZ, BvRedunXYZ, BvBigSizeXYZ, BvSmallSizeXYZ, sliceLen,
                              save_dir, cfg_level)
        if dataType == "TIF":
            bigSizeXYZ = cut_infos["bigSizeXYZ"]
            imgSizeXYZ = cut_infos["imgSizeXYZ"]
            rXYZ = cut_infos["rXYZ"]
            sliceNumberXYZ = cut_infos["sliceNumberXYZ"]
            # cut_img_dir = cut_infos["cut_img_dir"]
            cut_swc_dir = cut_infos["cut_swc_dir"]

            stem = str(Path(cfg_path).stem)
            savePath = join(save_dir, f"{stem}.swc")

            workLen = 1
            for s in sliceNumberXYZ:
                workLen *= s

            self.ImgSplice(bigSizeXYZ, imgSizeXYZ, rXYZ, sliceNumberXYZ, workLen, savePath, cut_swc_dir, 100, 0)
        time.sleep(1)
        print("Splicing complete!")

    """整个图像拼接"""

    def BigImgSplice(self, CutWorkFilesDir, BvSliceNumberXYZ, BvRedunXYZ, BvBigSizeXYZ, BvSmallSizeXYZ, sliceLen,
                     save_dir, cfg_level):
        lv = 2 ** cfg_level
        print(f"Level ratio: {(1 / lv):.3f}x")
        splice_work_dir = join(save_dir, "splice_work_dir")
        os.makedirs(splice_work_dir, exist_ok=True)
        branch_work_dir = join(save_dir, "branch_work_dir")
        os.makedirs(branch_work_dir, exist_ok=True)
        direction_work_dir = join(save_dir, "direction_work_dir")
        os.makedirs(direction_work_dir, exist_ok=True)
        save_path = join(save_dir, "splice.swc")
        save_branch_path = join(save_dir, "branch.txt")
        save_direction_path = join(save_dir, "direction.txt")
        i = 0
        swc_add = 0
        swc_add_ = 0
        self.dirLines = []
        self.branchLines = []
        self.swcLines = []

        for nx in range(BvSliceNumberXYZ[0]):
            for ny in range(BvSliceNumberXYZ[1]):
                for nz in range(BvSliceNumberXYZ[2]):
                    stem = f"{nx}_{ny}"
                    bv_txt_name = stem + ".txt"
                    bv_swc_name = stem + ".swc"
                    bv_cut_info_name = stem + ".json"

                    self.bv_CutWorkFilesDir = join(CutWorkFilesDir, stem)
                    bv_cut_info_path = join(self.bv_CutWorkFilesDir, bv_cut_info_name)
                    bv_swc_path = join(splice_work_dir, bv_swc_name)
                    bv_branch_path = join(branch_work_dir, bv_txt_name)
                    bv_direction_path = join(direction_work_dir, bv_txt_name)

                    total_progress = (i + 1) / sliceLen * 100

                    if not os.path.exists(bv_cut_info_path):
                        print(f"{bv_cut_info_path} Path does not exist！")
                        return
                    with open(bv_cut_info_path) as f:
                        bv_cut_info = json.loads(f.read())

                    bigSizeXYZ = bv_cut_info["bigSizeXYZ"]
                    imgSizeXYZ = bv_cut_info["imgSizeXYZ"]
                    rXYZ = bv_cut_info["rXYZ"]
                    sliceNumberXYZ = bv_cut_info["sliceNumberXYZ"]
                    # cut_img_dir = bv_cut_info["cut_img_dir"]
                    cut_swc_dir = bv_cut_info["cut_swc_dir"]

                    workLen = 1
                    for s in sliceNumberXYZ:
                        workLen *= s

                    self.ImgSplice(bigSizeXYZ, imgSizeXYZ, rXYZ, sliceNumberXYZ, workLen, bv_swc_path,
                                   cut_swc_dir, total_progress, i, bv_branch_path=bv_branch_path,
                                   bv_direction_path=bv_direction_path)

                    i += 1

                    swc_add, swc_add_ = self.ImgSplice_2(BvBigSizeXYZ, BvSmallSizeXYZ, BvRedunXYZ,
                                                         [nx, ny, nz], bv_swc_path, swc_add, swc_add_, lv=lv,
                                                         bv_branch_path=bv_branch_path,
                                                         bv_direction_path=bv_direction_path)

        # 一次性写入文件
        with open(save_path, "w") as f0:
            f0.writelines(self.swcLines)

        with open(save_branch_path, 'w') as f1:
            f1.writelines(self.branchLines)

        with open(save_direction_path, 'w') as f2:
            f2.writelines(self.dirLines)


    def ImgSplice(self, bigSize, imgSize, r, sliceNumber, workLen, savePath, cut_swc_dir, total_progress, i,
                  bv_branch_path=None, bv_direction_path=None):
        swc_lines_all = []
        dir_lines_all = []
        branch_lines_all = []

        block_size_x, block_size_y, block_size_z = imgSize
        br_x = block_size_x - r[0]
        br_y = block_size_y - r[1]
        br_z = block_size_z - r[2]

        count = 0
        swc_add = 0
        swc_add_ = 0
        sTime = time.time()
        calculate_radius_dir = join(self.bv_CutWorkFilesDir, "calculate_radius")
        # 遍历所有图像并拼接到结果数组中
        for nz in range(sliceNumber[2]):  # ProcessZ轴方向
            for ny in range(sliceNumber[1]):  # ProcessY轴方向
                for nx in range(sliceNumber[0]):  # ProcessX轴方向
                    swc_name = f"{nz}_{ny}_{nx}.swc"
                    txt_name = f"{nz}_{ny}_{nx}.txt"
                    # swc_name = f"{nx}_{ny}.swc"
                    # swc_path = join(cut_swc_dir, swc_name)
                    swc_path = join(calculate_radius_dir, "RadiiSwc", swc_name)
                    branch_path = join(calculate_radius_dir, "Branch", txt_name)
                    direction_path = join(calculate_radius_dir, "DirectionDiameter", txt_name)

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
                        branchData = np.loadtxt(branch_path, ndmin=2)
                        directionData = np.loadtxt(direction_path, ndmin=2)
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

                                # for key, value in all_points.items():
                                #     rad = r_dict.get(tuple(key)) if r_dict.get(tuple(key)) else 0
                                #     if value == -1:
                                #         line = (f"{swc_add + 1} "
                                #                 f"{0} "
                                #                 f"{(key[0] - xmin + start_x):.3f} "
                                #                 f"{(key[1] - ymin + start_y):.3f} "
                                #                 f"{(key[2] - zmin + start_z):.3f} "
                                #                 f"{rad} "
                                #                 f"{value}\n")
                                #     else:
                                #         line = (f"{swc_add + 1} "
                                #                 f"{0} "
                                #                 f"{(key[0] - xmin + start_x):.3f} "
                                #                 f"{(key[1] - ymin + start_y):.3f} "
                                #                 f"{(key[2] - zmin + start_z):.3f} "
                                #                 f"{rad} "
                                #                 f"{value + swc_add_}\n")
                                #     swc_lines_all.append(line)
                                #     swc_add += 1

                                for bd in branchData:
                                    branch_lines_all.append(
                                        f"{(bd[0] - xmin + start_x):.3f} {(bd[1] - ymin + start_y):.3f} {(bd[2] - zmin + start_z):.3f} "
                                        f"{bd[3]} {bd[4]} {bd[5]} "
                                        f"{bd[6]} {bd[7]} {bd[8]}\n")
                                for dd in directionData:
                                    dir_lines_all.append(
                                        f"{dd[0]} {(dd[1] - xmin + start_x):.3f} {(dd[2] - ymin + start_y):.3f} {(dd[3] - zmin + start_z):.3f} "
                                        f"{dd[4]} {dd[5]} {dd[6]} "
                                        f"{dd[7]} {dd[8]} {dd[9]} {dd[10]}\n"
                                    )
                    else:
                        branchData = np.loadtxt(branch_path, ndmin=2)
                        directionData = np.loadtxt(direction_path, ndmin=2)
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

                                for bd in branchData:
                                    branch_lines_all.append(
                                        f"{(bd[0] + start_x):.3f} {(bd[1] + start_y):.3f} {(bd[2] + start_z):.3f} "
                                        f"{bd[3]} {bd[4]} {bd[5]} "
                                        f"{bd[6]} {bd[7]} {bd[8]}\n")
                                for dd in directionData:
                                    dir_lines_all.append(
                                        f"{dd[0]} {(dd[1] + start_x):.3f} {(dd[2] + start_y):.3f} {(dd[3] + start_z):.3f} "
                                        f"{dd[4]} {dd[5]} {dd[6]} "
                                        f"{dd[7]} {dd[8]} {dd[9]} {dd[10]}\n"
                                    )

                    swc_add_ = swc_add

                    if (count + 1) % 1 == 0:
                        userTime = time.time() - sTime
                        surplusTime = userTime / (count + 1) * (workLen - count - 1)
                        logInfo = '[Total progress %.2f%%] [Splicing progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                            total_progress, (count + 1) / workLen * 100, userTime, surplusTime)
                        print(logInfo)
                        # self.progress0.emit(logInfo, max(i, count))
                    count += 1

        # 一次性写入文件
        with open(savePath, "w") as f0:
            f0.writelines(swc_lines_all)

        with open(bv_branch_path, 'w') as f1:
            f1.writelines(branch_lines_all)

        with open(bv_direction_path, 'w') as f2:
            f2.writelines(dir_lines_all)

    def ImgSplice_2(self, bigSize, imgSize, r, sliceNumber, bv_swc_path, swc_add, swc_add_, lv=1,
                    bv_branch_path=None, bv_direction_path=None):

        block_size_x, block_size_y, block_size_z = imgSize
        br_x = block_size_x - r[0]
        br_y = block_size_y - r[1]
        br_z = block_size_z - r[2]

        nx, ny, nz = sliceNumber

        # is_surpass = False

        # 计算块的起始索引（边缘块从末尾向前取 block_size）
        start_z = min(nz * br_z, bigSize[2] - block_size_z)
        start_y = min(ny * br_y, bigSize[1] - block_size_y)
        start_x = min(nx * br_x, bigSize[0] - block_size_x)
        # 计算结束索引
        # end_z = start_z + block_size_z
        # end_y = start_y + block_size_y
        # end_x = start_x + block_size_x

        if bigSize[2] - block_size_z < 0:  # 大图边缘切块尺寸不足，大图尺寸不足
            start_z = 0
        elif start_z == bigSize[2] - block_size_z and nz * br_z != bigSize[2] - block_size_z:  # 边缘冗余
            # 倒数第二块的结束索引减冗余，作为最后一块的起始索引，避免过多追踪结果重合
            start_z = (nz - 1) * br_z + block_size_z - r[2]
            # end_z = bigSize[2]

        if bigSize[1] - block_size_y < 0:  # 大图不足填充生成的小图
            start_y = 0
        elif start_y == bigSize[1] - block_size_y and ny * br_y != bigSize[1] - block_size_y:  # 边缘冗余
            start_y = (ny - 1) * br_y + block_size_y - r[1]
            # end_y = bigSize[1]

        if bigSize[0] - block_size_x < 0:  # 大图不足填充生成的小图
            start_x = 0
        elif start_x == bigSize[0] - block_size_x and nx * br_x != bigSize[0] - block_size_x:  # 边缘冗余
            start_x = (nx - 1) * br_x + block_size_x - r[0]
            # end_x = bigSize[0]

        # if is_surpass:  # 超过原图
        #     swcData = np.loadtxt(bv_swc_path, ndmin=2)
        #     if len(swcData):
        #         if len(swcData[0]) == 7:
        #             swcDataLs = SplitSwcData(swcData)
        #             dx = end_x - start_x
        #             dy = end_y - start_y
        #             dz = end_z - start_z
        #             xmin, xmax, ymin, ymax, zmin, zmax = (block_size_x - dx, block_size_x,
        #                                                   block_size_y - dy, block_size_y,
        #                                                   block_size_z - dz, block_size_z)
        #             xmax -= 1
        #             ymax -= 1
        #             zmax -= 1
        #             cut_center = np.array([xmax + xmin, ymax + ymin, zmax + zmin]) / 2  # Center
        #             cut_v = np.array([xmax - xmin, ymax - ymin, zmax - zmin]) / 2  # 方向向量
        #             cut_r = np.linalg.norm(cut_v)
        #
        #             line_list = []
        #             r_dict = {}
        #
        #             for swcData in swcDataLs:
        #                 x0 = swcData[..., 2]
        #                 y0 = swcData[..., 3]
        #                 z0 = swcData[..., 4]
        #
        #                 x0_min = np.min(x0)
        #                 x0_max = np.max(x0)
        #                 y0_min = np.min(y0)
        #                 y0_max = np.max(y0)
        #                 z0_min = np.min(z0)
        #                 z0_max = np.max(z0)
        #
        #                 center = np.array([x0_max + x0_min, y0_max + y0_min, z0_max + z0_min]) / 2  # Center
        #                 v = np.array([x0_max - x0_min, y0_max - y0_min, z0_max - z0_min]) / 2  # 方向向量
        #                 cr = np.linalg.norm(v)
        #                 dist = np.linalg.norm(cut_center - center)  # 中心间距
        #                 if dist < cr + cut_r:
        #                     for ii, item in enumerate(swcData):
        #                         # Radius
        #                         r_dict[tuple(item[2: 5])] = item[5]
        #
        #                         if item[-1] == -1:
        #                             continue
        #                         p0 = item[2: 5]
        #                         index0 = int(float(item[-1]))
        #                         index1 = index0 - 1
        #                         p1 = swcData[index1, 2: 5]
        #
        #                         is_p0 = 0
        #                         is_p1 = 0
        #                         for vi in range(3):
        #                             if np.abs(cut_center[vi] - p0[vi]) >= np.abs(cut_v[vi]):
        #                                 is_p0 += 1
        #                                 break
        #                             if np.abs(cut_center[vi] - p1[vi]) >= np.abs(cut_v[vi]):
        #                                 is_p1 += 1
        #                                 break
        #                         if not is_p0 and not is_p1:
        #                             line_list.append([tuple(p1), tuple(p0)])
        #
        #             if len(line_list):
        #                 groups = group_segments(line_list)
        #                 del line_list
        #                 all_points = group_recombine(groups)
        #             else:
        #                 all_points = {}
        #
        #             for key, value in all_points.items():
        #                 rad = r_dict.get(tuple(key)) if r_dict.get(tuple(key)) else 0
        #                 if value == -1:
        #                     line = f"{swc_add + 1} {0} {(key[0] - xmin + start_x):.3f} {(key[1] - ymin + start_y):.3f} {(key[2] - zmin + start_z):.3f} {rad} {value}\n"
        #                 else:
        #                     line = f"{swc_add + 1} {0} {(key[0] - xmin + start_x):.3f} {(key[1] - ymin + start_y):.3f} {(key[2] - zmin + start_z):.3f} {rad} {value + swc_add_}\n"
        #                 ff.write(line)
        #                 swc_add += 1
        # else:
        branchData = np.loadtxt(bv_branch_path, ndmin=2)
        directionData = np.loadtxt(bv_direction_path, ndmin=2)
        swcData = np.loadtxt(bv_swc_path, ndmin=2)
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
                                    f"{((item[2] + start_x) * lv):.3f} "
                                    f"{((item[3] + start_y) * lv):.3f} "
                                    f"{((item[4] + start_z) * lv):.3f} "
                                    f"{item[5]} -1\n")
                            else:
                                self.swcLines.append(
                                    f"{swc_add + 1} "
                                    f"{item[1]} "
                                    f"{((item[2] + start_x) * lv):.3f} "
                                    f"{((item[3] + start_y) * lv):.3f} "
                                    f"{((item[4] + start_z) * lv):.3f} "
                                    f"{item[5]} "
                                    f"{item[6] + swc_add_}\n")
                        else:
                            self.swcLines.append(
                                f"{swc_add + 1} "
                                f"{item[1]} "
                                f"{((item[2] + start_x) * lv):.3f} "
                                f"{((item[3] + start_y) * lv):.3f} "
                                f"{((item[4] + start_z) * lv):.3f} "
                                f"{item[5]} "
                                f"{swc_add}\n")
                        swc_add += 1
                    swc_add_ += len(swcData)

                for bd in branchData:
                    self.branchLines.append(
                        f"{(bd[0] + start_x):.3f} {(bd[1] + start_y):.3f} {(bd[2] + start_z):.3f} "
                        f"{bd[3]} {bd[4]} {bd[5]} "
                        f"{bd[6]} {bd[7]} {bd[8]}\n")
                for dd in directionData:
                    self.dirLines.append(
                        f"{dd[0]} {(dd[1] + start_x):.3f} {(dd[2] + start_y):.3f} {(dd[3] + start_z):.3f} "
                        f"{dd[4]} {dd[5]} {dd[6]} "
                        f"{dd[7]} {dd[8]} {dd[9]} {dd[10]}\n"
                    )

        swc_add_ = swc_add
        return swc_add, swc_add_


if __name__ == "__main__":
    cfg_path = r"D:\SY\VesselData\bv_data\00_02_00_bv\data_save\PredictResults\CutWorkFiles\cut_infos.json"
    save_dir = r"D:\SY\VesselData\bv_data\00_02_00_bv\data_save\splice"
    os.makedirs(save_dir, exist_ok=True)

    with open(cfg_path, "r") as f:
        cut_infos = json.loads(f.read())
    img_splice_dict = {
        "cut_infos": cut_infos,
        "cfg_path": cfg_path,
        "save_dir": save_dir
    }
    ImgSplice = ImgSpliceQThread(img_splice_dict=img_splice_dict)
    ImgSplice.run()
    # ImgSplice.wait()
