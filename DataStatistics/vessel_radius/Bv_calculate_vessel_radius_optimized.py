# -*- coding: utf-8 -*-
import os
import time
import shutil
import tifffile as tiff
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from functools import partial
import warnings
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from DataInterpolation import data_add


# 常量定义
EPSILON = 1e-6
MAX_STEPS = 100
STEP_SIZE = 1
THRESH = 0.5


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

    swcData = np.loadtxt(swcPath, ndmin=2)  # 读取骨架文件
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
    return dfImg2

def autoAdjustGray(img):
    """优化版本：使用numpy向量化操作优化灰度调整"""
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
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


def calculate_distance(center, edge_point, resolution_ratio=(1, 1, 3 / 1.625)):
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
    pre_indices = np.where(parent_indices == -1, 1, parent_indices - 1).astype(int)
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
            if item2[-1] + 1 == item2[0]:
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
                current_edt = edt[round(p0[2]), round(p0[1]), round(p0[0])]
                radius = (radius + current_edt) / 2
                radii.append(radius)
            else:
                current_edt = edt[round(p0[2]), round(p0[1]), round(p0[0])]
                radii.append(max(current_edt, 0.1))
        
        # 计算平均半径
        if np.sum(radii) > 0:
            avg_radii = [np.mean(radii[i:i+2]) for i in range(0, len(radii), 2)]
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
    
    # 一次性写入文件
    with open(RadiiSwcPath, 'w') as f:
        f.writelines(swc_lines_all)
    
    with open(DirectionDiameterPath, 'w') as f2:
        f2.writelines(dir_lines_all)
    
    with open(BranchPath, 'w') as f4:
        f4.writelines(branch_lines_all)


def process_single_image(img_name, img_dir, swc_dir, RadiiSwcDir, DirectionDiameterDir, BranchDir, resolution_ratio):
    """处理单个图像，用于并行处理"""
    swc_name = img_name.replace(".tif", ".swc")
    txt_name = img_name.replace(".tif", ".txt")
    
    img_path = os.path.join(str(img_dir), img_name)
    # seg_path = os.path.join(str(seg_dir), img_name)
    swc_path = os.path.join(str(swc_dir), swc_name)
    
    RadiiSwcPath = os.path.join(RadiiSwcDir, swc_name)
    DirectionDiameterPath = os.path.join(DirectionDiameterDir, txt_name)
    BranchPath = os.path.join(BranchDir, txt_name)

    # 优化图像读取
    img = tiff.imread(img_path)
    img = ((1 * img - img.min()) / (max(1, img.max() - img.min())) * 255).astype(np.uint8)

    # 标签处理
    shapes = img.shape[::-1]
    mask = swc_to_mask(shapes, swc_path)
    mask = (mask > 0).astype(np.uint8) * 255

    # 优化分割结果处理
    # seg = tiff.imread(seg_path)
    # seg = (seg > 0).astype(np.uint8) * 255

    # 优化信号提取
    signal = autoAdjustGray(img.copy())
    mask = np.array(mask, dtype=np.uint16) + np.array(signal, dtype=np.uint16)
    signal = (mask > 0).astype(bool)

    # CalculateEDT
    edt = distance_transform_edt(signal)

    # 计算半径
    calculate_radii(swc_path, RadiiSwcPath, DirectionDiameterPath, BranchPath, edt, resolution_ratio)

    return True


def calculate_vessel_radius(img_dir, swc_dir, save_dir, resolution_ratio=(1, 1, 1), max_workers=None):
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
    DirectionDiameterDir = os.path.join(save_dir, "DirectionDiameter")
    RadiiSwcDir = os.path.join(save_dir, "RadiiSwc")
    BranchDir = os.path.join(save_dir, "Branch")
    
    for dir_path in [DirectionDiameterDir, RadiiSwcDir, BranchDir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path, exist_ok=True)
    
    # 获取图像列表
    img_names = [n for n in os.listdir(img_dir) if ".tif" in n]  # 限制只处理第一张图像
    total_len = len(img_names)
    
    if total_len == 0:
        print("No image files to process were found.")
        return
    
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
        resolution_ratio=resolution_ratio
    )

    # 并行处理图像
    if total_len > 1 and max_workers != 1:  # 如果只有一张图像，使用串行处理避免进程创建开销
        futs = []
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            for name in img_names:
                futs.append(exe.submit(process_func, name))  # 返回 Future

            # 每完成一个 Future 就更新一次进度条
            results = []
            for f in tqdm(as_completed(futs), total=len(futs)):
                results.append(f.result())

        # from tqdm.contrib.concurrent import process_map  # 多进程带进度条
        # # from functools import partial
        #
        # # 1. 把所有常量先拼成 partial
        # _process_one = partial(process_single_image,
        #                        img_dir=img_dir,
        #                        swc_dir=swc_dir,
        #                        RadiiSwcDir=RadiiSwcDir,
        #                        DirectionDiameterDir=DirectionDiameterDir,
        #                        BranchDir=BranchDir,
        #                        resolution_ratio=resolution_ratio)
        #
        # # 2. process_map 直接调这个 partial
        # results = process_map(_process_one, img_names, max_workers=max_workers)
        #
        # success = sum(results)
        # print(f"Success {success}/{len(results)} 张")

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


if __name__ == '__main__':
    # 保持与原版本相同的运行方式
    root = r"D:\SY\VesselData\bv_data\00_02_00_bv\data_save\PredictResults\CutWorkFiles\0_0"
    # root = r"D:\LuoJi\20251110\train_data"
    img_dir = os.path.join(root, "images")
    # seg_dir = os.path.join(root, "segment")
    swc_dir = os.path.join(root, "swc")
    save_dir = os.path.join(root, "calculate_radius")
    resolution_ratio = (1, 1, 1)  # x,y,z

    # 调用优化版本的函数
    calculate_vessel_radius(img_dir, swc_dir, save_dir, resolution_ratio)

    # Bv_calculate_vessel_radius_optimized.py

    # 处理大数据格式，计算半径
    # print()
    # rootDir = r"D:\SY\VesselData\bv_data\00_02_00_bv\data_save\PredictResults\CutWorkFiles"
    # names = [n for n in os.listdir(rootDir) if "." not in n]
    # print(names)
    # for name in names:
    #     root = os.path.join(rootDir, name)
    #     img_dir = os.path.join(root, "images")
    #     swc_dir = os.path.join(root, "swc")
    #     save_dir = os.path.join(root, "calculate_radius")
    #     swc_add_dir = os.path.join(save_dir, "swc_interpolation")
    #     # resolution_ratio = (1, 1, 3 / 1.625)  # x,y,z
    #     resolution_ratio = (1, 1, 1)  # x,y,z
    #     data_add(swc_dir, swc_add_dir)
    #     # 调用优化版本的函数
    #     calculate_vessel_radius(img_dir, swc_add_dir, save_dir, resolution_ratio)

    # from Swc_Splice import ImgSpliceQThread
    # import json
    #
    # cfg_path = r"D:\SY\VesselData\bv_data\00_02_00_bv\data_save\PredictResults\CutWorkFiles\cut_infos.json"
    # save_dir = r"D:\SY\VesselData\bv_data\00_02_00_bv\data_save\splice"
    # os.makedirs(save_dir, exist_ok=True)
    #
    # with open(cfg_path, "r") as f:
    #     cut_infos = json.loads(f.read())
    # img_splice_dict = {
    #     "cut_infos": cut_infos,
    #     "cfg_path": cfg_path,
    #     "save_dir": save_dir
    # }
    # ImgSplice = ImgSpliceQThread(img_splice_dict=img_splice_dict)
    # ImgSplice.run()
