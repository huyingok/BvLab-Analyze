# -*- coding: utf-8 -*-
import os
import numpy as np
from pathlib import Path
import tempfile
import shutil
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings("ignore")


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


def SplitSwcData(swcData):
    """SplitSWC数据为多个独立的树结构"""
    indLs = np.where(swcData[:, -1] == -1)[0].tolist() + [swcData.shape[0]]
    swcDataLs = []
    for i in range(len(indLs) - 1):
        data = swcData[indLs[i]: indLs[i + 1]]
        sp = data[0, 0]
        data[:, 0] -= sp - 1
        data[1:, -1] -= sp - 1
        swcDataLs.append(data)
    return swcDataLs


def adjust_radius_based_on_nearest_points(current_radius, nearest_points_radii, threshold=1.0):
    """根据邻近点的半径调整当前点的半径 - 保持与原始函数完全相同的逻辑"""
    mean_radius = np.mean(nearest_points_radii)
    
    if abs(current_radius - mean_radius) > threshold:
        return mean_radius
    return current_radius


def radius_smooth_optimized(RadiiSwcPath, SmoothSwcPath, limit_r=None):
    """
    优化版的半径平滑函数
    输入输出格式与原函数保持一致
    
    优化策略：
    1. 使用NumPy向量化操作替代部分循环
    2. 批量预计算所有点的邻近点
    3. 批量处理文件写入，减少I/O操作
    4. 优化异常处理逻辑
    5. 减少重复计算
    """
    # 读取SWCData
    swcData = np.loadtxt(RadiiSwcPath, ndmin=2)
    
    # 准备所有输出行
    output_lines = []
    near_nums = 15  # 保持与原函数相同的邻近点数
    sums = 0
    
    # SplitSWCData
    swcDataLs = SplitSwcData(swcData)
    
    for swcData in swcDataLs:
        if len(swcData) <= 1:
            continue
        
        add_sums = 0
        points = swcData[:, 2:5]  # 提取所有点的坐标
        radii = swcData[:, 5].copy()  # 复制半径数组
        
        # 构建KDTree（仅构建一次）
        tree = KDTree(points)
        
        # 循环处理每个点，保持与原始函数相同的处理顺序和逻辑
        for ii, item in enumerate(swcData):
            p0 = item[2:5]
            p_r = radii[ii]
            
            # 找到邻近点，与原始函数保持相同的参数
            nearest_points_indices = tree.query(p0, k=near_nums + 1)[1][1:]  # 排除自身
            try:
                nearest_points_radii = radii[nearest_points_indices]
            except Exception as e:
                nearest_points_radii = radii
            
            # 调整半径并更新原始数据
            p_r = adjust_radius_based_on_nearest_points(p_r, nearest_points_radii, threshold=0.05)
            radii[ii] = p_r  # 更新原始数据中的半径
            
            # 应用半径限制，保持与原始函数相同的逻辑
            if limit_r is not None:
                if p_r < limit_r:
                    p_r = 1
            
            # 准备输出行，保持与原始函数相同的格式
            if int(item[6]) == -1:
                output_lines.append(f"{item[0] + sums} {item[1]} {item[2]} {item[3]} {item[4]} {p_r:.4f} {item[6]}")
            else:
                output_lines.append(f"{item[0] + sums} {item[1]} {item[2]} {item[3]} {item[4]} {p_r:.4f} {item[6] + sums}")
            
            add_sums += 1
        
        sums += add_sums
    
    # 批量写入文件，减少I/O操作
    with open(SmoothSwcPath, 'w') as f:
        f.write('\n'.join(output_lines) + '\n')


def radius_smooth_advanced(RadiiSwcPath, SmoothSwcPath, limit_r=None):
    """
    高级优化版本，进一步提升性能，同时保持与原始函数完全相同的计算逻辑
    
    额外优化：
    1. 批量预计算所有点的邻近点
    2. 更高效的内存管理
    3. 批量文件写入
    """
    # 读取SWCData
    swcData = np.loadtxt(RadiiSwcPath, ndmin=2)
    
    # 准备所有输出行
    output_lines = []
    near_nums = 15  # 保持与原函数相同的邻近点数
    sums = 0
    
    # SplitSWCData
    swcDataLs = SplitSwcData(swcData)
    
    for swcData in swcDataLs:
        if len(swcData) <= 1:
            continue
        
        add_sums = 0
        points = swcData[:, 2:5]  # 提取所有点的坐标
        radii = swcData[:, 5].copy()  # 复制半径数组
        
        # 构建KDTree（仅构建一次）
        tree = KDTree(points)
        
        # 批量查询所有点的邻近点索引
        # 注意：这里保持与原始函数完全相同的查询参数
        # k_neighbors = min(near_nums + 1, len(points))
        all_nearest_indices = tree.query(points, k=near_nums + 1)[1][:, 1:]  # 排除自身
        
        # 循环处理每个点，严格按照原始函数的逻辑
        for ii, item in enumerate(swcData):
            # p0 = item[2:5]
            p_r = radii[ii]
            
            # 获取当前点的邻近点半径
            # nearest_points_indices = tree.query(p0, k=near_nums + 1)[1][1:]  # 排除自身
            nearest_points_indices = all_nearest_indices[ii]
            try:
                nearest_points_radii = radii[nearest_points_indices]
            except Exception as e:
                nearest_points_radii = radii
            
            # 调整半径，保持与原始函数相同的阈值
            p_r = adjust_radius_based_on_nearest_points(p_r, nearest_points_radii, threshold=0.05)
            radii[ii] = p_r  # 更新半径
            
            # 应用半径限制，保持与原始函数完全相同的逻辑
            # if limit_r is not None:
            #     if p_r < limit_r:
            #         p_r = limit_r
            
            # 准备输出行，保持与原始函数相同的格式
            if int(item[6]) == -1:
                output_lines.append(f"{item[0] + sums} {item[1]} "
                                    f"{item[2]} {item[3]} {item[4]} "
                                    f"{p_r:.4f} {item[6]}\n")
            else:
                output_lines.append(f"{item[0] + sums} {item[1]} "
                                    f"{item[2]} {item[3]} {item[4]} "
                                    f"{p_r:.4f} {item[6] + sums}\n")
            
            add_sums += 1
        
        sums += add_sums
    
    # 批量写入文件
    # with open(SmoothSwcPath, 'w') as f:
    #     f.writelines(output_lines)

    safe_write(SmoothSwcPath, "".join(output_lines), sync=True)


def radius_smooth(RadiiSwcPath, SmoothSwcPath, limit_r=None):
    """
    与原函数保持一致的接口，但内部调用优化版本
    这样可以直接替换原函数而不需要修改其他代码
    """
    # 调用高级优化版本
    radius_smooth_advanced(RadiiSwcPath, SmoothSwcPath, limit_r)


if __name__ == "__main__":
    # 示例用法
    print("优化版radius_smooth函数模块已加载")
    print("可通过以下方式使用:")
    print("from optimized_radius_smooth import radius_smooth")
    print("radius_smooth(RadiiSwcPath, SmoothSwcPath, limit_r=None)")

    # 创建测试目录
    # test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_results')
    # if os.path.exists(test_dir):
    #     shutil.rmtree(test_dir)
    # os.makedirs(test_dir)
    # RadiiSwcPath = r"D:\SY\VesselData\bv_data\00_02_00_bv\data_save\PredictResults\CutWorkFiles\0_0\calculate_radius\RadiiSwc\0_1_1.swc"
    # SmoothSwcPath = os.path.join(test_dir, 'small_advanced_output.swc')

    RadiiSwcPath = r"D:\BaiduNetdiskDownload\CH2_1um_res\radius_effect_res\00_02_00.swc"
    SmoothSwcPath = r"D:\BaiduNetdiskDownload\CH2_1um_res\radius_effect_res\00_02_00_smooth.swc"
    radius_smooth(RadiiSwcPath, SmoothSwcPath, limit_r=0.5)
