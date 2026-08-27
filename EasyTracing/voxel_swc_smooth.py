# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage, spatial
from typing import List, Tuple
import tifffile


def smooth_swc_with_fixed_points(swc_data: np.ndarray,
                                 iterations: int = 3,
                                 window_size: int = 3) -> np.ndarray:
    """
    使用移动平均平滑SWC骨架，保持点数不变

    Args:
        swc_data: (n, 7)的SWC数据数组
        iterations: 平滑迭代次数
        window_size: 滑动窗口大小（奇数）

    Returns:
        平滑后的SWCData
    """
    smoothed_data = swc_data.copy()

    # 构建节点连接关系
    id_to_index = {row[0]: i for i, row in enumerate(swc_data)}
    children_map = {}
    for row in swc_data:
        parent = row[6]
        if parent > 0:
            children_map.setdefault(parent, []).append(row[0])

    for _ in range(iterations):
        new_positions = []

        for i, row in enumerate(smoothed_data):
            node_id = row[0]
            x, y, z = row[2:5]

            # 收集邻居节点（父节点和子节点）
            neighbors = []

            # 添加父节点
            if row[6] > 0 and row[6] in id_to_index:
                parent_idx = id_to_index[row[6]]
                neighbors.append(smoothed_data[parent_idx][2:5])

            # 添加子节点
            if node_id in children_map:
                for child_id in children_map[node_id]:
                    child_idx = id_to_index[child_id]
                    neighbors.append(smoothed_data[child_idx][2:5])

            if neighbors:
                neighbors = np.array(neighbors)
                # 计算加权平均位置
                new_pos = np.mean(neighbors, axis=0)
                # 与原位置加权混合
                alpha = 0.5  # 混合系数
                smoothed_data[i][2:5] = alpha * new_pos + (1 - alpha) * np.array([x, y, z])

    return smoothed_data


def image_constrained_smoothing(swc_data: np.ndarray,
                                segmentation: np.ndarray,
                                iterations: int = 5) -> np.ndarray:
    """
    在分割图像约束下进行骨架平滑

    Args:
        swc_data: SWCData
        segmentation: 三维分割图像
        iterations: 迭代次数

    Returns:
        平滑后的SWCData
    """
    from skimage.morphology import binary_dilation

    smoothed_data = swc_data.copy()

    for _ in range(iterations):
        for i, row in enumerate(smoothed_data):
            node_id = row[0]
            x, y, z = row[2:5]
            radius = row[5]
            parent_id = row[6]

            # 获取当前点的局部区域
            x_int, y_int, z_int = int(x), int(y), int(z)

            # 确保坐标在图像范围内
            x_int = max(0, min(segmentation.shape[2] - 1, x_int))
            y_int = max(0, min(segmentation.shape[1] - 1, y_int))
            z_int = max(0, min(segmentation.shape[0] - 1, z_int))

            # 提取局部二值区域
            label_value = segmentation[z_int, y_int, x_int]
            if label_value == 0:
                continue

            # 获取该标签的连通分量
            local_mask = (segmentation == label_value)

            # 在当前点周围搜索更中心的位置
            search_radius = max(2, int(radius * 2))
            z_min = max(0, z_int - search_radius)
            z_max = min(segmentation.shape[0], z_int + search_radius + 1)
            y_min = max(0, y_int - search_radius)
            y_max = min(segmentation.shape[1], y_int + search_radius + 1)
            x_min = max(0, x_int - search_radius)
            x_max = min(segmentation.shape[2], x_int + search_radius + 1)

            local_region = local_mask[z_min:z_max, y_min:y_max, x_min:x_max]

            if np.any(local_region):
                # 计算区域中心
                indices = np.argwhere(local_region)
                center = np.mean(indices, axis=0)

                # 转换为全局坐标
                new_z = z_min + center[0]
                new_y = y_min + center[1]
                new_x = x_min + center[2]

                # 温和地更新位置
                smoothed_data[i][2:5] = 0.3 * np.array([new_x, new_y, new_z]) + \
                                        0.7 * np.array([x, y, z])

    return smoothed_data


from scipy.interpolate import splprep, splev


def spline_smooth_swc(swc_data: np.ndarray,
                      segment_ids: List[int] = None,
                      s: float = 0.5) -> np.ndarray:
    """
    使用B样条曲线平滑SWC骨架段

    Args:
        swc_data: SWCData
        segment_ids: 需要平滑的片段IDList（如None则平滑所有连续段）
        s: 平滑参数（0-1之间，越小越平滑）

    Returns:
        平滑后的SWCData
    """
    smoothed_data = swc_data.copy()

    # 如果没有指定片段，找出所有连续段
    if segment_ids is None:
        # 找出所有端点（没有子节点或只有一个父节点的点）
        id_to_index = {row[0]: i for i, row in enumerate(swc_data)}
        children_map = {}
        for row in swc_data:
            parent = row[6]
            if parent > 0:
                children_map.setdefault(parent, []).append(row[0])

        # 找出所有端点和连接点
        segments = []
        visited = set()

        for i, row in enumerate(swc_data):
            node_id = row[0]
            if node_id in visited:
                continue

            # 如果是端点
            if node_id not in children_map or len(children_map[node_id]) == 0:
                # 从端点开始追踪路径
                current = node_id
                segment = []
                while current > 0:
                    if current in visited:
                        break
                    segment.append(current)
                    visited.add(current)

                    # 找到父节点
                    idx = id_to_index[current]
                    parent = swc_data[idx][6]
                    current = parent

                if len(segment) > 2:
                    segments.append(segment)

    # 对每个片段应用样条平滑
    for segment in segments:
        if len(segment) < 3:  # 太短的片段不处理
            continue

        # 获取坐标
        points = []
        for node_id in segment:
            idx = id_to_index[node_id]
            points.append(smoothed_data[idx][2:5])

        points = np.array(points)

        # B样条拟合
        try:
            tck, u = splprep(points.T, s=s, k=min(3, len(points) - 1))

            # 在原始参数位置评估样条
            new_points = splev(u, tck)
            new_points = np.array(new_points).T

            # 更新坐标
            for i, node_id in enumerate(segment):
                idx = id_to_index[node_id]
                smoothed_data[idx][2:5] = new_points[i]
        except:
            # 如果样条拟合失败，跳过该片段
            continue

    return smoothed_data


def complete_swc_smoothing_pipeline(swc_path: str,
                                    segmentation_path: str = None,
                                    output_path: str = None,
                                    methods: List[str] = ['moving_average', 'spline'],
                                    params: dict = None) -> np.ndarray:
    """
    完整的SWC平滑流程

    Args:
        swc_path: SWC文件路径
        segmentation_path: 分割图像路径（Optional）
        output_path: 输出文件路径
        methods: 使用的平滑方法列表
        params: 各方法参数

    Returns:
        平滑后的SWCData
    """
    # 读取SWCFile
    swc_data = np.loadtxt(swc_path)

    if params is None:
        params = {
            'moving_average': {'iterations': 3, 'window_size': 3},
            'image_constrained': {'iterations': 5},
            'spline': {'s': 0.5}
        }

    # 应用平滑方法
    for method in methods:
        if method == 'moving_average':
            swc_data = smooth_swc_with_fixed_points(
                swc_data,
                **params['moving_average']
            )
        elif method == 'spline' and len(swc_data) > 3:
            swc_data = spline_smooth_swc(swc_data, **params['spline'])

    # Optional：基于分割图像的进一步约束
    if segmentation_path and 'image_constrained' in methods:
        try:
            import tifffile
            segmentation = tifffile.imread(segmentation_path)
            swc_data = image_constrained_smoothing(
                swc_data,
                segmentation,
                **params['image_constrained']
            )
        except:
            print("Warning: Could not apply image constraint smoothing")

    # 保存结果
    if output_path:
        np.savetxt(output_path, swc_data, fmt='%d %d %.2f %.2f %.2f %.4f %d')

    return swc_data


# 示例使用
if __name__ == "__main__":
    from pathlib import Path
    from tqdm import tqdm
    import os
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_54"
    root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\pt3"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_54_add"
    root = Path(root)

    swc_dir = root.joinpath("results_voxel_swc")
    save_dir = root.joinpath("results_voxel_smooth")
    os.makedirs(save_dir, exist_ok=True)

    swc_names = [n for n in os.listdir(swc_dir) if ".swc" in n]

    for name in tqdm(swc_names):
        swc_path = swc_dir.joinpath(name)
        save_path = save_dir.joinpath(name)
        # Method1：简单移动平均
        # swc_data = np.loadtxt(swc_path)
        # smoothed = smooth_swc_with_fixed_points(swc_data, iterations=3)

        # Method2：结合分割图像
        # segmentation = tifffile.imread("segmentation.tif")
        # smoothed = image_constrained_smoothing(swc_data, segmentation)

        # Method3：样条平滑
        # smoothed = spline_smooth_swc(swc_data, s=0.3)

        # 完整流程
        smoothed = complete_swc_smoothing_pipeline(
            str(swc_path),
            # segmentation_path=str(),
            output_path=str(save_path),
            # methods=['image_constrained']
            methods=['moving_average', 'spline']
            # methods=['moving_average', 'image_constrained', 'spline']
        )
