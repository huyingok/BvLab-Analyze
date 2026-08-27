# -*- coding: utf-8 -*-
import time
import numpy as np
import os
from tqdm import tqdm
import tifffile as tiff
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes


def skeletonize_3d(image):
    """
    3D Skeletonization - Lee Algorithm

    Parameters:
    -----------
    image : ndarray
        3D 二值图像

    Returns:
    --------
    skeleton : ndarray
        3D 单像素宽骨架
    """
    binary = image > 0 if image.dtype != bool else image
    skeleton = skeletonize(binary)
    return skeleton


def binarize_3d(image):
    # 假设原先是按深度方向独立填充
    binary_volume = np.stack([binary_fill_holes(slice_2d) for slice_2d in image], axis=0)

    # 再沿高度方向（axis=1）填充
    binary_volume = np.stack([binary_fill_holes(slice_2d) for slice_2d in np.moveaxis(binary_volume, 1, 0)], axis=0)
    binary_volume = np.moveaxis(binary_volume, 0, 1)

    # 再沿宽度方向（axis=2）填充
    binary_volume = np.stack([binary_fill_holes(slice_2d) for slice_2d in np.moveaxis(binary_volume, 2, 0)], axis=0)
    binary_volume = np.moveaxis(binary_volume, 0, 2)

    return binary_volume


def fixed_skeletonize_3d(binary_volume, prune=False, binary=True):
    # 填充中空
    if binary:
        binary_volume = binarize_3d(binary_volume)
    else:
        binary_volume = np.stack([binary_fill_holes(slice_2d) for slice_2d in binary_volume], axis=0)

    centerline = skeletonize_3d(binary_volume)
    centerline[centerline > 0] = 255

    # 填充中空
    if binary:
        centerline = binarize_3d(centerline)
    else:
        centerline = np.stack([binary_fill_holes(slice_2d) for slice_2d in centerline], axis=0)
    centerline = skeletonize_3d(centerline)

    # Postprocessing：连接中心线点
    if np.sum(centerline) > 0:
        if prune:
            centerline = prune_skeleton_advanced(centerline, min_branch_length=11, max_branch_distance=None)
            # 连接中心线点
            centerline = postprocess_centerline(centerline, binary_volume)

            centerline = prune_skeleton_advanced(centerline, min_branch_length=11, max_branch_distance=None)
        else:
            # 连接中心线点
            centerline = postprocess_centerline(centerline, binary_volume)

    return centerline


# 修复后的Voxel ScopingAlgorithm
def fixed_voxel_scoping(binary_volume, distance_threshold=None, angle_threshold=45, prune=False):
    """
    修复后的Voxel ScopingAlgorithm
    参数自适应调整
    """
    # 1. 计算距离变换
    distance_map = distance_transform_edt(binary_volume)

    # 2. 自动确定distance_threshold
    if distance_threshold is None:
        max_distance = np.max(distance_map)
        distance_threshold = min(10, max(5, int(max_distance * 0.7)))
        # print(f"自动设置distance_threshold: {distance_threshold}")

    # 3. 使用简化的但更可靠的方法
    centerline = np.zeros_like(binary_volume, dtype=bool)

    # Method：对于每个体素，检查在多个方向上的对称性
    # 只处理距离值足够大的点
    min_distance_for_center = 0.01
    candidate_mask = (distance_map >= min_distance_for_center) & binary_volume

    candidate_points = np.argwhere(candidate_mask)
    # print(f"候选中心点数: {len(candidate_points)}")

    if len(candidate_points) == 0:
        # print("Warning: 没有候选中心点，尝试降低min_distance_for_center")
        min_distance_for_center = 1.0
        candidate_mask = (distance_map >= min_distance_for_center) & binary_volume
        candidate_points = np.argwhere(candidate_mask)
        # print(f"新候选中心点数: {len(candidate_points)}")

    # cos_threshold = math.cos(math.radians(angle_threshold))

    # 定义要检查的方向
    directions = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                # 只检查主要方向，减少计算量
                if abs(dz) + abs(dy) + abs(dx) == 1:
                    directions.append((dz, dy, dx))

    # directions += [(-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0),
    #                (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
    #                (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1)]

    for (z, y, x) in tqdm(candidate_points):
        # if i % 1000 == 0 and i > 0:
        #     print(f"处理进度: {i}/{len(candidate_points)}")
        distance = distance_map[z, y, x]
        scope_size = min(int(distance), distance_threshold)

        if scope_size < 0.01:
            continue

        # 检查多个方向的对称性
        symmetric_directions = 0
        total_directions = 0

        for dz, dy, dx in directions:
            total_directions += 1

            # 检查正方向
            found_positive = False
            for step in range(1, scope_size + 1):
                nz, ny, nx = z + dz * step, y + dy * step, x + dx * step
                if (0 <= nz < binary_volume.shape[0] and
                        0 <= ny < binary_volume.shape[1] and
                        0 <= nx < binary_volume.shape[2]):

                    if binary_volume[nz, ny, nx]:
                        found_positive = True
                        break

            # 检查负方向
            found_negative = False
            for step in range(1, scope_size + 1):
                nz, ny, nx = z - dz * step, y - dy * step, x - dx * step
                if (0 <= nz < binary_volume.shape[0] and
                        0 <= ny < binary_volume.shape[1] and
                        0 <= nx < binary_volume.shape[2]):

                    if binary_volume[nz, ny, nx]:
                        found_negative = True
                        break

            # 如果正负方向都有体素，说明这个方向是对称的
            if found_positive and found_negative:
                symmetric_directions += 1

        # 如果大多数方向都是对称的，则认为是中心点
        if total_directions > 0 and symmetric_directions / total_directions >= 1.0:
            centerline[z, y, x] = True

    # print(f"找到中心线点: {np.sum(centerline)}")
    # 填充中空

    # Postprocessing：连接中心线点
    if np.sum(centerline) > 0:
        if prune:
            # 连接中心线点
            centerline = postprocess_centerline(centerline, binary_volume)
            # 去除毛刺
            centerline = prune_skeleton_advanced(centerline, min_branch_length=11, max_branch_distance=None)
            # 连接中心线点
            centerline = postprocess_centerline(centerline, binary_volume)
        else:
            centerline = postprocess_centerline(centerline, binary_volume)
    return centerline


def prune_skeleton_advanced(skeleton, min_branch_length=10, max_branch_distance=None):
    """
    增强的骨架修剪：移除短分支，并可选择基于到主干距离的修剪。

    Parameters:
    -----------
    skeleton : ndarray (bool)
        骨架图像。
    min_branch_length : int
        分支最短长度阈值。
    max_branch_distance : float or None
        分支起点到主干的最大距离（基于距离变换），用于识别细长毛刺。若为 None，则只使用长度。

    Returns:
    --------
    pruned : ndarray (bool)
        修剪后的骨架。
    """
    import networkx as nx

    skel = skeleton.astype(bool)
    if np.sum(skel) == 0:
        return skel

    ndim = skel.ndim
    coords = np.argwhere(skel)
    coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))

    # 邻域偏移（与之前相同）
    if ndim == 2:
        offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx == 0 and dy == 0)]
    else:
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                   if not (dx == 0 and dy == 0 and dz == 0)]

    for i, c in enumerate(coords):
        for off in offsets:
            neighbor = tuple(c[j] + off[j] for j in range(ndim))
            if neighbor in coord_to_idx:
                j = coord_to_idx[neighbor]
                if j > i:
                    G.add_edge(i, j)

    endpoints = [node for node in G.nodes if G.degree(node) == 1]
    branch_points = [node for node in G.nodes if G.degree(node) > 2]

    # 如果需要距离信息，预先计算距离变换
    if max_branch_distance is not None:
        # 对原骨架计算距离变换，得到每个像素到背景的距离（值越大表示越远离边界）
        dist_map = distance_transform_edt(~skel)  # 注意：背景为True，前景为False
        # 距离值在骨架像素上表示该像素到背景的最近距离（即局部厚度）

    to_remove = set()
    for ep in endpoints:
        if ep in to_remove:
            continue

        path = []
        current = ep
        prev = None
        length = 0
        while True:
            path.append(current)  # 添加当前节点
            length += 1

            neighbors = list(G.neighbors(current))
            if prev is not None and prev in neighbors:
                neighbors.remove(prev)

            if not neighbors:
                # 到达另一个端点（孤立线段）
                break
            if len(neighbors) > 1:
                # 遇到分支点，Stop，但不将分支点加入 path
                break

            prev = current
            current = neighbors[0]

        # 判断是否删除
        should_remove = False
        if length < min_branch_length:
            should_remove = True
        elif max_branch_distance is not None:
            avg_dist = np.mean([dist_map[tuple(coords[idx])] for idx in path])
            if avg_dist < max_branch_distance:
                should_remove = True

        if should_remove:
            to_remove.update(path)

    pruned = np.zeros_like(skel, dtype=bool)
    for idx, coord in enumerate(coords):
        if idx not in to_remove:
            pruned[tuple(coord)] = True

    return pruned


def postprocess_centerline(centerline, binary_volume):
    """后处理中心线"""
    from scipy import ndimage

    # 1. 移除孤立点
    # labeled, num_features = ndimage.label(centerline)
    # if num_features > 0:
    #     # 计算每个组件的大小
    #     component_sizes = ndimage.sum(centerline, labeled, range(num_features + 1))
    #
    #     # 只保留足够大的组件
    #     min_component_size = max(1, np.sum(centerline) * 0.01)
    #     mask = component_sizes >= min_component_size
    #     centerline = mask[labeled]

    # 2. 形态学操作连接断点
    structure = ndimage.generate_binary_structure(3, 1)

    # 先膨胀连接近邻点
    dilated = ndimage.binary_dilation(centerline, structure, iterations=1)

    # 只保留在原始血管内的膨胀结果
    dilated = dilated & binary_volume

    # 然后细化回单像素宽度
    from skimage.morphology import skeletonize
    centerline_connected = skeletonize(dilated)

    return centerline_connected


def dilate_3d_mask_print(image, radius=1, kernel='ball'):
    """
    对 3-D 二值掩膜图像做形态学膨胀

    Parameters
    ----------
    image : SimpleITK.Image 或 numpy.ndarray
        输入的 3D 二值图像
    radius : int, optional
        膨胀半径（voxel 数），默认为 1
    kernel : str, optional
        结构元素类型：'ball' 或 'box'，默认为 'ball'

    Returns
    -------
    SimpleITK.Image
        膨胀后的二值掩膜（UInt8 类型，0/1）
    """
    # 如果是 numpy Array，Convert To SimpleITK Image
    if isinstance(image, np.ndarray):
        image = sitk.GetImageFromArray(image)

    # 1. 二值化
    mask = image > 0.5

    # 2. 选择核类型
    kernel_type = sitk.sitkBall if kernel == 'ball' else sitk.sitkBox

    # 3. 执行膨胀
    dilated = sitk.BinaryDilate(mask, [radius] * 3, kernel_type)

    # 4. 返回 UInt8 类型的结果
    return sitk.Cast(dilated, sitk.sitkUInt8)


def dilate_3d_mask(src_path, dst_path, radius=1, kernel='ball'):
    """
    对 3-D 二值掩膜 TIF 做形态学膨胀
    radius : 膨胀半径（ voxel 数）
    kernel : 'ball' 或 'box'
    """
    # 1. 读图并二值化
    img = sitk.ReadImage(src_path)
    mask = img > 0.5          # 保证 0/1 二值

    # 2. 执行膨胀
    dilated = sitk.BinaryDilate(mask,
                                [radius] * 3,      # 三个轴半径相同
                                sitk.sitkBall if kernel == 'ball' else sitk.sitkBox)

    # 3. 写回
    sitk.WriteImage(sitk.Cast(dilated, sitk.sitkUInt8),
                    dst_path, useCompression=True)


def remove_small_skeleton_branches(skeleton, min_length=10):
    """
    移除短的骨架分支
    """
    labeled = label(skeleton)
    props = regionprops(labeled)

    mask = np.zeros_like(skeleton, dtype=bool)

    for prop in props:
        if prop.area >= min_length:
            mask[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = True

    return mask


def save_image(image, save_path, th=None):
    if image.dtype != np.uint8:
        image = 1 * image
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    if th is not None:
        image[image < th] = 0
    tiff.imwrite(save_path, image, compression="lzw")


def read_images():
    # 创建或加载三维二值图像（例如血管分割结果）
    # 假设 shape = (depth, height, width)

    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\signal_data_contrast"
    # root = r"D:\LuoJi\20251110\train_data\filter\FilterResults"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_54_add"
    root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\pt3-2"

    # signal_dir = os.path.join(root, "signal")
    # signal_dir = os.path.join(root, "seg")
    # signal_dir = os.path.join(root, "predict")
    signal_dir = os.path.join(root, "predict_res")

    signal_op_dir = os.path.join(root, "signal_op")

    # save_dir = os.path.join(root, "results_voxel2")
    # save_dir = os.path.join(root, "results")
    save_dir = os.path.join(root, "results_voxel")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(signal_op_dir, exist_ok=True)
    names = [n for n in os.listdir(signal_dir) if n.lower().endswith('.tif')]

    for i, name in enumerate(names):
        # if name != '0016-6_6_9-1_2_0.tif':
        # if i != 0:
        #     continue

        print(i + 1, '/', len(names), name)
        signal_path = os.path.join(signal_dir, name)
        save_path = os.path.join(save_dir, name)

        test_volume = tiff.imread(signal_path)
        test_volume[test_volume > 0] = 255

        # signal_op_path = os.path.join(signal_op_dir, name)
        # save_image(test_volume, signal_op_path)
        # dilate_3d_mask(signal_op_path, signal_op_path, radius=1, kernel='ball')
        # test_volume = tiff.imread(signal_op_path)

        # test_volume, _ = calculate_distance_field(test_volume, max_iterations=200)
        # save_image(test_volume, save_path, th=113)
        # test_volume = tiff.imread(signal_op_path)

        if np.max(test_volume) == 0:
            save_image(test_volume, save_path)
            continue
        # ApplyVoxel ScopingAlgorithm
        fixed_centerline = fixed_voxel_scoping(test_volume)
        # 移除小的孤立骨架段
        fixed_centerline = remove_small_skeleton_branches(fixed_centerline, min_length=5)

        save_image(fixed_centerline, save_path)

        # dilate_3d_mask(save_path, save_path, radius=1, kernel='ball')
        # res = tiff.imread(save_path)
        # res, _ = calculate_distance_field(res, max_iterations=200)
        # res[res > 2] = 2
        # # save_image(res, save_path, th=113)
        # save_image(res, save_path)


# 运行测试
if __name__ == "__main__":
    start_time = time.time()
    read_images()
    print(time.time() - start_time)
