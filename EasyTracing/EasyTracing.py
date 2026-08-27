import time

import numpy as np
import os
from pathlib import Path
import tempfile
import tifffile

from EasyTracing.voxel_scoping_centerline import (save_image, dilate_3d_mask, fixed_voxel_scoping,
                                                  remove_small_skeleton_branches, fixed_skeletonize_3d)
from EasyTracing.Skeleton_3D_to_swc import VascularSkeletonExtractor
from EasyTracing.voxel_swc_smooth import complete_swc_smoothing_pipeline

# from voxel_scoping_centerline import (save_image, dilate_3d_mask, fixed_voxel_scoping, remove_small_skeleton_branches,
#                                       fixed_skeletonize_3d)
# from Skeleton_3D_to_swc import VascularSkeletonExtractor
# from voxel_swc_smooth import complete_swc_smoothing_pipeline

from fast_distance_field.fast_distance_field import calculate_distance_field


def voxel_scoping_centerline_tracing(img_path, swc_save_path, extractor, signal_op_dir=None, skeletonize=False,
                                     prune=False, binary=True):
    signal_img = tifffile.imread(img_path)

    signal_img = ((signal_img - signal_img.min()) / (signal_img.max() - signal_img.min()) * 255).astype(np.uint8)
    signal_img[signal_img > 0] = 255

    os.makedirs(os.path.dirname(swc_save_path), exist_ok=True)

    if np.max(signal_img) == 0:  # 空数据
        safe_write(swc_save_path, "", sync=True)
        return

    # 信号数据追踪
    if signal_op_dir is not None:
        signal_op_path = os.path.join(signal_op_dir, Path(img_path).name)
        save_image(signal_img, signal_op_path)
        dilate_3d_mask(signal_op_path, signal_op_path, radius=1, kernel='ball')
        signal_img = tifffile.imread(signal_op_path)
        # signal_img, _ = calculate_distance_field(signal_img, max_iterations=200)
        # signal_img[signal_img < 0.8] = 0
        # signal_img = ((signal_img - signal_img.min()) / (signal_img.max() - signal_img.min()) * 255).astype(np.uint8)
        # signal_img[signal_img > 0] = 255
        # save_image(signal_img, signal_op_path)
    # ApplyVoxel ScopingAlgorithm
    if skeletonize:
        fixed_centerline = fixed_skeletonize_3d(signal_img, prune=prune, binary=binary)
    else:
        fixed_centerline = fixed_voxel_scoping(signal_img, prune=prune)
    # 移除小的孤立骨架段
    fixed_centerline = remove_small_skeleton_branches(fixed_centerline, min_length=5)

    image = fixed_centerline.copy()
    if image.dtype != np.uint8:
        image = 1 * image
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    # tifffile.imwrite("centerline.tif", image, compression="lzw")

    if np.max(signal_img) == 0:  # 空数据
        safe_write(swc_save_path, "", sync=True)
        return

    G = extractor.build_skeleton_graph(image)
    if G.number_of_nodes() == 0:
        # print("Error: Graph construction failed")
        safe_write(swc_save_path, "", sync=True)
        return

    # Convert ToSWC
    swc_nodes = extractor.connected_components_to_swc(G)
    swc_lines = []
    # 头部信息
    swc_lines.append("# SWC format for vascular skeleton\n")
    swc_lines.append("# Created by VascularSkeletonExtractor\n")
    swc_lines.append("# Format: ID Type X Y Z Radius ParentID\n")
    swc_lines.append(f"# Total nodes: {len(swc_nodes)}\n")
    # 节点数据
    for node in swc_nodes:
        node['type'] = 0
        line = (f"{node['id']} {node['type']} "
                f"{node['x']:.3f} {node['y']:.3f} {node['z']:.3f} "
                f"{node['radius']:.3f} {node['parent_id'] if node['parent_id'] > 0 else -1}\n")
        swc_lines.append(line)
    safe_write(swc_save_path, ''.join(swc_lines), sync=True)
    # 平滑处理
    complete_swc_smoothing_pipeline(
        str(swc_save_path),
        # segmentation_path=str(),
        output_path=str(swc_save_path),
        # methods=['image_constrained']
        methods=['moving_average', 'spline']
        # methods=['moving_average', 'image_constrained', 'spline']
    )


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


if __name__ == '__main__':
    extractor = VascularSkeletonExtractor(
        vessel_scale=1.0,
        threshold=0.1,
        min_vessel_length=2
    )

    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\results"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\testData\pre_seg"
    # root = r"D:\GSS\test_mask\mask"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\train2\testData\train_res"
    root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\LARGE_DATA_140\FilterResults\predict_sig"
    # save_dir = os.path.join(Path(root).parent, "results_swc")
    # save_dir = os.path.join(Path(root).parent, "swc")
    save_dir = os.path.join(Path(root).parent, "swc2")
    # signal_op_dir = os.path.join(Path(img_path).parent, "results")
    os.makedirs(save_dir, exist_ok=True)

    signal_op_dir = os.path.join(Path(root).parent, "signal_op")
    os.makedirs(signal_op_dir, exist_ok=True)

    start_time = time.time()

    for filename in [n for n in os.listdir(root) if ".tif" in n]:
        # if filename != "cut_1_3_9_0_0_7_0002_02_02.tif":
        #     continue
        img_path = os.path.join(root, filename)
        # img_path = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\pt6_methods_error\test\images\cut_1_3_9_0_0_7_0002_02_02.tif"
        # img_path = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\results\cut_0_5_6_0_0_6_0000_00_00.tif"
        # save_path = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\pt6_methods_error\test\swc\cut_1_3_9_0_0_7_0002_02_02.swc"
        save_path = os.path.join(save_dir, filename.replace(".tif", ".swc"))

        voxel_scoping_centerline_tracing(img_path, save_path, extractor, signal_op_dir=signal_op_dir, skeletonize=False)
        # voxel_scoping_centerline_tracing(img_path, save_path, extractor, skeletonize=False)

    print(time.time() - start_time)
