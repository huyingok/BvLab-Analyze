from multiprocessing import Pool, cpu_count
import time
import numpy as np
import os
from pathlib import Path
import tempfile
import tifffile
from functools import partial
import gc
import atexit
import signal
from voxel_scoping_centerline import (save_image, dilate_3d_mask, fixed_voxel_scoping, remove_small_skeleton_branches,
                                      fixed_skeletonize_3d)
from voxel_swc_smooth import complete_swc_smoothing_pipeline

from fast_distance_field.fast_distance_field import calculate_distance_field

# 将临时文件目录设置到D盘，避免C盘空间占用
TEMP_DIR = r"D:\Temp"
os.makedirs(TEMP_DIR, exist_ok=True)
tempfile.tempdir = TEMP_DIR
os.environ['TMP'] = TEMP_DIR
os.environ['TEMP'] = TEMP_DIR


def cleanup_temp_dir():
    """程序退出时清理临时目录"""
    try:
        if os.path.exists(TEMP_DIR) and os.path.isdir(TEMP_DIR):
            for temp_file in os.listdir(TEMP_DIR):
                try:
                    temp_path = os.path.join(TEMP_DIR, temp_file)
                    if os.path.isfile(temp_path):
                        os.remove(temp_path)
                except:
                    pass
            try:
                os.rmdir(TEMP_DIR)
            except:
                pass
    except:
        pass


def force_memory_cleanup():
    """强制清理所有可能的内存泄漏源"""
    try:
        # 清理SimpleITK对象缓存
        import SimpleITK as sitk
        sitk.ProcessObject_AbortAll()
    except:
        pass
    
    try:
        # 清理networkxCache
        import networkx as nx
        nx.clear_cachedfunctions()
    except:
        pass
    
    # 强制运行完整GC
    gc.collect(2)
    
    try:
        # 尝试释放内存给OS
        import ctypes
        ctypes.CDLL('msvcrt.dll')._heapmin()
    except:
        pass


def init_worker():
    """子进程初始化函数"""
    # 在子进程中也设置临时目录
    tempfile.tempdir = TEMP_DIR
    os.environ['TMP'] = TEMP_DIR
    os.environ['TEMP'] = TEMP_DIR
    # 更激进的GCSettings，更频繁回收
    gc.set_threshold(30, 3, 3)
    # DisableSimpleITKMultithreaded，减少内存碎片
    try:
        import SimpleITK as sitk
        sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    except:
        pass
    # 初始化时清理一次内存
    force_memory_cleanup()


# 注册退出处理函数
atexit.register(cleanup_temp_dir)
# 处理中断信号
try:
    signal.signal(signal.SIGTERM, lambda s, f: cleanup_temp_dir())
    signal.signal(signal.SIGINT, lambda s, f: cleanup_temp_dir())
except:
    pass


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


def voxel_scoping_centerline_tracing(img_path, swc_save_path, extractor, signal_op_dir=None, skeletonize=False,
                                     prune=True, binary=True):
    signal_img = tifffile.imread(img_path)

    # if skeletonize:
    #     signal_img[signal_img < 103] = 0

    signal_img = ((signal_img - signal_img.min()) / (signal_img.max() - signal_img.min()) * 255).astype(np.uint8)
    signal_img[signal_img > 0] = 255

    os.makedirs(os.path.dirname(swc_save_path), exist_ok=True)

    if np.max(signal_img) == 0:  # 空数据
        safe_write(swc_save_path, "", sync=True)
        del signal_img
        force_memory_cleanup()
        return

    # 信号数据追踪
    if signal_op_dir is not None:
        signal_op_path = os.path.join(signal_op_dir, Path(img_path).name)
        save_image(signal_img, signal_op_path)
        dilate_3d_mask(signal_op_path, signal_op_path, radius=2, kernel='ball')
        signal_img = tifffile.imread(signal_op_path)
        # signal_img, dist_field = calculate_distance_field(signal_img, max_iterations=200)
        # del dist_field
        # # signal_img[signal_img < 0.8] = 0
        # signal_img = ((signal_img - signal_img.min()) / (signal_img.max() - signal_img.min()) * 255).astype(np.uint8)
        # signal_img[signal_img > 0] = 255
        # save_image(signal_img, signal_op_path)
        force_memory_cleanup()

    # ApplyVoxel ScopingAlgorithm
    if skeletonize:
        fixed_centerline = fixed_skeletonize_3d(signal_img, prune=prune, binary=binary)
    else:
        fixed_centerline = fixed_voxel_scoping(signal_img, prune=prune)

    # signal_img不再需要，立即删除
    del signal_img
    force_memory_cleanup()

    # fixed_centerline = signal_img

    # 移除小的孤立骨架段
    fixed_centerline = remove_small_skeleton_branches(fixed_centerline, min_length=5)

    image = fixed_centerline.copy()
    # fixed_centerline在这里之后可以删除
    del fixed_centerline
    force_memory_cleanup()
    
    if image.dtype != np.uint8:
        image = 1 * image
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    # tifffile.imwrite("centerline.tif", image, compression="lzw")

    if np.max(image) == 0:  # 空数据
        safe_write(swc_save_path, "", sync=True)
        del image
        force_memory_cleanup()
        return

    G = extractor.build_skeleton_graph(image)
    # image在这里之后可以删除
    del image
    force_memory_cleanup()
    
    if G.number_of_nodes() == 0:
        # print("Error: Graph construction failed")
        safe_write(swc_save_path, "", sync=True)
        del G
        force_memory_cleanup()
        return

    # Convert ToSWC
    swc_nodes = extractor.connected_components_to_swc(G)
    # G在这里之后可以删除
    del G
    force_memory_cleanup()
    
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
    # swc_nodes在这里之后可以删除
    del swc_nodes
    force_memory_cleanup()
    
    safe_write(swc_save_path, ''.join(swc_lines), sync=True)
    del swc_lines
    
    # 平滑处理
    complete_swc_smoothing_pipeline(
        str(swc_save_path),
        # segmentation_path=str(),
        output_path=str(swc_save_path),
        # methods=['image_constrained']
        methods=['moving_average', 'spline']
        # methods=['moving_average', 'image_constrained', 'spline']
    )
    # 最终彻底清理
    force_memory_cleanup()


def process_single_file(filename, root, save_dir, signal_op_dir, extractor_params, skeletonize=False, prune=True,
                        binary=True):
    """单个文件的处理函数，将在子进程中运行"""
    if not filename.endswith(".tif"):
        return
    
    # 重新创建 extractor Instance（避免 pickling Issue）
    from Skeleton_3D_to_swc import VascularSkeletonExtractor  # 替换为实际导入
    extractor = VascularSkeletonExtractor(**extractor_params)

    img_path = os.path.join(root, filename)
    save_path = os.path.join(save_dir, filename.replace(".tif", ".swc"))

    # if not os.path.exists(save_path):
    # 调用核心处理函数
    voxel_scoping_centerline_tracing(
        img_path, save_path, extractor,
        signal_op_dir=signal_op_dir,
        skeletonize=skeletonize,
        prune=prune,
        binary=binary
    )
    # 清理内存
    del extractor
    force_memory_cleanup()


def main():
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\LARGE_DATA_140\FilterResults\predict_sig"
    # root = r"D:\GSS\test_mask\mask"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_9\dist_train_res2"
    # root = r"D:\BaiduNetdiskDownload\test\dist_train_res2"
    # root = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\images\dist_train_res2"
    # root = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\CD1-E_no1_iso3um_stitched_segmentation\converted_slices\cut_segmentation"
    # root = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\CD1-E_no1_iso3um_stitched_segmentation\converted_slices\bv_data_cut"
    # root = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\images\cut_dist_train_res2"
    # root = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\CD1-E_no1_iso3um_stitched_skeleton\converted_slices_skeleton\cut_skeleton"
    # root = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\CD1-E_no1_iso3um_stitched_segmentation\converted_slices\cut_segmentation"
    # root = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\CD1-E_no1_iso3um_stitched_segmentation\converted_slices\dist_train_res2"
    # root = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\CD1-E_no1_iso3um_stitched_skeleton\converted_slices_skeleton\cut_skeleton"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\X-ray-plate\cut_skeleton"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\X-ray-plate\testData\dist_train_res2"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\X-ray-plate\testData\origin_res\predict_seg_seg"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\X-ray-plate\trainData_test\origin_res\predict_seg_seg"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\X-ray-plate\trainData_test\new_res\dist_train_res2"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\trainData_40\new_res\dist_train_res2"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\trainData_40\origin_res\predict_seg_seg"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\Data_108\origin_res\predict_seg_seg"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\testData\origin_res\predict_seg_seg"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\testData\new_res\dist_train_res2"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\simulation_data\signal_op_simulation_new_dist_seg_dist_seg"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\simulation_data\simulation_images"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\X-ray-plate\trainData_test\op\predict_dist_seg_dist_seg"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\simulation_data\op\predict_dist_seg"
    # root = r"D:\SY\10GTestData\10GVesselTestData\predict_dist_seg_dist_seg"
    # root = r"D:\SY\10GTestData\10GVesselTestData\Vessel_bv_res\FilterResults\predict_dist_seg_dist_seg"
    # root = r"D:\SY\xueguan\xueguan_512_test\union_results"
    # root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\testData\nnUnet_mask_dist"
    # root = r"D:\SY\10GTestData\10GVesselTestData\Vessel_DataSet\testData\nnUnet_mask"
    # root = r"D:\SY\10GTestData\10GVesselTestData\Vessel_DataSet\testData\nnUnet_mask_dist"
    # root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\testData\Unet_seg_73"
    # root = r"D:\BaiduNetdiskDownload\val\Predict_label2"
    root = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\TrainDataSet\mask_cr2"
    # root = r"D:\SY\10GTestData\10GVesselTestData\big_image_o\images"
    # root = r"D:\SY\10GTestData\10GVesselTestData\Vessel_bv_res\FilterResults\predict_seg_seg"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\X-ray-plate\trainData_test\new_res\dist_train_res2"
    # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\X-ray-plate\trainData_test\origin_res_2\predict_seg_seg"

    # save_dir = os.path.join(Path(root).parent, "swc2")
    # save_dir = os.path.join(Path(root).parent, "nnUnet_swc")
    # save_dir = os.path.join(Path(root).parent, "Unet_swc_73")
    # save_dir = os.path.join(Path(root).parent, "Predict_label2_swc")
    save_dir = os.path.join(Path(root).parent, "mask_cr2_swc")
    # save_dir = os.path.join(Path(root).parent, "cut_swc")
    # save_dir = os.path.join(Path(root).parent, "new_swc")
    # save_dir = os.path.join(Path(root).parent, "origin_swc")
    # save_dir = os.path.join(Path(root).parent, "cut_skeleton_swc")
    os.makedirs(save_dir, exist_ok=True)

    # signal_op_dir = os.path.join(Path(root).parent, "signal_op")
    # signal_op_dir = os.path.join(Path(root).parent, "signal_op_new")
    # signal_op_dir = os.path.join(Path(root).parent, "signal_op_simulation_new")
    # os.makedirs(signal_op_dir, exist_ok=True)
    signal_op_dir = None

    # extractor Parameter（固定）
    extractor_params = {
        "vessel_scale": 1.0,
        "threshold": 0.1,
        "min_vessel_length": 2
    }

    # 获取所有 .tif 文件列表
    tif_files = [f for f in os.listdir(root) if f.endswith(".tif")]

    start_time = time.time()

    # 清理临时目录中遗留的临时文件
    os.makedirs(TEMP_DIR, exist_ok=True)
    if os.path.exists(TEMP_DIR) and os.path.isdir(TEMP_DIR):
        for temp_file in os.listdir(TEMP_DIR):
            try:
                temp_path = os.path.join(TEMP_DIR, temp_file)
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
            except:
                pass

    # 创建进程池，每个子进程处理2个文件后重启，彻底防止内存泄漏
    num_workers = min(cpu_count() - 2, 6)  # 进一步限制最大进程数
    with Pool(processes=num_workers, maxtasksperchild=2, initializer=init_worker) as pool:
        # 使用 partial 固定公共参数
        worker = partial(
            process_single_file,
            root=root,
            save_dir=save_dir,
            signal_op_dir=signal_op_dir,
            extractor_params=extractor_params,
            skeletonize=True,
            prune=False,
            binary=False
        )
        # 使用imap逐个处理，减少内存压力
        list(pool.imap_unordered(worker, tif_files, chunksize=1))

    # 最终清理临时目录
    if os.path.exists(TEMP_DIR) and os.path.isdir(TEMP_DIR):
        for temp_file in os.listdir(TEMP_DIR):
            try:
                temp_path = os.path.join(TEMP_DIR, temp_file)
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
            except:
                pass
        try:
            os.rmdir(TEMP_DIR)
        except:
            pass

    print(f"Multi-process processing completed, time elapsed: {time.time() - start_time:.2f} Seconds")


if __name__ == "__main__":
    main()

