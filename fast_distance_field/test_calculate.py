# import time
#
# from fast_distance_field import calculate_distance_field
# import tifffile
# import os
# import numpy as np
#
#
# # root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\predict"
# root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\seg"
# save_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\seg_dist_field"
# os.makedirs(save_dir, exist_ok=True)
#
# start = time.time()
#
# for file in os.listdir(root):
#     if ".tif" in file:
#         path = os.path.join(root, file)
#         signal_img = tifffile.imread(path)
#         signal_img, _ = calculate_distance_field(signal_img, max_iterations=200)
#         signal_img = ((signal_img - signal_img.min()) / (signal_img.max() - signal_img.min()) * 255).astype(np.uint8)
#         tifffile.imwrite(os.path.join(save_dir, file), signal_img, compression="lzw")
#
#     print(time.time() - start)


import time
from fast_distance_field import calculate_distance_field
import tifffile
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Optional：用于进度条

# ==================== 配置路径 ====================
# parent_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\X-ray-plate\testData"
# parent_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\X-ray-plate"
# parent_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\X-ray-plate\testData\origin_res"
# parent_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\X-ray-plate\trainData_test"
# parent_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\simulation_data\op"
# parent_dir = r"D:\SY\10GTestData\10GVesselTestData"
# parent_dir = r"D:\SY\10GTestData\10GVesselTestData\Vessel_bv_res\FilterResults"
# parent_dir = r"D:\SY\xueguan\xueguan_512_test"
# parent_dir = r"D:\SY\10GTestData\10GNeuralTestData\Neural_DataSet\testData"
parent_dir = r"D:\SY\10GTestData\10GVesselTestData\Vessel_DataSet\testData"
# parent_dir = r"D:\SY\nnUNet"
# parent_dir = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\testData"
# parent_dir = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\TrainDataSet"
# parent_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\simulation_data"
# parent_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Radius_Data"
# parent_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Radius_Data\op"
# parent_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\X-ray-plate\trainData_test\op"
# parent_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\testData\op"

# root = os.path.join(parent_dir, "predict")
# root = os.path.join(parent_dir, "predict_dist_seg")
# root = os.path.join(parent_dir, "simulation_images")
# root = os.path.join(parent_dir, "Predict")
# root = os.path.join(parent_dir, "signal_op_simulation_new_dist_seg")
# root = os.path.join(parent_dir, "dist_train_res")
# root = os.path.join(parent_dir, "imagesTs_results_postprocessing_tif")
# root = os.path.join(parent_dir, "nnUnet_mask_cr")
# root = os.path.join(parent_dir, "cut_union_results2")
# root = os.path.join(parent_dir, "union_results2")
root = os.path.join(parent_dir, "nnUnet_mask")
# root = os.path.join(parent_dir, "nnUnet_mask_10")
# root = os.path.join(parent_dir, "seg_dist_seg")
# root = os.path.join(parent_dir, "seg_signal")

# save_dir = os.path.join(parent_dir, "predict_dist_field")
# save_dir = os.path.join(parent_dir, "simulation_images_dist")
# save_dir = os.path.join(parent_dir, "predict_dist")
# save_dir = os.path.join(parent_dir, "predict_dist_seg_dist")
# save_dir = os.path.join(parent_dir, "signal_op_simulation_new_dist_seg_dist")
# save_dir = os.path.join(parent_dir, "dist_train_res_dist_field")
# save_dir = os.path.join(parent_dir, "imagesTs_results_postprocessing_tif_dist")
# save_dir = os.path.join(parent_dir, "nnUnet_mask_cr_dist")
# save_dir = os.path.join(parent_dir, "union_results2_dist")
save_dir = os.path.join(parent_dir, "nnUnet_mask_dist")
# save_dir = os.path.join(parent_dir, "nnUnet_mask_10_dist")
# save_dir = os.path.join(parent_dir, "seg_dist_seg_dist")
# save_dir = os.path.join(parent_dir, "seg_signal_dist")

# root = r"D:\ZXQ\C00\filter\FilterResults\predict"
# root = r"D:\ZXQ\C00\filter\FilterResults\dist_train_res"

# root = r"D:\BaiduNetdiskDownload\DATA\predict"
# root = r"D:\BaiduNetdiskDownload\DATA\dist_train_res"

# root = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\CD1-E_no1_iso3um_stitched_segmentation\converted_slices\cut_segmentation"
# root = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\CD1-E_no1_iso3um_stitched_segmentation\converted_slices\predict2"
# root = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\CD1-E_no1_iso3um_stitched_segmentation\converted_slices\dist_train_res"

# root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_9\predict"
# root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\testData2\predict"
# root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\testData2\dist_train_res"
# root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\trainData_40\predict"
# root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\Data_108\predict"
# root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\field_train1\dist_train_res"
# root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_9\dist_train_res"
# root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\trainData_40\dist_train_res"
# root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\Data_108\dist_train_res"
# root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\predict_dist_field_small_size\PredictResults\PredictWorkFiles\seg"
# root = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\train2\testData\images"

# save_dir = r"D:\BaiduNetdiskDownload\DATA\predict_dist_field"
# save_dir = r"D:\BaiduNetdiskDownload\DATA\dist_train_res_dist_field"

# save_dir = r"D:\BaiduNetdiskDownload\images\predict_dist_field"
# save_dir = r"D:\BaiduNetdiskDownload\images\dist_train_res_dist_field"
# save_dir = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\CD1-E_no1_iso3um_stitched_segmentation\converted_slices\cut_segmentation_dist_field"
# save_dir = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\CD1-E_no1_iso3um_stitched_segmentation\converted_slices\predict2_dist_field"
# save_dir = r"D:\ZXQ\CD1-E_no1_iso3um_stitched\CD1-E_no1_iso3um_stitched_segmentation\converted_slices\dist_train_res_dist_field"

# save_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_9\predict_dist_field"
# save_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\testData2\predict_dist_field"
# save_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\testData2\dist_train_res_dist_field"
# save_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\trainData_40\predict_dist_field"
# save_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\Data_108\predict_dist_field"
# save_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\field_train1\dist_train_res_dist_field"
# save_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_9\dist_train_res_dist_field"
# save_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\trainData_40\dist_train_res_dist_field"
# save_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\LARGE_DATA_140\filter\Data_108\dist_train_res_dist_field"
# save_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\predict_dist_field_small_size\PredictResults\PredictWorkFiles\seg_dist_field"
# save_dir = r"D:\SY\xueguan\Voxel_To_Model_Lunwen\Data\Data_108\ml_train\train2\testData\images_dist_field"
os.makedirs(save_dir, exist_ok=True)


# ==================== 处理函数 ====================
def process_single_file(args):
    """处理单个文件（必须在顶层定义以支持pickleSerialization）"""
    file, root_dir, save_dir = args
    try:
        path = os.path.join(root_dir, file)
        signal_img = tifffile.imread(path)
        signal_img, _ = calculate_distance_field(signal_img, max_iterations=200)
        signal_img = ((signal_img - signal_img.min()) / (signal_img.max() - signal_img.min()) * 255).astype(np.uint8)
        tifffile.imwrite(os.path.join(save_dir, file), signal_img, compression="lzw")
        return True, file, None
    except Exception as e:
        return False, file, str(e)


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 获取所有tifFile
    tif_files = [f for f in os.listdir(root) if ".tif" in f.lower()]
    total = len(tif_files)
    print(f"📁 Found {total} TIFF files")

    # 准备参数列表
    args_list = [(f, root, save_dir) for f in tif_files]

    # 设置进程数（根据CPU和内存调整）
    num_workers = min(cpu_count(), 6)  # 建议6个，避免内存不足
    print(f"🚀 Starting {num_workers} processes for parallel processing...\n")

    start = time.time()
    success_count = 0
    failed_files = []

    # 多进程执行
    with Pool(processes=num_workers) as pool:
        # 使用imap显示实时进度
        for i, (success, file, error) in enumerate(pool.imap(process_single_file, args_list), 1):
            if success:
                success_count += 1
                print(f"[{i}/{total}] ✓ {file}")
            else:
                failed_files.append((file, error))
                print(f"[{i}/{total}] ✗ {file}: {error}")

    # Statistical results
    elapsed = time.time() - start
    print(f"\n{'=' * 50}")
    print(f"✅ Success: {success_count}/{total}")
    if failed_files:
        print(f"❌ Failed: {len(failed_files)} 个")
        for f, e in failed_files:
            print(f"   - {f}: {e}")
    print(f"⏱️  Total time: {elapsed:.2f} Seconds")
    print(f"⚡  Average speed: {total / elapsed:.2f} files/second")
    print(f"{'=' * 50}")
