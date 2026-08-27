import os
import random
import time

import tifffile as tiff


import numpy as np
import tifffile as tiff
from scipy.ndimage import distance_transform_edt, gaussian_laplace
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# def extract_3d_centerline(image_path, sigma=1.0, threshold=-0.1, min_length=0):
#     # 1. 读取三维 .tif Image
#     image_3d = tiff.imread(image_path)
#     binary_image_3d = (image_3d > 0).astype(np.uint8)  # 假设图像已经二值化，非零为前景
#
#     # 2. Distance transform
#     distance_3d = distance_transform_edt(binary_image_3d)
#
#     # 3. 应用三维高斯拉普拉斯算子 (LoG)
#     log_3d = gaussian_laplace(distance_3d, sigma=sigma)
#
#     # 4. 阈值化提取中心线
#     centerline_3d = (log_3d < threshold).astype(np.uint8)
#
#     # 5. 细化操作（使用 skimage 的 skeletonize）
#     centerline_3d = skeletonize(centerline_3d)
#
#     # 6. 筛选长度较长的线条（三维连通性分析）
#     from scipy.ndimage import label, generate_binary_structure
#     structure = generate_binary_structure(3, 1)  # 三维连通性结构
#     labeled_array, num_features = label(centerline_3d, structure=structure)
#
#     for label in range(1, num_features + 1):
#         component = (labeled_array == label)
#         if np.sum(component) < min_length:
#             centerline_3d[component] = 0
#
#     return centerline_3d
#
# # 测试代码
# if __name__ == "__main__":
#     start = time.time()
#     # image_path = r"D:\xueguan\TrainDateSet3\train_data\test\save\1_2_1_1.tif"  # 替换为你的三维 .tif 图像路径
#     image_path = r"D:\xueguan\TrainDateSet3\train_data\test\centerline\1_2_1_1.tif"  # 替换为你的三维 .tif 图像路径
#     centerline_3d = extract_3d_centerline(image_path)
#     end = time.time()
#     print(end - start)
#
#     # 显示结果（显示一个切片）
#     slice_index = centerline_3d.shape[0] // 2  # 取中间切片
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Original Slice")
#     plt.imshow(tiff.imread(image_path)[slice_index], cmap='gray')
#     plt.subplot(1, 2, 2)
#     plt.title("Extracted Centerline Slice")
#     plt.imshow(centerline_3d[slice_index], cmap='gray')
#     plt.show()
#     centerline_3d = np.float32(centerline_3d)
#     centerline_3d = (((centerline_3d - np.min(centerline_3d)) / (np.max(centerline_3d) - np.min(centerline_3d))) * 255).astype(np.uint8)
#
#     tiff.imwrite(r"D:\xueguan\TrainDateSet3\train_data\test\save\1_2_1_1_cut.tif", centerline_3d)

# 测试代码
# if __name__ == "__main__":
#     image_path = r"D:\xueguan\TrainDateSet3\train_data\test\save\1_2_1_1.tif"  # 替换为你的三维 .tif 图像路径
#     centerline_3d = extract_3d_centerline(image_path)
#
#     tiff.imwrite(r"D:\xueguan\TrainDateSet3\train_data\test\save\1_2_1_1_cut.tif", centerline_3d)

# dist_path = r"D:\xueguan\TrainDateSet3\train_data\128\centerline\1_2_1_1.tif"
# dist_path = r"D:\BaiduNetdiskDownload\exp078\2\mask1.tif"
# dist_path = r"D:\BaiduNetdiskDownload\exp078\2\dist1.tif"

dist_path = r"D:\xueguan\TrainDateSet3\train_data\test\centerline\1_2_1_1.tif"
dist = tiff.imread(dist_path)
# centerline = tiff.imread(centerline_path)

dist[dist > 0] = 255
# dist[dist < 255] = 0
# dist[dist < 85] = 0

# print(len(dist[dist > 0]))
# print(len(dist[dist <= 0]))
# print(len(dist[dist <= 0]) / len(dist[dist > 0]))

# tiff.imwrite(r"D:\xueguan\TrainDateSet3\train_data\128\save\1_2_1_1_cut.tif", dist)
tiff.imwrite(r"pre.tif", dist)
