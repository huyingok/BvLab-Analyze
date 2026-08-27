import os
import tifffile as tiff


# # root = r"D:\xueguan\TrainDateSet3\train_data\128\data"
# root = r"D:\xueguan\TrainDateSet3\train_data\128\dist"
# dist_path = os.path.join(root, "dist.tif")
# dist_swc_path = os.path.join(root, "dist_swc.swc")
# dist = tiff.imread(dist_path)
# with open(dist_swc_path, "w") as f:
#     count = 0
#     for i in range(dist.shape[0]):
#         for j in range(dist.shape[1]):
#             for k in range(dist.shape[2]):
#                 T = dist[i, j, k]
#                 if T == 255:
#                     f.write(f"{count} 0 {k} {j} {i} 0 {count}\n")
#                     count += 1
# f.close()
# print(count)


# dist_dir = r"D:\xueguan\TrainDateSet3\train_data\128\dist-10"
dist_dir = r"D:\xueguan\TrainDateSet2\train\dist-30"
# dist_swc_dir = r"D:\xueguan\TrainDateSet3\train_data\128\dist_swc"
dist_swc_dir = r"D:\xueguan\TrainDateSet2\train\dist_swc"
os.makedirs(dist_swc_dir, exist_ok=True)
dist_names = os.listdir(dist_dir)
for ii, dist_name in enumerate(dist_names):
    dist_path = os.path.join(dist_dir, dist_name)
    dist = tiff.imread(dist_path)
    dist_swc_path = os.path.join(dist_swc_dir, f"{dist_name.split('.')[0]}.swc")
    with open(dist_swc_path, "w") as f:
        count = 0
        for i in range(dist.shape[0]):
            for j in range(dist.shape[1]):
                for k in range(dist.shape[2]):
                    T = dist[i, j, k]
                    if T > 200:
                        f.write(f"{count} 0 {k} {j} {i} 0 {count}\n")
                        count += 1
    f.close()
    print(ii + 1, dist_name, dist.min(), dist.max(), dist.mean(), dist.std(), count)

