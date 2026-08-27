import numpy as np
import os
import tifffile as tiff
from scipy.ndimage import gaussian_filter
import utils


def generate_list(n, k):
    # 创建一个长度为 n 的全零数组
    arr = np.zeros(n, dtype=int)
    # 随机选择 k 个位置设置为 1
    indices = np.random.choice(n, k, replace=False)
    arr[indices] = 1
    return arr.tolist()


def CutData(image, mask, centerline, dist, txy, name, save_dir, save_images_dir, save_masks_dir, save_centerlines_dir, save_dists_dir, train_list, val_list):
    # 切成 8 块
    index0 = image.shape[0] - txy[0]
    index1 = image.shape[1] - txy[1]
    index2 = image.shape[2] - txy[2]
    images = [image[i:i + txy[0], j:j + txy[1], k:k + txy[2]]
              for i in (0, index0) for j in (0, index1) for k in (0, index2)]

    images.append(image[
                  int(index0 / 2):int(index0 / 2) + txy[0],
                  int(index1 / 2):int(index1 / 2) + txy[1],
                  int(index2 / 2):int(index2 / 2) + txy[2]
                  ])

    masks = [mask[i:i + txy[0], j:j + txy[1], k:k + txy[2]]
             for i in (0, index0) for j in (0, index1) for k in (0, index2)]

    masks.append(mask[
                 int(index0 / 2):int(index0 / 2) + txy[0],
                 int(index1 / 2):int(index1 / 2) + txy[1],
                 int(index2 / 2):int(index2 / 2) + txy[2]
                 ])

    centerlines = [centerline[i:i + txy[0], j:j + txy[1], k:k + txy[2]]
                   for i in (0, index0) for j in (0, index1) for k in (0, index2)]

    centerlines.append(centerline[
                       int(index0 / 2):int(index0 / 2) + txy[0],
                       int(index1 / 2):int(index1 / 2) + txy[1],
                       int(index2 / 2):int(index2 / 2) + txy[2]
                       ])

    dists = [dist[i:i + txy[0], j:j + txy[1], k:k + txy[2]]
             for i in (0, index0) for j in (0, index1) for k in (0, index2)]

    dists.append(dist[
                 int(index0 / 2):int(index0 / 2) + txy[0],
                 int(index1 / 2):int(index1 / 2) + txy[1],
                 int(index2 / 2):int(index2 / 2) + txy[2]
                 ])
    # arr = generate_list(len(masks), 6)
    # 检查每块的形状
    for i, m in enumerate(masks):
        if np.max(m) <= 0 or len(m[m > 0]) < 6000:
            print(len(m[m > 0]))
            continue

        # if not (i == 0 or i == len(masks) - 2):
        #     continue

        # if arr[i]:
        #     # 随机翻转
        #     image, mask = random_flip(images[i], m)
        #     if np.random.rand() > 0.5:
        #         # 随机增强，
        #         image = change_img(image)
        #     tiff.imwrite(os.path.join(save_images_dir, name.replace(".tif", f"_{i}_{i}.tif")), image)
        #     tiff.imwrite(os.path.join(save_masks_dir, name.replace(".tif", f"_{i}_{i}.tif")), mask)
        #     if name in train_list:
        #         with open(os.path.join(save_dir, "train.txt"), "a", encoding="utf-8") as f:
        #             f.write(name.replace(".tif", f"_{i}_{i}.tif") + "\n")
        #     elif name in val_list:
        #         with open(os.path.join(save_dir, "val.txt"), "a", encoding="utf-8") as f:
        #             f.write(name.replace(".tif", f"_{i}_{i}.tif") + "\n")
        #     else:
        #         with open(os.path.join(save_dir, "test.txt"), "a", encoding="utf-8") as f:
        #             f.write(name.replace(".tif", f"_{i}_{i}.tif") + "\n")

        # tiff.imwrite(os.path.join(save_images_dir, name.replace(".tif", f"_{i}.tif")), images[i])
        # tiff.imwrite(os.path.join(save_masks_dir, name.replace(".tif", f"_{i}.tif")), m)
        # tiff.imwrite(os.path.join(save_centerlines_dir, name.replace(".tif", f"_{i}.tif")), centerlines[i])
        # if name in train_list:
        #     with open(os.path.join(save_dir, "train.txt"), "a", encoding="utf-8") as f:
        #         f.write(name.replace(".tif", f"_{i}.tif") + "\n")
        # elif name in val_list:
        #     with open(os.path.join(save_dir, "val.txt"), "a", encoding="utf-8") as f:
        #         f.write(name.replace(".tif", f"_{i}.tif") + "\n")
        # else:
        #     with open(os.path.join(save_dir, "test.txt"), "a", encoding="utf-8") as f:
        #         f.write(name.replace(".tif", f"_{i}.tif") + "\n")

        # if i == len(masks) - 2:
        #     # 随机翻转
        #     image, mask = random_flip(images[i], m)
        #     image = change_img(image)
        #     rename = name.replace(".tif", "_1.tif")
        # else:
        #     image, mask = images[i], m
        #     rename = name
        image, mask = images[i], m
        rename = name

        tiff.imwrite(os.path.join(save_images_dir, rename), image)
        tiff.imwrite(os.path.join(save_masks_dir, rename), mask)
        tiff.imwrite(os.path.join(save_centerlines_dir, rename), centerlines[i])
        tiff.imwrite(os.path.join(save_dists_dir, rename), dists[i])

        if name in train_list:
            with open(os.path.join(save_dir, "train.txt"), "a", encoding="utf-8") as f:
                f.write(rename + "\n")
        elif name in val_list:
            with open(os.path.join(save_dir, "val.txt"), "a", encoding="utf-8") as f:
                f.write(rename + "\n")
        else:
            with open(os.path.join(save_dir, "test.txt"), "a", encoding="utf-8") as f:
                f.write(rename + "\n")
        break


def change_txt(root):
    train_txt = os.path.join(root, "train.txt")
    val_txt = os.path.join(root, "val.txt")
    train_list = []
    val_list = []
    with open(train_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            train_list.append(line)
    f.close()
    with open(val_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            val_list.append(line)
    f.close()
    return train_list, val_list


def change_img(image, tyx=(128, 128, 128)):
    aug_choices = np.random.choice([0, 1], 4)
    # aug_choices = [0, 0, 0, 1]

    if aug_choices[0]:
        # 随机生成伽马值并进行伽马校正
        image = gaussian_filter(image, np.random.uniform(0, 2))
    # 随机添加噪声
    if aug_choices[1]:
        bit_shift = int(np.random.triangular(left=0, mode=8, right=14, size=1))
        im = utils.to_16_bit(image)
        # im = utils.to_8_bit(image)
        # imgi[k] = utils.normalize99(im>>bit_shift)
        image = utils.rescale(im >> bit_shift)
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    # 随机进行位移增强
    if aug_choices[2]:
        # 假设 image 是一个 uint8 类型的数组
        border_inds = utils.border_indices(tyx)  # 假设 border_inds 是有效的索引
        # 先将 image Convert To float 类型
        image_float = image.astype(np.float32)
        # 执行乘法操作
        image_float.flat[border_inds] *= np.random.uniform(0, 1)
        # 将结果转换回 uint8 类型
        image = image_float.astype(np.uint8)

        # border_inds = utils.border_indices(tyx)
        # image.flat[border_inds] *= np.random.uniform(0, 1)
    # 随机调整边界像素: 随机将一些像素设置为 0 或 1
    if aug_choices[3]:
        indices = np.random.rand(*tyx) < 0.001
        image[indices] = np.random.choice([0, 1], size=np.count_nonzero(indices))

    return image


def random_flip(image, mask):
    """
    随机翻转三维数据块
    :param data: 三维数据块，形状为 (depth, height, width)
    :return: 翻转后的三维数据块
    """
    a = 0
    while a == 0:
        # 随机选择是否沿X轴翻转
        if np.random.rand() > 0.5:
            a += 1
            image = image[:, :, ::-1]  # 沿X轴翻转
            mask = mask[:, :, ::-1]  # 沿X轴翻转
        # # 随机选择是否沿Y轴翻转
        if np.random.rand() > 0.5:
            a += 1
            image = image[:, ::-1, :]  # 沿Y轴翻转
            mask = mask[:, ::-1, :]  # 沿Y轴翻转
        # 随机选择是否沿Z轴翻转
        if np.random.rand() > 0.5:
            a += 1
            image = image[::-1, :, :]  # 沿Z轴翻转
            mask = mask[::-1, :, :]  # 沿Z轴翻转
    return image, mask


if __name__ == '__main__':
    # root = r'D:\xueguan\TrainDateSet3\train_data'
    root = r'D:\xueguan\TrainDateSet2'
    images_dir = os.path.join(root, 'images')
    # masks_dir = os.path.join(root, 'mask')
    # masks_dir = os.path.join(root, 'mySignal_big')
    # centerline_dir = os.path.join(root, 'centerline-3')
    # dist_dir = os.path.join(root, 'dist-30')
    masks_dir = os.path.join(root, 'Predict_Signal')
    centerline_dir = os.path.join(root, 'centerline-1')
    dist_dir = os.path.join(root, 'Predict-dist')

    # save_dir = os.path.join(root, '128-random')
    save_dir = os.path.join(root, '128-predict')
    os.makedirs(save_dir, exist_ok=True)

    save_images_dir = os.path.join(save_dir, 'images')
    save_masks_dir = os.path.join(save_dir, 'mask')
    save_centerlines_dir = os.path.join(save_dir, 'centerline')
    save_dists_dir = os.path.join(save_dir, 'dist')
    os.makedirs(save_images_dir, exist_ok=True)
    os.makedirs(save_masks_dir, exist_ok=True)
    os.makedirs(save_centerlines_dir, exist_ok=True)
    os.makedirs(save_dists_dir, exist_ok=True)
    images_names = [l for l in os.listdir(images_dir) if (".tif" in l and ".tiff" not in l)]
    txy = (128, 128, 128)

    train_list, val_list = change_txt(root)

    for image_name in images_names:
        try:
            image_path = os.path.join(images_dir, image_name)
            mask_path = os.path.join(masks_dir, image_name)
            centerline_path = os.path.join(centerline_dir, image_name)
            dist_path = os.path.join(dist_dir, image_name)

            image = tiff.imread(image_path)
            mask = tiff.imread(mask_path)
            centerline = tiff.imread(centerline_path)
            dist = tiff.imread(dist_path)

            CutData(image, mask, centerline, dist, txy, image_name, save_dir, save_images_dir, save_masks_dir, save_centerlines_dir, save_dists_dir, train_list, val_list)
        except Exception as e:
            print(e)

    # m = tiff.imread(r'D:\xueguan\TrainDateSet3\train_data\128\images\1_2_0_0.tif')
    # print(len(m[m > 0]))
    # Example
    # flipped_data = random_flip(m)
    # print(flipped_data.shape)  # 输出翻转后的数据形状
    # flipped_data = change_img(m)
    # tiff.imwrite(r"test.tif", flipped_data)
