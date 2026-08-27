import time
import tifffile as tiff
import torch
import eikonal
import os
import skimage.io
import numpy as np
# from src.cellpose_omni import plot


# def dist_grad():
#     nclasses = 3
#     root = r"D:\omnipose-main2\train3d\tests\images\dist2"
#     result_dir = r"D:\omnipose-main2\train3d\tests\images\grad2"
#     os.makedirs(result_dir, exist_ok=True)
#     dist_names = [l for l in os.listdir(root) if (".tif" in l and ".tiff" not in l)]
#     dist_list = []
#     for ii, image_name in enumerate(dist_names):
#         image_path = os.path.join(root, image_name)
#         image = skimage.io.imread(image_path)
#         image = np.transpose(image, (2, 1, 0))
#         C = 1  # 假设通道数为 1
#         image = np.expand_dims(image, axis=0)  # 添加两个维度
#         image = np.tile(image, (C, 1, 1, 1))  # 复制数据
#         dist_list.append(image)
#
#     data = torch.tensor(dist_list)
#     gradient = eikonal.gradient_from_eikonal(data)
#     data = data.cpu().detach().numpy()
#     gradient = gradient.cpu().detach().numpy()
#
#     for i in range(len(dist_names)):
#         print(dist_names[i])
#         grad_path = os.path.join(result_dir, dist_names[i].replace(".tif", "_grad.tif"))
#         grad = gradient[i]
#         grad = np.transpose(grad, (0, 3, 2, 1))
#
#         tiff.imwrite(grad_path, np.transpose(grad, (1, 2, 3, 0)))
#         tiff.imwrite(os.path.join(result_dir, dist_names[i].replace(".tif", "_grad_color.tif")),
#                      plot.dx_to_circ(grad, transparency=True)
#                      if nclasses > 1 else np.zeros(
#                          data[i].shape + (3 + True,), np.uint8)
#                      )
#         g = tiff.imread(grad_path)
#         g = g[:, :, :, 0] + g[:, :, :, 1] + g[:, :, :, 2]
#         gint8 = ((g - g.min()) / (g.max() - g.min()) * 255).astype(np.uint8)
#
#         gint8_path = os.path.join(result_dir, dist_names[i].replace(".tif", "_grad_uint8.tif"))
#         tiff.imwrite(gint8_path, gint8)
#
#
# if __name__ == "__main__":
#     dist_grad()


def dist_grad():
    nclasses = 3

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # root = r"D:\xueguan\TrainDateSet3\Train2\test2\folder"
    # root = r"D:\omnipose-main\train3d"
    # root = r"D:\xueguan\TrainDateSet3\Train2\test2\docs\test_files"

    # root = r"D:\xueguan\TrainDateSet3\train_data\test"
    # root = r"D:\xueguan\TrainDateSet3\train_data\128\78\1"
    # root = r"D:\xueguan\TrainDateSet3\train_data\128\data"
    # root = r"D:\xueguan\TrainDateSet3\train_data\128\test_dist"
    # root = r"D:\xueguan\TrainDateSet3\train_data\ttt"
    root = r"D:\xueguan\TrainDateSet2\train"

    # img_dir = os.path.join(root, "images")  # Image
    # mask_dir = os.path.join(root, "masks")  # Tag
    # mask_dir = os.path.join(root, "masks_big")  # Tag
    mask_dir = os.path.join(root, "mySignal_big")  # Tag
    # mask_dir = os.path.join(root, "mask")  # Tag
    # mask_dir = root
    result_dir = os.path.join(root, "results2")  # Result
    result_uint8_dir = os.path.join(root, "results2_uint8")  # Result
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_uint8_dir, exist_ok=True)

    # images_names = [l for l in os.listdir(img_dir) if (".tif" in l and ".tiff" not in l)]
    # images_names = ["net_seg1.tif"]
    # images_names = ["mask.tif"]
    images_names = ["merge3_4_6_0.tif"]

    train_images = []
    train_masks = []

    start = time.time()

    for ii, image_name in enumerate(images_names):
        # image_path = os.path.join(img_dir, image_name)
        # image = skimage.io.imread(image_path)
        # train_images.append(image)

        mask_path = os.path.join(mask_dir, image_name)
        mask = skimage.io.imread(mask_path)
        mask = np.transpose(mask, (2, 1, 0))

        # 添加维度，并扩展大小
        C = 1  # 假设通道数为 1
        mask = np.expand_dims(mask, axis=0)  # 添加两个维度
        mask = np.tile(mask, (C, 1, 1, 1))  # 复制数据
        # mask = np.array(mask, dtype=np.float32)

        train_masks.append(mask)

        print(mask.shape)  # Output: (1, Z, Y, X)

    train_masks = torch.tensor(train_masks, device=device)  # 直接将数据移动到 GPU
    print(train_masks.shape)  # [B, C, Z, Y, X]

    # 计算欧几里得距离场
    distance = eikonal.solve_eikonal(train_masks, eps=1e-5, min_steps=200, use_triton=False)

    # distance = r"D:\xueguan\TrainDateSet3\train_data\ttt\merge2_2_5_0.tif"
    # distance = skimage.io.imread(distance)
    # distance = np.transpose(distance, (2, 1, 0))
    # C = 1  # 假设通道数为 1
    # distance = np.expand_dims(distance, axis=0)  # 添加两个维度
    # distance = np.tile(distance, (C, C, 1, 1, 1))  # 复制数据
    # distance = torch.tensor(distance, device=device)  # 直接将数据移动到 GPU
    # distance = distance.float()
    gradient = eikonal.gradient_from_eikonal(distance)

    print(time.time() - start)

    # distance = distance.squeeze(1)  # 指定移除第 1 维
    # gradient = gradient.squeeze(1)  # 指定移除第 1 维

    print("Distance shape:", distance.shape)
    print("Gradient shape:", gradient.shape)

    distance = distance.cpu().detach().numpy()
    gradient = gradient.cpu().detach().numpy()

    dist_list = []
    grad_list = []

    for i in range(len(images_names)):
        dist_path = os.path.join(result_dir, images_names[i].replace(".tif", "_dist.tif"))
        grad_path = os.path.join(result_dir, images_names[i].replace(".tif", "_grad.tif"))

        dist = distance[i]
        grad = gradient[i]

        dist = np.transpose(dist, (0, 3, 2, 1))
        grad = np.transpose(grad, (0, 3, 2, 1))

        tiff.imwrite(dist_path, np.transpose(dist, (1, 2, 3, 0)))
        tiff.imwrite(grad_path, np.transpose(grad, (1, 2, 3, 0)))
        tiff.imwrite(os.path.join(result_dir, images_names[i].replace(".tif", "_grad_color.tif")),
                     plot.dx_to_circ(grad, transparency=True)
                     if nclasses > 1 else np.zeros(
                         dist.shape + (3 + True,), np.uint8)
                     )

        d = tiff.imread(dist_path)
        d = np.squeeze(d, axis=3)
        dint8 = ((d - d.min()) / (d.max() - d.min()) * 255).astype(np.uint8)

        g = tiff.imread(grad_path)
        g = g[:, :, :, 0] + g[:, :, :, 1] + g[:, :, :, 2]
        gint8 = ((g - g.min()) / (g.max() - g.min()) * 255).astype(np.uint8)

        dint8_path = os.path.join(result_uint8_dir, images_names[i].replace(".tif", "_dist_uint8.tif"))
        gint8_path = os.path.join(result_uint8_dir, images_names[i].replace(".tif", "_grad_uint8.tif"))
        tiff.imwrite(dint8_path, dint8)
        tiff.imwrite(gint8_path, gint8)


def dist_grad2():
    nclasses = 3
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = r"D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D_DDP\TrainDataSet"
    mask_dir = os.path.join(root, "mask")  # Tag
    result_dir = os.path.join(root, "dist")  # Result
    os.makedirs(result_dir, exist_ok=True)
    images_names = [l for l in os.listdir(mask_dir) if (".tif" in l and ".tiff" not in l)][:1]
    train_masks = []

    for ii, image_name in enumerate(images_names):
        # image_path = os.path.join(img_dir, image_name)
        # image = skimage.io.imread(image_path)
        # train_images.append(image)
        mask_path = os.path.join(mask_dir, image_name)
        mask = skimage.io.imread(mask_path)
        mask = np.transpose(mask, (2, 1, 0))
        # 添加维度，并扩展大小
        C = 1  # 假设通道数为 1
        mask = np.expand_dims(mask, axis=0)  # 添加两个维度
        mask = np.tile(mask, (C, 1, 1, 1))  # 复制数据
        # mask = np.array(mask, dtype=np.float32)
        train_masks.append(mask)
        # print(mask.shape)  # Output: (1, Z, Y, X)

    train_masks = torch.tensor(np.array(train_masks), device=device)  # 直接将数据移动到 GPU
    for i in range(len(train_masks)):

        if len(train_masks) == i + 1:
            t = train_masks[i:]
        else:
            t = train_masks[i:i+1]
        start = time.time()
        # 计算欧几里得距离场
        distance = eikonal.solve_eikonal(t, eps=1e-5, min_steps=200, use_triton=False)
        print(i, time.time() - start)
        distance = distance.cpu().detach().numpy()
        dist = distance[0]
        dist = np.transpose(dist, (0, 3, 2, 1))
        dist = np.transpose(dist, (1, 2, 3, 0))
        d = np.squeeze(dist, axis=3)
        dint8 = ((d - d.min()) / (d.max() - d.min()) * 255).astype(np.uint8)
        dint8_path = os.path.join(result_dir, images_names[i])
        tiff.imwrite(dint8_path, dint8)


if __name__ == "__main__":
    dist_grad2()
