# -*- coding: utf-8 -*-
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import shutil
import torch
from os.path import join
import os, tifffile
import numpy as np
from DataLoader import GetMultiTypeMemoryDataSetAndCropQxz, GetMultiTypeMemoryDataSetAndCropQxz2
from LossPy import LSDLoss, EvalScore
from torch.utils.data import DataLoader
from models.model import LoadModel


def CellDataBigImgPredict(bigImg, modelPath):
    device = torch.device("cuda:0")
    r = np.array([32, 32, 20])
    rSp = np.array([10, 10, 5])
    batchSize = 1
    imgSize = np.array([272, 272, 112])  # xyz
    bigSize = bigImg.shape[::-1]  # xyz

    # 加载网络
    modelCfg = {
        'name': 'UNet3D',
        # number of input channels to the model
        'in_channels': 4,
        # number of output channels
        'out_channels': 1,
        'fieldSpace': 1,
        # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
        'layer_order': 'gcr',
        # number of features at each level of the U-Net
        'f_maps': [16, 32, 64, 128, 256],
        # 'f_maps': [32, 64, 128, 256, 512],
        # 'f_maps': [32, 64, 128, 256, 512],
        # number of groups in the groupnorm
        'num_groups': 8,
        # apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax
        # this is only relevant during inference, during training the network outputs logits and it is up to the loss function
        # to normalize with Sigmoid or Softmax
        'final_sigmoid': True,
        # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
        'is_segmentation': True
    }
    model = LoadModel(modelCfg, modelPath)
    model.to(device)
    model.eval()
    batchImg = torch.zeros([batchSize, 1, imgSize[2], imgSize[1], imgSize[0]], dtype=torch.float32, device=device)
    coorInfo = [[] for i in range(batchSize)]
    resImg = torch.zeros([bigSize[2], bigSize[1], bigSize[0]], dtype=torch.uint8, device=device)

    sliceNumber = np.ceil((bigSize - imgSize) / (imgSize - r)).astype(np.int32) + 1
    count = 0
    resImg[...] = 0
    with torch.no_grad():
        for nz in range(sliceNumber[2]):
            for ny in range(sliceNumber[1]):
                for nx in range(sliceNumber[0]):
                    sp = (imgSize - r) * [nx, ny, nz]
                    ep = np.min([sp + imgSize, bigSize], axis=0)
                    sp = np.min([sp, ep - imgSize], axis=0)
                    count %= batchSize
                    img = bigImg[sp[2]: ep[2], sp[1]: ep[1], sp[0]: ep[0]].astype(np.float32)
                    img = (img - img.mean()) / img.std()
                    batchImg[count, 0] = torch.from_numpy(img)
                    rsp2 = rSp * np.sign([nx, ny, nz])
                    rep2 = np.sign(bigSize - ep) * rSp
                    sp += rsp2
                    ep -= rep2
                    rep3 = imgSize - rep2
                    coorInfo[count] = [sp[2], ep[2], sp[1], ep[1], sp[0], ep[0], rsp2, rep3]
                    if count == batchSize - 1:
                        segs = model(batchImg)
                        segs = (segs * 255).to(torch.uint8)
                        for coor, seg in zip(coorInfo, segs):
                            # if self.resImg[coor[0]: coor[1], coor[2]: coor[3], coor[4]: coor[5]].shape != seg[0,
                            #                                                                            coor[6][2]:
                            #                                                                            coor[7][2],
                            #                                                                            coor[6][1]:
                            #                                                                            coor[7][1],
                            #                                                                            coor[6][0]:
                            #                                                                            coor[7][
                            #                                                                                0]].shape:
                            #     print('shape error')
                            resImg[coor[0]: coor[1], coor[2]: coor[3], coor[4]: coor[5]] = torch.maximum(
                                resImg[coor[0]: coor[1], coor[2]: coor[3], coor[4]: coor[5]],
                                seg[0, coor[6][2]:coor[7][2], coor[6][1]:coor[7][1], coor[6][0]:coor[7][0]])
                    count += 1
    if count != batchSize:
        with torch.no_grad():
            segs = model(batchImg[:count])
            segs = (segs * 255).to(torch.uint8)
            for coor, seg in zip(coorInfo[:count], segs):
                # self.resImg[coor[0]: coor[1], coor[2]: coor[3], coor[4]: coor[5]] = seg[0, coor[6][2]:,
                #                                                                     coor[6][1]:, coor[6][0]:]
                resImg[coor[0]: coor[1], coor[2]: coor[3], coor[4]: coor[5]] = torch.maximum(
                    resImg[coor[0]: coor[1], coor[2]: coor[3], coor[4]: coor[5]],
                    seg[0, coor[6][2]:coor[7][2], coor[6][1]:coor[7][1], coor[6][0]:coor[7][0]])
    segResImg = resImg.detach().cpu().numpy()
    # segResImg[segResImg < 103] = 0
    segResImg[segResImg < 3] = 0
    return segResImg


def CellDataPredict():
    datPath = r'D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D_DDP\TrainDataSet'
    testTxt = r"val.txt"
    modelPath = r'D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D_DDP\ModelSave\CellTrainModel\exp000\supernet_00051_best_0.2135.pth'
    savePath = r'D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D_DDP\TrainDataSet\predict_eval'
    batchSize = 1
    imgSize = np.array([272, 272, 112], dtype=np.int32)  # xyz
    # device = torch.device('cuda:4')
    device = torch.device('cuda:0')
    imgPath = os.path.join(datPath, "images")
    maskPath = os.path.join(datPath, "mask")
    dataset = GetMultiTypeMemoryDataSetAndCropQxz(datPath, testTxt, imgSize, imgPath, maskPath)
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=0)

    if os.path.isdir(savePath): shutil.rmtree(savePath)
    os.makedirs(savePath)

    # 加载网络
    modelCfg = {
        'name': 'UNet3D',
        # number of input channels to the model
        'in_channels': 4,
        # number of output channels
        'out_channels': 1,
        'fieldSpace': 1,
        # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
        'layer_order': 'gcr',
        # number of features at each level of the U-Net
        'f_maps': [16, 32, 64, 128, 256],
        # 'f_maps': [32, 64, 128, 256, 512],
        # 'f_maps': [24, 48, 96, 192, 384],
        # number of groups in the groupnorm
        'num_groups': 8,
        # apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax
        # this is only relevant during inference, during training the network outputs logits and it is up to the loss function
        # to normalize with Sigmoid or Softmax
        'final_sigmoid': True,
        # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
        'is_segmentation': True
    }
    model = LoadModel(modelCfg, modelPath)
    model.to(device)
    model.eval()
    # Loss
    loss_criterion = LSDLoss()
    lsLen = len(loader)
    # Evaluate
    eval_metric = EvalScore()
    eval_metric.to(device)
    evalLs = []
    # 保存信息
    for kk, (img, mask, name) in enumerate(loader):
        if img.shape[0] != batchSize: continue
        img = img.to(device)

        # mask = tifffile.imread(r'D:\HJ\Soma_Localization\DataPreprocess\HNU_data\data_cut1\test_mask/0026-46_31_58-0_1_0.tif')
        # mask = mask / 255.0
        # mask = np.expand_dims(np.expand_dims(mask, axis=0), axis=0).astype(np.float32)
        # mask = torch.from_numpy(mask)

        mask = mask.to(device)
        with torch.no_grad():
            seg = model(img)
            # loss = loss_criterion(seg, mask)
            eval = eval_metric(seg, mask)
            imgName = os.path.splitext(name[0])[0]
            # if eval.item() < 0.7:
            evalLs.append(eval.item())
            seg2 = (seg * 255).to(torch.uint8).cpu().numpy()[0, 0]
            # seg2[seg2 < 103] = 0
            # seg2[seg2 > 0] = 255
            tifffile.imwrite(join(savePath, imgName + '.tif'), seg2.astype(np.uint8), compression="lzw")

            tmpStr = "%d | %d [Name: %s] [Eval: %f] " % (
                kk + 1, lsLen, imgName, eval
            )
            print(tmpStr[:-1])
            print()
    # print(evalLs)
    print("Min Eval: %f\nMax Eval: %f\nMean Eval: %f\n\n" % (min(evalLs), max(evalLs), np.mean(evalLs)))


def CellDataPredict2():
    # imageDir = r"D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D_DDP\TrainDataSet\images"
    # modelPath = r'D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D_DDP\ModelSave\CellTrainModel\exp000\supernet_00051_best_0.2135.pth'
    # saveDir = r'D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D_DDP\TrainDataSet\predict'

    modelPath = r"D:\SY\10GTestData\10GCellTestData\Cell_model\KS_Pro_00072_best_0.8719.pth"
    imageDir = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\testData\images"
    saveDir = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\testData\Unet_seg"

    batchSize = 1
    # imgSize = np.array([272, 272, 112], dtype=np.int32)
    device = torch.device('cuda:0')
    dataset = GetMultiTypeMemoryDataSetAndCropQxz2(imageDir, os.listdir(imageDir))
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=0)

    if os.path.isdir(saveDir):
        shutil.rmtree(saveDir)
    os.makedirs(saveDir, exist_ok=True)

    # 加载网络
    modelCfg = {
        'name': 'UNet3D',
        # number of input channels to the model
        'in_channels': 4,
        # number of output channels
        'out_channels': 1,
        'fieldSpace': 1,
        # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
        'layer_order': 'gcr',
        # number of features at each level of the U-Net
        'f_maps': [16, 32, 64, 128, 256],
        # 'f_maps': [32, 64, 128, 256, 512],
        # 'f_maps': [24, 48, 96, 192, 384],
        # number of groups in the groupnorm
        'num_groups': 8,
        # apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax
        # this is only relevant during inference, during training the network outputs logits and it is up to the loss function
        # to normalize with Sigmoid or Softmax
        'final_sigmoid': True,
        # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
        'is_segmentation': True
    }
    model = LoadModel(modelCfg, modelPath)
    model.to(device)
    model.eval()

    lsLen = len(loader)

    # 保存信息
    for kk, (img, name) in enumerate(loader):
        start_time = time.time()
        if img.shape[0] != batchSize:
            continue
        img = img.to(device)
        with torch.no_grad():
            seg = model(img)
            imgName = os.path.splitext(name[0])[0]
            seg2 = (seg * 255).to(torch.uint8).cpu().numpy()[0, 0]
            # seg2[seg2 < 103] = 0
            # seg2[seg2 > 0] = 255
            # seg2[seg2 < 3] = 0
            # seg2[seg2 < 63] = 0
            tifffile.imwrite(join(saveDir, f"{imgName}.tif"), seg2.astype(np.uint8), compression="lzw")

            tmpStr = "%d | %d [Name: %s] [Time: %.4f] " % (
                kk + 1, lsLen, imgName, (time.time() - start_time)
            )
            print(tmpStr[:-1])
            # print()


def CellPrediction():
    from ModelPredictPy import ModelPredictClass

    modelPath = r"D:\SY\10GTestData\10GCellTestData\Cell_model\KS_Pro_00072_best_0.8719.pth"
    imageDir = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\testData\images"
    saveDir = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\testData\Unet_seg2"

    device = torch.device('cuda:0')
    model = ModelPredictClass(modelPath, device=device)

    if os.path.isdir(saveDir):
        shutil.rmtree(saveDir)
    os.makedirs(saveDir, exist_ok=True)

    names = [n for n in os.listdir(imageDir) if ".tif" in n]
    lsLen = len(names)

    # 保存信息
    for kk, name in enumerate(names):
        start_time = time.time()
        img = tifffile.imread(os.path.join(imageDir, name))
        if not np.any(img):
            continue
        seg = model(img)
        # seg[seg < 3] = 0

        # seg[seg < 103] = 0

        seg[seg > 103] = 255  # 细胞，神经
        seg[seg < 255] = 0

        tifffile.imwrite(join(saveDir, name), seg, compression="lzw")

        tmpStr = "%d | %d [Name: %s] [Time: %.4f] " % (
            kk + 1, lsLen, name, (time.time() - start_time)
        )
        print(tmpStr[:-1])


if __name__ == '__main__':
    # CellDataPredict()

    startTime = time.time()
    # CellDataPredict2()  # 97

    CellPrediction()  # 99
    print("--- %s seconds ---" % (time.time() - startTime))

    # # model_path = r"D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D_DDP\ModelSave\CellTrainModel\exp000\supernet_00051_best_0.2135.pth"
    # model_path = r"D:\SY\10GTestData\10GCellTestData\Cell_model\KS_Pro_00072_best_0.8719.pth"
    # # images_dir = r"D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D_DDP\TrainDataSet\images"
    # images_dir = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\TrainDataSet\images"
    # save_dir = r"D:\SY\10GTestData\10GCellTestData\Cell_DataSet\TrainDataSet\Unet_seg"
    # os.makedirs(save_dir, exist_ok=True)
    # ls = [l for l in os.listdir(images_dir) if ".tif" in l]
    # lenLs = len(ls)
    # for i, name in enumerate(ls):
    #     start_time = time.time()
    #     path = join(images_dir, name)
    #     save_path = join(save_dir, name)
    #     img = tifffile.imread(path)
    #     segResImg = CellDataBigImgPredict(img, model_path)
    #     tifffile.imwrite(save_path, segResImg, compression="lzw")
    #     print("%s [Name: %s ] [Time: %fs]" % (f"{i + 1} | {lenLs}", name, time.time() - start_time))

'''
1 | 5 [Name: 0030-29_25_26-1_1_0] [Eval: 0.789886

2 | 5 [Name: 0042-9_7_16-0_1_0] [Eval: 0.947854

3 | 5 [Name: 0013-5_17_18-1_0_0] [Eval: 0.819762

4 | 5 [Name: 0023-47_13_20-1_1_0] [Eval: 0.840790

5 | 5 [Name: 0024-19_25_26-0_0_0] [Eval: 0.955436

Min Eval: 0.789886
Max Eval: 0.955436
Mean Eval: 0.870746
'''
