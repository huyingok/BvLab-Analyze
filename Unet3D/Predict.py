'''Predict'''
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil
import torch
from models.UnetModel2 import UNet3D
# from models import SuperSegM
from os.path import join
import os, tifffile
import numpy as np
from DataLoader import GetMultiTypeMemoryDataSetAndCropQxz
from LossPy import SmoothL1Loss, LSDLoss, EvalScore
from Util import SoftDiceLoss, BCEFocalLoss
from torch.utils.data import DataLoader
from models.model import LoadModel
from torch import nn as nn
from LossPy import BCEDiceLoss, ComputePR


def JsonSort(name):
    path = r'G:\qxz\DataSet\BrainSegDataSet301\DataSet4\Predict3\info.json'
    savePath = join(os.path.dirname(path), name)
    with open(path, 'r') as f:
        data = json.loads(f.read())
        data = sorted(data, key=lambda x: x[-1].split(' ')[0])
        with open(savePath, 'w') as f:
            f.write(json.dumps(data))

def DataSort(add, data):
    data = sorted(data, key=lambda x: x[-1].split(' ')[0])
    with open(add, 'w') as f:
        f.write(json.dumps(data))


if __name__ == '__main__':
    # datPath = r"D:\NeronDataSet\ZjHospital(0036)Dataset1\SegMentDataSet\DataSetSplit128-3"
    # testTxt = r"test.txt"
    # modelPath = r'./ModelSave/exp001/supernet_00015.pth'
    # savePath = r'D:\NeronDataSet\ZjHospital(0036)Dataset1\SegMentDataSet\DataSetSplit128-3\Predict/'

    datPath = r"D:\mytask\task1\Train_data\all"  # 根目录
    testTxt = r"test.txt"  # 测试集
    modelPath = r"D:\UNet3D\Unet3D\ModelSave\exp012\supernet_000.pth"  # Model
    savePath = r"D:\mytask\task1\Train_data\all\Predict/"  # 预测结果
    imgName = "images"
    maskName = "centerline-3"

    # datPath = r"D:\UNet3D\OriDataSet\correct"
    # testTxt = r"test.txt"
    # modelPath = r'D:\UNet3D\Unet3DLSD-main\ModelSave\exp067\supernet_00.pth'
    # savePath = r'D:\UNet3D\OriDataSet\correct\Predict/'
    # batchSize = 2
    batchSize = 1
    # imgSize = np.array([128, 128, 128], dtype=np.int32)
    imgSize = np.array([192, 192, 192], dtype=np.int32)
    device = torch.device('cuda:0')
    dataset = GetMultiTypeMemoryDataSetAndCropQxz(datPath, testTxt, imgSize, imgName, maskName)
    loader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=0)

    if os.path.isdir(savePath): shutil.rmtree(savePath)
    os.makedirs(savePath)

    # saveResDir = join(savePath, 'result')
    # if os.path.isdir(saveResDir): shutil.rmtree(saveResDir)
    # os.makedirs(saveResDir)

    # saveResDir2 = join(savePath, 'result2')
    # if os.path.isdir(saveResDir2): shutil.rmtree(saveResDir2)
    # os.makedirs(saveResDir2)

    # cfgPath = join(savePath, 'info.json')
    # cfgPath2 = join(savePath, 'info2.json')
    # cfgPath3 = join(savePath, 'info3.json')
    # 加载网络
    modelCfg = {
        'name': 'UNet3D',
        # number of input channels to the model
        'in_channels': 16,
        # number of output channels
        'out_channels': 1,
        # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
        'layer_order': 'gcr',
        # number of features at each level of the U-Net
        # 'f_maps': [16, 32, 64, 128, 256],
        'f_maps': [16, 32, 64, 128, 256, 512],
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
    # 保存信息
    # saveInfos = []
    # saveInfos2 = []
    # saveInfos3 = []
    # sigmod = nn.Sigmoid()
    eval_list = []
    eval_dict = {}
    for kk, (img, mask, name) in enumerate(loader):
        if img.shape[0] != batchSize:
            continue
        img = img.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            seg = model(img)
            # loss = loss_criterion(seg, mask)
            # eval = eval_metric(seg, mask)
            loss = loss_criterion(img, seg, mask)
            eval = eval_metric(img, seg, mask)

            # seg = sigmod(seg)
            seg2 = (seg * 255).to(torch.uint8).cpu().numpy()[0, 0]
            imgName = os.path.splitext(name[0])[0]
            tifffile.imwrite(join(savePath, imgName + '.tif'), seg2)

            tmpStr = "[Name: %s] [Eval: %f]\n" % (
                imgName, eval
            )
            eval_num = float(eval.cpu().numpy())
            eval_list.append(eval_num)
            print(imgName, eval_num)
            eval_dict[imgName] = str(eval_num)
            # saveInfos.append([join(saveResDir, name[0]), join(datPath, 'mask', name[0]), '%3f' % (eval)])
            # with open(cfgPath, 'w') as f:
            #     f.write(json.dumps(saveInfos))
            # saveInfos2.append([join(datPath, 'mask', name[0]), join(datPath, 'images', name[0]), '%3f' % (eval)])
            # with open(cfgPath2, 'w') as f:
            #     f.write(json.dumps(saveInfos2))
            # saveInfos3.append([join(saveResDir, name[0]), join(datPath, 'images', name[0]), '%3f' % (eval)])
            # with open(cfgPath3, 'w') as f:
            #     f.write(json.dumps(saveInfos3))
        if kk % 10 == 0:
            print('%d | %d' % (kk, lsLen))
    print(np.mean(eval_list))
    with open("loss_eval.txt", "w") as f:
        for item in eval_dict:
            f.write(f"{item} {eval_dict[item]}\n")
    # DataSort(cfgPath, saveInfos)
    # DataSort(cfgPath2, saveInfos2)
    # DataSort(cfgPath3, saveInfos3)

'''
1_2_13_0.tif

'''
