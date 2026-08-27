import os, importlib
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from Seg_Net_main_PyQt.models.model import LoadModel
from torch.utils.data import DataLoader
from Seg_Net_main_PyQt.DataLoader import GetMultiTypeMemoryDataSetAndCropQxz
import numpy as np
import torch, os
from tensorboardX import SummaryWriter
from Seg_Net_main_PyQt.Net import Trainer
from Seg_Net_main_PyQt.MyUtil import GetLossOptimiLr
from pathlib import Path
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

'''
cmd
activate QxzDeep
cd logs
tensorboard --logdir "./" --host=0.0.0.0
'''


from PyQt5.QtCore import QThread, pyqtSignal


class Train2(QThread):
    finish = pyqtSignal(int)
    error2 = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        super(Train2, self).__init__()
        self.win = kwargs.get('win')
        self.netObj2 = None
        self.stop_2 = False

    def run(self):
        try:
            self.netObj2 = None
            self.stop_2 = False
            rootPath = self.win.t2_dict['rootPath']
            imgPath = self.win.t2_dict['imgPath']
            maskPath = self.win.t2_dict['maskPath']
            distPath = self.win.t2_dict['distPath']
            shapes = self.win.t2_dict['shapes']
            batchSize = self.win.t2_dict['batchSize']
            epoch = self.win.t2_dict['epoch']
            th = self.win.t2_dict['th']

            imageName = str(Path(imgPath).name)
            maskName = str(Path(maskPath).name)
            distName = str(Path(distPath).name)

            # rootPath = r'D:\xueguan\TrainDateSet2\128-predict'  # data path
            # imageName = "images"
            # maskName = "centerline"
            # distName = "dist"
            # batchSize = 2
            # epoch = 2
            # shapes = [128, 128, 128]

            trainTxt = "train.txt"  # txt file for training
            valTxt = "val.txt"  # txt file for validation
            imgSize = np.array(shapes, dtype=np.int32)  # Img size
            # imgSize = np.array([192, 192, 192], dtype=np.int32)         # Img size
            device = torch.device('cuda:0')
            logPath = './logs/'  # log dir
            if not os.path.isdir(logPath):
                os.makedirs(logPath)
            logName = len(os.listdir(logPath))
            expName = 'exp%s' % str(logName).zfill(3)
            logAdd = './logs/' + expName
            while True:
                if os.path.isdir(logAdd):
                    logName += 1
                    logAdd = './logs/exp%s' % str(logName).zfill(3)
                else:
                    break
            writer = SummaryWriter(logAdd)
            savePath = r'./ModelSave2/%s' % expName  # path for saving model.

            # load data.
            train_dataset = GetMultiTypeMemoryDataSetAndCropQxz(rootPath, trainTxt, imgSize, imageName, maskName,
                                                                distName=distName)
            train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=0)
            val_dataset = GetMultiTypeMemoryDataSetAndCropQxz(rootPath, valTxt, imgSize, imageName, maskName,
                                                              distName=distName)
            val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=True, num_workers=0)
            if not os.path.isdir(savePath):
                os.makedirs(savePath)
            # load net.
            modelCfg = {
                'name': 'UNet3D',
                # number of input channels to the model
                'in_channels': 16,
                # 'dist_channels': 16,
                # number of output channels
                'out_channels': 1,
                # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
                'layer_order': 'gcr',
                # number of features at each level of the U-Net
                # 'f_maps_1': [8, 16],
                # 'f_maps_2': [16, 32, 64, 128],
                # 'addMapsId': 1,
                'f_maps': [16, 32, 64, 128, 256],
                # number of groups in the groupnorm
                'num_groups': 8,
                # apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax
                # this is only relevant during inference, during training the network outputs logits and it is up to the loss function
                # to normalize with Sigmoid or Softmax
                'final_sigmoid': True,
                # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
                'is_segmentation': True
            }
            # model = LoadModel(modelCfg, r'./ModelSave/exp006/supernet_00000.pth')
            model = LoadModel(modelCfg)
            model.to(device)
            model.train(True)
            loss_criterion, optimizer, lr_scheduler, eval_metric, eval_PR = GetLossOptimiLr(model, th)
            eval_metric.to(device)
            eval_PR.to(device)

            netObj = Trainer(train_loader, val_loader, model, loss_criterion, optimizer, lr_scheduler, eval_metric, eval_PR,
                             rootPath=rootPath, modelPath=savePath, device=device, batchSize=batchSize)
            self.netObj2 = netObj
            if not self.stop_2:
                netObj.Train(turn=epoch, writer=writer)
                self.finish.emit(2)
            del netObj
        except Exception as e:
            self.error2.emit(str(e))
            import sys
            import traceback
            print("=== Error message ===")
            print(f"异常类型: {type(e).__name__}")
            print(f"Error message: {e}")
            print("\n=== 错误位置 ===")
            tb = sys.exc_info()[2]
            frame = traceback.extract_tb(tb)[0]
            print(f"  File: {frame.filename}")
            print(f"  行号: {frame.lineno}")
            print(f"  Function: {frame.name}")
            print(f"  代码: {frame.line}\n")

    def stop2(self):
        if self.netObj2:
            self.netObj2.Stop2()
        else:
            self.stop_2 = True

    def turn_count2(self):
        if self.netObj2:
            # print("self.netObj2.tt2, self.netObj2.modSavePath", self.netObj2.tt2, self.netObj2.modSavePath)
            return self.netObj2.tt2, self.netObj2.modSavePath
        else:
            return 0, ""

def Train(rootPath, imgPath, maskPath, distPath, shapes, batchSize, epoch, th):
    imageName = str(Path(imgPath).name)
    maskName = str(Path(maskPath).name)
    distName = str(Path(distPath).name)

    rootPath = r'D:\xueguan\TrainDateSet2\128-predict'  # data path
    imageName = "images"
    maskName = "centerline"
    distName = "dist"
    batchSize = 2
    epoch = 500
    shapes = [128, 128, 128]

    trainTxt = "train.txt"  # txt file for training
    valTxt = "val.txt"  # txt file for validation
    imgSize = np.array(shapes, dtype=np.int32)         # Img size
    # imgSize = np.array([192, 192, 192], dtype=np.int32)         # Img size
    device = torch.device('cuda:0')
    logPath = './logs/'             # log dir
    if not os.path.isdir(logPath):
        os.makedirs(logPath)
    logName = len(os.listdir(logPath))
    expName = 'exp%s' % str(logName).zfill(3)
    logAdd = './logs/' + expName
    while True:
        if os.path.isdir(logAdd):
            logName += 1
            logAdd = './logs/exp%s' % str(logName).zfill(3)
        else:
            break
    writer = SummaryWriter(logAdd)
    savePath = r'./ModelSave2/%s' % expName          # path for saving model.
    # load data.
    train_dataset = GetMultiTypeMemoryDataSetAndCropQxz(rootPath, trainTxt, imgSize, imageName, maskName, distName=distName)
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=0)
    val_dataset = GetMultiTypeMemoryDataSetAndCropQxz(rootPath, valTxt, imgSize, imageName, maskName, distName=distName)
    val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=True, num_workers=0)
    if not os.path.isdir(savePath):
        os.makedirs(savePath)
    # load net.
    modelCfg = {
        'name': 'UNet3D',
        # number of input channels to the model
        'in_channels': 16,
        # 'dist_channels': 16,
        # number of output channels
        'out_channels': 1,
        # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
        'layer_order': 'gcr',
        # number of features at each level of the U-Net
        # 'f_maps_1': [8, 16],
        # 'f_maps_2': [16, 32, 64, 128],
        # 'addMapsId': 1,
        'f_maps': [16, 32, 64, 128, 256],
        # number of groups in the groupnorm
        'num_groups': 8,
        # apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax
        # this is only relevant during inference, during training the network outputs logits and it is up to the loss function
        # to normalize with Sigmoid or Softmax
        'final_sigmoid': True,
        # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
        'is_segmentation': True
    }
    # model = LoadModel(modelCfg, r'./ModelSave/exp006/supernet_00000.pth')
    model = LoadModel(modelCfg)
    model.to(device)
    model.train(True)
    loss_criterion, optimizer, lr_scheduler, eval_metric, eval_PR = GetLossOptimiLr(model, th)
    eval_metric.to(device)
    eval_PR.to(device)

    netObj = Trainer(train_loader, val_loader, model, loss_criterion, optimizer, lr_scheduler, eval_metric, eval_PR, rootPath=rootPath, modelPath=savePath, device=device, batchSize=batchSize)
    netObj.Train(turn=epoch, writer=writer)


if __name__ == '__main__':
    rootPath = r'D:\xueguan\TrainDateSet2\128-predict'  # data path
    imageName = "images"
    imgPath = os.path.join(rootPath, imageName)
    maskName = "centerline"
    maskPath = os.path.join(rootPath, maskName)
    distName = "dist"
    distPath = os.path.join(rootPath, distName)
    batchSize = 2
    epoch = 500
    shapes = [128, 128, 128]
    th = 10

    Train(rootPath, imgPath, maskPath, distPath, shapes, batchSize, epoch, th)
