# -*- coding: utf-8 -*-
import time

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

from models.model import LoadModel
from torch.utils.data import DataLoader
from DataLoader import GetMultiTypeMemoryDataSetAndCropQxz
import numpy as np
import torch, os
from tensorboardX import SummaryWriter
from Net import Trainer
from MyUtil import GetLossOptimiLr, TakeNotesLoss
from os.path import join
import warnings
warnings.filterwarnings("ignore")

'''
Visualization
cmd
activate QxzDeep
cd logs
tensorboard --logdir "./" --host=0.0.0.0

exp016 [Unet9] [尺寸: 192x192x192] [batch: 4]
exp017 [Unet5] [尺寸: 192x192x192] [batch: 1]
exp018 [Unet5] [尺寸: 192x192x192] [batch: 1]
exp019 [Unet9] [尺寸: 192x192x192] [batch: 4] [model 32 64 128 256 512]
exp020 [Unet5] [尺寸: 192x192x192] [batch: 2] [model 8 16 32 64 128 256]
exp021 [Unet5] [尺寸: 192x192x192] [batch: 2] [model 8 16 32 64 128]
exp022 [Unet10] [尺寸: 192x192x192] [batch: 2] [model 32 64 128 256 512] 双卷积采样
exp023 [Unet10] [尺寸: 192x192x192] [batch: 2] [model 32 64 128 256 512 1024] 单卷积采样

exp030 *DSJY-Soma-Section-DataSet* [UnetModel_V2.py] [尺寸: 272x272x144] [batch: 1] [model 32 64 128 256 512]
'''


class CfgClass:
    def __init__(self):
        # self.rootPath = r'D:\CellNeuralBloodVessel\IntegratePose\CellUnet3D-DDP\TrainDataSet'
        self.rootPath = r'D:\SY\cell_data\data1\save\MakeResults\DivideResults'
        self.batchSize = 1
        self.epochs = 5
        self.mainRankId = 0
        self.valBatchSize = 2
        self.valCount = 300
        self.writer = None
        self.trainLoader = None
        self.valLoader = None
        self.curRankId = 0
        self.savePath = ''
        self.device = None
        self.wordSize = 1


def reduce_tensor(tensor, world_size):
    # 用于平均所有gpu上的运行结果，比如loss
    # Reduces the tensor data across all machines
    # Example: If we print the tensor, we can get:
    # tensor(334.4330, device='cuda:1') *********************, here is cuda:  cuda:1
    # tensor(359.1895, device='cuda:3') *********************, here is cuda:  cuda:3
    # tensor(263.3543, device='cuda:2') *********************, here is cuda:  cuda:2
    # tensor(340.1970, device='cuda:0') *********************, here is cuda:  cuda:0
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def cleanup():
    dist.destroy_process_group()


def delete_oldest_file(directory):
    # 获取目录中的所有文件
    files = [l for l in os.listdir(directory) if ".pth" in l]
    if not files:
        print("Directory is empty, no files to delete.")
        return

    # 创建一个列表，存储文件的路径和创建时间
    files_with_ctime = []
    for file in files:
        file_path = join(directory, file)
        if os.path.isfile(file_path):  # 确保是文件而不是目录
            creation_time = os.path.getctime(file_path)
            files_with_ctime.append((file_path, creation_time))

    # 按创建时间排序，最早的文件在前面
    files_with_ctime.sort(key=lambda x: x[1])

    # 删除最早的文件
    oldest_file_path = files_with_ctime[0][0]
    os.remove(oldest_file_path)
    # print(f"已删除最早的文件：{oldest_file_path}")


def Train(cfg):
    # 加载数据
    trainTxt = r"train.txt"  # 训练的txt名称
    valTxt = r"val.txt"  # 验证的txt名称
    imgSize = np.array([272, 272, 144], dtype=np.int32)  # 图像尺寸
    # imgSize = np.array([272, 272, 112], dtype=np.int32)  # 图像尺寸
    curRankId = cfg.curRankId
    rootPath = cfg.rootPath

    imgPath = os.path.join(rootPath, "images")
    maskPath = os.path.join(rootPath, "mask")

    train_dataset = GetMultiTypeMemoryDataSetAndCropQxz(rootPath, trainTxt, imgSize, imgPath, maskPath)
    trainSampler = DistributedSampler(train_dataset)
    cfg.trainLoader = DataLoader(train_dataset, batch_size=cfg.batchSize, sampler=trainSampler, pin_memory=True)

    val_dataset = GetMultiTypeMemoryDataSetAndCropQxz(rootPath, valTxt, imgSize, imgPath, maskPath)
    valSampler = DistributedSampler(val_dataset)
    cfg.valLoader = DataLoader(val_dataset, batch_size=cfg.valBatchSize, sampler=valSampler, pin_memory=True)

    s_gpu_id = cfg.s_gpu_id

    # 加载网络
    # modelCfg = {
    #     'name': 'UNet3D',
    #     # number of input channels to the model
    #     'in_channels': 16,
    #     # number of output channels
    #     'out_channels': 1,
    #     # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
    #     'layer_order': 'gcr',
    #     # number of features at each level of the U-Net
    #     # 'f_maps_1': [8, 16],
    #     # 'f_maps_2': [16, 32, 64, 128],
    #     # 'addMapsId': 1,
    #     # 'f_maps': [8, 16, 32, 64, 128],
    #     'f_maps': [16, 32, 64, 128, 256],
    #     # 'f_maps': [32, 64, 128, 256, 512, 1024],
    #     # number of groups in the groupnorm
    #     'num_groups': 8,
    #     # apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax
    #     # this is only relevant during inference, during training the network outputs logits and it is up to the loss function
    #     # to normalize with Sigmoid or Softmax
    #     'final_sigmoid': True,
    #     # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
    #     'is_segmentation': True
    # }
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

        # 'f_maps': [64, 128, 256, 512, 1024],
        # 'f_maps': [32, 64, 128, 256, 512],
        'f_maps': [16, 32, 64, 128, 256],
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
    model = LoadModel(modelCfg)
    device = cfg.device
    model.to(device)
    # SynchronousBN
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # 获取损失优化器学习率
    loss_criterion, optimizer, lr_scheduler, eval_metric = GetLossOptimiLr(model)
    eval_metric.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[curRankId + s_gpu_id],
                                                      output_device=curRankId + s_gpu_id, find_unused_parameters=True)
    model.train(True)
    # Train
    train_small_losses = TakeNotesLoss()
    train_big_losses = TakeNotesLoss()
    evalVal = TakeNotesLoss()
    lastEvalVal = -1e2
    iter_count = 0
    cfg.valCount = min(cfg.valCount, len(cfg.trainLoader))
    start_time = time.time()
    for epoch in range(cfg.epochs):
        torch.cuda.empty_cache()
        torch.set_grad_enabled(True)
        model.train()
        trainSampler.set_epoch(epoch)
        epoch_start_time = time.time()
        for kk, (img, mask, name) in enumerate(cfg.trainLoader):
            if img.shape[0] != cfg.batchSize:
                continue
            img = img.to(device)
            mask = mask.to(device)
            seg = model(img)
            loss = loss_criterion(seg, mask)[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            reduced_loss = reduce_tensor(loss.data, cfg.wordSize).item()
            if curRankId == 0:
                train_small_losses.update(reduced_loss)
                train_big_losses.update(reduced_loss)
            if (iter_count + 1) % 10 == 0 and curRankId == 0:
                tmpLoss = train_small_losses.update2()
                cfg.writer.add_scalar('Loss/TrainSmallLoss', tmpLoss, train_small_losses.id)
                print('TRAIN [Epoch %d | %d] [Proce %d | %d] [Loss %.4f]' % (
                epoch, cfg.epochs, kk, len(cfg.trainLoader), tmpLoss))
            if (iter_count + 1) % cfg.valCount == 0:
                dist.barrier()
                torch.cuda.empty_cache()
                if curRankId == 0:
                    cfg.writer.add_scalar('Loss/TrainBigLoss', train_big_losses.update2(), train_big_losses.id)
                model.eval()
                valSampler.set_epoch(epoch)
                with torch.no_grad():
                    for kk, (img, mask, name) in enumerate(cfg.valLoader):
                        img = img.to(device)
                        mask = mask.to(device)
                        seg = model(img)
                        eval = eval_metric(seg, mask)
                        evalVal.update(eval)
                    dist.barrier()
                    curEvalVal = evalVal.update2()
                    curEvalVal = reduce_tensor(curEvalVal, cfg.wordSize)
                    if curRankId == 0:
                        cfg.writer.add_scalar('Eval/EvalVal', curEvalVal, evalVal.id)
                    curEvalVal = curEvalVal
                    if curEvalVal > lastEvalVal:
                        if curRankId == 0:
                            print('Validation loss decreased, saving model')
                            torch.save({'state_dict': model.module.state_dict(), 'param': optimizer},
                                       os.path.join(cfg.savePath,
                                                    "supernet_%s_best_%.4f.pth" % (str(epoch).zfill(5), curEvalVal)))
                            if len([l for l in os.listdir(cfg.savePath) if ".pth" in l]) > 3:
                                delete_oldest_file(cfg.savePath)
                        lastEvalVal = curEvalVal
                    # else:
                    #     if curRankId == 0:
                    #         torch.save({'state_dict': model.module.state_dict(), 'param': optimizer},
                    #                    os.path.join(cfg.savePath,
                    #                                 "supernet_%s.pth" % (str(epoch).zfill(5))))
                    lr_scheduler.step(curEvalVal)
                    lr = optimizer.param_groups[0]['lr']
                    if curRankId == 0:
                        cfg.writer.add_scalar('TrainParam/Lr', lr, evalVal.id)
                        print('VAL [Epoch %d | %d] [EvalVal; %.4f] [Lr: %f] [ValLen %d]' % (
                        epoch, cfg.epochs, curEvalVal, lr, len(cfg.valLoader)))
                torch.cuda.empty_cache()
                model.train()
            iter_count += 1
        print(time.time() - epoch_start_time)
    print(time.time() - start_time)
    cleanup()


if __name__ == '__main__':
    # pip install h5py
    # pip install torchmetrics

    logPath = './logs/'  # 日志路径

    cfg = CfgClass()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    s_gpu_id = 0
    cfg.wordSize = 4
    cfg.s_gpu_id = s_gpu_id
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.distributed.init_process_group(backend="gloo", init_method='env://', rank=0, world_size=1)
    # torch.distributed.init_process_group(backend="gloo")
    cfg.curRankId = torch.distributed.get_rank()
    torch.cuda.set_device(cfg.curRankId + s_gpu_id)
    cfg.device = torch.device("cuda", cfg.curRankId + s_gpu_id)
    if cfg.curRankId == 0:
        if not os.path.isdir(logPath): os.makedirs(logPath)
        logName = len(os.listdir(logPath))
        expName = 'exp%s' % str(logName).zfill(3)
        logAdd = './logs/' + expName
        while True:
            if os.path.isdir(logAdd):
                logName += 1
                logAdd = './logs/exp%s' % str(logName).zfill(3)
            else:
                break
        cfg.writer = SummaryWriter(logAdd)
        cfg.savePath = r'./ModelSave/CellTrainModel/%s' % expName  # 模型保存路径
        if not os.path.isdir(cfg.savePath):
            os.makedirs(cfg.savePath)
    # cfg.wordSize = torch.cuda.device_count()
    Train(cfg)



'''
--nproc_per_node=4 Train-DDP-main.py
'''
