# -*- coding: utf-8 -*-
import os
import time
from os.path import join
import json
# 必须在任何 import torch 之前执行
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from Unet3D.models.model import LoadModel
from torch.utils.data import DataLoader
from Unet3D.DataLoader import GetMultiTypeMemoryDataSetAndCropQxz
import numpy as np
import torch
from tensorboardX import SummaryWriter
from Unet3D.MyUtil import GetLossOptimiLr
from pathlib import Path
import sys
import traceback
import subprocess
import socket
import random
import cv2
import tifffile as tiff
from config import exe_cfg, cfgPath
from PyQt5.QtCore import QThread, pyqtSignal


import gpu_device_use


class TakeNotesLoss:
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.id = -1

    def update(self, value):
        self.sum += value
        self.count += 1

    def update2(self):
        # print(self.sum, self.count)
        if self.count != 0:
            tmp = self.sum / self.count
        else:
            tmp = 0
        self.sum = 0
        self.count = 0
        self.id += 1
        return tmp


class VesselDataTrainQThread(QThread):
    finish0 = pyqtSignal(str)  # 训练完成
    progress0 = pyqtSignal(str)  # 训练进度文本
    progress1 = pyqtSignal(str, int)  # 详细进度
    error0 = pyqtSignal(str)  # 报错文本
    update_modSavePath = pyqtSignal(str)  # 模型更新
    update_plot = pyqtSignal(str, str, str)  # 画图
    preview0 = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # 验证结果预览
    show_preview = pyqtSignal(bool)  # 是否训练可视化

    def __init__(self, *args, **kwargs):
        super(VesselDataTrainQThread, self).__init__()
        self.win = kwargs.get('win')
        self.logger = self.win.logger
        self.init_value()

    def init_value(self):
        self.lr = 0.0002
        self.weight_decay = 0.00001
        self.batchSize = 1  # Batch
        self.backwardNumber = 8  # 多少次反向传播一次
        self.device = None  # 设备
        self.new_text = ""  # 训练结束返回文本
        self.train_stop = False  # 是否停止训练
        self.start_train = False  # 是否开始训练
        self.epochs = 500  # 训练轮次
        self.rootPath = ""  # 训练图像文件夹根目录
        self.imgSize = None  # 图像尺寸xyz
        self.imgName = "images"  # 训练图像文件夹名称
        self.maskName = "mask"  # 监督标签文件夹名称
        self.writer = None  # 使用tensorboardX的SummaryWriter写入日志
        self.modelPath = ""  # 模型保存路径

        self.trainLoader = None
        self.valLoader = None

        self.valBatchSize = 2
        self.valCount = 150  # 多少次验证一次
        self.early_stop_patience = 100
        self.model = None  # Model
        self.loss_criterion = None
        self.optimizer = None
        self.lr_scheduler = None
        self.eval_metric = None
        self.modSavePath = ""

        self.make_config_path = ""  # 配置文件路径
        self.logAdd = ""  # 原日志路径
        self.checkpoint_path = ""  # 检查点保存路径
        self.make_config = {}  # 配置文件内容
        self.is_keep_on = False  # 是否继续训练
        self.checkpoints = None  # 检查点

    def run(self):
        """运行训练线程，处理模型训练的完整流程"""
        self.new_text = ""
        try:
            gpu_id = gpu_device_use.get_gpu_utilization()
            if gpu_id is not None:
                torch.cuda.set_device(gpu_id)
                self.device = torch.device("cuda", gpu_id)
            else:
                self.new_text = "No available GPU"
                return

            # 初始化训练状态
            self._initialize_training_state()

            # 执行训练
            if not self.train_stop:
                self._execute_training_pipeline()

        except Exception as e:
            self._handle_training_exception(e)
        finally:
            self._cleanup_resources()
            self.start_train = False
            time.sleep(2)
            self.finish0.emit(self.new_text)

    def _initialize_training_state(self):
        """初始化训练状态变量"""
        self.train_stop = False
        self.start_train = True

    def _execute_training_pipeline(self):
        """执行完整的训练流程：数据加载、模型配置、训练执行"""
        # 获取训练参数
        imgPath = self.win.data_train_dict['imgDir']
        shapes = self.win.data_train_dict['shapes']  # xyz
        self.epochs = self.win.data_train_dict['epochs']
        self.make_config_path = self.win.data_train_dict['makerInfo_path']  # 配置文件
        self.is_keep_on = self.win.data_train_dict.get('is_keep_on', self.is_keep_on)

        self.rootPath = str(Path(imgPath).parent.absolute())
        self.imgSize = np.array(shapes, dtype=np.int32)  # 图像尺寸xyz
        self.imgName = str(Path(imgPath).name)  # 训练图像文件夹名称
        self.maskName = "mask"  # 监督标签文件夹名称

        # 初始化训练环境
        self.train_init()

        # 加载数据
        train_loader, val_loader = self._load_training_data()
        self.trainLoader = train_loader
        self.valLoader = val_loader

        # 配置模型
        model_config = self._get_model_config()

        # 执行训练
        self.start_trainer(model_config)

        # 训练结束后关闭writer
        self.writer.close()

    def _load_training_data(self):
        """加载训练和验证数据集"""
        trainTxt = r"train.txt"  # 训练的txt名称
        valTxt = r"val.txt"  # 验证的txt名称

        # 加载训练数据集
        train_dataset = GetMultiTypeMemoryDataSetAndCropQxz(
            self.rootPath, trainTxt, self.imgSize, self.imgName, self.maskName
        )
        # DataLoader Yes PyTorch 中用于批量加载数据的工具，shuffle=True 表示在每个 epoch 开始时随机打乱数据，num_workers=0 表示不使用多线程加载数据
        train_loader = DataLoader(train_dataset, batch_size=self.batchSize, shuffle=True, num_workers=0)

        # 加载验证数据集
        val_dataset = GetMultiTypeMemoryDataSetAndCropQxz(
            self.rootPath, valTxt, self.imgSize, self.imgName, self.maskName
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batchSize, shuffle=True, num_workers=0)

        return train_loader, val_loader

    def _get_model_config(self):
        """获取模型配置参数"""
        # modelCfgDict = {
        #     'Unet3D_V3': {  # Unet3D_V3网络推荐配置
        #         # 输入通道 （不可更改）
        #         'in_channels': 8,
        #         # 输出通道 （不可更改）
        #         'out_channels': 1,
        #         # 采样间隔
        #         'fieldSpace': 1,
        #         # 每一层卷积数量
        #         'f_maps': [32, 64, 128, 256, 512],
        #         # 是否使用sigmoid
        #         'final_sigmoid': True,
        #     },
        #     'Unet3D_V2': {  # Unet3D_V2网络推荐配置
        #         # 输入通道 （不可更改）
        #         'in_channels': 16,
        #         # 输出通道 （不可更改）
        #         'out_channels': 1,
        #         # 每一层卷积数量
        #         'f_maps': [16, 32, 64, 128, 256],
        #         # 是否使用sigmoid
        #         'final_sigmoid': True,
        #     },
        #     'Unet3D_V1': {  # Unet3D_V1网络推荐配置
        #         'in_channels': 1,
        #         'out_channels': 1,
        #         'f_maps': [16, 32, 64, 128, 256],
        #         'final_sigmoid': True,
        #     }
        # }
        # 加载网络
        modelCfg = {
            'name': 'UNet3D',
            # number of input channels to the model
            'in_channels': 16,  #
            # number of output channels
            'out_channels': 1,
            # determines the order of operators in a single layer (gcr - GroupNorm+Conv3d+ReLU)
            'layer_order': 'gcr',
            # number of features at each level of the U-Net
            'f_maps': [16, 32, 64, 128, 256, 512],  # 定义每个层级的特征图数量，Convolutional layer
            # 'f_maps_1': [8, 16],
            # 'f_maps_2': [16, 32, 64, 128],
            # 'addMapsId': 1,
            # number of groups in the groupnorm
            'num_groups': 8,  # GroupNorm 中的分组数
            # apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax
            # this is only relevant during inference, during training the network outputs logits and it is up to the loss function
            # to normalize with Sigmoid or Softmax
            'final_sigmoid': True,  # 是否在最后一层应用 Sigmoid Activation function
            # if True applies the final normalization layer (sigmoid or softmax), otherwise the networks returns the output from the final convolution layer; use False for regression problems, e.g. de-noising
            'is_segmentation': True  # 是否用于分割任务
        }
        return modelCfg

    def _handle_training_exception(self, exception):
        """处理训练过程中的异常"""
        self.error0.emit(str(exception))
        print("Training ended with exception")

        # 记录详细错误信息
        self.logger.error("\n=== Error message ===")
        self.logger.error(f"Exception type: {type(exception).__name__}")
        self.logger.error(f"Error message: {exception}")
        self.logger.error("=== Error location ===")

        # 获取并记录堆栈信息
        tb = sys.exc_info()[2]
        for frame in traceback.extract_tb(tb):
            self.logger.error(f"  File: {frame.filename}")
            self.logger.error(f"  Line number: {frame.lineno}")
            self.logger.error(f"  Function: {frame.name}")
            self.logger.error(f"  Code: {frame.line}\n")

    def _cleanup_resources(self):
        """清理训练过程中使用的资源"""
        # 清理分布式训练环境
        if hasattr(self, 'model'):
            # 清理模型相关资源
            attributes_to_clean = ['model', 'trainLoader', 'valLoader', 'device',
                                   'writer', 'loss_criterion', 'optimizer',
                                   'lr_scheduler', 'eval_metric']
            for attr in attributes_to_clean:
                if hasattr(self, attr):
                    try:
                        delattr(self, attr)
                    except Exception:
                        pass

        # 清理 GPU Cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cache cleared")

    def load_train_checkpoint(self):
        # 尝试加载检查点
        self.model.load_state_dict(self.checkpoints['model_state'])
        self.optimizer.load_state_dict(self.checkpoints['optimizer_state'])
        self.lr_scheduler.load_state_dict(self.checkpoints['scheduler_state'])  # 恢复学习率调度器状态
        self.start_epoch = self.checkpoints['epoch'] + 1
        self.small_epoch = self.checkpoints['small_epoch'] + 1
        self.big_epoch = self.checkpoints['big_epoch'] + 1
        self.eval_epoch = self.checkpoints['eval_epoch'] + 1
        iter_count = self.checkpoints['iter_count']
        lastEvalVal = self.checkpoints.get('lastEvalVal', 0)
        patience_counter = self.checkpoints.get('patience_counter', 0)
        print(f"Resuming training from checkpoint, starting epoch: {self.start_epoch}")
        return iter_count, lastEvalVal, patience_counter

    def start_trainer(self, modelCfg):
        try:
            iter_count = 0  # 全局迭代计数器，用于跟踪当前迭代次数

            if self.valBatchSize is None:
                self.valBatchSize = self.batchSize
            self.valCount = 150  # 多少次验证一次

            view_count = 0
            lastEvalVal = 0  # 用于记录上一次验证集上最佳的评估指标值，以便判断是否需要保存模型
            patience_counter = 0  # 早停机制：验证精度连续未提升轮数计数器

            # 初始化模型
            self.model = LoadModel(modelCfg)
            self.model.to(self.device)

            # 获取损失函数、Optimizer、学习率调度器和评估指标
            self.loss_criterion, self.optimizer, self.lr_scheduler, self.eval_metric = GetLossOptimiLr(self.model,
                                                                                                       learning_rate=self.lr,
                                                                                                       weight_decay=self.weight_decay)
            self.eval_metric.to(self.device)

            self.start_epoch = 0
            self.small_epoch = 0
            self.big_epoch = 0
            self.eval_epoch = 0
            if self.is_keep_on and self.checkpoints is not None:  # 继续训练
                # 读取检查点
                iter_count, lastEvalVal, patience_counter = self.load_train_checkpoint()

            self.model.train(True)

            # 初始化训练跟踪变量
            train_small_losses = TakeNotesLoss()  # 记录短期（每10次迭代）的平均损失
            train_big_losses = TakeNotesLoss()  # 记录长期（每 valCount 次迭代）的平均损失
            evalVal = TakeNotesLoss()  # 用于记录验证集上的评估指标
            self.valCount = min(self.valCount, len(self.trainLoader))  # 多少次验证一次

            progress1_count = 0
            start_time = time.time()
            train_loader_length = len(self.trainLoader)

            self.train_round = self.eval_epoch

            # 主训练循环
            for epoch in range(self.epochs):
                epoch += self.start_epoch
                if self.train_stop:
                    return

                # 初始化第一个epoch的进度信息
                if epoch == 0:
                    text = f"TRAIN [Epoch {epoch} | {self.epochs}] [Proce {0} | {train_loader_length}] [Loss {1:.4f}]"
                    self.progress0.emit(text)

                # 清理缓存并设置训练模式
                torch.cuda.empty_cache()
                torch.set_grad_enabled(True)
                self.model.train()

                # 批次训练循环
                for batch_idx, (img, mask, name) in enumerate(self.trainLoader):  # 加载数据
                    if self.train_stop:  # 检查是否需要停止训练
                        return

                    # 跳过不符合批次大小的数据
                    if img.shape[0] != self.batchSize:
                        continue

                    # 训练一个批次
                    reduced_loss = self._train_one_batch(img, mask)
                    train_small_losses.update(reduced_loss.item())
                    train_big_losses.update(reduced_loss.item())

                    # 每10个迭代且是验证点时的处理
                    is_10th_iter = (iter_count + 1) % 10 == 0
                    is_val_point = (iter_count + 1) % self.valCount == 0
                    is_val10th_iter = self.valCount % 10 == 0

                    # 记录训练进度和损失
                    if is_10th_iter:
                        self._record_training_loss(epoch, batch_idx, train_small_losses)

                        # 记录详细进度
                        if is_val_point:
                            if is_val10th_iter:
                                userTime = time.time() - start_time
                                surplusTime = userTime / (progress1_count + 1) * (self.valCount - progress1_count - 1)
                                logInfo2 = 'Round %d progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                                    (self.train_round + 1), (progress1_count + 1) / self.valCount * 100, userTime, surplusTime)
                                self.progress1.emit(logInfo2, 0)
                            else:
                                self._update_progress_info(progress1_count, start_time)
                            start_time = time.time()
                            progress1_count = 0
                        else:
                            if batch_idx != 0:
                                userTime = time.time() - start_time
                                surplusTime = userTime / (progress1_count + 1) * (self.valCount - progress1_count - 1)
                                logInfo2 = 'Round %d progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                                    (self.train_round + 1), (progress1_count + 1) / self.valCount * 100, userTime, surplusTime)
                                self.progress1.emit(logInfo2, 0)
                            progress1_count += 1
                    else:
                        # 更新进度信息
                        if is_val_point:
                            self._update_progress_info(progress1_count, start_time)
                            start_time = time.time()
                            progress1_count = 0
                        else:
                            userTime = time.time() - start_time
                            surplusTime = userTime / (progress1_count + 1) * (self.valCount - progress1_count - 1)
                            logInfo2 = 'Round %d progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                                (self.train_round + 1), (progress1_count + 1) / self.valCount * 100, userTime, surplusTime)
                            self.progress1.emit(logInfo2, progress1_count)
                            progress1_count += 1

                    # 每valCount个迭代进行一次验证
                    if is_val_point:
                        # _perform_validation返回新的评估值和早停计数器
                        new_eval_val, view_count, patience_counter = self._perform_validation(epoch, train_big_losses, evalVal,
                                                                                              lastEvalVal, view_count, patience_counter)
                        if self.train_stop:
                            return
                        if new_eval_val is not None:
                            lastEvalVal = new_eval_val

                        self.train_round = evalVal.id + self.eval_epoch + 1

                    iter_count += 1
                    torch.cuda.empty_cache()

                self.model.eval()
                with torch.no_grad():
                    # 保存当前状态
                    torch.save({
                        # 检查点参数
                        'epoch': epoch,
                        'small_epoch': train_small_losses.id + self.small_epoch,
                        'big_epoch': train_big_losses.id + self.big_epoch,
                        'eval_epoch': evalVal.id + self.eval_epoch,
                        'model_state': self.model.state_dict(),
                        'iter_count': iter_count,
                        'optimizer_state': self.optimizer.state_dict(),
                        'scheduler_state': self.lr_scheduler.state_dict(),  # 保存学习率调度器状态
                        # 训练模型参数
                        'state_dict': self.model.state_dict(),
                        'lastEvalVal': lastEvalVal,
                        'patience_counter': patience_counter,
                        'param': self.optimizer,
                    }, self.checkpoint_path)

                    if (self.checkpoint_path != self.make_config.get('checkpoint_path', None) or
                            self.logAdd != self.make_config.get('log_path', None)):
                        print("Updating configuration file!")
                        self.make_config['checkpoint_path'] = self.checkpoint_path
                        self.logAdd = self.make_config['log_path']
                        with open(self.make_config_path, 'w') as f:  # 更新配置文件
                            f.write(json.dumps(self.make_config, indent=4))
                torch.cuda.empty_cache()
                self.model.train(True)
                # 如果最后一个批次没有达到backwardNumber次，也需要更新参数
                if iter_count % self.backwardNumber != 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        except Exception as e:
            torch.cuda.empty_cache()
            print("End")
            print("\n=== Error message ===")
            print(f"Exception type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("=== Error location ===")
            tb = sys.exc_info()[2]
            for frame in traceback.extract_tb(tb):
                print(f"  File: {frame.filename}")
                print(f"  Line number: {frame.lineno}")
                print(f"  Function: {frame.name}")
                print(f"  Code: {frame.line}\n")
        finally:
            torch.cuda.empty_cache()
            return

    def _train_one_batch(self, img, mask):
        """训练单个批次的数据

        Args:
            img: 输入图像
            mask: 目标掩码

        Returns:
            loss: 损失值
        """
        img = img.to(self.device)
        mask = mask.to(self.device)
        seg = self.model(img)  # 使用网络 self.super_net 对图像进行分割，得到分割结果 seg
        loss = self.loss_criterion(img, seg, mask)[0]  # 返回的损失值是一个张量
        self.optimizer.zero_grad()
        loss.backward()  # Backpropagation
        self.optimizer.step()  # 参数更新
        return loss

    def _record_training_loss(self, epoch, kk, train_small_losses):
        """记录训练损失信息"""
        tmpLoss = train_small_losses.update2()
        self.writer.add_scalar('Loss/TrainSmallLoss', tmpLoss, train_small_losses.id + self.small_epoch)

        # 发送进度信息
        text = f"TRAIN [Epoch {epoch} | {self.epochs}] [Proce {kk} | {len(self.trainLoader)}] [Loss {tmpLoss:.4f}]"
        self.progress1.emit(text, 1)

        # # 写入损失日志文件
        # with open(self.text_loss_path, "a", encoding="utf-8") as f:
        #     f.write(f"{iter_count + 1} {tmpLoss:.4f}\n")
        # f.close()

    def _update_progress_info(self, progress1_count, start_time):
        """更新并显示训练进度信息"""
        userTime = time.time() - start_time
        surplusTime = userTime / (progress1_count + 1) * (self.valCount - progress1_count - 1)
        logInfo2 = 'Round %d progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
            (self.train_round + 1), (progress1_count + 1) / self.valCount * 100, userTime, surplusTime)
        # if self.valCount % 10 == 0:
        #     progress1_count = 0
        self.progress1.emit(logInfo2, progress1_count)

    def _perform_validation(self, epoch, train_big_losses, evalVal, lastEvalVal, view_count, patience_counter):
        """执行验证过程并返回更新后的评估值和早停计数器"""
        torch.cuda.empty_cache()
        self.writer.add_scalar('Loss/TrainBigLoss', train_big_losses.update2(), train_big_losses.id + self.big_epoch)
        self.model.eval()
        try:
            with torch.no_grad():
                random_v = random.randint(0, len(self.valLoader) - 1)
                eval_start_time = time.time()
                for batch_idx, (img, mask, name) in enumerate(self.valLoader):
                    if self.train_stop:  # 检查是否需要停止训练
                        return None, view_count, patience_counter
                    if img.shape[0] != self.valBatchSize:
                        continue
                    img = img.to(self.device)
                    mask = mask.to(self.device)
                    # 使用网络对验证集进行分割
                    seg = self.model(img)
                    # 计算评估指标
                    eval = self.eval_metric(img, seg, mask)
                    evalVal.update(eval)

                    eval_userTime = time.time() - eval_start_time
                    eval_surplusTime = eval_userTime / (batch_idx + 1) * (len(self.valLoader) - batch_idx - 1)
                    eval_logInfo = 'Round %d progress %.2f%%] [Time elapsed %ds] [Estimated remaining time %ds]' % (
                        (self.train_round + 1), (batch_idx + 1) / len(self.valLoader) * 100, eval_userTime, eval_surplusTime)

                    if batch_idx == random_v and (epoch + 1) % 10 == 0:
                        ns = seg.cpu().detach().numpy()  # 训练结果--Tag
                        ma = mask.cpu().detach().numpy()  # 目标标签
                        imgs = img.cpu().detach().numpy()  # 原图
                        ns_i = ns[:, 0]
                        ma_i = ma[:, 0]
                        img_i = imgs[:, 0]
                        self.save_result(ns_i, ma_i, img_i)
                        self.progress1.emit(eval_logInfo, 0)
                    else:
                        self.progress1.emit(eval_logInfo, batch_idx)

                    torch.cuda.empty_cache()

                curEvalVal = evalVal.update2()
                self.writer.add_scalar('Eval/EvalVal', curEvalVal, evalVal.id + self.eval_epoch)
                curEvalVal = curEvalVal

                if view_count < 2 and curEvalVal > 0:
                    view_count += 1
                if view_count == 2 or epoch >= 5:
                    self.show_preview.emit(True)

                # 早停机制：检查验证精度是否提升
                if curEvalVal > lastEvalVal:
                    # 精度提升，重置计数器
                    patience_counter = 0
                else:
                    # 精度未提升，计数器累加
                    patience_counter += 1
                    self.progress0.emit(f'Validation accuracy has not improved for {patience_counter} consecutive rounds')
                    
                    # 触发早停
                    if patience_counter >= self.early_stop_patience:
                        self.progress0.emit(f'Early stopping triggered! Validation accuracy has not improved for {self.early_stop_patience} consecutive rounds, training stops automatically')
                        self.train_stop = True
                        return None, view_count, patience_counter

                # 保存最佳模型
                if curEvalVal > lastEvalVal:
                    self.progress0.emit('Validation loss decreased, saving model')
                    self.modSavePath = os.path.abspath(join(self.modelPath, "supernet_%s_best_%.4f.pth" % (
                        str(epoch).zfill(5), curEvalVal)))
                    # Update
                    torch.save({'state_dict': self.model.state_dict(), 'param': self.optimizer},
                               self.modSavePath)
                    # Copy
                    self.update_modSavePath.emit(self.modSavePath)
                    # 如果保存的模型数大于一，去掉之前的模型
                    # if len([l for l in os.listdir(self.modelPath) if ".pth" in l]) > 1:
                    #     self.delete_oldest_file(self.modelPath)
                    lastEvalVal = curEvalVal

                    # with open(self.text_save_model_path, "a", encoding="utf-8") as f:
                    #     f.write(f"{iter_count + 1} 验证集损失减少,保存模型 {curEvalVal:.4f}\n")
                    # f.close()

                # 更新学习率
                self.lr_scheduler.step(curEvalVal)
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('TrainParam/Lr', lr, evalVal.id + self.eval_epoch)
                text = f"VAL [Epoch {epoch} | {self.epochs}] [EvalVal: {curEvalVal:.4f}] [Lr: {lr:.8f}]"
                self.progress0.emit(text)

            return lastEvalVal, view_count, patience_counter

        finally:
            torch.cuda.empty_cache()
            if self.train_stop:
                return None, view_count, patience_counter
            self.model.train()

    def save_result(self, net_seg, mask, img):
        root = self.rootPath
        i = random.randint(0, len(mask) - 1)

        path = f"{root}/TrainResultsPreview/{i + 1}/"
        os.makedirs(path, exist_ok=True)
        net_seg_i = ((net_seg[i] - net_seg[i].min()) / (net_seg[i].max() - net_seg[i].min()) * 255).astype(np.uint8)
        mask_i = ((mask[i] - mask[i].min()) / (mask[i].max() - mask[i].min()) * 255).astype(np.uint8)
        img_i = ((img[i] - img[i].min()) / (img[i].max() - img[i].min()) * 255).astype(np.uint8)

        net_seg_i[net_seg_i < 103] = 0

        seg_name = f"net_seg{i + 1}.tif"
        tiff.imwrite(join(path, seg_name), net_seg_i, compression="lzw")

        tiff.imwrite(join(path, f"mask{i + 1}.tif"), mask_i, compression="lzw")
        tiff.imwrite(join(path, f"img{i + 1}.tif"), img_i, compression="lzw")

        img_i_2d = self.MaxProject(img_i, 1)
        mask_i_2d = self.MaxProject(mask_i, 0)
        pre_i_2d = self.MaxProject(net_seg_i, 0)

        self.preview0.emit(img_i_2d, mask_i_2d, pre_i_2d)

    def MaxProject(self, img, is_enhance):
        img = np.max(img, axis=0)
        img = self.normalize_and_scale_to_uint8(img)
        if is_enhance:
            self.enhance_contrast_histogram_equalization(img)
        return img

    def enhance_contrast_histogram_equalization(self, img):
        """
        使用直方图均衡化增强图像对比度。

        Parameter:
            img (numpy.ndarray): 输入图像。
        返回:
            numpy.ndarray: 对比度增强后的图像。
        """
        # 应用直方图均衡化
        img_eq = cv2.equalizeHist(img)
        return img_eq

    def normalize_and_scale_to_uint8(self, data):
        """
        将灰度值归一化到0到1，然后缩放到0到255，并转换为uint8类型。

        Parameter:
            data (numpy.ndarray): 输入数据。
        返回:
            numpy.ndarray: 转换后的uint8Data。
        """
        data_min = np.min(data)
        data_max = np.max(data)
        normalized_data = (data - data_min) / (data_max - data_min)
        scaled_data = (normalized_data * 255).astype(np.uint8)
        return scaled_data

    def delete_oldest_file(self, directory):
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

    def train_init(self):
        with open(self.make_config_path, 'r') as f:  # 读取配置文件
            self.make_config = json.loads(f.read())

        # 更新训练参数
        self.lr = self.make_config.get("learning_rate", self.lr)
        self.weight_decay = self.make_config.get("weight_decay", self.weight_decay)
        self.batchSize = self.make_config.get("batch_size", self.batchSize)
        self.valBatchSize = self.make_config.get("val_batch_size", self.valBatchSize)
        self.valCount = self.make_config.get("val_counts", self.valCount)
        self.early_stop_patience = self.make_config.get("early_stop_patience", self.early_stop_patience)

        self.logAdd = self.make_config.get("log_path", "")  # 获取原来的日志文件路径

        # if self.is_keep_on:  # self.checkpoint_path存在
        #     self.checkpoint_path = self.make_config.get("checkpoint_path", "")  # 获取原来的检查点文件路径

        results_dir = self.win.data_train_dict['results_dir']  # 保存路径
        checkpoint_dir = join(results_dir, "CheckPoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_path = join(checkpoint_dir, "checkpoint.pth")  # 检查点文件路径

        if self.is_keep_on and self.checkpoint_path:
            try:
                # 加载检查点
                self.checkpoints = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            except Exception as e:
                # Load failed
                print("Checkpoints load failed!")
                self.checkpoints = None
        is_keep_on = False
        if self.is_keep_on and self.checkpoints is not None:
            logAdd = self.make_config.get("log_path", None)
            if logAdd is not None:
                if os.path.isdir(logAdd):  # 存在日志路径
                    small_epoch = self.checkpoints['small_epoch']
                    # 使用tensorboardX的SummaryWriter写入日志
                    # self.writer = SummaryWriter(logAdd)
                    # purge_step参数确保从指定步数继续记录
                    # 避免TensorBoard显示重复的步数范围
                    # 保持日志目录不变，确保数据连续性
                    self.writer = SummaryWriter(log_dir=logAdd, purge_step=small_epoch)
                    is_keep_on = True
                    expName = Path(logAdd).stem
        if not is_keep_on:
            logPath = exe_cfg.cfg['ExePath']['vessel_logs_path']  # 日志路径
            if not os.path.isdir(logPath):
                os.makedirs(logPath)

            logName = len(os.listdir(logPath))
            expName = 'exp%s' % str(logName).zfill(3)
            logAdd = join(logPath, expName)
            while True:
                if os.path.isdir(logAdd):
                    logName += 1
                    logAdd = join(logPath, f"exp{str(logName).zfill(3)}")
                else:
                    break
            logAdd = os.path.abspath(logAdd)
            # 确保日志目录存在
            os.makedirs(logAdd, exist_ok=True)
            # 使用tensorboardX的SummaryWriter写入日志
            self.writer = SummaryWriter(logAdd)
            # 更新日志文件
            self.make_config['log_path'] = str(logAdd)
        # 启动 TensorBoard
        print(logAdd)

        # if not os.path.exists(logAdd):
        #     os.makedirs(logAdd)
        #     # 设置端口和接口
        # port = exe_cfg.cfg['Port']['vessel_port']  # 默认端口是6008，你可以根据需要更改
        # host = exe_cfg.cfg['Host']['vessel_host']  # 默认主机地址是127.0.0.1，你可以根据需要更改
        #
        # subprocess.Popen([
        #     "tensorboard", f"--logdir={logAdd}", f"--host={host}", f"--port={port}"], shell=True)

        if not os.path.exists(logAdd):
            os.makedirs(logAdd)
            # 设置端口和接口
        port = exe_cfg.cfg['Port']['vessel_port']  # 默认端口是6008，你可以根据需要更改
        host = exe_cfg.cfg['Host']['vessel_host']  # 默认主机地址是127.0.0.1，你可以根据需要更改

        """启动 TensorBoard"""
        while self.is_port_in_use(host, port):
            print(f"TensorBoard is already running on port {port}.")
            port = random.randint(6006, 6100)
            exe_cfg.cfg['Port']['vessel_port'] = str(port)

        with open(cfgPath, 'w') as configfile:
            exe_cfg.cfg.write(configfile)

        # 尝试启动TensorBoard的函数
        def start_tensorboard():
            # Method1: 尝试使用TensorFlow的TensorBoardModule（这应该是随程序打包的）
            try:
                import tensorboard.program
                import mimetypes
                # EnsureJavaScript文件的MIME类型正确设置
                mimetypes.add_type('application/javascript', '.js')
                mimetypes.add_type('text/css', '.css')
                mimetypes.add_type('application/json', '.json')

                tb = tensorboard.program.TensorBoard()
                tb.configure(
                    argv=[
                        None,
                        f"--logdir", f"{logAdd}",
                        f"--host", f"{host}",
                        f"--port", f"{port}",
                        # '--bind_all'  # 允许从任何IP访问
                    ]
                )
                url = tb.launch()
                print(f"TensorBoard service started (using TensorFlow module): {url}")
                return True
            except Exception as e:
                print(f"Failed to start TensorBoard using TensorFlow module: {e}")

            # Method2: 尝试使用系统中的TensorBoard命令
            try:
                # 使用命令行参数方式启动TensorBoard
                command = [
                    "tensorboard",
                    f"--logdir={logAdd}",
                    f"--host={host}",
                    f"--port={port}"
                ]

                # 启动TensorBoardProcess
                subprocess.Popen(command)
                print(f"TensorBoard service started (using system command), access at: http://{host}:{port}")
                return True
            except Exception as e:
                print(f"Failed to start TensorBoard using system command: {e}")

            # Method3: 如果前两种方法都失败，提供手动启动指引
            print("Failed to start TensorBoard. Please start it manually:")
            print(f"1. Open a command prompt")
            print(f"2. Run command: tensorboard --logdir={logAdd} --host={host} --port={port}")
            print(f"3. Access in browser: http://{host}:{port}")
            return False

        # 启动TensorBoard
        start_tensorboard()

        self.modelPath = join(exe_cfg.cfg['ExePath']['vessel_models_path'], f"{expName}")  # 模型保存路径
        if not os.path.isdir(self.modelPath):
            os.makedirs(self.modelPath)

        # self.text_eval_path = join(self.savePath, 'text_eval.txt')
        # self.text_save_model_path = join(self.savePath, 'text_save_model.txt')
        # self.text_loss_path = join(self.savePath, 'text_loss.txt')
        # with open(self.text_eval_path, 'w', encoding='utf-8') as f:
        #     f.write("")
        # with open(self.text_save_model_path, 'w', encoding='utf-8') as f:
        #     f.write("")
        # with open(self.text_loss_path, 'w', encoding='utf-8') as f:
        #     f.write("")

    def to_stop(self):
        self.train_stop = True
        time.sleep(3)
        try:
            # 删除模型引用
            # if hasattr(self, 'model'):
            #     del self.model
            #     print("模型已删除")
            # del self.trainLoader
            # del self.valLoader
            # del self.device
            # del self.writer
            # del self.loss_criterion
            # del self.optimizer
            # del self.lr_scheduler
            # del self.eval_metric
            # 清理 GPU Cache
            torch.cuda.empty_cache()
            print("GPU cache cleared")
        except Exception as e:
            print(e)
            print("Stop")
        finally:
            time.sleep(3)
            self.progress1.emit("Training stopped!", 0)

    def is_port_in_use(self, host, port):
        """检查指定端口是否被占用"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((str(host), int(float(port)))) == 0