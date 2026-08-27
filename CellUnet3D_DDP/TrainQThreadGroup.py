# -*- coding: utf-8 -*-
import random
import time
import sys
import tifffile
import traceback
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
import cv2
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
from .models.model import LoadModel
from torch.utils.data import DataLoader
from .DataLoader import GetMultiTypeMemoryDataSetAndCropQxz
import numpy as np
import torch
from tensorboardX import SummaryWriter
from .MyUtil import GetLossOptimiLr, TakeNotesLoss
from os.path import join
import cc3d
from speed_cc3d import speed_cc3d
from scipy.spatial import distance
import warnings
warnings.filterwarnings("ignore")
import socket
import subprocess


from config import exe_cfg, cfgPath
import gpu_device_use

from PyQt5.QtCore import QThread, pyqtSignal


class CellDataTrainQThread(QThread):
    finish0 = pyqtSignal(str)
    progress0 = pyqtSignal(str)
    progress1 = pyqtSignal(str, int)
    error0 = pyqtSignal(str)
    update_modSavePath = pyqtSignal(str)
    update_plot = pyqtSignal(str, str, str)
    preview0 = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    show_preview = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super(CellDataTrainQThread, self).__init__()
        self.win = kwargs.get('win')
        self.train_stop = False
        self.start_train = False
        self.s_gpu_id = 0
        self.wordSize = 4
        self.curRankId = 0
        self.device = None
        self.writer = None
        self.savePath = ""
        self.trainLoader = None
        self.valLoader = None
        self.rootPath = ""
        self.imgSize = None
        self.batchSize = 1
        self.valBatchSize = 2
        self.epochs = 500
        self.valCount = 150

        self.text_eval_path = ""
        self.text_save_model_path = ""
        self.text_loss_path = ""

        self.modSavePath = ""
        self.logger = self.win.logger

    def run(self):
        """运行训练线程，处理模型训练的完整流程"""
        self.new_text = ""
        try:
            gpu_id = gpu_device_use.get_gpu_utilization()
            if gpu_id is not None:
                torch.cuda.set_device(gpu_id)
                self.device = torch.device("cuda", gpu_id)
            else:
                self.new_text = "没有可用的GPU"
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

        self.rootPath = str(Path(imgPath).parent.absolute())
        self.imgSize = np.array(shapes, dtype=np.int32)  # 图像尺寸xyz

        # 初始化训练环境
        self.train_init()

        # 加载数据
        train_loader, val_loader, train_sampler, val_sampler = self._load_training_data(imgPath)
        self.trainLoader = train_loader
        self.valLoader = val_loader

        # 配置模型
        model_config = self._get_model_config()
        
        # 执行训练
        self.trainer(model_config, train_sampler, val_sampler)
        
    def _load_training_data(self, imgPath):
        """加载训练和验证数据集"""
        trainTxt = r"train.txt"  # 训练的txt名称
        valTxt = r"val.txt"      # 验证的txt名称
        maskPath = join(self.rootPath, "mask")

        # 加载训练数据集
        train_dataset = GetMultiTypeMemoryDataSetAndCropQxz(
            self.rootPath, trainTxt, self.imgSize, imgPath, maskPath
        )
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batchSize, sampler=train_sampler, pin_memory=True
        )

        # 加载验证数据集
        val_dataset = GetMultiTypeMemoryDataSetAndCropQxz(
            self.rootPath, valTxt, self.imgSize, imgPath, maskPath
        )
        val_sampler = DistributedSampler(val_dataset)
        val_loader = DataLoader(
            val_dataset, batch_size=self.valBatchSize, sampler=val_sampler, pin_memory=True
        )

        return train_loader, val_loader, train_sampler, val_sampler
        
    def _get_model_config(self):
        """获取模型配置参数"""
        return {
            'name': 'UNet3D',
            'in_channels': 4,
            'out_channels': 1,
            'fieldSpace': 1,
            'layer_order': 'gcr',
            'f_maps': [16, 32, 64, 128, 256],
            'num_groups': 8,
            'final_sigmoid': True,
            'is_segmentation': True
        }
        
    def _handle_training_exception(self, exception):
        """处理训练过程中的异常"""
        self.error0.emit(str(exception))
        print("训练异常结束")
        
        # 记录详细错误信息
        self.logger.error("\n=== Error message ===")
        self.logger.error(f"异常类型: {type(exception).__name__}")
        self.logger.error(f"Error message: {exception}")
        self.logger.error("=== 错误位置 ===")
        
        # 获取并记录堆栈信息
        tb = sys.exc_info()[2]
        for frame in traceback.extract_tb(tb):
            self.logger.error(f"  File: {frame.filename}")
            self.logger.error(f"  行号: {frame.lineno}")
            self.logger.error(f"  Function: {frame.name}")
            self.logger.error(f"  代码: {frame.line}\n")
            
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
            print("GPU内存缓存已清除")

    def trainer(self, modelCfg, trainSampler, valSampler):
        try:
            # 初始化模型
            model = LoadModel(modelCfg)
            device = self.device
            model.to(device)

            # SynchronousBN
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            # 获取损失函数、Optimizer、学习率调度器和评估指标
            self.loss_criterion, self.optimizer, self.lr_scheduler, self.eval_metric = GetLossOptimiLr(model)
            self.eval_metric.to(device)

            # 初始化分布式训练模型
            self.model = torch.nn.parallel.DistributedDataParallel(model,
                                                                   device_ids=[self.curRankId + self.s_gpu_id],
                                                                   output_device=self.curRankId + self.s_gpu_id,
                                                                   find_unused_parameters=False)
            self.model.train(True)

            # 初始化训练跟踪变量
            train_small_losses = TakeNotesLoss()
            train_big_losses = TakeNotesLoss()
            evalVal = TakeNotesLoss()
            lastEvalVal = 0
            iter_count = 0
            view_count = 0
            progress1_count = 0
            start_time = time.time()
            self.valCount = min(self.valCount, len(self.trainLoader))
            train_loader_length = len(self.trainLoader)

            # 主训练循环
            for epoch in range(self.epochs):
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
                trainSampler.set_epoch(epoch)

                # 批次训练循环
                for kk, (img, mask, name) in enumerate(self.trainLoader):
                    if self.train_stop:
                        return
                    if img.shape[0] != self.batchSize:
                        continue

                    # 训练一个批次
                    reduced_loss = self._train_one_batch(img, mask, device)
                    if self.curRankId == 0:
                        train_small_losses.update(reduced_loss)
                        train_big_losses.update(reduced_loss)

                    # 每10个迭代且是验证点时的处理
                    is_10th_iter = (iter_count + 1) % 10 == 0
                    is_val_point = (iter_count + 1) % self.valCount == 0
                    is_val10th_iter = self.valCount % 10 == 0

                    # 记录训练进度和损失
                    if is_10th_iter and self.curRankId == 0:
                        self._record_training_loss(epoch, kk, train_small_losses, iter_count)

                        # 记录详细进度
                        if is_val_point:
                            if is_val10th_iter:
                                userTime = time.time() - start_time
                                surplusTime = userTime / (progress1_count + 1) * (self.valCount - progress1_count - 1)
                                logInfo2 = '[第 %d 轮训练进度 %.2f%%] [已耗时 %ds] [预计剩余时间 %ds]' % (
                                    (epoch + 1), (progress1_count + 1) / self.valCount * 100, userTime, surplusTime)
                                self.progress1.emit(logInfo2, 0)
                            else:
                                self._update_progress_info(epoch, progress1_count, start_time)
                            start_time = time.time()
                            progress1_count = 0
                        else:
                            if kk != 0:
                                userTime = time.time() - start_time
                                surplusTime = userTime / (progress1_count + 1) * (self.valCount - progress1_count - 1)
                                logInfo2 = '[第 %d 轮训练进度 %.2f%%] [已耗时 %ds] [预计剩余时间 %ds]' % (
                                    (epoch + 1), (progress1_count + 1) / self.valCount * 100, userTime, surplusTime)
                                self.progress1.emit(logInfo2, 0)
                            progress1_count += 1
                    else:
                        # 更新进度信息
                        with torch.no_grad():
                            if is_val_point and self.curRankId == 0:
                                self._update_progress_info(epoch, progress1_count, start_time)
                                start_time = time.time()
                                progress1_count = 0
                            else:
                                userTime = time.time() - start_time
                                surplusTime = userTime / (progress1_count + 1) * (self.valCount - progress1_count - 1)
                                logInfo2 = '[第 %d 轮训练进度 %.2f%%] [已耗时 %ds] [预计剩余时间 %ds]' % (
                                    (epoch + 1), (progress1_count + 1) / self.valCount * 100, userTime, surplusTime)
                                self.progress1.emit(logInfo2, progress1_count)
                                progress1_count += 1

                    # 每valCount个迭代进行一次验证
                    if is_val_point:
                        # _perform_validation返回新的评估值
                        new_eval_val, view_count = self._perform_validation(epoch, train_big_losses, evalVal, lastEvalVal,
                                                                            iter_count, view_count, valSampler)
                        if new_eval_val is not None:
                            lastEvalVal = new_eval_val

                    iter_count += 1

        except Exception as e:
            torch.cuda.empty_cache()
            print("结束")
            print("\n=== Error message ===")
            print(f"异常类型: {type(e).__name__}")
            print(f"Error message: {e}")
            print("=== 错误位置 ===")
            tb = sys.exc_info()[2]
            for frame in traceback.extract_tb(tb):
                print(f"  File: {frame.filename}")
                print(f"  行号: {frame.lineno}")
                print(f"  Function: {frame.name}")
                print(f"  代码: {frame.line}\n")
        finally:
            return
                
    def _train_one_batch(self, img, mask, device):
        """训练一个批次的数据"""
        img = img.to(device)
        mask = mask.to(device)
        seg = self.model(img)
        loss = self.loss_criterion(seg, mask)[0]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self.reduce_tensor(loss.data, self.wordSize).item()
        
    def _record_training_loss(self, epoch, kk, train_small_losses, iter_count):
        """记录训练损失信息"""
        tmpLoss = train_small_losses.update2()
        self.writer.add_scalar('Loss/TrainSmallLoss', tmpLoss, train_small_losses.id)

        text = f"TRAIN [Epoch {epoch} | {self.epochs}] [Proce {kk} | {len(self.trainLoader)}] [Loss {tmpLoss:.4f}]"
        self.progress1.emit(text, 1)
        
        # 写入损失日志文件
        with open(self.text_loss_path, "a", encoding="utf-8") as f:
            f.write(f"{iter_count + 1} {tmpLoss:.4f}\n")
        f.close()
        
    def _update_progress_info(self, epoch, progress1_count, start_time):
        """更新并显示训练进度信息"""
        with torch.no_grad():
            userTime = time.time() - start_time
            surplusTime = userTime / (progress1_count + 1) * (self.valCount - progress1_count - 1)
            logInfo2 = '[第 %d 轮训练进度 %.2f%%] [已耗时 %ds] [预计剩余时间 %ds]' % (
                (epoch + 1), (progress1_count + 1) / self.valCount * 100, userTime, surplusTime)
            # if self.valCount % 10 == 0:
            #     progress1_count = 0
            self.progress1.emit(logInfo2, progress1_count)
            
    def _perform_validation(self, epoch, train_big_losses, evalVal, lastEvalVal, iter_count, view_count, valSampler):
        """执行验证过程并返回更新后的评估值"""
        dist.barrier()
        torch.cuda.empty_cache()
        if self.curRankId == 0:
            self.writer.add_scalar('Loss/TrainBigLoss', train_big_losses.update2(), train_big_losses.id)
        self.model.eval()
        valSampler.set_epoch(epoch)
        
        try:
            with torch.no_grad():
                random_v = random.randint(0, len(self.valLoader) - 1)
                eval_start_time = time.time()
                for kk, (img, mask, name) in enumerate(self.valLoader):
                    if self.train_stop:
                        return None
                    img = img.to(self.device)
                    mask = mask.to(self.device)
                    seg = self.model(img)
                    eval = self.eval_metric(seg, mask)
                    evalVal.update(eval)

                    eval_userTime = time.time() - eval_start_time
                    eval_surplusTime = eval_userTime / (kk + 1) * (len(self.valLoader) - kk - 1)
                    eval_logInfo = '[第 %d 轮验证进度 %.2f%%] [已耗时 %ds] [预计剩余时间 %ds]' % (
                        (epoch + 1), (kk + 1) / len(self.valLoader) * 100, eval_userTime, eval_surplusTime)

                    if kk == random_v and (epoch + 1) % 10 == 0:
                        ns = seg.cpu().detach().numpy()  # 训练结果--Tag
                        ma = mask.cpu().detach().numpy()  # 目标标签
                        imgs = img.cpu().detach().numpy()  # 原图
                        ns_i = ns[:, 0]
                        ma_i = ma[:, 0]
                        img_i = imgs[:, 0]
                        self.save_result(ns_i, ma_i, img_i)
                        self.progress1.emit(eval_logInfo, 0)
                    else:
                        self.progress1.emit(eval_logInfo, kk)

                dist.barrier()
                curEvalVal = evalVal.update2()
                curEvalVal = self.reduce_tensor(curEvalVal, self.wordSize)
                if self.curRankId == 0:
                    self.writer.add_scalar('Eval/EvalVal', curEvalVal, evalVal.id)

                if view_count < 2 and curEvalVal > 0:
                    view_count += 1
                if view_count == 2:
                    self.show_preview.emit(True)

                # 保存最佳模型
                if curEvalVal > lastEvalVal:
                    if self.curRankId == 0:
                        self.progress0.emit('验证集损失减少,保存模型')
                        self.modSavePath = join(self.savePath, "supernet_%s_best_%.4f.pth" % (
                            str(epoch).zfill(5), curEvalVal))
                        # Update
                        torch.save({'state_dict': self.model.module.state_dict(), 'param': self.optimizer},
                                   self.modSavePath)
                        # Copy
                        self.update_modSavePath.emit(self.modSavePath)
                        # 如果保存的模型数大于一，去掉之前的模型
                        if len([l for l in os.listdir(self.savePath) if ".pth" in l]) > 1:
                            self.delete_oldest_file(self.savePath)
                    lastEvalVal = curEvalVal

                    with open(self.text_save_model_path, "a", encoding="utf-8") as f:
                        f.write(f"{iter_count + 1} 验证集损失减少,保存模型 {curEvalVal:.4f}\n")
                    f.close()
                    
                # 更新学习率
                self.lr_scheduler.step(curEvalVal)
                lr = self.optimizer.param_groups[0]['lr']
                if self.curRankId == 0:
                    self.writer.add_scalar('TrainParam/Lr', lr, evalVal.id)
                    
                    text = f"VAL [Epoch {epoch} | {self.epochs}] [EvalVal; {curEvalVal:.4f}] [Lr: {lr}] [ValLen {len(self.valLoader)}]"
                    self.progress0.emit(text)
                    
                    with open(self.text_eval_path, "a", encoding="utf-8") as f:
                        f.write(f"{iter_count + 1} {curEvalVal:.4f}\n")
                    f.close()
            
            return curEvalVal, view_count
            
        finally:
            torch.cuda.empty_cache()
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

        # 预测结果
        mask_to_skeleton_swc(net_seg_i, join(path, f"net_predict{i + 1}.swc"))
        pre_i = swc_to_mask(net_seg_i, join(path, f"net_predict{i + 1}.swc"))

        tifffile.imwrite(join(path, f"net_predict{i + 1}.tif"), pre_i, compression="lzw")
        tifffile.imwrite(join(path, f"mask{i + 1}.tif"), mask_i, compression="lzw")
        tifffile.imwrite(join(path, f"img{i + 1}.tif"), img_i, compression="lzw")

        img_i_2d = self.MaxProject(img_i, 1)
        mask_i_2d = self.MaxProject(mask_i, 0)
        pre_i_2d = self.MaxProject(pre_i, 0)

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

    def reduce_tensor(self, tensor, world_size):
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

    def cleanup(self):
        dist.destroy_process_group()

    def delete_oldest_file(self, directory):
        # 获取目录中的所有文件
        files = [l for l in os.listdir(directory) if ".pth" in l]
        if not files:
            print("目录为空，没有文件可删除。")
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
        logPath = exe_cfg.cfg['ExePath']['cell_logs_path']  # 日志路径
        # 设置环境变量，指定当前进程可以使用的 GPU 设备
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

        # 初始化分布式训练环境
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", init_method='env://', rank=0, world_size=1)

        # 获取当前进程的排名
        self.curRankId = torch.distributed.get_rank()

        # torch.cuda.set_device(self.curRankId + self.s_gpu_id)
        # self.device = torch.device("cuda", self.curRankId + self.s_gpu_id)
        # self.device = torch.device("cuda", 1)

        # 在单GPU环境中，可以直接使用单GPU训练逻辑
        if self.curRankId == 0:
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

            # 启动 TensorBoard
            print(logAdd)

            # if not os.path.exists(logAdd):
            #     os.makedirs(logAdd)
            #     # 设置端口和接口
            # port = exe_cfg.cfg['Port']['cell_port']  # 默认端口是6008，你可以根据需要更改
            # host = exe_cfg.cfg['Host']['cell_host']  # 默认主机地址是127.0.0.1，你可以根据需要更改
            #
            # subprocess.Popen([
            #     "tensorboard", f"--logdir={logAdd}", f"--host={host}", f"--port={port}"], shell=True)

            if not os.path.exists(logAdd):
                os.makedirs(logAdd)
                # 设置端口和接口
            port = exe_cfg.cfg['Port']['cell_port']  # 默认端口是6008，你可以根据需要更改
            host = exe_cfg.cfg['Host']['cell_host']  # 默认主机地址是127.0.0.1，你可以根据需要更改

            """启动 TensorBoard"""
            while self.is_port_in_use(host, port):
                print(f"TensorBoard is already running on port {port}.")
                port = random.randint(6006, 6100)
                exe_cfg.cfg['Port']['cell_port'] = str(port)

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
                    print(f"TensorBoard服务已启动（使用TensorFlowModule）: {url}")
                    return True
                except Exception as e:
                    print(f"使用TensorFlow模块启动TensorBoardFailed: {e}")
                    
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
                    print(f"TensorBoard服务已启动（使用系统命令），访问地址: http://{host}:{port}")
                    return True
                except Exception as e:
                    print(f"使用系统命令启动TensorBoardFailed: {e}")
                    
                # Method3: 如果前两种方法都失败，提供手动启动指引
                print("TensorBoard启动失败，建议手动启动：")
                print(f"1. 打开命令提示符")
                print(f"2. 运行命令: tensorboard --logdir={logAdd} --host={host} --port={port}")
                print(f"3. 在浏览器中访问: http://{host}:{port}")
                return False
            
            # 启动TensorBoard
            start_tensorboard()
            
            # 即使TensorBoard启动失败，训练也可以继续

            # subprocess.Popen([
            #     "tensorboard", f"--logdir {logAdd}", f"--host {host}", f"--port {port}"])

            # # 获取当前程序的环境变量
            # current_env = os.environ.copy()
            #
            # # 定义命令
            # command = [
            #     "tensorboard",
            #     f"--logdir={logAdd}",
            #     f"--host={host}",
            #     f"--port={port}"
            # ]
            #
            # # 启动子进程
            # process = subprocess.Popen(command, env=current_env)

            # 使用 TensorFlow 的 TensorBoard 模块启动 TensorBoard
            # tb = tensorboard.program.TensorBoard()
            # tb.configure(
            #     argv=[
            #         None,
            #         f"--logdir", f"{logAdd}",
            #         f"--host", f"{host}",
            #         f"--port", f"{port}",
            #         # '--bind_all'
            #     ]
            # )
            # try:
            #     url = tb.launch()
            #     print(f"TensorBoard is running at {url}")
            #
            #     # 等待几秒，Ensure TensorBoard 服务已经启动
            #     # time.sleep(2)
            #
            #     # # 打开浏览器
            #     # webbrowser.open(url)
            # except Exception as e:
            #     print(f"Failed to start TensorBoard: {e}")

            self.savePath = join(exe_cfg.cfg['ExePath']['cell_models_path'], f"{expName}")  # 模型保存路径
            if not os.path.isdir(self.savePath):
                os.makedirs(self.savePath)

            self.text_eval_path = join(self.savePath, 'text_eval.txt')
            self.text_save_model_path = join(self.savePath, 'text_save_model.txt')
            self.text_loss_path = join(self.savePath, 'text_loss.txt')
            with open(self.text_eval_path, 'w', encoding='utf-8') as f:
                f.write("")
            with open(self.text_save_model_path, 'w', encoding='utf-8') as f:
                f.write("")
            with open(self.text_loss_path, 'w', encoding='utf-8') as f:
                f.write("")

    def to_stop(self):
        self.train_stop = True
        time.sleep(3)
        # try:
        #     # 删除模型引用
        #     if hasattr(self, 'super_net'):
        #         del self.model
        #         print("模型已删除")
        #     del self.trainLoader
        #     del self.valLoader
        #     del self.device
        #     del self.writer
        #     del self.loss_criterion
        #     del self.optimizer
        #     del self.lr_scheduler
        #     del self.eval_metric
        #     # 清理 GPU Cache
        #     torch.cuda.empty_cache()
        #     print("GPU 缓存已清理")
        # except Exception as e:
        #     print("Stop")

    def is_port_in_use(self, host, port):
        """检查指定端口是否被占用"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((str(host), int(float(port)))) == 0


def mask_to_skeleton_swc(mask, savePath):
    with open(savePath, "w", encoding="utf-8") as f:
        pass

    thre = 5000
    mask2 = mask.copy()
    mask2[mask2 <= 103] = 0
    mask2[mask2 > 0] = 255
    labels_out, N = cc3d.connected_components(mask2, connectivity=26, return_N=True, out_dtype=np.uint32)

    flag = 1
    if N != 0:
        out1 = speed_cc3d.group_type(labels_out)
        # out1Ls = len(out1)
        # out1Ls_step = 1
        center_point = []
        for i, key in enumerate(out1):
            # if i % out1Ls_step == 0:
            #     ProcessQueue.put((f"{((i + 1) / out1Ls):.2f}%", i))
            # print('进度：%f' % (i / len(out1) * 100) + '%')
            point = np.array(out1[key], dtype=np.int32)
            if len(point) <= 297 or len(point) >= thre:
                continue
            dist = distance.cdist(point, point)
            dist_max = np.max(dist)
            rhos = []
            for j in range(len(point)):
                rhos.append(mask[point
                [j, 2], point[j, 1], point[j, 0]])
            max_rho = max(rhos)
            # 计算每个点的最小距离
            sigmas = np.zeros(len(point))
            nearest_neighbor = np.zeros(len(point))
            sorted_id = sorted(range(len(rhos)), key=lambda k: rhos[k], reverse=True)
            for j, index in enumerate(sorted_id):
                if j == 0:
                    sigmas[index] = 20
                    continue
                higher_rho_idx = sorted_id[:j]
                sigmas[index] = np.min(dist[index, higher_rho_idx])
                temp = np.argmin(dist[index, higher_rho_idx]).astype(int)
                nearest_neighbor[index] = higher_rho_idx[temp]
            # plt.scatter(rhos, sigmas)
            # plt.show()
            # center_idx = sorted_id[0]
            center_idx = np.where(sigmas > 5)[0]
            for o in range(len(center_idx)):
                p = point[center_idx[o]]
                tmp = np.array([flag, 0, p[0], p[1], p[2], 0, -1])
                center_point.append(tmp)
                flag += 1
        if len(center_point) != 0:
            center_point = np.array(center_point)
            np.savetxt(savePath, center_point)
        # else:
        #     temp_p = np.array([0, 0, 0, 0, 0, 0, -1]).reshape(1, -1)
        #     np.savetxt(savePath, temp_p)


def swc_to_mask(seg2, swc_path):
    shapes = np.array(seg2.shape)[::-1]  # xyz

    kernelLen = 10
    imgShape = np.array(shapes, dtype=np.int32)  # xyz
    maxD = ((kernelLen ** 2) * 3) ** 0.5
    minD = 0.0697
    maskMax = -np.log(minD)
    dfImg = np.zeros(imgShape[::-1], dtype=np.float32)  # zyx
    kernelArr = []
    for z in range(-kernelLen, kernelLen + 1):
        for y in range(-kernelLen, kernelLen + 1):
            for x in range(-kernelLen, kernelLen + 1):
                kernelArr.append([x, y, z])
    kernelArr = np.array(kernelArr, dtype=np.int32)

    swcData = np.loadtxt(swc_path, ndmin=2)
    dfImg[...] = maxD
    for item in swcData:
        if len(item) < 7:
            continue
        p0 = item[2: 5]  # xyz
        curPc = GetPcKernelPc(np.array([p0]), kernelArr, imgShape)
        d = np.linalg.norm(p0 - curPc, axis=1)
        dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]] = np.min([dfImg[curPc[:, 2], curPc[:, 1], curPc[:, 0]], d],
                                                              axis=0)
    dfImg = -np.log(minD + dfImg / maxD * (1 - minD))
    dfImg2 = (dfImg / maskMax * 255).astype(np.uint8)
    # dfImg2_2d = MaxProject(dfImg2)
    return dfImg2

def GetPcKernelPc(pc, kernelArr, imgShape):
    curPc = (pc[:, None] + kernelArr[None]).reshape([-1, 3])
    curPc = np.round(curPc).astype(np.int32)
    curPc[curPc < 0] = 0
    curPc[curPc[:, 2] > imgShape[2] - 1, 2] = imgShape[2] - 1
    curPc[curPc[:, 1] > imgShape[1] - 1, 1] = imgShape[1] - 1
    curPc[curPc[:, 0] > imgShape[0] - 1, 0] = imgShape[0] - 1
    curPc = np.unique(curPc, axis=0)
    return curPc


if __name__ == '__main__':
    # import tensorboard.program
    # import subprocess
    # import time
    logAdd = r"D:\CellNeuralBloodVessel\IntegratePose\logs\cell_logs\exp001"
    host = r"localhost"
    port = 6008

    # tb = tensorboard.program.TensorBoard()
    # tb.configure(
    #     argv=[
    #         None,
    #         fr"--logdir={logAdd}",
    #         f"--host={host}",
    #         f"--port={port}",
    #     ]
    # )
    # url = tb.launch()
    # print(f"TensorBoard is running at {url}")

    # subprocess.Popen([
    #     "tensorboard", f"--logdir={logAdd}", f"--host={host}", f"--port={port}"], shell=True)
    #
    # time.sleep(100)
