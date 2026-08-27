import os
import time
from os.path import join
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from Unet3D.models.model import LoadModel
from torch.utils.data import DataLoader
from Unet3D.DataLoader import GetMultiTypeMemoryDataSetAndCropQxz
import numpy as np
import torch
from tensorboardX import SummaryWriter
from Unet3D.Net import Trainer
from Unet3D.MyUtil import GetLossOptimiLr
from pathlib import Path
import sys
import traceback
import subprocess
import socket
import random
from config import exe_cfg, cfgPath
from PyQt5.QtCore import QThread, pyqtSignal


import gpu_device_use


class NeuralTrainQThread(QThread):
    finish0 = pyqtSignal(str)  # 训练完成
    progress0 = pyqtSignal(str)  # 训练进度文本
    progress1 = pyqtSignal(str, int)  # 详细进度
    error0 = pyqtSignal(str)  # 报错文本
    update_modSavePath = pyqtSignal(str)  # 模型更新
    update_plot = pyqtSignal(str, str, str)  # 画图
    preview0 = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # 验证结果预览
    show_preview = pyqtSignal(bool)  # 是否训练可视化

    def __init__(self, *args, **kwargs):
        super(NeuralTrainQThread, self).__init__()
        self.win = kwargs.get('win')
        self.netObj = None
        self.netObj_stop = False
        self.start_train = False

        self.logger = self.win.logger

    def run(self):
        try:
            self.netObj = None
            self.netObj_stop = False
            self.start_train = True
            batchSize = 1  # Batch
            imgPath = self.win.data_train_dict['imgDir']  # 图像文件加路径
            shapes = self.win.data_train_dict['shapes']  # xyz
            epoch = self.win.data_train_dict['epochs']  # 轮次

            rootPath = str(Path(imgPath).parent)
            imgName = str(Path(imgPath).name)
            maskName = "mask"

            imgSize = np.array(shapes, dtype=np.int32)  # 图像尺寸xyz

            trainTxt = r"train.txt"  # 训练的txt名称
            valTxt = r"val.txt"  # 验证的txt名称
            # batchSize = 1  # 每次训练时处理的样本数量
            # batchSize = 2  # 每次训练时处理的样本数量
            backwardNumber = 8  # 多少次反向传播一次

            gpu_id = gpu_device_use.get_gpu_utilization()
            if gpu_id is not None:
                torch.cuda.set_device(gpu_id)
                # torch.cuda.set_device(self.curRankId + self.s_gpu_id)
                # self.device = torch.device("cuda", self.curRankId + self.s_gpu_id)
                device = torch.device("cuda", gpu_id)
            else:
                self.finish0.emit("No available GPU")
                return
            # device = torch.device('cuda:0')  # 使用设备
            # device = torch.device('cpu')
            # logPath = './logs/'  # 日志路径
            logPath = exe_cfg.cfg['ExePath']['neural_logs_path']  # 日志路径

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
            writer = SummaryWriter(logAdd)
            # 启动 TensorBoard
            print(logAdd)

            if not os.path.exists(logAdd):
                os.makedirs(logAdd)
                # 设置端口和接口
            port = exe_cfg.cfg['Port']['neural_port']  # 默认端口是6008，你可以根据需要更改
            host = exe_cfg.cfg['Host']['neural_host']  # 默认主机地址是127.0.0.1，你可以根据需要更改

            """启动 TensorBoard"""
            while self.is_port_in_use(host, port):
                print(f"TensorBoard is already running on port {port}.")
                port = random.randint(6006, 6100)
                exe_cfg.cfg['Port']['neural_port'] = str(port)

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

            # subprocess.Popen([
            #     "tensorboard", f"--logdir={logAdd}", f"--host={host}", f"--port={port}"])

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

            savePath = join(exe_cfg.cfg['ExePath']['neural_models_path'], f"{expName}")  # 模型保存路径
            if not os.path.isdir(savePath):
                os.makedirs(savePath)

            # 加载数据
            train_dataset = GetMultiTypeMemoryDataSetAndCropQxz(rootPath, trainTxt, imgSize, imgName, maskName)
            # DataLoader Yes PyTorch 中用于批量加载数据的工具，shuffle=True 表示在每个 epoch 开始时随机打乱数据，num_workers=0 表示不使用多线程加载数据
            train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, num_workers=0)
            val_dataset = GetMultiTypeMemoryDataSetAndCropQxz(rootPath, valTxt, imgSize, imgName, maskName)
            val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=True, num_workers=0)

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
            # model = LoadModel(modelCfg, r'./ModelSave/exp030/supernet_00040.pth')
            model = LoadModel(modelCfg)
            model.to(device)  # 将模型移动到GPU或CPU
            model.train(True)  # 将模型设置为训练模式
            # 获取损失优化器学习率
            # 获取损失函数、Optimizer、学习率调度器和评估指标
            # 损失函数通常用于衡量模型输出与真实标签之间的差异。
            # 优化器用于更新模型参数，以最小化损失函数。
            # 学习率调度器用于动态调整学习率
            loss_criterion, optimizer, lr_scheduler, eval_metric = GetLossOptimiLr(model)
            eval_metric.to(device)  # 将评估指标移动到GPU或CPU
            # Train: Trainer封装训练和验证的逻辑
            netObj1 = Trainer(train_loader, val_loader, model, loss_criterion, optimizer, lr_scheduler,
                              eval_metric, backwardNumber=backwardNumber, modelPath=savePath, rootPath=rootPath,
                              device=device, batchSize=batchSize,
                              progress0=self.progress0, progress1=self.progress1,
                              update_modSavePath=self.update_modSavePath,
                              preview0=self.preview0, show_preview=self.show_preview)
            self.netObj = netObj1
            if not self.netObj_stop:
                # netObj.Train(turn=2, writer=writer)
                # Train500个 epoch
                self.netObj.Train(turns=epoch, writer=writer)
            del netObj1
            del device
            del self.netObj
            del model
            del loss_criterion
            del optimizer
            del lr_scheduler
            del eval_metric
            self.start_train = False
            self.finish0.emit("")
            print("Training finished")
        except Exception as e:
            self.start_train = False
            self.error0.emit(str(e))
            self.logger.error("\n=== Error message ===")
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Error message: {e}")
            self.logger.error("=== Error location ===")
            tb = sys.exc_info()[2]
            for frame in traceback.extract_tb(tb):
                self.logger.error(f"  File: {frame.filename}")
                self.logger.error(f"  Line number: {frame.lineno}")
                self.logger.error(f"  Function: {frame.name}")
                self.logger.error(f"  Code: {frame.line}\n")

    def train_stop(self):
        if self.netObj:
            self.netObj.Train_Stop()
        else:
            self.netObj_stop = True
        time.sleep(3)

    def is_port_in_use(self, host, port):
        """检查指定端口是否被占用"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((str(host), int(float(port)))) == 0