import os
# 必须在任何 import torch 之前执行
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch, importlib
from Unet3D.LossPy import BCEDiceLoss, ComputePR, LSDLoss, EvalScore
from torch import nn as nn


def GetLossOptimiLr(model, learning_rate=0.0002, weight_decay=0.00001):
    optimizer_config = {
        # 'learning_rate': 0.0002,
        'learning_rate': learning_rate,
        # weight decay
        'weight_decay': weight_decay
    }
    lr_config = {
        # reduce learning rate when evaluation metric plateaus
        'name': 'ReduceLROnPlateau',
        # use 'max' if eval_score_higher_is_better=True, 'min' otherwise
        'mode': 'max',
        # factor by which learning rate will be reduced
        'factor': 0.5,
        # number of *validation runs* with no improvement after which learning rate will be reduced
        'patience': 15
    }
    # Loss
    alpha = 1
    beta = 1
    # loss_criterion = BCEDiceLoss(alpha, beta)
    # loss_criterion = nn.BCEWithLogitsLoss(pos_weight=None)
    # loss_criterion = nn.L1Loss(reduction='none')
    # Loss function: 加权损失函数
    loss_criterion = LSDLoss()
    # Optimizer
    learning_rate = optimizer_config['learning_rate']  # 它是一个正数，控制每次参数更新的步长
    weight_decay = optimizer_config.get('weight_decay', 0)
    betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
    eps = 1e-8  # 为了数值稳定性而添加到分母中的小常数。防止除零错误
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay, eps=eps)
    # Learning rate
    class_name = lr_config.pop('name')  # 'ReduceLROnPlateau'
    m = importlib.import_module('torch.optim.lr_scheduler')  # 动态导入指定的模块
    clazz = getattr(m, class_name)  # 从导入的模块 m 中获取名为 ReduceLROnPlateau 的类
    lr_config['optimizer'] = optimizer
    lr_scheduler = clazz(**lr_config)  # 使用 lr_config 字典中的参数实例化 clazz Class，创建学习率调度器对象 lr_scheduler
    # Evaluate
    # eval_metric = ComputePR(0.3, 0.7, 0.5)
    # eval_metric = nn.L1Loss(reduction='none')
    # eval_metric = LSDLoss()
    # 评估指标
    eval_metric = EvalScore()
    return loss_criterion, optimizer, lr_scheduler, eval_metric