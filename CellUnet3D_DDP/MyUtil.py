# -*- coding: utf-8 -*-
import os
# 必须在任何 import torch 之前执行
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import torch, importlib
from CellUnet3D_DDP.LossPy import BCEDiceLoss, ComputePR, LSDLoss, EvalScore
from torch import nn as nn


def GetLossOptimiLr(model, learning_rate=0.0002, weight_decay=0.00001):
    optimizer_config = {
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
    loss_criterion = LSDLoss()
    # Optimizer
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', 0)
    betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    # Learning rate
    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    lr_config['optimizer'] = optimizer
    lr_scheduler = clazz(**lr_config)
    # Evaluate
    # eval_metric = ComputePR(0.3, 0.7, 0.5)
    # eval_metric = nn.L1Loss(reduction='none')
    # eval_metric = LSDLoss()
    eval_metric = EvalScore()
    return loss_criterion, optimizer, lr_scheduler, eval_metric


class TakeNotesLoss:
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.id = -1

    def update(self, value):
        self.sum += value
        self.count += 1

    def update2(self):
        if self.count != 0:
            tmp = self.sum / self.count
        else:
            tmp = 0
        self.sum = 0
        self.count = 0
        self.id += 1
        return tmp
