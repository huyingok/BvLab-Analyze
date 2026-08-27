from torch.nn import functional as F
import tifffile, torch, os
from os.path import join
import torch.nn as nn
import numpy as np
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
# from sklearn.metrics import precision_recall_curve
from torchmetrics.classification import BinaryRecall, BinaryPrecision
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    # assert input.dim() == 3 or not reduce_batch_first
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):#logits,
        num = targets.size(0)
        smooth = 1
        #probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        mm1 = (m1 * m1)
        mm2 = (m2 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (mm1.sum(1) + mm2.sum(1) + smooth)
        score = 1 - score.sum() / num
        BCE = F.binary_cross_entropy(probs, targets, reduction='mean')
        return 0.6 * score + 0.4 * BCE

class DiceLoss(nn.Module):
    """
    Ldice(s, m) = 1 - 2 * (|s ∩ m| + 1e-5)/(|s ∪ m| + 1e-5)
    """
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5
        self.thre = 99. / 255

    def forward(self, predict, target):
        num = predict.size(0)
        pre = predict.view(num, -1)
        pre[pre > self.thre] = 1
        pre[pre <= self.thre] = 0
        tar = target.view(num, -1)
        tar[tar > self.thre] = 1
        tar[tar <= self.thre] = 0
        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        return score

class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""
    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


class ComputePR(nn.Module):
    def __init__(self, wp, wr, thre):
        super(ComputePR, self).__init__()
        self.wp = wp
        self.wr = wr
        self.thre = thre
        self.pre = BinaryPrecision()
        self.rec = BinaryRecall()

    def forward(self, input, target):
        input[input > self.thre] = 1
        input[input < 1] = 0
        target[target > self.thre] = 1
        target[target < 1] = 0
        if not input.is_contiguous():
            # 在.view()之前使用.contiguous()
            input = input.contiguous()
        input = input.view(-1)
        target = target.view(-1)
        p = self.pre(input, target)
        r = self.rec(input, target)
        return p, r, p * self.wp + r * self.wr

class LSDLoss(nn.Module):
    def __init__(self):
        super(LSDLoss, self).__init__()
        self.alpha = 0.5
        self.maxThre = 3. / 255
        self.maxThre2 = 103 / 255
        self.epsilon = 1e-5
        self.thre = 99. / 255
        self.pos_weight = torch.tensor([3.0], device=torch.device("cuda:0"))  # 正样本权重
        self.bcelog_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_weight)
        self.l1_loss = nn.L1Loss(reduction='none')
        self.sigmod = nn.Sigmoid()

    def forward(self, input, mask, ngrad, gradient):
        # mask2 = mask > self.maxThre
        # mask2_sum = max(1, mask2.sum())
        # mask3 = mask > self.maxThre2
        # mask3_sum = max(1, mask3.sum())

        bceWeight = self.bcelog_loss(input, mask)
        bceLoss = bceWeight.mean()
        # bceLoss = bceWeight.mean() * 0.6
        # bceLoss = bceWeight.mean() * 0.5 + bceWeight[mask2].sum() / mask2_sum + bceWeight[mask3].sum() / mask3_sum
        # 可微操作
        pre = torch.sigmoid((input - self.thre) * 100)
        tar = torch.sigmoid((mask - self.thre) * 100)
        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        # 非可微操作
        # num = input.size(0)
        # pre = input.view(num, -1)
        # pre[pre > self.thre] = 1
        # pre[pre <= self.thre] = 0
        # tar = mask.view(num, -1)
        # tar[tar > self.thre] = 1
        # tar[tar <= self.thre] = 0
        # intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        # union = (pre + tar).sum(-1).sum()
        # score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        # hybrid = bceLoss + score

        # ngrad = self.sigmod(ngrad)
        l1 = self.l1_loss(ngrad, gradient)
        l1Loss = l1.mean()
        # l1Loss = l1.mean() * 0.4
        # l1Loss = l1.mean() * 0.5 + l1[mask2].sum() / mask2_sum + l1[mask3].sum() / mask3_sum
        # loss2 = 1 - dice_coeff(input, mask, reduce_batch_first=True)
        # rawloss = [hybrid, l1Loss]
        rawloss = [bceLoss, l1Loss, score]
        totalLoss = sum(rawloss)
        return totalLoss, rawloss
        # return l1Loss, [l1Loss]
        # return bceLoss, [bceLoss]


class EvalScore(nn.Module):
    def __init__(self):
        super(EvalScore, self).__init__()
        self.alpha = 0.5
        self.maxThre = 3. / 255
        self.maxThre2 = 103 / 255
        self.epsilon = 1e-5
        self.thre = 99. / 255
        self.pos_weight = torch.tensor([10.0], device=torch.device("cuda:0"))  # 正样本权重
        self.bcelog_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_weight)
        self.l1_loss = nn.L1Loss(reduction='none')
        self.sigmod = nn.Sigmoid()

    def forward(self, input, mask, ngrad, gradient):
        # mask2 = mask > self.maxThre
        # mask2_sum = max(1, mask2.sum())
        # mask3 = mask > self.maxThre2
        # mask3_sum = max(1, mask3.sum())
        bceWeight = self.bcelog_loss(input, mask)
        bceLoss = bceWeight.mean()
        # bceLoss = bceWeight.mean() * 0.6

        # 可微操作
        pre = torch.sigmoid((input - self.thre) * 100)
        tar = torch.sigmoid((mask - self.thre) * 100)
        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        hybrid = (bceLoss + score)

        # ngrad = self.sigmod(ngrad)
        l1 = self.l1_loss(ngrad, gradient)
        l1Loss = l1.mean()
        # l1Loss = l1.mean() * 0.4
        # l1Loss = l1.mean() * 0.5 + l1[mask2].sum() / mask2_sum + l1[mask3].sum() / mask3_sum
        # return 1 - l1Loss
        # return 1 - bceLoss
        return 2 - (hybrid + l1Loss)


class DistLSDLoss(nn.Module):
    def __init__(self, th):
        super(DistLSDLoss, self).__init__()
        # 定义了两个阈值, 区分前景、背景或特定区域
        self.epsilon = 1e-5
        self.thre = float(th) / 255
        self.alpha = 0.2
        self.beta = 1 - self.alpha
        self.maxThre = 3. / 255
        # self.maxThre = 50. / 255
        self.maxThre2 = 103 / 255
        # self.maxThre2 = 150 / 255
        # self.maxThre3 = 133 / 255
        self.pos_weight = torch.tensor([10.0], device=torch.device("cuda:0"))  # 正样本权重
        self.bcelog_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_weight)
        # L1Loss计算输入x和目标y之间的差的绝对值，然后根据reduction参数的设置来决定如何处理这些差值。
        # 如果reduction设置为'mean'（Default value），则会计算所有差值的平均值；如果设置为'sum'，则会计算所有差值的总和。如果设置为'none'，则不会对差值进行任何处理，直接返回
        self.l1_loss = nn.L1Loss(reduction='none')  # Loss function, 不对损失值进行汇总, 返回每个像素的损失值
        self.sigmod = nn.Sigmoid()  # Activation function，它将输入值映射到0和1之间的输出

    """加权损失函数"""
    def forward(self, input, dist):
        # 计算输入图像 input 和目标距离场 dist 之间的差异
        # dist2 = dist > self.maxThre
        # dist2_sum = max(1, dist2.sum())
        # dist3 = dist > self.maxThre2
        # dist3_sum = max(1, dist3.sum())
        #
        # back1 = input > self.maxThre2
        # back1_sum = max(1, back1.sum())

        bceWeight = self.bcelog_loss(input, dist)
        bceLoss = bceWeight.mean()

        # 可微操作
        pre = self.sigmod((input - self.thre) * 100)
        tar = self.sigmod((dist - self.thre) * 100)
        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        hybrid = bceLoss + score

        # # # 将输入 input 通过 Sigmoid Activation function，将其值归一化到 [0, 1] 范围内
        # input = self.sigmod(input)
        # l1 是一个张量，每个像素位置的值表示输入和目标之间的绝对误差
        # l1 = self.l1_loss(input, dist)
        # l1Loss = l1.mean()
        # 最终的加权损失
        # l1Loss = (l1.mean() +  # 计算全局 L1 损失的平均值
        #           l1[dist2].sum() / dist2_sum +  # 计算掩码 mask2 区域内的 L1 Loss，并对其进行归一化（除以该区域的像素数量）
        #           l1[dist3].sum() / dist3_sum +  # 计算掩码 mask3 区域内的 L1 Loss，并进行归一化
        #           l1[back1].sum() / back1_sum)  # 计算掩码 back1 区域内的 L1 Loss，并进行归一化
        # l1Loss = (l1.mean() +  # 计算全局 L1 损失的平均值
        #           l1[dist2].sum() / dist2_sum +  # 计算掩码 mask2 区域内的 L1 Loss，并对其进行归一化（除以该区域的像素数量）
        #           l1[dist3].sum() / dist3_sum)  # 计算掩码 mask3 区域内的 L1 Loss，并进行归一化

        # totalLoss = [l1Loss * self.alpha, bceLoss * self.beta]
        # return l1Loss, [l1Loss]
        # return bceLoss, [bceLoss]
        return hybrid, [hybrid]
        # return sum(totalLoss), totalLoss


class DistEvalScore(nn.Module):
    def __init__(self, th):
        super(DistEvalScore, self).__init__()
        self.epsilon = 1e-5
        self.thre = float(th) / 255
        self.alpha = 0.2
        self.beta = 1 - self.alpha
        self.maxThre = 3. / 255
        # self.maxThre = 50. / 255
        self.maxThre2 = 103 / 255
        # self.maxThre2 = 150 / 255
        # self.maxThre3 = 133 / 255
        self.pos_weight = torch.tensor([10.0], device=torch.device("cuda:0"))  # 正样本权重
        self.bcelog_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_weight)
        self.l1_loss = nn.L1Loss(reduction='none')
        self.sigmod = nn.Sigmoid()

    def forward(self, input, dist):
        # dist2 = dist > self.maxThre
        # dist2_sum = max(1, dist2.sum())
        # dist3 = dist > self.maxThre2
        # dist3_sum = max(1, dist3.sum())
        #
        # back1 = input > self.maxThre2
        # back1_sum = max(1, back1.sum())
        bceWeight = self.bcelog_loss(input, dist)
        bceLoss = bceWeight.mean()

        # 可微操作
        pre = self.sigmod((input - self.thre) * 100)
        tar = self.sigmod((dist - self.thre) * 100)
        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        hybrid = bceLoss + score

        # input = self.sigmod(input)
        # l1 = self.l1_loss(input, dist)
        # l1Loss = l1.mean()
        # l1Loss = (l1.mean() +
        #           l1[dist2].sum() / dist2_sum +
        #           l1[dist3].sum() / dist3_sum +
        #           l1[back1].sum() / back1_sum)
        # l1Loss = (l1.mean() +
        #           l1[dist2].sum() / dist2_sum +
        #           l1[dist3].sum() / dist3_sum)
        # totalLoss = [l1Loss * self.alpha, bceLoss * self.beta]
        # return 2 - l1Loss
        # return 1 - bceLoss
        return 2 - hybrid
        # return 2 - sum(totalLoss)


if __name__ == '__main__':
    # maskPath = r'G:\qxz\DataSet\BrainSegDataSet301\DataSet2\OriDataSetSpilt2\mask'
    # prePath = r'G:\qxz\DataSet\BrainSegDataSet301\DataSet2\OriDataSetSpilt2\Predict2\result'
    maskPath = r'D:\omnipose-main2\train3d\tests\masks'
    # maskPath = r'D:\xueguan\TrainDateSet3\Train2\radius_mask\mask'
    # prePath = r'D:\omnipose-main2\train3d\tests\masks'
    prePath = r'D:\omnipose-main2\train3d\tests\images\mask'
    name = 'merge1_5_3_0.tif'
    # name = '1_2_0_0.tif'
    # name = '00012_1_1_1.tif'
    mask = tifffile.imread(join(maskPath, name)) / 255.
    mask = np.expand_dims(mask, axis=0).astype(np.float32)
    # mask1 = np.stack([mask, mask], axis=0)
    mask1 = torch.from_numpy(mask)

    seg = tifffile.imread(join(prePath, name)) / 255.
    seg = np.expand_dims(seg, axis=0).astype(np.float32)
    # seg1 = np.stack([seg, seg], axis=0)
    seg1 = torch.from_numpy(seg)

    LSDLoss = LSDLoss()
    loss = LSDLoss(mask1, seg1)
    print("LSDLoss: ", loss[0])

    precise_dice_loss = SoftDiceLoss()
    loss = precise_dice_loss(mask1, seg1)
    print("SoftDiceLoss: ", loss)

    dice = DiceLoss()
    loss = dice(mask1, seg1)
    print("DiceLoss: ", loss)
