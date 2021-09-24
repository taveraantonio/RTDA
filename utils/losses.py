import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import numpy as np
#import matplotlib.pyplot as plt


def get_target_tensor(input_tensor, mode):
    # Source tensor = 0.0
    # Target tensor =  1.0
    source_tensor = torch.FloatTensor(1).fill_(0.0)
    target_tensor = torch.FloatTensor(1).fill_(1.0)
    source_tensor = source_tensor.expand_as(input_tensor)
    target_tensor = target_tensor.expand_as(input_tensor)
    if mode == 'source':
        return source_tensor
    elif mode == 'target':
        return target_tensor


def get_target_tensor_mc(input_tensor, mode):
    # Source tensor = 0.0
    # Target tensor =  1.0
    source_tensor = torch.FloatTensor(1).fill_(0.0)
    target_tensor = torch.FloatTensor(1).fill_(1.0)
    #source_tensor = source_tensor.expand_as(input_tensor)
    source_tensor = source_tensor.expand((input_tensor.shape[0], input_tensor.shape[2], input_tensor.shape[3]))
    #target_tensor = target_tensor.expand_as(input_tensor)
    target_tensor = target_tensor.expand((input_tensor.shape[0], input_tensor.shape[2], input_tensor.shape[3]))
    if mode == 'source':
        return source_tensor
    elif mode == 'target':
        return target_tensor

# Cross Entropy loss used for the semantic segmentation model
class CrossEntropy2d(nn.Module):

    def __init__(self, reduction='mean', ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
        Args:
        predict:(n, c, h, w)
        target:(n, h, w)
        weight (Tensor, optional): a manual rescaling weight given to each class.
                                   If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction=self.reduction)

        return loss


class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=0.25, gamma=2.0, balance_index=2, size_average=True, ignore_label=255):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6
        self.ignore_label = ignore_label

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float,int)):
            assert 0<self.alpha<1.0, 'alpha should be in `(0,1)`)'
            assert balance_index >-1
            alpha = torch.ones((self.num_class))
            alpha *= 1-self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha,torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):
        logit = F.softmax(logit, dim=1)
        n, c, h, w = logit.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        logit = logit.transpose(1, 2).transpose(2, 3).contiguous()
        logit = logit[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        target = target.view(-1, 1)

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            alpha = self.alpha.to(logpt.device)
            alpha_class = alpha.gather(0,target.view(-1))
            logpt = alpha_class*logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index):
        super().__init__()
        self.epsilon = 1e-5
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = output.size()[1]
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)

        intersection = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice_score = 2. * intersection / (denominator + self.epsilon)
        return torch.mean(1. - dice_score)


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes+1, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target[:, :classes, ...]
