import csv
import os
from os.path import join
import numpy as np

import numpy as np
import torch
import torch.nn.functional as F
import argparse
def miou(ypred, ytrue, eps=1e-6):
    """caculate miou with ypred and ytrue for all class,include background,

    Args:
        ypred (_type_): ont hot, output  softmax layer
        ytrue (_type_): ont hot
        eps (_type_, optional): _description_. Defaults to 1e-6.

    Returns:
        _type_: miou
    """
    outputs = torch.argmax(ypred, dim=1, keepdim=True).type(torch.int64)
    outputs = torch.zeros_like(ypred).scatter_(dim=1, index=outputs, src=torch.tensor(1.0)).type(torch.int8)
	# ytrue = torch.unsqueeze(ytrue, dim=1).type(torch.int64)
	# ytrue = torch.zeros_like(ypred).scatter_(dim=1, index=ytrue, src=torch.tensor(1.0)).type(torch.int8)
    ytrue = ytrue.type(torch.int8)

    inter = (outputs & ytrue).type(torch.float32).sum(dim=(2,3))
    union = (outputs | ytrue).type(torch.float32).sum(dim=(2,3))
    iou = inter / (union + eps)
    return iou.mean()
def miou_weight( ignore_labels=None,mode="sigmoid"):
    ignore_labels=ignore_labels
    
    def miou_weight(ypred, ytrue, eps=1e-6):
        """caculate miou with ypred and ytrue for all class,include background,

        Args:
            ypred (_type_): ont hot, output  softmax layer
            ytrue (_type_): ont hot
            eps (_type_, optional): _description_. Defaults to 1e-6.

        Returns:
            _type_: miou
        """
        ypred=ypred.cpu()
        outputs = torch.argmax(ypred, dim=1, keepdim=True).cpu()
        ypred = torch.zeros_like(ypred).scatter_(1, outputs, 1).cpu().numpy()

        #ytrue = ytrue.cpu().numpy()
        ytrue=torch.clamp(ytrue, min=0, max=1).cpu().numpy()

        for lab in ignore_labels:
            ypred[:,lab]=0
            ytrue[:,lab]=0
        # ytrue = torch.unsqueeze(ytrue, dim=1).type(torch.int64)
        # ytrue = torch.zeros_like(ypred).scatter_(dim=1, index=ytrue, src=torch.tensor(1.0)).type(torch.int8)
        ytrue = ytrue.astype(np.int8)
        ypred = ypred.astype(np.int8)

        inter = np.sum(ypred & ytrue)
        union = np.sum(ypred | ytrue)
        iou = inter / (union + eps)
        return iou
    return miou_weight
def iou_score(output, target): # binary , only support 0 1
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)
def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()


    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
        
# def str2bool(v):
#     if v.lower() in ['true', 1]:
#         return True
#     elif v.lower() in ['false', 0]:
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')


# def count_params(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

