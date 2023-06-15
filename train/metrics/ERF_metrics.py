
import torch
import numpy as np

def miou_ERF(ignore_labels=[],eps=1e-6,**kwargs):
    def calculate_iou(ypred, ytrue):
        seed = torch.sigmoid(ypred[:,-3:]).cpu().detach().numpy()
        ypred = seed > 0.5
        #pred=pred[:,-1]
        ytrue=ytrue.cpu().detach().numpy()
        # print(ypred.shape,ytrue.shape)
        for lab in ignore_labels:
            ypred[:,lab]=0
            ytrue[:,lab]=0
        ytrue = ytrue.astype(np.int8)
        ypred = ypred.astype(np.int8)

        inter = np.sum(ypred & ytrue)
        union = np.sum(ypred | ytrue)
        iou = inter / (union + eps)
        return iou
    return calculate_iou

def miou_wnet(ignore_labels=[],eps=1e-6,**kwargs):
    def calculate_iou(ypred, ytrue):
        ytrue=ytrue.cpu().detach().numpy()
        ypred=ypred[0][:,:]
        seed = torch.sigmoid(ypred).cpu().detach().numpy()
        ypred = seed > 0.5
        for lab in ignore_labels:
            ypred[:,lab]=0
            ytrue[:,lab]=0
        ytrue = ytrue.astype(np.int8)
        ypred = ypred.astype(np.int8)

        inter = np.sum(ypred & ytrue)
        union = np.sum(ypred | ytrue)
        iou = inter / (union + eps)
        return iou
    return calculate_iou
