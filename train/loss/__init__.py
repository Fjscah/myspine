from .loss import  FocalLoss,DiceLoss,Dis_loss,ReconstructionLoss
from .ERF_single_loss import single_SpatialEmbLoss
from .ERF_multi_loss import multi_SpatialEmbLoss
from .Wnet_loss import W_SpatialEmbLoss
import torch.nn as nn
loss_dict={
    "FocalLoss":FocalLoss,
    "b_cross_entropy": nn.BCELoss,
    "Dis_loss": Dis_loss(FocalLoss("multiclass",2),ReconstructionLoss("l3")),
    "ERF_loss":single_SpatialEmbLoss,
    "ERF_multi_loss":multi_SpatialEmbLoss,
    "Wnet_loss":W_SpatialEmbLoss,
}


def get_lossfunc(name, loss_opts):
    if name in loss_dict:
        try:
            loss_class=loss_dict[name]
            return(loss_class(**loss_opts))
        except Exception as e:
            print(e)
            print(name,"loss_opts:",loss_opts)
            raise RuntimeError("Dataset {} not available".format(name))
   
        
    else:
        raise RuntimeError("Dataset {} not available".format(name))