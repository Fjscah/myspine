from .metrics import  miou_weight
from .ERF_metrics import miou_ERF,miou_wnet

import torch.nn as nn

metric_dict={
    "unet_miou":miou_weight,
    "ERF_miou": miou_ERF,
    "miou_wnet":miou_wnet,
    
}


def get_metricfunc(name, opts):
    if name in metric_dict:
        metric_class=metric_dict[name]
        return(metric_class(**opts))
   
        
    else:
        raise RuntimeError("Dataset {} not available".format(name))