import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import List,Any
import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torchsummary import summary
from skimage.morphology import remove_small_objects
from csbdeep.utils import normalize
#from train.trainers.visual import Visualizer
import colorcet as cc
from skimage.color import label2rgb
class BaseNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.kwargs=kwargs
    @property 
    def cur_device(self):
        return next(self.parameters()).device
    def get_visual_keys(self) -> List[str]:
        raise NotImplementedError

    def in_cluster(self, model:torch.nn.Module,img) -> Any:
        raise NotImplementedError
    
    def show_result(self,visualizer,visual_result):
        raise NotImplementedError
    def predict(self,img)-> Any:
        raise NotImplementedError
    def valid_im(self,im):
        if isinstance(im,np.ndarray):
            im=torch.from_numpy(im)
        if im.dim()==2:
            im=im.unsqueeze(0)
        if im.dim()==3:
            im=im.unsqueeze(0)
        return im.to(self.cur_device)
      