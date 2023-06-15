
import torch
from typing import List,Any
import numpy as np
class BaseCluster():
    
    def __init__(self) -> None:
        pass

    def get_visual_keys(self) -> List[str]:
        raise NotImplementedError

    def in_cluster(self, model:torch.nn.Module,img) -> Any:
        raise NotImplementedError
    
    def show_result(self,visualizer,visual_result):
        raise NotImplementedError
    def valid_im(self,im):
        if isinstance(im,np.ndarray):
            im=torch.from_numpy(im)
        if im.dim()==2:
            im=im.unsqueeze(0)
        if im.dim()==3:
            im=im.unsqueeze(0)
        return im

    