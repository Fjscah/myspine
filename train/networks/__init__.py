

from .unetplusplus import NestedUNet,UNet2d
from .ERFNet import Net,BranchedERFNet
from .wnet import WNet
model_dict={
   "unet2d":UNet2d, #num_classes, input_channels=3
   "unet++":NestedUNet ,#num_classes, input_channels=1, deep_supervision=False
   "ERFnet":BranchedERFNet, # num_classes, encoder=None,inputchannel=3
   "wnet": WNet,#input_channel=1,num_class=3,unet_dim=64, dis2emb_channels=8, emb_channels=8
    
}

def get_network(name,opt={}):
    if name in model_dict:
        model_class=model_dict[name]
        return(model_class(**opt))
   
        
    else:
        raise RuntimeError("Dataset {} not available".format(name))