#%%
import sys
from utils.yaml_config import YAMLConfig
from utils.basic_wrap import Logger,logit
from skimage.segmentation import relabel_sequential
import os
from train.dataset.enhancer import generate_crop_img_save
from train.dataset.dataloader import SliceLoader
from skimage.io import imread,imsave
import numpy as np
from train.trainers import trainer

#-----------------------#
# ===== Parameter ======#
#-----------------------# 
configfile=r"config\defalut_2d_seg_UnetL3.yaml"

get_seg_tif_flag=False
split_tif_flag  =True
#==================================================================================



default_configuration=YAMLConfig(
    configfile
    )
dict_a = default_configuration.config["Path"]
dict_b = default_configuration.config["Training"]
dict_c = default_configuration.config["Data"]
# logpath=os.path.join(os.path.abspath(dict_a["log_path"]),"model_info.log")
# logger=Logger(logpath)
# sys.stdout =logger
# logger.logger.info(
#             "\n=========logger path==============\n"+
#             dict_a["log_path"]+
#             "\n======================================\n"
#         )

        



spli_partions=[
    #[0.1,0.9],#1
    [0.2,0.8],#2
    #[0.25,0.75],#3
    # [0.3,0.7],#3
    # [0.4,0.6],#4
    # [0.5,0.5],#5
    # [0.6,0.4],#6
    # [0.7,0.3],#7
    # [0.8,0.2],#8
    # [0.9,0.1],#11
]

#%%
itern=20
# split dataset 
for split in spli_partions:
    if split_tif_flag:
        
        dataloader=SliceLoader(default_configuration)
        # tf.executing_eagerly()
        #trds,_,_=dataloader.get_dataset()
        #SliceLoader.show_data(trds)
        dataloader.crop_and_split_data(itern,norm=True,zoom=[1,2],split_partion=split)  # first split then crop
    trainmodel = trainer.Trainer()
    trainmodel.setting(default_configuration,use_gpu=True)
    trainmodel.train(denovo=True,premodel="")    
# %%
