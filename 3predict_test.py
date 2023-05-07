
import sys
sys.path.append(".")
import torch
from train.networks import unetplusplus
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from utils.yaml_config import YAMLConfig
from glob import glob
from utils import file_base, yaml_config


from utils.file_base import file_list
import colorsys
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from train.trainers.predict import Predict

import os
#%% set predict mode: sigle image or multi images(dir)
predictmode="single" # multi , sigle image or multi images(dir)
configfile=r"config\defalut_2d_seg_sim.yaml"
test_flag=True  #False, wheather to use test folder ( load test label together)

# if premodel not specify will search from yaml profile
premodel=""#r"../test-dataset/model/2Dtorchunet-b/seg\ep003-loss0.415.pth"
# if datas not specify will load test data folder ,datas supprt imgfile,imgfiles,imgdolder
datas=r"myspine-dataset/2D-morph-seg/img/"
datas=[r"myspine-dataset/2D-morph-seg/img/",r"myspine-dataset/2D-morph-seg/label/","-seg"]
datas=r"myspine-dataset\crimson-20200319-1.tif"
datas=None
# print(os.path.isdir(datas))
if configfile is not None:
    default_configuration=YAMLConfig(
        configfile
        )
    network_info=default_configuration.config["Network"]
   
    print("===== Train dataset ======")
    trainmodel = Predict()
    trainmodel.setting(default_configuration,use_gpu=False)
    trainmodel.predict(premodel=premodel,data=datas)


# elif predictmode=="signle":
#     pass
# elif predictmode=="multi":
#     if test_flag:
#        pass
#     else:
#         pass