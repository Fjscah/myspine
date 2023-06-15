#%%
import sys
from utils.yaml_config import YAMLConfig
from utils.basic_wrap import Logger,logit
from train.trainers import trainer
import os


#-----------------------#
# ===== Parameter ======#
#-----------------------# 
#==================================================================================
configfile=r"config\defalut_2d_ins_ERF.yaml"
premodel =""

#==================================================================================

#-----------------------#
# ===== Run train ======#
#-----------------------#
# #%%
default_configuration=YAMLConfig(
    configfile
    )
# train
print("===== Training ...... ======")
trainmodel = trainer.Trainer()
trainmodel.setting(default_configuration,use_gpu=True)
trainmodel.train(denovo=False,premodel="")



# %%
