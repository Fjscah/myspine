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

#-----------------------#
# ===== Parameter ======#
#-----------------------# 
configfile=r"config\defalut_2d_ins.yaml"

get_seg_tif_flag=False
crop_tif_flag   =True
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


#%%
imgdir=r"D:\code\test-dataset\result\img"
dendir=r"D:\code\test-dataset\result\den"
spinedir=r"D:\code\test-dataset\result\spine"
# seg dataset 
if get_seg_tif_flag:
    print("===== make spine tif ======")
    imgList = os.listdir(imgdir)
    segdir=dict_a["orilabel_path"]   
    imdir=dict_a["oriimg_path"]   
    for imgname in imgList:
        b,suffix=os.path.splitext(imgname)
        imgtif=os.path.join(imgdir,imgname)
        dentif=os.path.join(dendir,b+"den.tif")
        spinetif=os.path.join(spinedir,b+"spine.tif")
        segtif=os.path.join(segdir,b+"seg.tif")
        imtif=os.path.join(imdir,imgname)
        if os.path.exists(dentif) and os.path.exists(spinetif) :

            # os.rename(spinetif, os.path.join(absdir,b+"spine.tif"))
            print(dentif,spinetif)
            image=imread(imgtif)
            imden=imread(dentif)
            imspine=imread(spinetif)
            imspine=imspine.astype(np.uint16)
            
            imspine,_,_=relabel_sequential(imspine,2)
            print(imspine.shape,imden.shape)
            imspine[imspine<2]=imden[imspine<2]
            imsave(segtif,imspine)
            imsave(imtif,image)
        




#%%
# dataset create
if crop_tif_flag:
    itern=50 # one img crop times
    print("===== make dataset ======")
    cong=default_configuration
    laodir=cong.get_entry(['Path', 'label_path']) # label ori dir
    imodir=cong.get_entry(['Path', 'img_path']) # img ori dir
    ladir=cong.get_entry(['Path', 'orilabel_path'])
    imdir=cong.get_entry(['Path', 'oriimg_path'])
    w=cong.get_entry(['Data', 'input_sizexy'])
    nz=cong.get_entry(['Data', 'input_sizez'])
    if nz>1:
        outsize=(nz,w,w)# 3d
    else:
        outsize=(w,w)#2d
    note=cong.get_entry(['Path', 'label_suffix'])
    savetype=cong.get_entry(['Path', 'save_suffix'])


    generate_crop_img_save(imdir,ladir,imodir,laodir,outsize,note=note,hull=False,depth=nz,iter=itern,savetype=savetype,norm=True,zoom=2)
    print(w,nz,outsize,note)# first crop then split


#%%
# split dataset 
if split_tif_flag:
    print("===== split dataset ======")
    dataloader=SliceLoader(default_configuration)
    # tf.executing_eagerly()
    #trds,_,_=dataloader.get_dataset()
    #SliceLoader.show_data(trds)
    dataloader.split_data( )  
# %%
