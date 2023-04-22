import numpy as np
from skimage.util.shape import view_as_windows,view_as_blocks
from skimage.util import montage
from skimage.filters.thresholding import (threshold_isodata, threshold_li,
                                          threshold_mean, threshold_minimum,
                                          threshold_multiotsu,
                                          threshold_niblack, threshold_otsu,
                                          threshold_triangle, threshold_yen,threshold_local)
import sys
from skimage.io import imread
# sys.path.append("./")
# sys.path.append("../")
# sys.path.append("../../")
# sys.path.append("../../../")
# sys.path.append("../../../")

from .device import set_use_gpu
from ..dataset.dataloader import get_train_tranform,OrigionDatasetUnet2D
import numpy as np
from ..networks import unetplusplus
import matplotlib.pyplot as plt
# import napari
num_classes=3
from glob import glob
from matplotlib.widgets import MultiCursor

from utils.file_base import file_list
import colorsys
# from PIL import Image
# from skimage.segmentation import watershed,morphological_chan_vese
from utils.yaml_config import YAMLConfig
from utils.basic_wrap import Logger,logit
import os
from torch.utils.data import DataLoader
import torch
from utils import file_base
# from torchsummary import summary
def showims(*ims):
    fig,axes=plt.subplots(1,len(ims),sharex=True,sharey=True)
    # print(im1.shape)
    for i,im in enumerate(ims):
        #print(im.shape)
        axes[i].imshow(im,interpolation='none')
    # axes[1].imshow(im2,interpolation='none')
    # axes[2].imshow(im3,interpolation='none')
    multi = MultiCursor(fig.canvas, axes, color='r', lw=1, horizOn=True, vertOn=True)
    plt.show()
class Predict():
    def setting(self, configuration: YAMLConfig = None,use_gpu=True):

        if configuration == None:
            self.configuration = configuration
            return

        self.configuration = configuration
        self.network_info=self.configuration.config["Network"]
        dict_a = self.configuration.config["Path"]
        dict_b = self.configuration.config["Training"]
        dict_c = self.configuration.config["Data"]
        #print(dict_b)
        self.__dict__.update(dict_a)
        self.__dict__.update(dict_b)
        self.__dict__.update(dict_c)
        self.__dict__.update(self.network_info)
        self.Train_path=os.path.abspath(self.Train_path)
        if "z" in self.axes:
            self.imgshape = (self.input_sizez, self.input_sizexy,
                             self.input_sizexy,1)
        else:
            self.imgshape = (self.input_sizexy, self.input_sizexy,1)
       
        self.logger=Logger(os.path.join(os.path.abspath(self.log_path),"train_info.log"))
    
        self.initial_gpu(use_gpu)
        self.inital_model()
    def initial_gpu(self,use_gpu):
        device,ngpu,ncpu=set_use_gpu(use_gpu)
        self.use_gpu=use_gpu
        self.device=device
        self.ngpu=ngpu
        self.ncpu=ncpu    
               
    def load_weight(self,denovo,premodel):
        #-----------------------#
        #   Load model weights  #
        #-----------------------#
        checkpoint_save_path = premodel
        #checkpoint_save_path = ""  #r"models\M2d_seg\modelep100-loss0.011.h5"  # 模型参数保存路径
        if (not checkpoint_save_path) and (denovo is False):
            # find neweat weight file
            paths=file_base.file_list_bytime(self.model_path,".pth")
            checkpoint_save_path=paths[-1] if paths else ""   
        if checkpoint_save_path and os.path.exists(checkpoint_save_path):
            self.model.load_state_dict(torch.load(checkpoint_save_path))
            self.model.eval()
            self.logger.logger.info(
                "\n==============LOAD PRETRAINED model==============\n"+
                checkpoint_save_path+
                "\n=================================================\n"
            )
    
    def inital_model(self):
        
        network_type = self.configuration.get_entry(['Network', 'modelname'])
        num_classes=self.num_classes
        if "unet3d" == network_type:
            self.model = unet.UNet3D(self.configuration)
        elif "unet2d" == network_type:
            self.model =unetplusplus.UNet2d(num_classes,1)
        elif "unet++"==network_type:
            self.model=unetplusplus.NestedUNet(num_classes,1)
        self.model.load_network_set(self.network_info)
        self.model.to(self.device)
        #summary(self.model,(1,self.input_sizexy,self.input_sizexy))
        # self.show_train_info()
        
        # self.initial_loss()
    def test_epoch(self,valid_dataloader,model):

        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            
            
            for image, label in valid_dataloader:
                image = image.to(self.device)
                label = label.to(self.device)

                # compute output
                # self.model.predict_2d_img(image)

                output = model(image)[0]#.cpu().data.numpy()
                #print(image.shape)
                showims(image.squeeze(),label[0],output[0],output[1],output[2])



    
    @logit("error.log")
    def predict(self,premodel="",data=None):
        """_summary_

        Args:
            premodel (str, optional): _description_. Defaults to "".
            data (_type_, optional): Data None: use test folder, show and compare result ;Data folder/imagefile, show predict reshult. Defaults to None.

        Raises:
            Exception: _description_
        """
        if self.configuration is None:
            raise Exception('Please set Yaml configuration first!')
        #-----------------------#
        #  config load  #
        #-----------------------#
        Train_path=self.Train_path
        suffix=self.label_suffix # seg,spine,den
        num_classes=self.num_classes
        log_dir=self.log_path
        model=self.model
        train_trainform=get_train_tranform()
        self.load_weight(False,premodel)
        if data is None:
            #self.enhance_border=True if "border" in self.save_suffix else False
            test_datast=OrigionDatasetUnet2D(Train_path + "\\test",suffix,num_classes,iteration=0,des="test",transform=train_trainform)
            test_dataloader = DataLoader(test_datast,batch_size = 1,shuffle=False)
            self.test_epoch(test_dataloader,model)
        elif isinstance(data,list) and os.path.isdir(data[0]): # img folder, lab folder 
            
            imgfiles = glob(data[0]+'\\*.tif')
            labelfiles = glob(data[1]+'\\*.tif')
            pairs=file_base.pair_files(imgfiles,labelfiles,suffix=data[2])
            for img,lab in pairs:
                # print(img)
                lab=imread(lab)
                img=imread(img)
                
                ypred=model.predict_2d_img(img)
                showims(img,lab,ypred[0],ypred[1],ypred[2])
        elif isinstance(data,list) and os.path.isfile(data[0]): #imgfile list
            for im in data:
                #lab=imread(data[1])
                img=imread(im).astype("float32")
                ypred=model.predict_2d_img(img)
                showims(img,lab,ypred[0],ypred[1],ypred[2])
        elif os.path.isdir(data): # img dir
            imgfiles = glob(data+'\\*.tif')
           
            for img in imgfiles:
                print(img)
               
                img=imread(img).astype("float32")
                ypred=model.predict_2d_img(img)
                showims(img,ypred[0],ypred[1],ypred[2])
        else: # filename
            img=imread(data).astype("float32")
            ypred=model.predict_2d_img(img)
            showims(img,ypred[0],ypred[1],ypred[2])
            
        
def showtwo(im1,im2):
    fig,axes=plt.subplots(1,2,sharex=True,sharey=True)
    print(im1.shape)
    axes[0].imshow(im1)
    axes[1].imshow(im2)
    plt.show()

def split_patches(img,cropsize):
    imgpad=np.pad(img,[(0,s%c) for s,c in zip(img.shape,cropsize)])
    splitshape=[s//c for s,c in zip(imgpad.shape,cropsize)]
    B = view_as_blocks(imgpad, cropsize)
    print(B.shape)
    B=B.reshape(-1,*cropsize)
    return B,splitshape
def merge_patches(B,splitshape):
    arr_out = montage(B,grid_shape=splitshape)
    return arr_out
def predict_dir(modelpath=r"D:\spine\spinesoftware\myspine\models\M2d_den\modelep010-loss0.023.h5"):
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    model=unet.UNet2D()
    model.build(input_shape =(4,256,256,1))
    model.load_weights(modelpath)
#     model=tf.keras.models.load_model(
#     r"D:\spine\spinesoftware\myspine\models\M2d_den\model", custom_objects=None, compile=True, options=None
# )
    imgdir=r"C:\Users\ZLY\Desktop\Train\4D\deconvole_2D_one"


        
    imglist=file_list(imgdir)
    for imgf in imglist:
        print(imgf)
        img=imread(imgf)
        imgs,splitshape=split_patches(img,cropsize=(256,256))
        imgsn=[]
        for im in imgs:
            im=im.reshape(1,256,256,1).astype(np.float32)
            ppr=model.predict(im)[0]
            pr = ppr.argmax(axis=-1)
            imgsn.append(pr)
        imgn=merge_patches(imgsn,splitshape)
        imgn=imgn[0:img.shape[0],0:img.shape[1]]


        # res=np.max(out,axis=-1)
        showtwo(img,imgn)
    # print(out)
def predict_movie(imgf=r"D:\spine\Train\4D\deconvolve_2D\MAX_decon_20211111-24D.tif",
          modelpath=r"D:/spine/spinesoftware/myspine/models/M2d_den\modelep008-loss0.013.h5"):
    imgs=imread(imgf)
    model=unet.UNet2D()
#     model=tf.keras.models.load_model(
#     r"D:\spine\spinesoftware\myspine\models\M2d_den\model", custom_objects=None, compile=True, options=None
# )
    model.build(input_shape =(4,512,512,1))
    model.load_weights(modelpath)
    print("img shape : ",imgs.shape)
    imgns=[]
    for img in imgs:
        img=img.reshape(1,512,512,1).astype(np.float32)
        print(img.shape)
        ppr=model.predict(img)[0]
        pr = ppr.argmax(axis=-1)
        imgn = ppr.argmax(axis=-1)
        # imgss,splitshape=split_patches(img,cropsize=(256,256))
        # imgsn=[]
        # for im in imgss:
        #     im=im.reshape(1,256,256,1).astype(np.float32)
        #     ppr=model.predict(im)[0]
        #     pr = ppr.argmax(axis=-1)
        #     imgsn.append(pr)
        # imgn=merge_patches(imgsn,splitshape)
        # imgn=imgn[0:img.shape[0],0:img.shape[1]]
        imgns.append(imgn)
    return imgns,imgs
def predict_4D(imgf=r"D:\data\Train\4D\deconvolve_4D\decon_20211111-24D.tif",
          modelpath=r"D:/spine/spinesoftware/myspine/models/M2d_seg\modelep104-loss0.047.h5"):
    imgss=imread(imgf)[:1]
    adth=imgss[0]>threshold_otsu(imgss[0])
    # adth=morphological_chan_vese(imgss[0],20,adth,1,1,2)
    model=unet.UNet2D()
#     model=tf.keras.models.load_model(
#     r"D:\spine\spinesoftware\myspine\models\M2d_den\model", custom_objects=None, compile=True, options=None
# )
    model.build(input_shape =(4,512,512,1))
    model.load_weights(modelpath)
    print("img shape : ",imgss.shape)
    imgnss=[]
    for imgs in imgss:
        
        imgns=[]
        prs=[]
        for img in imgs:
            img=img.reshape(1,512,512,1).astype(np.float32)
            ppr=model.predict(img)[0]
            imgn = ppr.argmax(axis=-1)
            prs.append(ppr[...,2])
            
            # imgss,splitshape=split_patches(img,cropsize=(256,256))
            # imgsn=[]
            # for im in imgss:
            #     im=im.reshape(1,256,256,1).astype(np.float32)
            #     ppr=model.predict(im)[0]
            #     pr = ppr.argmax(axis=-1)
            #     imgsn.append(pr)
            # imgn=merge_patches(imgsn,splitshape)
            # imgn=imgn[0:img.shape[0],0:img.shape[1]]
            imgns.append(imgn)
        imgns=np.array(imgns)
        mask=np.max(imgs,axis=0)[None,...]
        mask=mask.reshape(1,512,512,1).astype(np.float32)
        ppr=model.predict(mask)[0]
        mask = ppr.argmax(axis=-1)==1
        mask=np.tile(mask,(imgns.shape[0],1,1))
        imgns[(imgns==2)*mask]=0
    imgnss.append(imgns)    
    return imgnss,imgss,prs,adth,mask
# a,b,c,d,e=predict_4D()
# viewer=napari.Viewer()
# viewer.add_image(np.array(b))
# viewer.add_image(np.array(c))
# viewer.add_image(np.array(e))
# viewer.add_image(np.array(d*a))
# viewer.add_labels(np.array(a))
# napari.run()
# from utils import yaml_config
# configfile=r"config\defalut_2d_seg.yaml"
# predictmodel = Predict()
# default_configuration=yaml_config.YAMLConfig(
#         configfile
#         )
# predictmodel.setting(default_configuration,use_gpu=False)
# # predictmodel.load_weight(False,"premodel")