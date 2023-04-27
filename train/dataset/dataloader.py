#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataloader.py
@Time    :   2022/04/26 19:28:21
@Author  :   Fjscah 
@Version :   1.0
@Contact :   www_010203@126.com
@License :   GNU, LNP2 group
@Desc    :   load dataset from folder
'''

# here put the import lib
from glob import glob

import scipy.ndimage
import sys
from skimage.morphology import binary_dilation, convex_hull_image, dilation,binary_erosion

import numpy as np
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations import transforms,RandomBrightnessContrast
from albumentations import RandomRotate90,VerticalFlip,HorizontalFlip
sys.path.append(".")
import math
import os
import shutil
from functools import partial
from random import shuffle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.io import imread, imsave

from utils import file_base, yaml_config
from utils.basic_wrap import timing
from utils.yaml_config import YAMLConfig

from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
from csbdeep.utils import normalize
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
import imgaug.augmenters as iaa
from torch.utils.data import DataLoader


from skimage.morphology import binary_dilation, convex_hull_image, dilation,binary_erosion

from . import enhancer 
from scipy.ndimage import distance_transform_edt

class CustomDatasetUnet2D(Dataset):
    def __init__(self, datafolder, suffix,classnum=3,transform=None, des="train",enhance_border=False,iteration=100):
        """initial construcotr
        Args:
           
            datafolder (string): img and label root directory
            suffix (string): label type suffix:like spine , den , seg
            classnum(int):indluce background 0.
            transform (function, optional):   enhance. Defaults to None.
            mode (string, optional): train,valid ,test. Defaults to train.
        """
        self.imgfiles = glob(datafolder+'\\img\\*.tif')
        self.labelfiles = glob(datafolder+'\\label\\*.tif')
        self.pairs=file_base.pair_files(self.imgfiles,self.labelfiles,suffix=suffix)
        self.length             = len(self.imgfiles)
        assert self.length>0, "data could not be empty, chlease check your dataset folder :"+datafolder
        self.classnum=classnum
        self.transform = transform
        self.iteration=iteration
        self.enhance_border=enhance_border
        self.des=des
        self.load_cache()
        # self.show_info()
        
    def load_cache(self):
        self.imgs=[]
        self.labs=[]
        for img_path,lab_path in zip(self.imgfiles,self.labelfiles):
            image = imread(img_path)
            label = imread(lab_path)
            label=np.array(label,dtype="int64")
            image=np.array(image,dtype="float32")
            # image=normalize(image)
            ytrue=self._preprocess_mask(label,enhance_border=self.enhance_border)
            func=ToTensor()
            image=func(image)# C H W
            self.imgs.append(image)
            self.labs.append(ytrue)
    def show_info(self):
        dicts=dict(
            save_suffix=self.save_suffix,
            imgshape=self.imgshape,
            modelname=self.model.__class__.__name__,
            layer_num=self.layer_num,
            batch_size=self.batch_size,
            optimizer_name=self.optimizer_name,
            initial_learning_rate=self.learning_rate,
            loss=self.loss.__class__.__name__,
            metric=self.metric.__name__,)
        kvs = [f"{k:<15}" + "\t:\t" + v.__repr__() for k,v in dicts.items()]
        self.logger.logger.info(
            "\n=========YAML TRAIN INFO==============\n"+
            "\n".join(kvs)+
            "\n======================================\n"
        )
    
    def __len__(self):
        return max(self.iteration,len(self.imgfiles))
 
    def __getitem__(self, idx):
        if idx> self.length-1:
            idx=np.random.randint(0,self.length-1)
        # print(oidx,idx)

        image = self.imgs[idx]
        y_one_hot = self.labs[idx]
      
        return image, y_one_hot 
    def _preprocess_mask(self,mask,enhance_border=False):
        """process mask to  ytrue for train,
        1. trans to support dtype
        2. return one hot
        3. to tensor
        
        if enhance_border enabled, will generate weighted ytrue that spine-joint-border will both mask 
            as background and target class(spine) region .so network' out put will be sigmod
        if use casual unet, the multi-lable-same-class will mask as same class id only

        Args:
            mask (ndarray): class label,support spine have multi label/ same label, but will all trans to same label
            enhance_border (bool, optional): whether execute border weight. Defaults to False.

        Returns:
            ndarray: if commom unet,will generate mask which element 0,1,2...
                    if border enabled , will generate
        """ 
        classnum=self.classnum
        if enhance_border:
            border=mask==classnum
        # fig,axs=plt.subplots(1,3,sharex=1,sharey=1)
        # for ax in axs:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # im=axs[0].imshow(mask,interpolation='none')
        # cbar=fig.colorbar(im, ax = axs[0])
        # # cbar.ax.set_title('neck/μm',fontsize=fontsize)
        # border_thin=mask==classnum-1
        # border_thin=border_thin>binary_erosion(border_thin)
        
        # mask = mask.astype(np.int64)
        mask[mask>classnum-1]=classnum-1
        func=ToTensor()
        mask1=func(mask) # C H W
        # to one-hot
        y_one_hot = make_one_hot(mask1,self.classnum) # C H W  
        if enhance_border:
            w0=5
            for lab in range(classnum):# edt trans
                mask2=mask==lab
                edt=distance_transform_edt(mask2)*0.5
                y_one_hot[lab,...][mask2]+=torch.from_numpy(w0*np.exp(-edt))[mask2]
            y_one_hot[0,...][border]=1+w0  #background and forground all set postive and weight set 2
            # y_one_hot[classnum-1,...][border]=1.5  
            # y_one_hot[classnum-1,...][border_thin]=1.5  
   
        # im=axs[1].imshow(y_one_hot[0,...])
        # cbar=fig.colorbar(im, ax = axs[1])
        # im=axs[2].imshow(y_one_hot[classnum-1,...])
        # cbar=fig.colorbar(im, ax = axs[2])
        # plt.show()
        return y_one_hot



class OrigionDatasetUnet2D(Dataset):
    def __init__(self, datafolder, suffix,classnum=3,transform=None, des="train",iteration=100):
        """initial construcotr
        Args:
           
            datafolder (string): img and label root directory
            suffix (string): label type suffix:like spine , den , seg
            classnum(int):indluce background 0.
            transform (function, optional):   enhance. Defaults to None.
            mode (string, optional): train,valid ,test. Defaults to train.
        """
        self.imgfiles = glob(datafolder+'\\img\\*.tif')
        self.labelfiles = glob(datafolder+'\\label\\*.tif')
        self.pairs=file_base.pair_files(self.imgfiles,self.labelfiles,suffix=suffix)
        self.length             = len(self.imgfiles)
        assert self.length>0, "data could not be empty, chlease check your dataset folder :"+datafolder
        self.classnum=classnum
        self.transform = transform
        self.iteration=iteration
  
        self.des=des
        self.load_cache()
        # self.show_info()
        
    def load_cache(self):
        self.imgs=[]
        self.labs=[]
        for img_path,lab_path in zip(self.imgfiles,self.labelfiles):
            image = imread(img_path)
            label = imread(lab_path)
            label=np.array(label,dtype="int64")
            image=np.array(image,dtype="float32")
            ytrue=label.copy()#self._preprocess_mask(label,enhance_border=self.enhance_border)
            func=ToTensor()
            image=func(image)# C H W
            self.imgs.append(image)
            self.labs.append(ytrue)
    def show_info(self):
        dicts=dict(
            save_suffix=self.save_suffix,
            imgshape=self.imgshape,
            modelname=self.model.__class__.__name__,
            layer_num=self.layer_num,
            batch_size=self.batch_size,
            optimizer_name=self.optimizer_name,
            initial_learning_rate=self.learning_rate,
            loss=self.loss.__class__.__name__,
            metric=self.metric.__name__,)
        kvs = [f"{k:<15}" + "\t:\t" + v.__repr__() for k,v in dicts.items()]
        self.logger.logger.info(
            "\n=========YAML TRAIN INFO==============\n"+
            "\n".join(kvs)+
            "\n======================================\n"
        )
    
    def __len__(self):
        return max(self.iteration,len(self.imgfiles))
 
    def __getitem__(self, idx):
        if idx> self.length-1:
            idx=np.random.randint(0,self.length-1)
        # print(oidx,idx)

        image = self.imgs[idx]
        y_one_hot = self.labs[idx]
      
        return image, y_one_hot 
   


class ClassDataset(Dataset):
    def __init__(self, X,Y,classnum=4,transform=ToTensor(), des="train",iteration=100):
        """initial construcotr
        Args:
           
            datafolder (string): img and label root directory
            suffix (string): label type suffix:like spine , den , seg
            classnum(int):indluce background 0.
            transform (function, optional):   enhance. Defaults to None.
            mode (string, optional): train,valid ,test. Defaults to train.
        """
        self.X=X
        self.Y=Y
        self.length             = len(X)
        assert self.length>0, "data could not be empty, chlease check your dataset folder :"
        self.classnum=classnum
        self.transform = transform
        self.iteration=iteration

        self.des=des
        
   
    def __len__(self):
        return max(self.iteration,self.length)
 
    def __getitem__(self, idx):
        if idx> self.length-1:
            idx=np.random.randint(0,self.length-1)
        # print(oidx,idx)

        image = self.X[idx]
        lab = self.Y[idx]
        if self.transform is not None:
           
            image = self.transform(image)
        # ont_hot_y=make_one_hot(lab,4)
        lab=torch.tensor(lab).long()
        return image, lab 



class SliceLoader:
    def setting(self, configuration: YAMLConfig = None):
        if configuration == None:
            self.configuration=None
            return
        self.configuration=configuration  
        dict_a=self.configuration.config["Path"]
        dict_b=self.configuration.config["Data"]
        dict_c=self.configuration.config["Training"]
        self.__dict__.update(dict_a)
        self.__dict__.update(dict_b)
        self.__dict__.update(dict_c)
        self.initial_filelist()
        self.inital_shape()
    def inital_shape(self):
        w=self.input_sizexy
        nz=self.input_sizez
    
    def initial_filelist(self):
        imfiles=file_base.file_list(self.data_path)
        lafiles=file_base.file_list(self.label_path)
        self.filepairs=file_base.pair_files(imfiles,lafiles,self.save_suffix)
        self.filenum=len(self.filepairs)
        shuffle(self.filepairs)
        
    def __init__(self, configuration: YAMLConfig = None):
        self.setting(configuration)
    
    def img_shape(self):
        if "z" in self.axes:
            self.shape=[self.input_sizez,self.input_sizexy,self.input_sizexy]
        else:
            self.shape=[self.input_sizexy,self.input_sizexy]
    @staticmethod
    def load_ds_dir(ddir,note="seg",suffix=".tif"):
        impath=os.path.join(ddir,"img")
        lapath=os.path.join(ddir,"label")
        imfiles=file_base.file_list(impath,suffix)
        lafiles=file_base.file_list(lapath,suffix)
        filepairs=file_base.pair_files(imfiles,lafiles,note)
        ims=[]
        las=[]
        for imfile,lafile in filepairs:
            im=imread(imfile)
            la=imread(lafile)
            # print(im.shape)
            im=im.reshape(list(im.shape)+[1]).astype(np.float32)
            la=la.reshape(list(im.shape)+[1]).astype(np.float32)
            # print(im.shape)
            ims.append(im)
            las.append(la)
        ims=np.array(ims)
        print("Data shape",ims.shape)
        print("======================")
        return  ims,np.array(las)
    @timing    
    def get_dataset(self):
        #使用并行化预处理num_parallel_calls 和预存数据prefetch来提升性能
        Train_path=self.Train_path
        trpath=os.path.join(Train_path,"train") 
        tepath=os.path.join(Train_path,"test") 
        vapath=os.path.join(Train_path,"valid") 
        batch_size=self.batch_size
        note=self.label_suffix
        filetype=self.filetype
        print("Dataset Folder:")
        print("\n".join([trpath,tepath,vapath]))
        print("=============================================")
        ds_train = SliceLoader.load_ds_dir(trpath,note,filetype) 
        ds_test = SliceLoader.load_ds_dir(tepath,note,filetype) 
        ds_valid = SliceLoader.load_ds_dir(vapath,note,filetype) 

        return ds_train,ds_valid,ds_test
    def split_data(self):
        Train_path=self.Train_path
        split_partion=self.partion # train,valid,test+=1
        accu_partion=[split_partion[0],split_partion[0]+split_partion[1],1]
        path_ts=["train","valid","test"]

        length=len(self.filepairs)
        self.filenum=len(self.filepairs)
        t=0
        for path_type,p in zip(path_ts,accu_partion):
            files=self.filepairs[t:int(p*length)]
            t=int(p*length)
            imgdir=os.path.join(Train_path,path_type+"/img")
            labdir=os.path.join(Train_path,path_type+"/label")
            file_base.remove_dir(imgdir)
            file_base.create_dir(imgdir)
            file_base.remove_dir(labdir)
            file_base.create_dir(labdir)
            for imfile,labfile in files:
                _,name,suffix=file_base.split_filename(imfile)
                shutil.copyfile(imfile, os.path.join(imgdir,name+suffix))
                _,name,suffix=file_base.split_filename(labfile)
                shutil.copyfile(labfile, os.path.join(labdir,name+suffix))
            print("complete folder : ", os.path.join(Train_path,path_type))
            
    def crop_and_split_data(self,itern=10,split_partion=[]):
        Train_path=self.Train_path
        if not split_partion:
            split_partion=self.partion # train,valid,test+=1
        accu_partion=[split_partion[0],split_partion[0]+split_partion[1],1]
        path_ts=["train","valid","test"]
        imdir=os.path.abspath(self.oridata_path)
        ladir=os.path.abspath(self.orilabel_path)
        imfiles=file_base.file_list(imdir)
        lafiles=file_base.file_list(ladir)
        pairs=file_base.pair_files(imfiles,lafiles,self.label_suffix)
        length=len(pairs)
        t=0
        w=self.input_sizexy
        nz=self.input_sizez
        if nz>1:
            outsize=(nz,w,w)# 3d
        else:
            outsize=(w,w)#2d
        lists=np.random.choice(length,length,False)
        files_splits=[]
        infos=[]
        for path_type,p in zip(path_ts,accu_partion):
            inds=lists[t:int(p*length)]
            inds=list(range(t,int(p*length)))
            files=[pairs[i] for i in inds]
            t=int(p*length)
            imgodir=os.path.join(Train_path,path_type+"/img")
            labodir=os.path.join(Train_path,path_type+"/label")
            file_base.remove_dir(imgodir)
            file_base.create_dir(imgodir)
            file_base.remove_dir(labodir)
            file_base.create_dir(labodir)
            cnt=enhancer.generate_crop_img_save_list(
                files,imgodir,labodir,outsize,
                depth=nz,
                iter=itern,savetype=self.save_suffix,
            )
            if files:
                files_splits.append(files)
            else:
                files.append([])
            infos.append({"complete folder : ": os.path.join(Train_path,path_type),"img num":len(files),"crop num":cnt})
            print("complete folder : ", os.path.join(Train_path,path_type),"img num",len(files),"crop num",cnt)
        for info,fs in zip(infos,files_splits):
            print(info)
            for f1,f2 in fs:
                print(f1,f2)
           
    
    
    @staticmethod
    def show_data(ds):
        def showtwo(im1,im2):
            fig,axes=plt.subplots(1,2,sharex=True,sharey=True)
            print(im1.shape)
            axes[0].imshow(im1[:,:,0])
            axes[1].imshow(im2[:,:,0])
            plt.show()
        imgs,labs=ds
        for i,(img,label) in enumerate(zip(imgs[:9],labs[0:9])):
            showtwo(img,label)         

#--


def transtonD(img,tarndim):
    if tarndim==4:
        if img.ndim==2:
            return img[None,...,None]
        elif img.ndim==3:
            return img[None,...]
    elif tarndim==3:
        if img.ndim==2:
            return img[None,...]
        elif img.ndim==3:
            return img



def get_train_tranform():
    train_transform = Compose([
        RandomRotate90(),
        VerticalFlip(),
        HorizontalFlip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            RandomBrightnessContrast(),#RandomContrast has been deprecated. Please use RandomBrightnessContrast
        ], p=1),
        # transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])
    return train_transform

    
def augtransform(aug=True,configuration: YAMLConfig=None):
    """transform image and label together for augmentation

    Args:
        ndim (int): img dimension
        aug (bool, optional): whether excuting augmentation. Defaults to True.
        configuration (YAMLConfig, optional): config for sugmentation. Defaults to None.

    Raises:
        Exception: _description_

    Returns:
        image and labels: _description_
    """
    if configuration is None:
        crop_or_pad_size=256
    else:
        crop_or_pad_size=configuration.get_entry(['Data', 'input_sizexy'])

    if aug:
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential([
                #iaa.PadToFixedSize(width=32, height=32),
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.5), # vertically flip 20% of all images
                
                sometimes(iaa.Affine(
                        scale={"x": (0.8, 1.5), "y": (0.8, 1.5)}, # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45), # rotate by -45 to +45 degrees
                        shear=(-16, 16), # shear by -16 to +16 degrees
                        order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                        cval=90, # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )),
                #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                sometimes(iaa.PiecewiseAffine(scale=(0.05, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                
                # iaa.CropToFixedSize(height=crop_or_pad_size,width=crop_or_pad_size,pad_cval=0),
                #iaa.GaussianBlur((0, 3.0)),
                #iaa.Affine(translate_px={"x": (-40, 40)}),
                #iaa.Crop(px=(0, 10))
            ])
    else:
        seq = iaa.Sequential([
                # iaa.CropToFixedSize(height=crop_or_pad_size,width=crop_or_pad_size,pad_cval=0),
                #iaa.GaussianBlur((0, 3.0)),
                #iaa.Affine(translate_px={"x": (-40, 40)}),
                #iaa.Crop(px=(0, 10))
            ])


    return seq

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = input.shape
    shape = list(shape)
    shape[0] = num_classes
    shape=tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(0, input, 1)

    return result
#----------------------------------------
# from data_loader.preprocessing import PaddedStackPatcher
def show_two(im1,im2):
    fig,axes=plt.subplots(1,2,sharex=True,sharey=True)
    axes[0].imshow(im1,extent=[0, im1.shape[0], 0, im1.shape[0]])
    axes[1].imshow(im2,extent=[0, im2.shape[0], 0, im2.shape[0]])
    plt.show()


def load_img(filename=""):
    if os.path.exists(filename):
        return imread(filename)
    else:
        raise FileNotFoundError("ERR file "+filename+" not exist \n")


if __name__=="__main__":
  
    datast=CustomDatasetUnet2D(r"D:\data\Train\Train\2D-2023-border\test","border",3,enhance_border=True)
    print(f"Train size: {len(datast)}")
    dataload=DataLoader(datast,4,shuffle=False,
                            pin_memory=True)
    for patch in dataload:
        img,ytrue=patch
        fig,axs=plt.subplots(2,1,sharex=1,sharey=1)
      
        axs[0].imshow(img[0,0,...])
        a=ytrue[0,-1,...]
        # print(a[a>1])
        axs[1].imshow(ytrue[0,-1,...])
        plt.show()  
        
        print(len(patch))
        # break
    # for i in range(5):
        
   
        