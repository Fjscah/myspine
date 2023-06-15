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
import skimage.measure
import sys
from skimage.morphology import binary_dilation, convex_hull_image, dilation,binary_erosion

import numpy as np

import math
import os
import shutil

from random import shuffle


import matplotlib.pyplot as plt
import numpy as np
from .distance_transorm import get_joint_border2,get_border2
from skimage.io import imread, imsave

from utils import file_base, yaml_config
from utils.basic_wrap import timing
from utils.yaml_config import YAMLConfig

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
from torchvision.transforms.functional import rotate
from . import enhancer 
from scipy.ndimage import distance_transform_edt

class SpineDataset(Dataset):
    def __init__(self, datafolder, suffix,classnum=3,transform=None, des="train",
                 enhance_border=False,iteration=100,task_type="seg",
                 make_dis=False):
        """initial construcotr
        Args:
           
            datafolder (string): img and label root directory
            suffix (string): label type suffix: like spine , den , seg,...
            classnum(int):indluce background 0. usualy 3 : bg, den, spine
            transform (function, optional):   enhance. Defaults to None. online transform #TODO
            mode (string, optional): train,valid ,test. Defaults to train.
        """
        self.imgfiles = glob(datafolder+'/img/*.tif')
        self.labelfiles = glob(datafolder+'/label/*.tif')
        self.suffix=suffix
        self.task_type=task_type
        self.pairs=file_base.pair_files(self.imgfiles,self.labelfiles,suffix=suffix)
        
        self.length             = len(self.pairs)
        print(datafolder,"find files:",self.length)
        assert self.length>0, "data could not be empty, chlease check your dataset folder :"+datafolder
        self.classnum=classnum
        self.transform = transform
        self.iteration=iteration
        self.enhance_border=enhance_border
        self.des=des
        self.use_dis=make_dis
        self.load_cache()
        # self.show_info()
        
    def load_cache(self):
        self.imgs=[]
        self.labs=[]
        for img_path,lab_path in self.pairs:
            #print(img_path,lab_path)
            #print(img_path,lab_path)
            image = imread(img_path) # H W 
            label = imread(lab_path) # H W
            label=np.array(label,dtype="int64")
            image=np.array(image,dtype="float32")
            # image=normalize(image)
            # ins : instance, only include spine; feature: include spine and den, with some transorm information, like one hot, binary, distance transform and so on
            ins,feature=self._preprocess_mask(label,process_method=self.task_type) # 2/3, H,W
            func=ToTensor()
            image=func(image)# C H W
            self.imgs.append(image)
            self.labs.append([ins,feature])
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
        return min(self.iteration,len(self.pairs))
 
    def __getitem__(self, idx):
        if idx> self.length-1:
            idx=np.random.randint(0,self.length-1)
        # print(oidx,idx)
        #idx=np.random.randint(0,self.length-1)
        image = self.imgs[idx]
        ins,y_one_hot = self.labs[idx]
        
        lab_ins,_ = ins.unique().sort()
        lab_inds=torch.arange(0,len(lab_ins))  
        for labi,ind_i in zip(lab_ins,lab_inds):
            ins[ins==labi]=ind_i
            
        #print(y_one_hot.max())
        # if  torch.rand(1) > 0.5:
        #     # image=torch.rot90(image,1,[1,2])
        #     # ins=torch.rot90(ins,1,[0,1])
        #     # y_one_hot=torch.rot90(y_one_hot,1,[0,1])
        #     angle=np.random.randint(0,360)
        #     image=rotate(image,angle)
        #     ins=rotate(ins,angle)
        #     y_one_hot=rotate(y_one_hot,angle)
        return image, ins,y_one_hot 
    def _preprocess_mask(self,label,process_method="mask"):
        """process mask to  ytrue for train,
        process_method : 
            seg     : label include bg 0,den 1,spine >1 ,   get ins(ignore den 1) and one hot (bg 0+den 1+spine 2), for instance seg
            den     : label only include den==1 ,           get ins and one hot as den==1  ,for semantic seg (bg+den)
            spine   : label only include spine==2 ,         get ins and one hot as den==1  ,for sematic seg (bg+den)
            dis     : label include bg 0,den 1,spine >1 ,   get ins(ignore den 1) and one hot (bg 0+den 1+spine 2), for instance seg
            ins     : label include bg 0,den 1,spine >1 ,   get ins(ignore den 1) and one hot (bg 0+den 1+spine 2), for instance seg
        """ 
        func=ToTensor()
        ins=label.copy()
        ins[ins<2]=0
        
        classnum=self.classnum
        
        if "seg" in process_method :
            mask=label.copy()
            mask[mask>classnum-1]=classnum-1
            
            # to one-hot
           
            ins=func(ins)
            mask=func(mask) # C H W
            y_one_hot = make_one_hot(mask,self.classnum) # C H W  
            joint=get_joint_border2(label,beginlabel=2)
            y_one_hot[0][joint]=2
            y_one_hot[2][joint]=2
        if "ins" in process_method:
            mask=label.copy()
            mask[mask>classnum-1]=classnum-1
            joint=get_border2(label,beginlabel=2)
            # to one-hot
           
            ins=func(ins)
            mask=func(mask) # C H W
            y_one_hot = make_one_hot(mask,self.classnum) # C H W  
            y_one_hot[0][joint]=1
            y_one_hot[2][joint]=0.2
        if "spine" in process_method:
            # mask=label.copy()
            # mask[mask==1]=0
            # mask[mask>classnum-1]=classnum-1
            mask=ins>0
            # to one-hot
   
            ins=func(ins)
            mask=func(mask) # C H W
            return ins,mask
            #y_one_hot = make_one_hot(mask,self.classnum)
    
            
            
        return ins,y_one_hot



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
        self.imgfiles = glob(datafolder+'/img/*.tif')
        self.labelfiles = glob(datafolder+'/label/*.tif')
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
        for img_path,lab_path in self.pairs:
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
        imfiles=file_base.file_list(self.img_path)
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
        ori_path=self.ori_path
        trpath=os.path.join(ori_path,"train") 
        tepath=os.path.join(ori_path,"test") 
        vapath=os.path.join(ori_path,"valid") 
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
        ori_path=self.ori_path
        split_partion=self.partion # train,valid,test+=1
        accu_partion=[split_partion[0],split_partion[0]+split_partion[1],1]
        path_ts=["train","valid","test"]

        length=len(self.filepairs)
        self.filenum=len(self.filepairs)
        t=0
        for path_type,p in zip(path_ts,accu_partion):
            files=self.filepairs[t:int(p*length)]
            t=int(p*length)
            imgdir=os.path.join(self.crop_path,path_type+"/img")
            labdir=os.path.join(self.crop_path,path_type+"/label")
            print("===== split dataset ======",path_type,":",len(files))
            file_base.remove_dir(imgdir)
            file_base.create_dir(imgdir)
            file_base.remove_dir(labdir)
            file_base.create_dir(labdir)
            for imfile,labfile in files:
                _,name,suffix=file_base.split_filename(imfile)
                shutil.copyfile(imfile, os.path.join(imgdir,name+suffix))
                _,name,suffix=file_base.split_filename(labfile)
                shutil.copyfile(labfile, os.path.join(labdir,name+suffix))
            print("complete folder : ", os.path.join(ori_path,path_type))
            
    def crop_and_split_data(self,itern=10,split_partion=[],zoom=1,norm=False):
        ori_path=self.ori_path
        if not split_partion:
            split_partion=self.partion # train,valid,test+=1
        accu_partion=[split_partion[0],split_partion[0]+split_partion[1],1]
        path_ts=["train","valid","test"]
        imdir=os.path.abspath(self.oriimg_path)
        ladir=os.path.abspath(self.orilabel_path)
        imfiles=file_base.file_list(imdir)
        lafiles=file_base.file_list(ladir)
        pairs=file_base.pair_files(imfiles,lafiles,self.label_suffix)
        length=len(pairs)
        print("===== split dataset ====== total files : ",length)
        
        w=self.input_sizexy
        nz=self.input_sizez
        if nz>1:
            outsize=(nz,w,w)# 3d
        else:
            outsize=(w,w)#2d
        lists=np.random.choice(length,length,False)
        files_splits=[]
        infos=[]
        t=0
        cnts={type:0 for  type in path_ts}
        for path_type,p in zip(path_ts,accu_partion):
            inds=lists[t:int(p*length)]
            inds=list(range(t,int(p*length)))
            files=[pairs[i] for i in inds]
            t=int(p*length)
            print("===== split dataset ======",path_type,":",len(files))
            imgodir=os.path.join(self.crop_path,path_type+"/img")
            labodir=os.path.join(self.crop_path,path_type+"/label")
            file_base.remove_dir(imgodir)
            file_base.create_dir(imgodir)
            file_base.remove_dir(labodir)
            file_base.create_dir(labodir)
        def func(zo):
            t=0
            print("========zoom========",zo)
            for path_type,p in zip(path_ts,accu_partion):
                inds=lists[t:int(p*length)]
                inds=list(range(t,int(p*length)))
                files=[pairs[i] for i in inds]
                t=int(p*length)
                print("===== split dataset ======",path_type,":",len(files))
                imgodir=os.path.join(self.crop_path,path_type+"/img")
                labodir=os.path.join(self.crop_path,path_type+"/label")
                
                cnt=cnts[path_type]
                cnt=enhancer.generate_crop_img_save_list(
                    files,imgodir,labodir,outsize,
                    depth=nz,
                    iter=itern,savetype=self.save_suffix,zoom=zo,norm=norm,startn=cnt
                )
                if files:
                    files_splits.append(files)
                else:
                    files.append([])
                cnts[path_type]=cnt
                infos.append({"complete folder : ": os.path.join(ori_path,path_type),"img num":len(files),"crop num":cnt})
                print("complete folder : ", os.path.join(ori_path,path_type),"img num",len(files),"crop num",cnt)
        if isinstance(zoom,list):

            for zo in zoom:
                func(zo)
        else:
            func(zoom)
        
        # for info,fs in zip(infos,files_splits):
        #     print(info)
        #     for f1,f2 in fs:
        #         print(f1,f2)
           
    
    
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



# def get_train_tranform():
#     train_transform = Compose([
#         RandomRotate90(),
#         VerticalFlip(),
#         HorizontalFlip(),
#         OneOf([
#             transforms.HueSaturationValue(),
#             transforms.RandomBrightness(),
#             RandomBrightnessContrast(),#RandomContrast has been deprecated. Please use RandomBrightnessContrast
#         ], p=1),
#         # transforms.Resize(config['input_h'], config['input_w']),
#         transforms.Normalize(),
#     ])
#     return train_transform

    
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
  
    pass
        # break
    # for i in range(5):
        
   
        