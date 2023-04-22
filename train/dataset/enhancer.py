#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   enhancer.py
@Time    :   2022/04/26 19:28:46
@Author  :   Fjscah 
@Version :   1.0
@Contact :   www_010203@126.com
@License :   GNU, LNP2 group
@Desc    :   enhance data offline/online
'''

# here put the import lib

from csbdeep.utils import Path, normalize
from skimage.morphology import binary_dilation, convex_hull_image, dilation,binary_erosion

from skimage.io import imread,imsave
import os
import sys

sys.path.append(".")

from skimage.measure import label, regionprops, regionprops_table
from utils import file_base
from scipy.ndimage import gaussian_laplace, minimum_filter,maximum_filter
import numpy as np
from sympy import im
import imgaug.augmenters as iaa
import imgaug as ia
# Standard scenario: You have N=16 RGB-images and additionally one segmentation
# map per image. You want to augment each image and its heatmaps identically.
import matplotlib.pyplot as plt
from .segment import fill_hulls,resortseg
from .dataloader import load_img,show_two
from utils.yaml_config import YAMLConfig
from .localthreshold import local_threshold
from skimage.filters.thresholding import (threshold_isodata, threshold_li,
                                          threshold_mean, threshold_minimum,
                                          threshold_multiotsu,
                                          threshold_niblack, threshold_otsu,
                                          threshold_triangle, threshold_yen,threshold_local)
import tifffile
def savelabel(arr,filename):
    tifffile.imwrite(
        filename,
        np.array(arr,dtype=np.uint16),
        imagej=True,
        photometric='minisblack',
        #metadata={'axes': 'TYX'},
    )
    print("save :",filename)
def savepr(arr,filename):
    tifffile.imwrite(
        filename,
        arr,
        imagej=True,
        photometric='minisblack',
        #metadata={'axes': 'TYX'},
    )
    print("save :",filename)
#-----------------------#
#  custom Transform online   #
#-----------------------#



#-----------------------#
#  custom Transform offline   #
#-----------------------#
def generate_crop_img_save(imdir,ladir,imodir,laodir,outsize,note="seg",hull=True,depth=10,iter=10,cval=None,denmode=False,savetype="seg"):
    """from pdir,load img and label,then crop or tanseform to generate more img and save in odir

    Args:
        imdir (_type_): _description_
        ladir (_type_): _description_
        imodir (_type_): _description_
        laodir (_type_): _description_
        outsize (_type_): xyshape
        note (str, optional): suffix for label file. Defaults to "seg".
        hull (bool, optional): whethre hull for roi point. Defaults to True.
        depth (int, optional): z. Defaults to 10.
        iter (int, optional): generate how much crop image for each image. Defaults to 10.
        denmode: use for only spine label no dendrite label
    Raises:
        FileNotFoundError: _description_

    Returns:
        _type_: None
    """
    imdir=os.path.abspath(imdir)
    ladir=os.path.abspath(ladir)
    imfiles=file_base.file_list(imdir)
    lafiles=file_base.file_list(ladir)
    file_base.remove_dir(imodir)
    file_base.remove_dir(laodir)
    file_base.create_dir(imodir)
    file_base.create_dir(laodir)
    pairs=file_base.pair_files(imfiles,lafiles,note)
    images=[]
    segmaps=[]
    #print(imdir,"\n",ladir,imfiles,lafiles,pairs)
    print("outsize :",outsize)
    
    for img,labelf in pairs:
        print(img,"\t",labelf)
        if depth>1:
            im=load_img(img)[:depth]
            lab=load_img(labelf)[:depth]
        else:
            im=load_img(img)
            lab=load_img(labelf)
        # print(im.shape)
        lab=lab.astype(np.uint16)
        if 'border' in savetype:
            border=get_joint_border2(lab,2)
            lab[lab>1]=2
            lab[border]=3
            # plt.imshow(lab)
            # plt.show()
        elif "spine" not in savetype:
            lab[lab>1]=2
        
        if denmode:
            lab=(lab>0).astype(np.uint16)
            adth=local_threshold(im).astype(np.uint16)
            print(lab.shape,adth.shape)
            adth[adth>0]=2
            adth[lab>0]=1
            lab=adth
        
        
        if hull:
            lab=fill_hulls(lab)
            lab,_=resortseg(lab)
        if depth>1:
            im=im.swapaxes(0,2)
            lab=lab.swapaxes(0,2)
        else:
            im=im[..., np.newaxis]
            lab=lab[..., np.newaxis]
        #show_two(im,lab)
        #im = normalize(im,1,99.8,axis=None)
        #im = normalize(im,1,99.8,axis=None)
        images.append(im)
        segmaps.append(lab)
    if hull:
        hulldir="hulldir"
        hulldir=os.path.join(file_base.get_parent_dir(imdir,1),hulldir)
        file_base.create_dir(hulldir)
        for seg,(img,labelf) in zip(segmaps,pairs):
            f=os.path.basename(labelf)
            of=os.path.join(hulldir,f)
            seg=seg.swapaxes(0,2)
            imsave(of,seg)
            
    if cval is None:
        # threshold_li()
        vmean=np.nanmean([np.mean(im[lab==0]) for im,lab in zip(images,segmaps)])
        vstd=np.nanmean([np.std(im[lab==0]) for im,lab in zip(images,segmaps)])
        print(vmean,vstd)
        cval=(int(max(0,vmean-vstd)),int(vmean+vstd))
        

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    # images = np.random.randint(0, 255, (16, 128, 128, 1), dtype=np.uint8)
    # segmaps = np.random.randint(0, 10, size=(16, 128, 128, 1), dtype=np.int32)
    # segmaps = np.expand_dims(np.sum(images,axis=-1),axis=-1)>180
    if len(outsize)==3:
        height=outsize[1]
        width=outsize[2]
    else:
        height=outsize[0]
        width=outsize[1]
        

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
                cval=cval, # if mode is constant, use a cval between 0 and 255
            # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
        sometimes(iaa.PiecewiseAffine(scale=(0.05, 0.05))), # sometimes move parts of the image around
        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
        iaa.CropToFixedSize(height=height,width=width),
        #iaa.GaussianBlur((0, 3.0)),
        #iaa.Affine(translate_px={"x": (-40, 40)}),
        #iaa.Crop(px=(0, 10))
    ])
    N=len(images)
    for it in range(iter):
        images_aug, segmaps_aug = seq(images=images, segmentation_maps=segmaps)
        for n,(im1,im2) in enumerate(zip(images_aug,segmaps_aug)):
            
            oim=os.path.join(imodir,str(it*N+n)+".tif")
            ola=os.path.join(laodir,str(it*N+n)+savetype+".tif")
            if depth>1:
                im1=im1.swapaxes(0,2)
                im2=im2.swapaxes(0,2)
            else:
                #show_two(im1,im2)
                im1=np.squeeze(im1)
                im2=np.squeeze(im2)
            # imsave(oim,im1)
            savepr(im1,oim)
            savelabel(im2,ola)
            # imsave(ola,im2)

def get_joint_border(label,ignore=[]):#bug
    label=label.copy()
    spine_label_c=np.zeros_like(label)
    labs=np.unique(label)
    for lab in labs:
        if lab in ignore:
            label[label==lab]=0
            continue
        mask=label==lab
        spine_label_c[binary_dilation(mask)>mask]=1
    spine_label_c[spine_label_c>(label>max(ignore))]=0    
    
    return    spine_label_c>0
def get_joint_border2(label,beginlabel=2):
    label=label.copy()
    label[label<beginlabel]=0
    mask=label>0
    labs=np.unique(label)
    footprint=np.ones((3,) * label.ndim, dtype=np.int8)
    border1=maximum_filter(label,footprint=footprint)>label
    label[label==0]=np.max(label)+1
    border2=minimum_filter(label,footprint=footprint)<label
    border=(border1 |border2) & mask
    return border
# def _preprocess_mask(self,mask,enhance_border=False):
#         """process mask to  ytrue for train,
#         1. trans to support dtype
#         2. return one hot
#         3. to tensor
        
#         if enhance_border enabled, will generate weighted ytrue that spine-joint-border will both mask 
#             as background and target class(spine) region .so network' out put will be sigmod
#         if use casual unet, the multi-lable-same-class will mask as same class id only

#         Args:
#             mask (ndarray): class label,support spine have multi label/ same label, but will all trans to same label
#             enhance_border (bool, optional): whether execute border weight. Defaults to False.

#         Returns:
#             ndarray: if commom unet,will generate mask which element 0,1,2...
#                     if border enabled , will generate
#         """ 
#         classnum=self.classnum
#         if enhance_border:
#             border=get_joint_border(mask,list(range(classnum-1)))
#         mask[mask>classnum-1]=classnum-1
#         border_thin=mask==classnum-1
#         border_thin=border_thin>binary_erosion(border_thin)
        
#         # mask = mask.astype(np.int64)
#         func=ToTensor()
#         mask=func(mask) # C H W
#         # to one-hot
#         y_one_hot = make_one_hot(mask,self.classnum) # C H W  
#         if enhance_border:
#             y_one_hot[0,...][border]=1.5  #background and forground all set postive and weight set 2
#             y_one_hot[classnum-1,...][border]=1.5  
#             # y_one_hot[classnum-1,...][border_thin]=1.5  
   
#         # fig,axs=plt.subplots(2,1,sharex=1,sharey=1)
#         # axs[0].imshow(y_one_hot[classnum-1,...])
#         # axs[1].imshow(border)
#         # plt.show()
#         return y_one_hot


  
        
def single_spine_box(img="",labelf=""):
    """get spine roi box img from labelfile and img file

    Args:
        img (str, optional): img file. Defaults to "".
        labdels (str, optional): label file. Defaults to "" , from 1 to num , if has dendrite from 2 to num+1.
    Return:
    """
    im=load_img(img)
    lab=load_img(labelf)
    regions = regionprops(lab)
    imgs=[]
    labs=[]
    ndim=im.ndim

    for prop in regions:
   
        # area=prop.area
        label=prop.label
        # index = prop.centroid
        box=prop.bbox
       
        obj=[slice(int(a1),int(a2),1) for a1,a2 in zip(box[:ndim],box[ndim::])]
        mm=im[obj]
       
        imgs.append(mm.copy())
        m=lab[obj]==label
        labs.append(m.copy())

    return imgs,labs

def split_img_to_roi(img="",labelf="",outdir=""):
    """return roi bounding box from whole label image

    Args:
        img (str, optional): img file. Defaults to "".
        labelf (str, optional): label file. Defaults to "".
        outdir (str, optional): directory to store img box. Defaults to "".
    """
    imgs,labs=single_spine_box(img,labelf)
    if not outdir:
        pdir=file_base.get_parent_dir(img,1)
        _,shorname,_=file_base.split_filename(img)
        ndir=os.path.join(pdir,shorname)
        file_base.create_dir(ndir)
    
    print("save roi forder",outdir)
    for n,(im,la)in enumerate(zip(imgs,labs)):
        newp=file_base.create_imgroi_path(img,n,"_roi",ndir)
        imsave(newp,im)
        newp2=file_base.create_imgroi_path(img,n,"_mask",ndir)
        #print("save ",newp,newp2)
        imsave(newp2,la)
        

#-----------------------#
# custom Test   #
#-----------------------#

# if __name__=="__main__":
#     #-----------------------#
#     #   test 1  #
#     #-----------------------#
#     # imgf=r"D:\spine\spinesoftware\myspine\data\Train\2D\20200319-1.proj\20200319-1.tif"   
#     # imgf=r"D:\spine\spinesoftware\myspine\data\Train\3D\20200319-1.proj\20200319-1.tif"   
#     # labf=r"D:\spine\spinesoftware\myspine\data\Train\2D\20200319-1.proj\segmentation.tif"
#     # labf=r"D:\spine\spinesoftware\myspine\data\Train\3D\20200319-1.proj\segment.tif"
#     # split_img_to_roi(imgf,labf)
    
    
    
#      #-----------------------#
#     #   test2  #
#     #-----------------------#
    
#     cong=default_configuration
#     laodir=cong.get_entry(['Path', 'label_path']) # label ori dir
#     imodir=cong.get_entry(['Path', 'data_path']) # img ori dir
#     ladir=cong.get_entry(['Path', 'orilabel_path'])
#     imdir=cong.get_entry(['Path', 'oridata_path'])
#     w=cong.get_entry(['Data', 'input_sizexy'])
#     nz=cong.get_entry(['Data', 'input_sizez'])
#     if nz>1:
#         outsize=(nz,w,w)# 3d
#     else:
#         outsize=(w,w)#2d
#     note=cong.get_entry(['Path', 'label_suffix'])
    
#     generate_crop_img_save(imdir,ladir,imodir,laodir,outsize,note=note,hull=False,depth=nz,iter=50)
#     print(w,nz,outsize,note)
    
