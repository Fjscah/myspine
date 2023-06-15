#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   skeleton_base.py
@Time    :   2021/01/18 15:25:28
@Author  :   Fjscah 
@Version :   1.0
@Contact :   www_010203@126.com
@Desc    :   some operator for skeleton, only for 2D/3D binary image
'''
from cProfile import label
from skimage.morphology import closing,remove_small_objects,opening,skeletonize_3d,dilation,cube,thin,disk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_local,apply_hysteresis_threshold
from tifffile import imwrite
from skimage.color import label2rgb
from scipy import ndimage

import skimage.morphology as morph
from ..seg.seg_base import remove_large_objects
# here put the import lib
import numpy as np
from numpy.lib.type_check import imag
from skimage import morphology
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
import sys


from ..utils import npixel
from ..utils.measure import vector_deg,euclidean
#√ multi skeletons int or bool return int
def label_trees(skeleton):
    '''
    skeleton is binary image, will return conected image , one tree one label,
    label form 1 to num(branchs)
    '''
    skeleton=skeleton>0
    label_tree_img=np.zeros_like(skeleton)+skeleton*(-1)
    skeleton_coords=[]
    h,w=label_tree_img.shape
    color_num=0
    for y in range(h):
        for x in range(w):
            if label_tree_img[y,x]==-1:
                color_num=color_num+1
                color_tree(label_tree_img,color_num,y,x)
    return label_tree_img,color_num

def color_tree(label_tree_img,colo_num,y,x):
    label_tree_img[y,x]=colo_num
    coords=npixel.connect_8(y,x)
    for cord in coords:
        if label_tree_img[cord[0],cord[1]]==-1:
            color_tree(label_tree_img,colo_num,cord[0],cord[1])

def tree_end_point_2d(skeleton_one_img):
    # bool or int
    end_points=[]
    if len(skeleton_one_img[skeleton_one_img>0])<2:
        y,x=np.argwhere(skeleton_one_img)[0]
        return [[y,x]]
    h,w=skeleton_one_img.shape
    for y in range(h):
        for x in range(w):
            if not skeleton_one_img[y,x]:
                continue
            coords=npixel.connect_8(y,x)
            count=0
            poss=[]
            for cord in coords:
                if cord[0]<0 or cord[0]>=h or cord[1]<0 or cord[1]>=w:
                    continue
                if skeleton_one_img[cord[0],cord[1]]:
                    count=count+1
                    poss.append([cord[0],cord[1]])
            if count==1:
                end_points.append([y,x])
            elif count==2:

                abss=abs(poss[0][0]-poss[1][0])+abs(poss[0][1]-poss[1][1])
                if abss<2:
                    end_points.append([y,x])

    return end_points


def get_min_pre_legth(label_one_tree_img,y,x,w,h):
    coords=npixel.connect_8(y,x)
    pos=[]
    lengthp=0
    for cord in coords:
        if cord[0]<0 or cord[0]>=h or cord[1]<0 or cord[1]>=w:
            continue
        if label_one_tree_img[cord[0],cord[1]]>0:
            if lengthp and label_one_tree_img[cord[0],cord[1]]<lengthp:
                lengthp=label_one_tree_img[cord[0],cord[1]]
                pos=cord

            elif not lengthp:
                lengthp=label_one_tree_img[cord[0],cord[1]]
                pos=cord
    return pos,lengthp

def label_length(label_one_tree_img,y,x,w,h):
    coords=npixel.connect_8(y,x)
    
    for cord in coords:
        if cord[0]<0 or cord[0]>=h or cord[1]<0 or cord[1]>=w:
            continue
        if label_one_tree_img[cord[0],cord[1]]==-1:
            label_one_tree_img[cord[0],cord[1]]=get_min_pre_legth(label_one_tree_img,cord[0],cord[1],w,h)[1]+1
            label_length(label_one_tree_img,cord[0],cord[1],w,h)
        #elif label_one_tree_img[cord[0],cord[1]]
    return label_one_tree_img


def caculate_length_from_start_point(binary_one_tree_img,start_p,end_points):
    label_one_tree_img=(binary_one_tree_img.copy()>0)*(-1)
    h,w=label_one_tree_img.shape
    y,x=start_p[0],start_p[1]
    label_one_tree_img[y,x]=1
    label_length(label_one_tree_img,y,x,w,h)
    lengths=[label_one_tree_img[cord[0],cord[1]] for cord in end_points]
    return lengths,label_one_tree_img
    
    
def get_branch_trace(label_length_img,star_p,end_p,w,h):
    trace=[end_p]
    current_p=end_p
    current_len=label_length_img[current_p[0],current_p[1]]
    while current_len>1:
        coords=npixel.connect_8(current_p[0],current_p[1])
        for cord in coords:
            if cord[0]<0 or cord[0]>=h or cord[1]<0 or cord[1]>=w:
                continue
            if label_length_img[cord[0],cord[1]] == current_len-1:
                current_len=current_len-1
                current_p=[cord[0],cord[1]]
                trace.append(current_p)
    return trace


def get_longest_branch_mask(binary_one_tree_img):
    # binary 255 or bool
    indexx,legth,trace=get_longest_branch(binary_one_tree_img)
    label_one_image=np.zeros_like(binary_one_tree_img)
    rows,cols=zip(*trace)
    #print(rows,cols,trace,lengthx)
    label_one_image[rows,cols]=255
    return label_one_image



# √  one skeleton int or bool
def get_longest_branch(binary_one_tree_img):
    # binary_one_tree_img is one label binary has one tree img
    binary_one_tree_img=binary_one_tree_img>0
    end_points=tree_end_point(binary_one_tree_img)
    h,w=binary_one_tree_img.shape
    length_max=0
    max_trace=None
    for start_p in end_points:
        lengths,label_length_img=caculate_length_from_start_point(binary_one_tree_img,start_p,end_points)
        length_temp=max(lengths)
        #print("length temp",length_temp)
        if length_max< length_temp:
            end_p=end_points[lengths.index(max(lengths))]
            indexx=[start_p,end_p]
            trace=get_branch_trace(label_length_img,start_p,end_p,w,h)
            #print('start-end',start_p,end_p,length_max,length_temp,trace)
            length_max=length_temp
    return indexx,length_max,trace

def distance_of_points_sets(set1,set2):
    pair=None
    min_length=np.inf
    for p1 in set1:
        y1,x1=p1
        for p2 in set2:
            y2,x2=p2
            d=np.sqrt((y2-y1)**2+(x2-x1)**2)
            if min_length>d:
                min_length=d 
                pair=[[y1,x1],[y2,x2]]
    return min_length,pair

def distances_of_sets(sets):
    distance_matrix=np.ones((len(sets),len(sets)))
    pairs_matrix=np.zeros((len(sets),len(sets),4),dtype=np.uint8)
    for n,set1 in enumerate(sets):
        for m,set2 in enumerate(sets):
            min_length,pair=distance_of_points_sets(set1,set2)
            #print('minlength',min_length)
            distance_matrix[n,m]=min_length
            pairs_matrix[n,m]=[pair[0][0],pair[0][1],pair[1][0],pair[1][1]]
            #print('pp',[pair[0][0],pair[0][1],pair[1][0],pair[1][1]])
    return distance_matrix,pairs_matrix


def color_skeleton_baseon_point(binary_one_tree_img,start_point):
    # binary_one_tree_img is one label binary has one tree img
    """
    start_point: if start_point not on skeleton ,will find nearest end point as tart_p
    """
    binary_one_tree_img=binary_one_tree_img>0
    marked_keleton_img=binary_one_tree_img*(-1)

    end_points=tree_end_point(binary_one_tree_img)
    w,h=binary_one_tree_img.shape

    min_length,pair=distance_of_points_sets([start_point],end_points)
    start_p=pair[1]
    marked_keleton_img[start_p[0],start_p[1]]=1

    marked_keleton_img=label_length(marked_keleton_img,start_p[0],start_p[1],w,h)
    return marked_keleton_img,start_p




def color_baseon_skeleton(marked_skeleton,mask):
    #print('set',set(marked_skeleton.ravel()))
    indexx,length,trace=get_longest_branch(marked_skeleton)
    #print('minlength',length,len(trace))
    fill_color_mask=np.zeros_like(marked_skeleton)
    h,w=mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y,x]:
                min_distance=np.inf
                for y2,x2 in trace:
                    d= (y2-y)**2+(x2-x)**2
                    if d<min_distance:

                        min_distance=d
                        fill_color_mask[y,x]=marked_skeleton[y2,x2]
                    #print(fill_color_mask[y,x])
    return fill_color_mask






#==↓==↓==↓==↓==↓==↓== 2D/3D skeleton==↓==↓==↓==↓==↓==↓#

def label_trees_end_points(label_tree_img):
    #return end point(leaf point mask) 2D/3D
    #√ multi labels for skeleton  int or bool , keep int or bool
    # label_tree_img canbe skeleon or labeled img
    leafmask=np.zeros_like(label_tree_img)
    label_tree_img=label_tree_img.copy()
    end_points=tree_end_point(label_tree_img)
        #print("leaf point",len(end_points))
    for point in end_points:
        leafmask[tuple(point)]=1
    label_tree_img=leafmask*label_tree_img

    return label_tree_img

def get_border_connection(skeimage,x,y,z=None,getlabel=False):
    # get degree of node, 
    # 2D:0-4
    # 3D:0-14
    # if you use this methos 
    if z : # 3d
        cube=skeimage[x-1:x+2,y-1:y+2,z-1:z+2]
        cube=cube.copy()
        cube[1,1,1]=0
        labels,num=ndi.label(cube)
    else:
        square=skeimage[x-1:x+2,y-1:y+2]
        square=square.copy()
        square[1,1]=0
        labels,num=ndi.label(square)
    return num


def tree_end_point(skeleton_one_img):
    # bool or int
    # return leaf points
    ndim=skeleton_one_img.ndim
    end_points=[]
    if not skeleton_one_img.any():return []
    inds=np.argwhere(skeleton_one_img)
    if inds.shape[0]<2:
        return inds
    mask=skeleton_one_img.copy()
    mask=np.pad(mask,[(1,1) for n in range(ndim)])
    
    for ind in inds:
        index=tuple(ind+1)
        if get_border_connection(mask,*index)<=1:
            end_points.append(list(ind))
    return end_points

def _label_leaf(label_ske_img,pos,traces=[]):
    # label_ske_img has been add frame!!!!
    # 0 backgournd 1 ske 2 leaf
    pos=tuple(pos)
    if not label_ske_img[pos]: return #has label
    if get_border_connection(label_ske_img,*pos)<3:# check conenction
        label_ske_img[pos]=2 #leaf
        traces.append(pos)
        coords=npixel.valid_connect_pixel(pos,label_ske_img.shape)
        poss=[]
        for pos in coords:
            pos=tuple(pos)
            if label_ske_img[pos]==1:
                poss.append(pos)
        for pos in poss:
            # double check , must ,otherwise error
            if get_border_connection(label_ske_img,*pos)>2: 
                return
        for pos in poss:
            _label_leaf(label_ske_img,pos,traces=traces)
    else: return
def cut_short_leaf(skeleton_one_img,leaf_length=None): # if length > leaf_length is not leaf
    skeleton_one_img=skeleton_one_img>0
    end_points=tree_end_point(skeleton_one_img)
    mask=skeleton_one_img.copy().astype(np.uint8) #inital 1 for ske ,  0 backgournd 1 ske 2 leaf
    ndim=skeleton_one_img.ndim
    mask=np.pad(mask,[(1,1) for n in range(ndim)])
    #print("leaf number :",len(end_points))
    another_end_points=[]
    for pos in end_points:
        pos=np.array(pos)+1
        traces=[]
        traces.append(pos)
        _label_leaf(mask,pos,traces=traces)
        another_end_points.append(traces)
    fibermask=mask==1
    #print("len traces",len(another_end_points))
    for traces in another_end_points:
        ind=max(len(traces)-4,0)
        vector1=np.array(traces[-1])-np.array(traces[ind])
        points1=npixel.valid_surround_pixel(traces[-1],1,fibermask.shape)
        flag=False
        for point in points1:
            if point  and fibermask[tuple(point)]:
                if get_border_connection(fibermask,*list(point))==1:
                    flag=True
        if not flag :continue
        points=npixel.valid_surround_pixel(traces[-1],11,fibermask.shape)
        fiberpoints=[]
        for point in points:
            if point  and fibermask[tuple(point)]:
                fiberpoints.append(point)
        #print("len fiberpoint : ",fiberpoints)
        if (fiberpoints or len(fiberpoints)>1):
            farpoint=None
            nearpoint=None
            maxdis=0
            mindis=np.inf
            for point in fiberpoints:
                dis=euclidean(traces[-1],point)
                if dis>maxdis:
                    maxdis=dis
                    farpoint=point
                if dis<mindis:
                    mindis=dis
                    nearpoint=point
            vector2=np.array(farpoint)-np.array(nearpoint)
            deg=vector_deg(vector1,vector2)
            print(deg)
            if deg<45:
                # remove end leaf(maybe fiber actually)
                print("find angle <30")
                for point in traces:
                    mask[tuple(point)]=1
    if ndim==2:
        mask=mask[1:-1,1:-1]
    elif ndim==3:
        mask=mask[1:-1,1:-1,1:-1] # label all leaf
    leafmask=mask==2
    
    # remove leaf
    if leaf_length:
        leafmask=remove_large_objects(leafmask,leaf_length,method="skimage")
    
    return skeleton_one_img ^ leafmask

def removeleaf(ad_th,mask=None,spine_maxlen=15):
    
    image1=morph.remove_small_holes(ad_th,9)  
    image2=remove_small_objects(image1,4)
    ndim=ad_th.ndim

    # image_open=opening(image)
    # image_open=remove_small_objects(image_open,100)
    image2=(image2).astype(np.uint8)

    #remove skeleton noise
    
    image_skeleton=skeletonize_3d(image2).astype(np.uint8)
    if mask is not None:
        image_skeleton=image_skeleton*mask
    if ndim==3:
        footprint=cube(5)
    elif ndim==2:
        footprint=disk(5)
    skedilate=dilation(image_skeleton,footprint)
        
    #image_fill=ndimage.binary_fill_holes(image_skeleton)
    image_skeleton=skeletonize_3d(skedilate)

    # label leaf
    fibermask=cut_short_leaf(image_skeleton,spine_maxlen)# size is diameter
  
    leafmask=fibermask ^ image_skeleton
    leafmask=leafmask & (~dilation(fibermask,footprint=footprint))
    
    # removeleaf2=cut_short_leaf_3d(removeleaf,6)# size is diameter
    # leafmask2=removeleaf ^ removeleaf2
    # leafmask2=leafmask2 & (~dilation(removeleaf2,footprint=cube(3)))
    
    #leafmask=np.logical_or(leafmask,leafmask2)
    leaflabel,num=ndi.label(leafmask,np.ones((3,)*ndim))
        
    print("leaf number ", num,"\nlabels:\n",set(list(leaflabel.ravel())))
    leaflabel[leaflabel>0]+=1
    fiberlabel=fibermask.astype(np.uint8)

    return fiberlabel,leaflabel
    
def _label_path(skele_img,label,p):
    points=npixel.valid_surround_pixel(p,1,skele_img.shape)
    v=label[tuple(p)]
    for pos in points:
        if pos is None: continue
        pos=tuple(pos)
        if label[pos] and label[pos]<=v+1 : continue
        if skele_img[pos]:
            label[pos]=v+1
            _label_path(skele_img,label,pos)
def _reverse_path(label,p):
    v=label[tuple(p)]
    traces=[tuple(p)]
    while(v>0):
        v-=1
        points=npixel.valid_surround_pixel(p,1,label.shape)
        for pos in points:
            if pos is None: continue
            pos=tuple(pos)
            if label[pos]==v :  
                p=pos
                traces.append(pos)
                break  
    return traces
                     
def get_longest_path(skeleton_img):
    end_points=tree_end_point(skeleton_img)
    max_traces=None
    max_length=0
    for p in end_points:
        mask=np.zeros_like(skeleton_img,dtype=np.uint16)
        mask[tuple(p)]=1
        _label_path(skeleton_img,mask,p)
        ind=tuple(np.unravel_index(np.argmax(mask), mask.shape))
        length=mask[ind]
        if length>max_length:
            max_length=length
            max_traces=_reverse_path(mask,ind)
    return max_traces,max_length
def get_longest_path_mask(skeleton_img):
    max_traces,max_length=get_longest_path(skeleton_img)
    mask=np.zeros_like(skeleton_img)
    for p in max_traces:
        mask[tuple(p)]=1
    return mask   


def _label_attach(leafmask,label,p):
    points=npixel.valid_surround_pixel(p,1,leafmask.shape)
    v=label[tuple(p)]
    for pos in points:
        if pos is None: continue
        pos=tuple(pos)
        if label[pos] and label[pos]<=v+1 : continue
        if leafmask[pos]:
            label[pos]=v+1
            _label_attach(leafmask,label,pos)

def _grow_attach(label,label2,p):
    points=npixel.valid_surround_pixel(p,1,label.shape)
    v=label[tuple(p)]
    lab=label2[tuple(p)]
    for pos in points:
        if pos is None: continue
        pos=tuple(pos)
        if label2[pos] : continue
        if label[pos]==v+1:
            label2[pos]=lab
            _grow_attach(label,label2,pos)
def leaf_label(skeleton_img):
    fibermask=get_longest_path_mask(skeleton_img)
    leafmask=skeleton_img>fibermask
    attach_mask=dilation(fibermask,np.ones((5,)*skeleton_img.ndim))*leafmask
    inds=np.argwhere(attach_mask)
    mask=np.zeros_like(skeleton_img,dtype=np.uint16)
    for ind in inds:
        index=tuple(ind)
        mask[index]=1
        _label_attach(leafmask,mask,index)
    labels,_=ndi.label(attach_mask,np.ones((3,)*skeleton_img.ndim))
    for ind in inds:
        index=tuple(ind)
        _grow_attach(mask,labels,index)
    return labels
    
def removeleaf2(ad_th,mask=None,spine_maxlen=15):
    
    image1=morph.remove_small_holes(ad_th,9)  
    image2=remove_small_objects(image1,4)
    ndim=ad_th.ndim

    # image_open=opening(image)
    # image_open=remove_small_objects(image_open,100)
    image2=(image2).astype(np.uint8)

    #remove skeleton noise
    
    image_skeleton=skeletonize_3d(image2).astype(np.uint8)
    if mask is not None:
        image_skeleton=image_skeleton*mask
    if ndim==3:
        footprint=cube(5)
    elif ndim==2:
        footprint=disk(5)
    # skedilate=dilation(image_skeleton,footprint)
        
    #image_fill=ndimage.binary_fill_holes(image_skeleton)
    # image_skeleton=skeletonize_3d(skedilate)
    
    fibermask=get_longest_path_mask(image_skeleton)
    leafmask=image_skeleton>fibermask
    attach_mask=dilation(fibermask,footprint)*leafmask
    
    inds=np.argwhere(attach_mask)
    mask=np.zeros_like(image_skeleton,dtype=np.uint16)
    for ind in inds:
        index=tuple(ind)
        mask[index]=1
        _label_attach(leafmask,mask,index)
    
    resleaf=leafmask>mask
    labels1,_1=ndi.label(resleaf,np.ones((3,)*image_skeleton.ndim))
    labels1[labels1>0]+=1
    labels,_=ndi.label(attach_mask,np.ones((3,)*image_skeleton.ndim))
    labels[labels>0]+=2+_1
    labels[labels1>labels]=labels1[labels1>labels]
    for ind in inds:
        index=tuple(ind)
        _grow_attach(mask,labels,index)
    
    labels[dilation(fibermask,footprint=footprint)]=0
    print("leaf number ",_1+_,"\nlabels:\n",set(list(labels.ravel())))
    
    fiberlabel=fibermask.astype(np.uint8)
    leaflabel=labels
    return fiberlabel,leaflabel,labels1# labels1: no attach label
  


#==↓==↓==↓==↓==↓==↓== 3D skeleton==↓==↓==↓==↓==↓==↓#



def label_trees_end_points_3d(labeled_ske):
    # return leaf point image
    # labeled_ske canbe skeleon or labeled img
    end_points=tree_end_point(labeled_ske)
    leafmask=np.zeros_like(labeled_ske)
    labeled_ske=labeled_ske.copy()
    for point in end_points:
        leafmask[tuple(point)]=1
    labeled_ske=leafmask*labeled_ske
    return labeled_ske

#TODO
def label_branchs_layers_onetree_3d(skeleton_img):
    end_points=tree_end_point(skeleton_img)
    pass
def label_branchs_onetree_3d(skeleton_img,label=True):
    # principle  :  >=3 Bifurcation remove # no layer , like decompose
    # it not return for fiddtren tree , so if you want to get braches for each tree,please 
    # run label tree first(ndi.label) , then for different label , run label_branches_one_tree
    #badding a frame
    mask=skeleton_img.copy()
    mask=np.pad(mask,((1,1),(1,1),(1,1)))

    xx,yy,zz = skeleton_img.shape
    for x in range(1,xx+1):
        for y in range(1,yy+1):
            for z in range(1,zz+1):
                cube=mask[x-1:x+2,y-1:y+2,z-1:z+2]
                cube=cube.copy()
                cube[1,1,1]=0
                labels,num=ndi.label(cube)
                if num>=3:
                    mask[x,y,z]=0
    mask=mask[1:xx+1,1:yy+1,1:zz+1]
    if label:
        return ndi.label(mask)
    return mask


#TODO
def cut_short_branch_3d(skeleton_one_img):
    pass









# def label_leaf(image):


#     oriimage=image.copy()
#     # Now we want to separate the two objects in image
#     # Generate the markers as local maxima of the distance to the background
#     #11 512 512
#     mean1=np.mean(image)
#     stdv1=np.std(image)

#     noiseimg=image.copy()

#     noise_mean=np.nanmean(noiseimg[noiseimg<mean1+3*stdv1])
#     noise_stdv=np.nanstd(noiseimg[noiseimg<mean1+3*stdv1])

#     sig_mean=np.nanmean(noiseimg[noiseimg>mean1+stdv1])
#     sig_stdv=np.nanstd(noiseimg[noiseimg>mean1+stdv1])
#     print("image mean: ",mean1, " stdv :",stdv1)
#     print("moise mean: ",noise_mean, " stdv :",noise_stdv)



#     #%%    ======Threshold========
#     ad_th=image.copy()
#     #border
#     ad_th_mask=apply_hysteresis_threshold(ad_th, noise_mean+2*noise_stdv,noise_mean+3*noise_stdv)


#     #ad_th[ad_th<noise_mean+3*noise_stdv]=0
#     #inner smller than mask
#     ad_th_inner=ad_th.copy()
#     for i in range(image.shape[0]):
#         ad_th_inner[i,:,:]=ad_th[i,:,:]>threshold_local(ad_th[i,:,:],35,'gaussian')

#     ad_th_inner[~ad_th_mask]=0

    
#     #ad_th[image<noise_mean+3*noise_stdv]=0

#     image1=morph.remove_small_holes(ad_th_inner,9)  
#     image2=remove_small_objects(image1,4)

#     # image_open=opening(image)
#     # image_open=remove_small_objects(image_open,100)
#     image2=(image2).astype(np.uint8)

#     #remove skeleton noise
#     image_skeleton=skeletonize_3d(image2).astype(np.uint8)
#     skedilate=dilation(image_skeleton,cube(5))
#     image_fill=ndimage.binary_fill_holes(image_skeleton)
#     image_skeleton=skeletonize_3d(skedilate)

#     # label leaf
#     removeleaf=cut_short_leaf_3d(image_skeleton,15)# size is diameter
#     leafmask=removeleaf ^ image_skeleton
#     points=tree_end_point(image_skeleton)
#     leafpointimg=label_trees_end_points_3d(image_skeleton)
#     #return leafpointimg*255



#     # imwrite("ske.tif",image_skeleton)
#     #give leaf diffent label
#     # split leafmask and fibermask for watershed
#     leafmask=leafmask & (~dilation(removeleaf,footprint=cube(3)))

#     leaflabel,num = ndi.label(leafmask,cube(3))


#     #backfround 0  leaf >=2
#     leaflabel=leaflabel+(leaflabel>0).astype(np.uint8)*1


#     leaf_projectz=np.max(leafmask,axis=0)





#     print("leaf number ", num,"labels",set(list(leaflabel.ravel())))

#     leafnodemask=label_trees_end_points(image_skeleton).astype(np.uint8)
#     # for showing pot that show spine pot 
#     leafnodelabel=leafnodemask*leaflabel

#     # fiber = 1
#     fiberlabel=removeleaf.astype(np.uint8)*(1)

#     # backfround 0  leaf >=2 fiber = 1
#     labels=leaflabel+fiberlabel



   
