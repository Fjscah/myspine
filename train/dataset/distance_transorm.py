import torch
import numpy as np
from scipy.ndimage import distance_transform_edt,distance_transform_cdt
from skimage.io import imread,imsave
from skimage.morphology import skeletonize
import napari
from csbdeep.utils import normalize
from gwdt import gwdt
from scipy.ndimage import gaussian_laplace, minimum_filter,maximum_filter
def outOfbound(index,shape):
    if np.min(index)<0:
        return True
    if np.min(shape-np.array(index))<=0:
        return True
    return False

def connect_pixel(point):
    # arbitary dim neighbour pixel
    point=list(point)
    linespace=[np.arange(p-1,p+2) for p in point]
    grids = np.meshgrid(*linespace,indexing="ij")
    grids=[grid.ravel() for grid in grids]
    points=[ index for index in zip(*grids)]
    points.pop(3**len(point)//2)
    return points

def valid_connect_pixel(point,size):
    points=connect_pixel(point)
    for n,p in enumerate(points):
        if outOfbound(p,size):
            points[n]=None
    return points
def resortseg(seg,start=1):
    labellist=list(np.unique(seg))
    labellist.sort()
    #print(labellist)
    arr=seg
    if 0 in labellist:
        labellist.remove(0)
    if len(labellist)>0 and labellist[0]<start:
        off=start-labellist[0]
    else:
        off=0
    seg[seg>0]+=off
    newl=start-1
    for newl,oldl in enumerate(labellist,start):
        oldl+=off
        if(newl!=oldl):
            #print(oldl,newl)
            arr[arr==oldl]=newl
    return arr,newl+1
def make_one_hot(label):
    # label = W H
    # one hot = W H ndim
    maxlab=np.max(label)
    eyes=np.eye(maxlab+1)
    outt=eyes[label]
    return outt
    outputs = torch.argmax(ypred, dim=1, keepdim=True).type(torch.int64)
    outputs = torch.zeros_like(ypred).scatter_(dim=1, index=outputs, src=torch.tensor(1.0)).type(torch.int8)
def get_joint_border2(label,beginlabel=2):
    # return join label pixel
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
def distance_edt_barrier(mask):
    # return distance matrix : initial =1,road =1- max distance(interger not float) ,wall = max border+1
    # here 1 is beginning,0 is road , -1 is wall, only 0 1 -1
    dismap=np.zeros_like(mask,"int")
    inds=np.argwhere(mask==1)
    points=list(inds)
    itern=1
    while points:
        new_points=[]
        for p in points:
            p=tuple(p)
            dismap[p]=itern
            ps=valid_connect_pixel(p,mask.shape)
            for pt in ps:
                if pt is None : continue
                pt=tuple(pt)
                if mask[pt]==0 and (not dismap[pt]):
                    new_points.append(pt)
        itern+=1
        points=new_points
    dismap[mask==-1]=itern+1
    return dismap
            
     
def make_label_distance_ske(label,skiplab=1):
    label=label.copy()
    label[label<=skiplab]=0
    label,_=resortseg(label,start=1)
    onthot_label=make_one_hot(label)
    outt=np.zeros_like(onthot_label,"float32")
    border=get_joint_border2(label,0)
    for dim in range(1,onthot_label.shape[-1]):
        lab_d=onthot_label[...,dim].copy()
        ske_d=skeletonize(lab_d)>border
        lab_dc=lab_d.copy()
        lab_dc-=1
        lab_dc[ske_d>0]=1        
        out_d=-distance_edt_barrier(lab_dc)
        out_d=out_d-np.min(out_d)
        # out_d[lab_d==0]=0
        if np.max(out_d)>0.1:
            out_d=(out_d/np.max(out_d))#*2
        out_d[lab_d<1]=0
        outt[...,dim]=out_d
    outt=np.sum(outt[...,1:],axis=-1)    
    # in_mask=label<=0
    # ex_out=distance_transform_edt(in_mask)
    # ex_out[ex_out>4]=4
    # ex_out=ex_out/4.0*0.5-0.5
    # outt[in_mask]=ex_out[in_mask]
    # outt[border]*=0.2
    return outt

# def distance_norm_center2border(ske,distance,indices):
#     norm_distance=np.zeros_like(distance,"float32")
#     inds=np.argwhere(ske>0)
#     ind2s=np.argwhere(distance>0)
#     for ind in inds:
#         point=tuple(ind)
#         dis=distance[point]
#         src_point=[indice[point] for indice in indices]
#         src_point=tuple(src_point)
#         distance[src_point]=min(dis,distance[src_point]) if distance[src_point] else dis
#     for ind in ind2s:
#         point=tuple(ind)
#         dis=distance[point]
#         src_point=[indice[point] for indice in indices]
#         max_dis=norm_distance[src_point]
#         norm_distance=
        
#         # while (distance[point]):
#         #     norm_distance[point]=max(norm_distance[point],distance[point]/dis)  
#         #     point=[indice[point] for indice in indices]
#         #     point=tuple(point)
#     return norm_distance
    
def make_label_distance_norm(label,skiplab=1):
    label=label.copy()
    label[label<=skiplab]=0
    label,_=resortseg(label,start=1)
    onthot_label=make_one_hot(label)
    outt=np.zeros_like(onthot_label,"float32")
    border=get_joint_border2(label,0)
    for dim in range(1,onthot_label.shape[-1]):
        lab_d=onthot_label[...,dim].copy()
        ske_d=skeletonize(lab_d)>border
        lab_dc=lab_d.copy()
        lab_dc-=1
        lab_dc[ske_d>0]=1        
        out_d1=distance_edt_barrier(lab_dc) # dis to center
        out_d2=distance_transform_edt(lab_d) # dis to border
        if np.max(out_d2)>0.1:
            norm_distance=np.where(out_d2==0,0,out_d2/(out_d2+out_d1-1)) 
            norm_distance=np.clip(norm_distance,0,1)
            norm_distance=np.log10(norm_distance*9+1)
            norm_distance[lab_d<1]=0
        # out_d=out_d-np.min(out_d)
        # # out_d[lab_d==0]=0
        #     out_d=(out_d/np.max(out_d))#*2
        # out_d[lab_d<1]=0
        outt[...,dim]=norm_distance
    outt=np.sum(outt[...,1:],axis=-1)    
    in_mask=label<=0
    # ex_out=distance_transform_edt(in_mask)
    # ex_out[ex_out>4]=4
    # ex_out=ex_out/4.0*0.5-0.5
    outt[in_mask]=-1#ex_out[in_mask]
    outt[border]*=-1
    return outt
def make_label_distance_edt(label,skiplab=1):
    label=label.copy()
    label[label<=skiplab]=0
    label,_=resortseg(label,start=1)
    onthot_label=make_one_hot(label)
    outt=np.zeros_like(onthot_label,"float")
    for dim in range(1,onthot_label.shape[-1]):
        out_d=distance_transform_edt(onthot_label[...,dim])
        if np.max(out_d)>0:
            out_d=out_d/np.max(out_d)
        outt[...,dim]=out_d
    outt=np.sum(outt[...,1:],axis=-1)    
    
    return outt
def make_label_distance_merge(label,skiplab=1):
    label,_=resortseg(label,start=1)
    onthot_label=make_one_hot(label)
    outt=np.zeros_like(onthot_label,"float")
    for dim in range(onthot_label.shape[-1]-1,skiplab,-1):
        lab_d=onthot_label[...,dim].copy()
        out_d1=distance_transform_edt(lab_d)
        if np.max(out_d1)>0:
            out_d1=out_d1/np.max(out_d1)
        ske_d=skeletonize(lab_d)
        lab_d-=1
        lab_d[ske_d>0]=1        
        out_d2=distance_edt_barrier(lab_d)
        out_d2=out_d2-np.min(out_d2)
        
        if np.max(out_d2)>0:
            out_d2=out_d2/np.max(out_d2)
            
        outt[...,dim]=(out_d1+out_d2)*0.5
    outt=np.sum(outt[...,1:],axis=-1)    
    return outt
if __name__=="__main__":
    filename=r"E:\data\myspine-dataset\2D-morph-seg\label\20200319-2-seg.tif"
    lab=imread(filename)
    # label=np.random.randint(0,3,(4,4))
    viewer=napari.Viewer()
    # # diss[lab<2]=-1
    diss=make_label_distance_ske(lab,1)
    diss2=make_label_distance_edt(lab,1)
    diss4=make_label_distance_norm(lab,1)
    # diss3=make_label_distance_merge(lab,1)
    # diss2[lab<2]=-1
    # diss=np.sum(diss,axis=-1)
    # diss=np.transpose(diss,(2,0,1))
    viewer.add_labels(lab)
    viewer.add_image(diss)
    viewer.add_image(diss2)
    viewer.add_image(diss4)
    # viewer.add_image(diss3)
    # print(label,)
    napari.run()