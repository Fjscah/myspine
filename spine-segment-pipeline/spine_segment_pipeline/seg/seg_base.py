

import numpy as np


#-----------------------#
#   point - mask  #
#-----------------------#
def get_point_from_mask(mask):
    indexs=np.argwhere(mask>0)
    points=list(indexs)
    # for ind in indexs:
    #     index=tuple(ind)
    #     points.append(index)
    return points

def get_mask_from_point(points,imgsize):
    mask=np.zeros(imgsize)
    points=np.around(points,0)
    points=np.array(points,dtype=np.uint64)
    indexs=list(points)
    mask[tuple(zip(*indexs))]=1

    return mask
from scipy import ndimage as ndi
def ndilable(image,offset=1):
    labels,num=ndi.label(image)
    labels[labels>0]+=offset
    return labels,num


def get_mask_box(mask):
    """get mask box from mask image

    Args:
        mask (ndarry): mask
    return:
        Box:[(x1,y1,[z1,...]),(x2,y2,[z2,...]))]
    """
    points=np.argwhere(mask)
    box=np.ptp(points,axis=0)
    
    return box

#-----------------------#
#   relabel  #
#-----------------------#

def resortseg(seg,start=1):
    """relabel seg from start

    Args:
        seg (label iamge): _description_
        start (int, optional): min label id. Defaults to 1.

    Returns:
        label image: relabel image from start
    """
    seg=seg.copy()
    labellist=list(np.unique(seg))
    labellist.sort()
    if 0 in labellist:
        labellist.remove(0)
    if len(labellist)>0 and labellist[0]<start:
        off=start-labellist[0]
        seg[seg>0]+=off # seg label > target label
    else:
        off=0
    
    newl=start-1
    for newl,oldl in enumerate(labellist,start):
        oldl+=off # seg label
        if(newl!=oldl):
            #print(oldl,newl)
            seg[seg==oldl]=newl # seg label = target label
    return seg,newl+1

def resortseg_truncate(seg,start=1):
    """ ignore label < start (keep label >= start), and resort rest label from start

    Args:
        seg (label img): _description_
        start (int, optional): _description_. Defaults to 1.

    Returns:
        label img: _description_
    """
    arr=(seg>=start)*seg
    labellist=list(np.unique(arr))
    if 0 in labellist:
        labellist.remove(0)
    newl=start-1
    for newl,oldl in enumerate(labellist,start):
        if(newl!=oldl):
            arr[arr==oldl]=newl
    # nstart=len(labellist)
    
    return arr,newl+1