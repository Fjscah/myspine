import sys
from operator import index
import matplotlib.pyplot as plt
import numpy as np
import skimage
from pip import main
from sklearn.preprocessing import binarize
from scipy.ndimage import generate_binary_structure
from skimage.transform import rotate
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import binary_dilation, cube, disk
from .measure import *
from .npixel import *
from . import npixel
gwdt_enable=False
try:
    from gwdt import gwdt
    gwdt_enable=True
except Exception as e:
    from .distance_transform import get_weighted_distance_transform
    print(e)

def path_spine_den(labels):
    """
    Args: straight line(vectors) from spine to nearset dendrite
        labels (int ndarray): den :1 , spine>1, background=0
    return : straight line from spine to nearset dendrite
    """
    fiber=labels!=1
    edt, inds = ndimage.distance_transform_edt(fiber, return_indices=True)
    las,areas,indexs=label_statics(labels)
    vectors=[]
    ndim=labels.ndim
    for index in indexs:
        index=np.round(index,0).astype(np.int)
        index=tuple(index)
        toindex=np.array([ind[index] for ind in inds])
        vector=np.zeros((2,ndim))
        vector[0,...]=index
        vector[1,...]=toindex-np.array(index)
        vectors.append(vector)
    return vectors
    
def constrain_path_spine_den(labels,mask):
    """return paths according to constain distance framsform
    path nlabel=len(paths), for each ele in path, npoint=len(path[0])
    that's [n,np,ndim] list
    for example : 
    [
        [[0,1,2],[0,3,2]],
        [[0,1,2],[0,1,4],[1,2,3]]
    ]

    Args:
        labels (np.ndarray): spine and fiber labels fiber=1, spine =2,3,4,,,,
        mask (np.ndarray):   neuron mask
    """
    labels=labels.copy()
    fiber=labels==1
    ndim=labels.ndim
    if(ndim==3) :footpoint=cube
    elif(ndim==2):footpoint=disk
    diskmatrix=np.zeros_like(labels)
    radius=1
    diff=fiber
    diskmatrix[diff]=radius
    while(diff.any()):
        radius+=1
        dfiber=binary_dilation(fiber,footpoint(3))
        dfiber=dfiber*mask
        diff=dfiber>fiber
        diskmatrix[diff]=radius
        fiber=dfiber
    las,areas,indexs=label_statics(labels)
    paths=[]
    connectlabels={}
    ndim=labels.ndim
    for la,index in zip(las,indexs):
        if(la==1): continue # fiber
        index=np.round(index,0).astype(np.int)
        index=tuple(index)
        path,labels=back_route(index,diskmatrix,labels)
        if len(path)>2:
            paths.append(path)
            #connectlabels[la]=passlabels

    # # merge label
    # lls={}
    # kls={}
    
    # def notindict(vs,lls):
    #     f=True
    #     for v in vs:
    #         if v in lls: 
    #             f=False
    #             break
    #     return f
    # def keysindict(vs,lls):
    #     keys=[]
    #     for v in vs:
    #         if v in lls: 
    #             keys.append(lls[v])
    #     keys=set(keys)
    # for key,vs in connectlabels.items():
    #     #  no any v has been in lls, create new dict
    #     if(notindict(vs,lls)):
    #         for v in vs:
    #             lls[v]=key
    #         kls[key]=set(vs)
    #     else:
    #         keys=keysindict(vs,lls)
    #         for k in keys:
    #             for v in kls[key]:
    #                 lls[v]=key 
    #             kls.pop(k)
    # print("kls",kls)
                
            
    
    
    
    return paths,labels
    
def back_route(index,distancemap,labels):
    size=distancemap.shape
    path=[]
    passlabel=[]
    la=labels[index]
    while(distancemap[index]>1):
        passlabel.append(labels[index])
        points=valid_connect_pixel(index,size) 
        for p in points:
            p=tuple(p)
            if distancemap[p]==distancemap[index]-1:
                path.append(p)
                index=p
                break
    passlabel=set(passlabel)
    if(len(passlabel)>1):
        for lab in passlabel:
            labels[labels==lab]=la
    return path,labels

def _trace_back(distance_transform,p):
    """ 
    trackback spine to dendrite, and get trace from den to spine
    p at den
    """
    p=tuple(p)
    v=distance_transform[p]
    traces=[p]
    mask=np.zeros_like(distance_transform)
    mask[p]=1
    shape=distance_transform.shape
    while(v>0):

        points=npixel.valid_surround_pixel(p,1,shape)
        minpos=None
        for pos in points:
            if pos is None: continue
            pos=tuple(pos)
            if distance_transform[pos]<v :  
                v=distance_transform[pos]
                minpos=pos
        if minpos is None: break
        traces.append(minpos)
        mask[minpos]=1
        p=minpos

    return traces,mask
def _trace_length(headmask,neckmask,denmask,centroid,dencord,lab=None):
    """ 
    tace head+neck trace to cal length, and get head diameter
    headmask neckmask denmask need be binary 01"""
    ndim=headmask.ndim
    shape=headmask.shape
    spinemask=((headmask+neckmask)>0)*1
    ls=denmask*spinemask*1
    inds=np.argwhere(ls>0)
    centroid=np.mean(np.argwhere(headmask>0),axis=0)
    dencord=np.mean(inds,axis=0)
    inds=list(inds)
    if ndim==3:
        angle=np.arctan2(centroid[2]-dencord[2],centroid[1]-dencord[1])*180/np.pi
    elif ndim==2:
        #print(centroid,dencord)
        angle=np.arctan2(centroid[1]-dencord[1],centroid[0]-dencord[0])*180/np.pi
    else:
        return None,None,None
    rotatemask=np.array(spinemask,dtype="uint8")*10
    if ndim==3:
        rotatemask=np.swapaxes(rotatemask,0,2)
    # plt.figure()
    # plt.imshow(rotatemask)
    # plt.colorbar()
    rotatemask=rotate(rotatemask,180-angle,resize=True,preserve_range=True)
    if ndim==3:
        rotatemask=np.swapaxes(rotatemask,0,2)
    rotatemask=rotatemask>4
    ps=np.argwhere(rotatemask)
    try:
        mins=np.min(ps,axis=0)
        maxs=np.max(ps,axis=0)
        wids=int((np.max(maxs-mins)+1)/2)
        mids=np.array((mins+maxs)/2,dtype="int")
    except:
        print(lab,angle,np.max(rotatemask))
        
        # plt.imshow(spinemask)
        # plt.figure()
        # plt.figure()
        # plt.imshow(neckmask)
        raise
    bindbox=[slice(max(mids[i]-wids,0), min(mids[i]+wids,rotatemask.shape[i]))   for i in range(ndim)]
    bindbox=tuple(bindbox)
    rotatemask=rotatemask[bindbox].copy()
  
    

    v=2
    cnts=[len(inds)]
    while(inds):
        points=[]
        #points=npixel.valid_surround_pixel(p,1,shape)
        minpos=None
        for ind in inds:
            ps=npixel.valid_surround_pixel(ind,1,shape)
            for pos in ps:
                if pos is None: continue
                pos=tuple(pos)
                if (not spinemask[pos]) or ls[pos]:continue
                points.append(pos)
                ls[pos]=v
        inds=points
        cnts.append(len(inds))
        v+=1
    length =np.max(ls)
    
    attachd_cnt=np.mean(cnts[0:len(cnts)//4+1])
    head=np.max(cnts[len(cnts)//2:])
    if ndim==3:
        head=np.sqrt(head/np.pi)
        attachd_cnt=np.sqrt(head/np.pi)
    head=head*0.7    
    attachd_cnt=attachd_cnt*0.7
    return length,head,attachd_cnt,rotatemask*lab


def spine_distance(img,labels,lab,searchbox,linemask):
    """caculate distance matrix by weighted img and get trace from spine to dendrite skeleton

    Args:
        img (ndarray): _description_
        labels (ndarray): _description_
        lab (int): _description_
        searchbox (tuple): _description_
        linemask (ndarray): _description_

    Returns:
        _type_: dendrite point while link cartain spine , change linemask
    """
    structure = generate_binary_structure(img.ndim, 3)
    def exclude_crop(v,b):
        if v<0:return 0
        if v>=b:return b-1
        return v
    inds=np.argwhere(labels==lab)
    shape=labels.shape
    centroid=np.mean(inds,axis=0)
    bindbox=[slice(
        exclude_crop(np.min(inds[:,i])-searchbox[i],shape[i]),
        exclude_crop(np.max(inds[:,i])+searchbox[i],shape[i])) 
             for i in range(labels.ndim)]
    bindbox=tuple(bindbox)
    cropimg=img[bindbox].copy()
    croplable=labels[bindbox]
    cropimg[croplable==lab]=0
    
    # plt.figure()
    if gwdt_enable:
        distance_transform=gwdt(cropimg, structure)
    else:
        distance_transform=get_weighted_distance_transform(cropimg)
    # plt.imshow(distance_transform)
    # plt.colorbar()
    cropdenmask=binary_dilation(croplable==1,structure)
    labdis=distance_transform*(croplable==1)
    if not labdis.any():
        #print("find no dendrite")
        return (centroid,None),None,None,None,None
    inds=np.argwhere(labdis>0)
    minind=None
    minv=np.inf
    for ind in inds:
        index=tuple(ind)    
        if labdis[index]<minv:
            minind=index
            minv=labdis[index]
    
    trace,maskcrop=_trace_back(distance_transform,minind)
    #print(trace)
    linemask[bindbox][maskcrop>0]=lab#maskcrop[maskcrop>0]
    dencord=[minind[i]+bindbox[i].start for i in range(labels.ndim) ]
    length,headlength,attachd_cnt,rotatemask=_trace_length(croplable==lab,maskcrop,cropdenmask,centroid,dencord,lab)
    
    return (centroid,dencord),length,headlength,attachd_cnt,rotatemask
from csbdeep.utils import normalize
def spines_distance(img,labels,searchbox=[10,20,20],imgweight=1,disweight=1):
    """caculate distance matrix by weighted img and get trace from spine to dendrite skeleton

    Args:
        img (ndarray): _description_
        labels (ndarray): include den +spine den=1 spine>=2
        searchbox (tuple): 2D or 3D searchbox 
        linemask (ndarray): _description_

    Returns:
        _type_: dendrite point while link cartain spine , change linemask
    """
    img=normalize(img)
    weightimg=(np.max(img)-img)
    weightimg=weightimg*imgweight
    weightimg=np.exp(weightimg)*disweight
    # weightimg=np.array(weightimg*weightimg,dtype="uint16")
    labs=set(np.unique(labels))
    if 0 in labs: labs.remove(0) #bg
    if 1 in labs: labs.remove(1)#den
    
    linemask=np.zeros_like(img,dtype="int16")
    corddict={}
    for lab in labs:
        cord,length,headlength,attachd_cnt,rotamask=spine_distance(weightimg,labels,lab,searchbox,linemask)
        corddict[lab]=[cord,length,headlength,attachd_cnt,rotamask]
    return corddict,linemask
    


def nei_label(labels,orientation):
    fiber=labels==1
    ndim=labels.ndim
    if(ndim==3) :footpoint=cube(3)
    elif(ndim==2):footpoint=disk(3)
    labs=list(np.unique(labels))
    labs.remove(0)
    labs.remove(1)
    labdict={}
    labcenters={}
    labos={}
    for lab in labs:
        
        mask=labels==lab
        mask=binary_dilation(mask,footpoint)>mask
        bbs=labels[mask]
        bbs=set(list(bbs))
        if(0 in bbs):
            bbs.remove(0)
        if(1 in bbs):
            bbs.remove(1)
        if(len(bbs)>0):
            labdict[lab]=bbs
    
    def oritation(regionmask,intensity):
        ori=np.nanmean(intensity)
        return ori
    props = regionprops(labels, orientation,extra_properties=(oritation,))
    for prop in props:
        lab=prop.label
        index=prop.centroid
        index=np.round(index,0).astype(np.int)
        index=tuple(index)
        labcenters[lab]=index
        labos[lab]=prop.oritation
    
    for key,vs in labdict.items():
        oinden=labcenters[key]
        oo=labos[key]
        for v in vs:
            to=labos[v]
            tinden=labcenters[v]
            #if np.mod(abs(oo-to),np.pi) < np.pi/4:
            tlab=labels[tinden]
            labels[labels==tlab]=key
            
    return labels
    
def trace_spine_den(labels,fibers,mask):
    """find a line from spine label to fiber, and line not cross to back region

    Args:
        labels (np.ndarray): spine labels ,1,2,3,4,
        fibers (np.ndarray): dendrite fiber, binary
        mask (np.ndarray):   neuron mask
    """
    indxs=np.argwhere(labels>2)
    labellist=[]
    route=np.zeros_like(mask,dtype=np.int8)
    paths=[]
    vetors=[]
    for ind in indxs:
        index=tuple(ind)
        if labels[index] in labellist: continue # has been resolved
        labellist.append(labels[index])
        endp=layerbylayer(index,fibers,mask,route)
        if (endp is None) : # not link to dendrite directly
            pass
        else:
            path,pathlabel=reverseRoute(endp,route,labels)
            pathlabel=set(pathlabel)
            path=np.array(path)
            paths.append(path)
    return paths
def reverseRoute(endp,route,labels):
    """return path from dentrite to spine,and labels pass by (for merge spine)

    Args:
        endp (tuple): dendrite node
        route (np.ndarray): map for get path
        labels (np.ndarray): spine labels

    Returns:
        path,pathlabels: retrun point path ,and label path
    """
    nextpoint=endp
    radius=route[endp]
    newp=None
    path=[endp]
    pathlabel=[]
    while(radius>1):
        radius-=1
        points=valid_connect_pixel(nextpoint,route.shape)
        for neip in points:
            if neip is None:
                continue
            neip=tuple(neip)
            if(route[neip]==radius):
                newp=neip
                path.append(newp)
                pathlabel.append(labels[newp])
                break
        nextp=newp   
    #last radius==1
    route=np.zeros_like(route)
    return path,pathlabel
            
def layerbylayer(start,fibers,mask,route):
    nextpoints=[start]
    route[start]=1
    radius=1
    while(nextpoints):
        newsp=[]
        radius+=1
        for p in nextpoints:
            points=valid_connect_pixel(p,mask.shape)
            for neip in points:
                if neip is None:
                    continue
                neip=tuple(neip)
                if(route[neip]): continue
                if (fibers[neip]): # find fiber node 
                    route[neip]=radius
                    return neip
                if(mask[neip]): # in signal mask
                    newsp.append(neip)
                    route[neip]=radius
        nextpoints=newsp
    return None
    
if __name__ == "__main__":
    m=np.arange(64).reshape((4,4,4))+1
    d=np.eye(4,4,4)>0
    m[d]=1
    print(m)
    vectors=path_spine_den(m)
    print(vectors)
    
    