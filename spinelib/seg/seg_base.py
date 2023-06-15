
import scipy.ndimage
from spinelib.imgio.utils import points_to_voxels
import scipy
import numpy as np
import scipy
from scipy import ndimage as ndi
import skimage.morphology as morph
from skimage import filters, morphology
from skimage.morphology import (binary_dilation, convex_hull_image, dilation,
                                erosion, remove_small_objects)
from ..utils import measure

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

def ndilable(image,offset=1):
    labels,num=ndi.label(image)
    labels[labels>0]+=offset
    return labels,num


#-----------------------#
#   convex hull  #
#-----------------------#

def fill_hull(image):
    """
    for each slice, Compute the convex hull of the given binary image and
    return a mask of the filled hull. 
    
    Adapted from:
    https://stackoverflow.com/a/46314485/162094
    https://gist.github.com/stuarteberg/8982d8e0419bea308326933860ecce30
    This version is slightly (~40%) faster for 3D volumes,
    by being a little more stingy with RAM.
    image :binary
    """
    # (The variable names below assume 3D input,
    # but this would still work in 4D, etc.)
    
    assert (np.array(image.shape) <= np.iinfo(np.int16).max).all(), \
        f"This function assumes your image is smaller than {2**15} in each dimension"
    
    points = np.argwhere(image).astype(np.int16)
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])

    # Instead of allocating a giant array for all indices in the volume,
    # just iterate over the slices one at a time.
    idx_2d = np.indices(image.shape[1:], np.int16)
    idx_2d = np.moveaxis(idx_2d, 0, -1)

    idx_3d = np.zeros((*image.shape[1:], image.ndim), np.int16)
    idx_3d[:, :, 1:] = idx_2d
    
    mask = np.zeros_like(image, dtype=bool)
    for z in range(len(image)):
        idx_3d[:,:,0] = z
        s = deln.find_simplex(idx_3d)
        mask[z, (s != -1)] = 1

    return mask

def fill_hulls(image,start=1):
    """
    for whole stack, Compute the convex hull of the given binary image and
    return a mask of the filled hull.
    
    Adapted from:
    https://stackoverflow.com/a/46314485/162094
    https://gist.github.com/stuarteberg/8982d8e0419bea308326933860ecce30
    This version is slightly (~40%) faster for 3D volumes,
    by being a little more stingy with RAM.
    iamge : labels int
    """
    # (The variable names below assume 3D input,
    # but this would still work in 4D, etc.)
    
    assert (np.array(image.shape) <= np.iinfo(np.int16).max).all(), \
        f"This function assumes your image is smaller than {2**15} in each dimension"
    
    labs=np.unique(image)
    mask = np.zeros_like(image)
    ndim=image.ndim
    
    if ndim==3:
        for lab in labs:
            if lab<start: 
                mask[image==lab]=lab
                continue
            points = np.argwhere(image==lab).astype(np.int16)
            if(len(points)<4):continue
            try:
                hull = scipy.spatial.ConvexHull(points)
                deln = scipy.spatial.Delaunay(points[hull.vertices])
            except:
                mask[image==lab] = lab
                continue

            # Instead of allocating a giant array for all indices in the volume,
            # just iterate over the slices one at a time.
            idx_2d = np.indices(image.shape[1:], np.int16)
            idx_2d = np.moveaxis(idx_2d, 0, -1)

            idx_3d = np.zeros((*image.shape[1:], image.ndim), np.int16)
            idx_3d[:, :, 1:] = idx_2d
            
            
            for z in range(len(image)):
                idx_3d[:,:,0] = z
                s = deln.find_simplex(idx_3d)
                mask[z, (s != -1)] = lab
    else:
        for lab in labs:
            if lab<start: 
                mask[image==lab]=lab
                continue
            points = np.argwhere(image==lab).astype(np.int16)
            if(len(points)<4):continue
            try:
                hull = scipy.spatial.ConvexHull(points)
                deln = scipy.spatial.Delaunay(points[hull.vertices])
            except:
                mask[image==lab] = lab
                continue

            idx = np.stack(np.indices(image.shape), axis = -1)
            out_idx = np.nonzero(deln.find_simplex(idx) + 1)
           
            mask[out_idx] = lab

    return mask

#-----------------------#
#    morphology #
#-----------------------#
def remove_large_objects(ar, max_size=64, connectivity=1,method="skimage"):
    """remove binary object larger than max_size

    Args:
        ar (_type_): _description_
        max_size (int, optional): _description_. Defaults to 64.
        connectivity (int, optional): _description_. Defaults to 1.
        method (str, optional): _description_. Defaults to "skimage".

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if "skimage" == method:
        mask =mask>0
        mask1=morphology.remove_small_objects(mask,max_size,connectivity=mask.ndim)
        return mask ^ mask1
    out = ar.copy()
    if max_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        footprint = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, footprint, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")
    too_small = component_sizes > max_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0
    return out

def remove_small_lable(mask,thsize):
    """remove each label's noise pixel (smaller than thsize)

    Args:
        mask (_type_): _description_
        thsize (_type_): _description_

    Returns:
        _type_: _description_
    """
    labs=measure.unique_labs(mask)
    newmask=np.zeros_like(mask)
    for lab in labs:
        mm=mask==lab
        mask2=remove_small_objects(mm,thsize)
        if not mask.any():
            mask2=mm
        newmask[mask2]=lab
   
    return newmask  


