from dask.array.image import imread as da_imread
from matplotlib import image
from skimage import io
from skimage import color
from PIL import Image
from skimage import util
import numpy as np
import tifffile
def test_da_imread():
    stacks = da_imread("Stabilized_Concatenate_561.tif")
    print(stacks.shape)
def ski_imread(filename):
    return io.imread(filename)
def show_img_info(images):
    shape=images.shape
    dtype=images.dtype
    ndim=images.ndim
    print("image shape :",shape)
    print("image dtype :",dtype)
    print("image ndim :",ndim)
def CvToGray(image):
    if (1):
        return color.rgb2gray(image)
    elif(2):
        img=Image(image)
        img_gray=img.convert('L')
        return img_gray
    elif(3):
        import cv2
        return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
def Cvt8bit(image):
    return util.img_as_ubyte(image)
def pad2even(image):
    pad=[(0,s%2) for s in image.shape]
    image=np.pad(image,pad,'constant')
    return image

def savepr(arr,filename):
    tifffile.imwrite(
        filename,
        arr,
        imagej=True,
        photometric='minisblack',
        #metadata={'axes': 'TYX'},
    )
    print("save :",filename)
def savelabel(arr,filename):
    tifffile.imwrite(
        filename,
        np.array(arr,dtype=np.uint16),
        imagej=True,
        photometric='minisblack',
        #metadata={'axes': 'TYX'},
    )
    print("save :",filename)