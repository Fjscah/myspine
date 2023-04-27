from pint import Measurement
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea,QFileDialog,QMessageBox
from qtpy.QtCore import Qt
import napari
import functools
import numbers
import os
import time
import scipy.ndimage as ndi
from typing import List, Union
from warnings import warn
from .cflow.localthreadhold import all_threshold_func,local_threshold,local_threshold_23d
from .cflow.blob import peakfilter
# from .cflow import *
from .utils.measure import label_series_statics
from .utils.file_base import create_dir
import napari
import numpy as np
from skimage.io import imread,imsave
from skimage.morphology import binary_dilation
from skimage.segmentation import watershed
from csbdeep.utils import (
    _raise,
    axes_check_and_normalize,
    axes_dict,
    load_json,
    normalize,
)
import torch
from magicgui import magicgui
from magicgui import widgets as mw

from psygnal import Signal

from napari.utils.notifications import show_info

# from .networks.unetplusplus import UNet2d,NestedUNet,CNN
from .imgio import napari_base
from .seg import segment
from .seg import unetseg
from .strutensor.hessian import StructureEvi,enhance_ridges_2d

import time
import pickle
from skimage.morphology import remove_small_objects
import numpy as np
import os
from .utils.file_base import split_filename
from .utils.spine_struct import spines_distance,gwdt_enable


#-----------------------#
#   paras  #
#-----------------------#
paras=dict(
    steps=2, # caculate stick gradient step lenth 1-4, larger more error , smaller than spine diameter
    sigma=(2,2,2), # z,y,x
    sigma_hessian = 2,
    sigma_supp = 100,
    countth=20, # fiber largest pixels, include z stack, that to say lenth*stackn
    spine_radius=20, # for iteration to growth area
    lambda1=1,
    lambda2=1.5, # lambda2 need large than lambda1
    maxspinesize=800
)

edit_mode = [
    ("None", 0),
    ("Add", 1),
    ("Delete", 2),
]
grow_mode = [
    ("chan_vesc", 0),
    ("geodesic", 1),
]
modify_mode = [
    ("modify", 0),
    ("reset", 1),
]
strategy_mode= [
    ("Single", 0),
    ("MIP", 1),
]
segment_mode=[
    ("Auto", 0),
    ("Custom", 1),
]
Axes_type=["xy","txy","zxy","tzxy"]
DEFAULTS = dict(norm_image=True,
                input_scale="None",
                perc_low=1.0,
                perc_high=99.8,
                norm_axes="ZYX",
                prob_thresh=0.5,
                nms_thresh=0.4,
                editmode=edit_mode[0][1],
                growmode=grow_mode[0][1],
                strategy_mode=strategy_mode[0][1],
                segment_mode=segment_mode[0][1],
                )

Bmethod_func,Bmethod_name=all_threshold_func()
Instance_funcname=["peak","background"]
Instance_funcdict={
    "peak":unetseg.instance_unetmask_bypeak,
    "background":unetseg.instance_unetmask_by_border,
}
net_method=["unet2d","unet++2d","unet3d"]
measurement_choices=["all","area",
                     "mean_intensity",
                     "centroid"]#"total_intensity"

Spot_method=["structure","hessian","CNN","SVM"]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#   classify loader  #
#-----------------------#
# svm function
def predict_cls(clff,xs):
    clf,pca,scaler,_=clff
    xs=np.array([x.ravel() for x  in xs])
    # print(xs.shape)
    xs=scaler.transform(xs)
    xs=pca.transform(xs)
    y_pred = clf.predict(xs)
    return y_pred
def filter_p(points,lables):
    pps=[]
    pps2=[]
    for p,l in zip(points,lables):
        if l>0:
            pps.append(p)
        else:
            pps2.append(p)
    return pps,pps2

def filter_pd(points,image,clfs):
    ndim=image.ndim
    if ndim==3:
        #boxsize=clfs[-1]
        imagemax=np.max(image,axis=0)
        dpsimg=segment.cropfromseed(imagemax,points[...,1:],boxsize=clfs[-1])
    elif ndim==2:
        dpsimg=segment.cropfromseed(image,points,boxsize=clfs[-1])
    # print("ndim",ndim)
    y_pred=predict_cls(clfs,dpsimg)
    # print(y_pred)
    points,points2=filter_p(points,y_pred)
    return points,points2


def load_file_model(filepath):
    lmodel=torch.jit.load(filepath,map_location=torch.device('cpu'))
    # except Exception as e:
    #     show_info(e)
    #     return None
    return lmodel
  
def reshape_data(XX,shape):
    X=[]
    for x in XX:
        # x=filters.sobel(x)
        x=x.reshape(*tuple(shape))
        x=x.astype(np.float32)
        X.append(x)
    return np.array(X)
def modify_mask(mask,sizeth=4):
    denmask=mask==1
    spinemask=mask==2
    denmask=remove_small_objects(denmask,min_size=sizeth)
    spinemask=remove_small_objects(spinemask,min_size=sizeth)
    mask=spinemask*2+denmask
    return mask

def model_predict(points,image,modle,boxsize):
    if image.ndim==3:
        image=np.max(image,axis=0)
        #modle.summary()
        dpsimg=segment.cropfromseed(image,points[...,1:],boxsize=boxsize)
    elif image.ndim==2:
        dpsimg=segment.cropfromseed(image,points,boxsize=boxsize)
    

    dpsimg=reshape_data(dpsimg,[1,]+list(boxsize))
    # y_p=modle(dpsimg)
    # y_pred=y_p.argmax(axis=-1)
    #modle.to("cpu")
    y_p=modle(torch.tensor(dpsimg))
    y_pred=torch.argmax(y_p,dim=1).cpu().data.numpy()
    points,points2=filter_p(points,y_pred)
    return points,points2
# lmodel=load_default_model()
def th_img(imgs,all_lls,binary_func):
    global_th=binary_func(imgs[0])
    lss=np.zeros_like(all_lls)
    for i in range(len(imgs)):
        img=imgs[i]
        adth=img>global_th
        #corner=peakfilter(img,3,0,use_gaussian=True)*adth
        ls=all_lls[i].copy()
        #ls[~adth]=0
        lss[i]=ls
    return lss




# widget 1 : training label widget
def plugin_wrapper1():

    #-----------------------#
    #   call back  #
    #-----------------------#
    DEBUG = False

    def change_handler(*widgets, init=True, debug=DEBUG):
        def decorator_change_handler(handler):
            @functools.wraps(handler)
            def wrapper(*args):
                source = Signal.sender()
                emitter = Signal.current_emitter()
                if debug:

                    print(
                        f'{str(emitter.name).upper()}: {source.name} = {args!r}'
                    )
                return handler(*args)

            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(widget.value)
            return wrapper

        return decorator_change_handler

    #-----------------------#
    #   GUI  #
    #-----------------------#
    @magicgui(
        image=dict(label="grayscale Image"),
        binaryimage=dict(label="binary Image"),
        spinepoints=dict(label="spine point"),
        denpoints=dict(label="dendrite point"),
        labelmask=dict(label="segment layer"),
        ridge_button=dict(widget_type="CheckBox", text="ridge",value=False ),
        denmask=dict(label="den layer"),
        # ambiguspoints=dict(label="ambigus point"),
        preprocess1=dict(widget_type="Label",
                         label="<br><b>1. threshold</b>"),
        axes=dict(widget_type="LineEdit", label="Image Axes", value="zyx"),
        bmethod=dict(
            widget_type="ComboBox",
            label="Binary method",
            choices=Bmethod_name+["custom"],
            value=Bmethod_name[-3],
        ),
        threshold_slide=dict(widget_type="LineEdit",label="threshold",value="0"),     
        adth_button=dict(widget_type="PushButton", text="binarylize"),
        preprocess2=dict(widget_type="Label",
                         label="<br><b>2. find seed point</b>"),
        # minrange=dict(widget_type="LineEdit", label="min range", value="5",tooltip=r"/sqrt(3)"),
        # maxrange=dict(widget_type="LineEdit", label="max range", value="9",tooltip=r"/sqrt(3)"),
        # anisotropy_factor=dict(widget_type="LineEdit", label="anisotropy_factor", value="2.0",tooltip=r"float,usually >1,fork from platymatch"),
        spotmethod=dict(
            widget_type="ComboBox",
            label="Spot method",
            choices=Spot_method,
            value=Spot_method[0],
        ),
        
        model_file=dict(widget_type="FileEdit",visible=True),
        radiusrange=dict(widget_type="LineEdit", label="radius", value="5"),
        peak_button=dict(widget_type="PushButton", text="find spot"),
        spine_button=dict(widget_type="PushButton", text="to spine",tooltip="short key : shift+s"),
        den_button=dict(widget_type="PushButton", text="to den",tooltip="short key : shift+d"),
        # ambigus_button=dict(widget_type="PushButton", text="ambiguous"),
        # editrad=dict(
        #     widget_type="RadioButtons",
        #     label="Mode",
        #     orientation="horizontal",
        #     choices=edit_mode,
        #     value=DEFAULTS["editmode"],
        #     tooltip="not work , wait TODO"
        # ),
        
        
        preprocess3=dict(widget_type="Label",
                         label="<br><b>3. grow spine area</b>"),
        growrad=dict(
            widget_type="RadioButtons",
            #visible=False,
            label="grow method",
            choices=grow_mode,
            value=DEFAULTS["growmode"],
            orientation="horizontal",
        ),
        searchbox=dict(widget_type="LineEdit", label="search box", value="11,35,35"),
         
        lambdav=dict(
            widget_type="FloatSpinBox",
            label="lambda",
            min=0.1,
            max=50,
            step=0.1,
            value=3,), 
        spinesize=dict(
            widget_type="SpinBox",
            label="spine max pixels",
            min=1,
            max=10000,
            step=1,
            value=100,), 
        
        grow_button=dict(widget_type="PushButton", text="run grow"),
        
        
        preprocess4=dict(widget_type="Label",
                         label="<br><b>4. manual modify spine</b>"),
        
        seg_button=dict(widget_type="PushButton", text="reseg",tooltip=" from 2"),
        startlabel=dict(
            widget_type="SpinBox",
            label="start",
            min=1,
            max=300,
            step=1,
            value=1,),
        add_button=dict(widget_type="PushButton", text="add label",tooltip=" from 1"),
        linkweight=dict(
            widget_type="FloatSpinBox",
            label="img weight",
            min=0.1,
            max=50,
            step=0.1,
            value=10,),
        link_button=dict(widget_type="PushButton", text="link den",tooltip="please check ridge enable"),
        
        savebox=dict(widget_type="LineEdit", label="search box", value="16,16",
                     tooltip=" size need 2^n *2^n"),
        
        save_button=dict(widget_type="PushButton", text="save"),
        layout="vertical",
        persist=True,
        call_button=False,  # if True auto add "run" button
    )
    def plugin(
        viewer: napari.Viewer,
        image: napari.layers.Image,
        binaryimage: napari.layers.Labels,
        spinepoints:napari.layers.Points,
        denpoints:napari.layers.Points,
        # ambiguspoints:napari.layers.Points,
        labelmask:napari.layers.Labels,
        ridge_button,
        denmask:napari.layers.Labels,
        axes,
        preprocess1,
        bmethod,
        threshold_slide,
        adth_button,
        preprocess2,
        # minrange,
        # maxrange,
        # anisotropy_factor,
        spotmethod,
        model_file,
        radiusrange,
        peak_button,
        spine_button,
        den_button,
        # ambigus_button,

        preprocess3,
        growrad,
        searchlabel,
        searchbox,
        # smoothing,
        lambdav,
        spinesize,
        grow_button,
        preprocess4,
        seg_button,
        startlabel,
        add_button,
        linkweight,
        link_button,
        savebox,
        save_button,
    ) -> List[napari.types.LayerDataTuple]:
        x = image.data
        
        axes = axes_check_and_normalize(axes, length=x.ndim)
        spotm=get_value_plugin("spotmethod") 
        #show_info("KKKKKKKKK") 
        if spotm in ["CNN","SVM"]:
            plugin.model_file.show() 
        else:
            plugin.model_file.hide()
        lambdav.hide()
        
        # smoothing.hide()
        
        threshold_slide.hide()
        widgets_valid(plugin.image, valid=False)
        widgets_valid(plugin.binaryimage, valid=False)
        widgets_valid(plugin.labelmask, valid=False)

    #-----------------------#
    #  utils for get value   #
    #-----------------------#
    def get_value_plugin(keystr):
        if keystr=="Binaty_func":
            methodname=plugin.bmethod.value
            if methodname=="custom":
                return "custom"
            func=Bmethod_func[methodname]
            return func
        if keystr=="growmethod":
            return int(plugin.growrad.value)
        if keystr=="searchbox":
            seachstr=plugin.searchbox.value
            lengths=seachstr.split(',')
            lengths=[1-int(l)%2+int(l) for  l in lengths]
            return tuple(lengths)
        if keystr=="savebox":
            seachstr=plugin.savebox.value
            lengths=seachstr.split(',')
            lengths=[int(l)%2+int(l) for  l in lengths]
            return tuple(lengths)
        if keystr=="radiusrange":
            return int(plugin.radiusrange.value)
        if keystr=="lambda":
            return float(plugin.lambdav.value)
        if keystr=="linkweight":
            return float(plugin.linkweight.value)
        
        if keystr=="spinesize":
            return int(plugin.spinesize.value)   
        if keystr=="spotrange":
            return int(plugin.minrange.value) ,int(plugin.maxrange.value),float(plugin.anisotropy_factor.value)
        if keystr=="sel_den_point":
            selectpoint=list(plugin.denpoints.value.selected_data)
            if selectpoint:
                return selectpoint,plugin.denpoints.value.data[selectpoint]
            else:
                return selectpoint,None
        if keystr=="sel_spine_point":
            selectpoint=list(plugin.spinepoints.value.selected_data)
            if selectpoint:
                return selectpoint,plugin.spinepoints.value.data[selectpoint]
            else:
                return selectpoint,None
        if keystr=="spotmethod":
            return plugin.spotmethod.value
        if keystr=="bmethod":
            return plugin.bmethod.value
        if keystr=="modelfile":
            return plugin.model_file.value
        if keystr=="startlabel":
            return int(plugin.startlabel.value)
        # if keystr=="sel_ambigus_point":
        #     selectpoint=list(plugin.ambiguspoints.value.selected_data)
        #     if selectpoint:
        #         return selectpoint,plugin.ambiguspoints.value.data[selectpoint]
        #     else:
        #         return selectpoint,None
          
            
    # widget_for_methodtype = {
    # StarDist2D: plugin.model2d,
    # StarDist3D: plugin.model3d,
    # CUSTOM_MODEL: plugin.model_folder,
    # }    
    
    #-----------------------#
    #   signal trigger  #
    #-----------------------#
    def widgets_valid(*widgets, valid):
        for widget in widgets:
            widget.native.setStyleSheet("" if valid else "background-color: lightcoral")
            
   # update = Updater()
    # -----------------------#
    #   event trigger  #
    # -----------------------#
    # set thresholds to optimized values for chosen model
    @change_handler(plugin.adth_button, init=False)
    def _adth_button():
        show_info("plugin image value type"+str(type(plugin.image.value.data)))
        x = plugin.image.value.data
        #x=minimum_filter(x, size=(3,) * x.ndim)
        #plugin.viewer.value.add_image(x, name="minimum filter")
        func=get_value_plugin("Binaty_func")
        if (isinstance(func,str)):
            th=float(plugin.threshold_slide.value)
            adth=x>th
            funcname="custom"
        elif "local_threshold" in func.__name__:
            adth=func(x)
            funcname=func.__name__
        else:
            adth=x>func(x)
            funcname=func.__name__
        # filopodia
        ndim=x.ndim

        image_layer=plugin.viewer.value.add_labels(adth, name="th_"+funcname)
        plugin.binaryimage.value=image_layer
        #noise_mean=np.mean(x[~adth])
        #x=x-noise_mean
        #x[x<0]=0
        #plugin.image.value.data=x
    @change_handler(plugin.ridge_button, init=False)
    def _ridge_button():
        ridge_f=plugin.ridge_button.value
        if ridge_f:
            img = plugin.image.value.data
            ridge=enhance_ridges_2d(img)
            image_layer=plugin.viewer.value.add_labels(ridge, name="ridge")
            plugin.denmask.value=image_layer

    @change_handler(plugin.peak_button, init=False)
    def _peak_button():
        img = plugin.image.value.data
        adth= plugin.binaryimage.value.data
        radius=get_value_plugin("radiusrange") 
        st=time.time()
        ndim=img.ndim
        spotmethod=get_value_plugin("spotmethod")
        corners=peakfilter(img,radius,0,use_gaussian=False)*adth
        if ndim==2:
            sigma = (2, 2)
        else:
            sigma = (1, 2, 2)
        steps = 2
        ridge_f=plugin.ridge_button.value
        if ridge_f:
            corners[plugin.denmask.value.data>0]=0
        points=np.array(napari_base.get_point(corners))
        if len(points)==0:
            messg=QMessageBox(None)
            messg.setText("find no any point , please ridge setting")
            messg.exec_()
            return
        if spotmethod=="hessian" or spotmethod=="structure":
            eigens, stick = StructureEvi(img,step=steps, sigma=sigma, method=spotmethod)
            adth2=local_threshold(stick)
            points=np.array(napari_base.get_point(corners*adth2))
            points2=np.array(napari_base.get_point(corners*~adth2))
        elif spotmethod=="CNN":
            mfile=get_value_plugin("modelfile")
            lmodel=load_file_model(mfile)
            if lmodel is None: return
            lmodel.to("cpu")
            points,points2=model_predict(points,img,lmodel,boxsize=(16,16))
        elif spotmethod=="SVM":
            #D:\spine\segment\some_analysis_for_spine_img\ipynb\clf.pickle
            #fclf=r"D:\code\myspine\models\svm_model\clf.pickle"
            mfile=get_value_plugin("modelfile")
            with open(mfile, 'rb') as f:
                clff = pickle.load(f)
            points,points2=filter_pd(points,img,clff)
        # corners=napari_base.get_mask_from_point(points,img.shape)
        
        et=time.time()
        show_info("find peak number : "+str(len(points))+"\nrun time:"+str(et-st)+"s\n")
        p1=plugin.viewer.value.add_points(points,size=4,opacity=0.5,name=spotmethod+"spine point",face_color="blue")
        p2=plugin.viewer.value.add_points(points2,size=4,opacity=0.5,name=spotmethod+"dendrite point",face_color="red")
        p1.bind_key('Shift-s', overwrite=False,func=_spine_button)
        p2.bind_key('Shift-s', overwrite=False,func=_spine_button)
        p1.bind_key('Shift-d', overwrite=False,func=_den_button)
        p2.bind_key('Shift-d', overwrite=False,func=_den_button)
        plugin.spinepoints.value=p1
        plugin.denpoints.value=p2
        #p2.bind_key('d', overwrite=False)(self.toggle_bb_visibility)
        
        #plugin.viewer.value.add_points([np.zeros(ndim)],size=4,opacity=0.5,name="ambigus point",face_color="yellow")
        #plugin.viewer.value.add_image(adth*adth2,name="distance")
    
    @change_handler(plugin.grow_button, init=False)
    def _grow_button():
        img = plugin.image.value.data
        adth= plugin.binaryimage.value.data
        points=plugin.spinepoints.value.data
        st=time.time()
        seedmask=napari_base.get_mask_from_point(points,img.shape) # cost too much ,to point cloud
        seedmask,_=ndi.label(seedmask)
        seedmask[seedmask>0]+=1
        growmethod=get_value_plugin("growmethod")
        if growmethod==1:
            growmethod="geo"
        elif growmethod==0:
            growmethod="chan"
        
        searchbox=get_value_plugin("searchbox")
        if len(searchbox)!=img.ndim:
            messg=QMessageBox(None)
            messg.setText("search box must be "+str(img.ndim)+" dimension!")
            messg.exec_()
            return
        num_iter=np.max(searchbox)
        lambdav=get_value_plugin("lambda")
        spinesize=get_value_plugin("spinesize")
        ls=segment.foreach_grow(img,num_iter,seedmask,searchbox,
                     sizeth=spinesize,adth=adth,method=growmethod,
                     smoothing=0,lambda1=1, lambda2=lambdav)
        seedmask[ls<1]=0
        points=napari_base.get_point(seedmask)
        plugin.spinepoints.value.data=points
        et=time.time()
        show_info("run grow method :"+growmethod+"\nrun time:"+str(et-st)+"s\n")
        image_layer=plugin.viewer.value.add_labels(ls, name="segment")
        plugin.labelmask.value=image_layer
    @change_handler(plugin.seg_button, init=False)
    def _seg_button():
        img = plugin.image.value.data
        adth= plugin.binaryimage.value.data
        preseg=plugin.labelmask.value.data
        points=plugin.spinepoints.value.data
        if not points.any(): return
        growmethod=get_value_plugin("growmethod")
        if growmethod==1:
            growmethod="geo"
        elif growmethod==0:
            growmethod="chan"
        show_info("run grow method :"+growmethod)
        searchbox=get_value_plugin("searchbox")
        num_iter=np.max(searchbox)
        lambdav=get_value_plugin("lambda")
        spinesize=get_value_plugin("spinesize")
        ls=segment.foreach_grow_points(img,num_iter,points,searchbox,
                     sizeth=spinesize,adth=adth,method=growmethod,
                     smoothing=0,lambda1=1, lambda2=lambdav,oldseg=preseg)
        plugin.labelmask.value.data=ls
        # seedmask[ls<1]=0
        # points=napari_base.get_point(seedmask)
        #plugin.points.value.data=points
        #plugin.viewer.value.add_labels(ls, name="segment")
    @change_handler(plugin.add_button, init=False)
    def _add_button():
        startlabel=get_value_plugin("startlabel")
        preseg=plugin.labelmask.value.data
        preseg,startlab=segment.resortseg(preseg,startlabel)
        plugin.labelmask.value.selected_label=startlab
    @change_handler(plugin.spine_button, init=False)
    def _spine_button(kvs=None):# den to spine
        indx,points=get_value_plugin("sel_den_point")
        print(indx)
        if indx:
            plugin.denpoints.value.data=np.delete(plugin.denpoints.value.data,indx,axis=0)
            plugin.spinepoints.value.data=np.concatenate((plugin.spinepoints.value.data,points),axis=0)
    @change_handler(plugin.link_button, init=False)
    def _link_button(kvs=None):# den to spine
        if not gwdt_enable: 
            messg=QMessageBox(None)
            messg.setText("Please install gwdt module first!")
            messg.exec_()
            return
          
        
        ridge_f=plugin.ridge_button.value
        if not ridge_f:
            messg=QMessageBox(None)
            messg.setText("please set dendrite mask first and enable ridge!")
            messg.exec_()
            return
        searchbox=get_value_plugin("searchbox")
        img = plugin.image.value.data
        preseg=plugin.labelmask.value.data
        den= plugin.denmask.value.data 
        ls=preseg.copy()
        ls[den>0]=1
        imgweight=get_value_plugin("linkweight")
        # radius=get_value_plugin("radiusrange") 
        corddict,linemask=spines_distance(img,ls,[20,20],
                                          imgweight=imgweight,
                                          )
        image_layer=plugin.viewer.value.add_labels(linemask, name="linemask")
        

    @change_handler(plugin.den_button, init=False)
    def _den_button(kvs=None):# den to spine
        indx,points=get_value_plugin("sel_spine_point")
        print(indx)
        
        if indx:
            plugin.spinepoints.value.data=np.delete(plugin.spinepoints.value.data,indx,axis=0)
            plugin.denpoints.value.data=np.concatenate((plugin.denpoints.value.data,points),axis=0)    
    # @change_handler(plugin.ambigus_button, init=False)
    # def _ambigus_button():# den to spine
    #     indx,points=get_value_plugin("sel_spine_point")
    #     print(indx)
    #     if indx:
    #         plugin.spinepoints.value.data=np.delete(plugin.spinepoints.value.data,indx,axis=0)
    #         plugin.ambiguspoints.value.data=np.concatenate((plugin.ambiguspoints.value.data,points),axis=0)
    #     indx,points=get_value_plugin("sel_den_point")
    #     print(indx)
    #     if indx:
    #         plugin.denpoints.value.data=np.delete(plugin.denpoints.value.data,indx,axis=0)
    #         plugin.ambiguspoints.value.data=np.concatenate((plugin.ambiguspoints.value.data,points),axis=0)    
            
    @change_handler(plugin.save_button, init=False)
    def _save_button():# den to spine
        boxsize=get_value_plugin("savebox")
        
        img = plugin.image.value.data
        img=np.array(img)
        imgname = plugin.image.value.name
        ndim=img.ndim
        
        label = plugin.labelmask.value.data
        label=label.astype(np.uint16)
        label,startlab=segment.resortseg(label,2)
        
        den= plugin.denmask.value.data.astype(np.uint16)
        seg=label.copy()

        seg[den>0]=1
        mask=seg.copy()
        mask[mask>1]=2
        
        savedir = QFileDialog.getExistingDirectory(None, "choose Save  as...", "")
        savedir=os.path.join(savedir,imgname)
        create_dir(savedir)
        # 将会生成的image文件
        imgfile=os.path.join(savedir,imgname+".tif")
        maskfile=os.path.join(savedir,imgname+"-mask.tif")
        spinefile=os.path.join(savedir,imgname+"-spine.tif")
        denfile=os.path.join(savedir,imgname+"-den.tif")
        segfile=os.path.join(savedir,imgname+"-seg.tif")
        imsave(imgfile,img)
        imsave(spinefile,label)
        imsave(denfile,den)
        imsave(segfile,seg)
        imsave(maskfile,mask)
        show_info("save file : "+"\n".join([imgfile,spinefile,denfile,segfile,maskfile]))
        if ndim==3:
            zimgfile=os.path.join(savedir,imgname+"-z.tif")
            zmaskfile=os.path.join(savedir,imgname+"-z-mask.tif")
            zspinefile=os.path.join(savedir,imgname+"-z-spine.tif")
            zdenfile=os.path.join(savedir,imgname+"-z-den.tif")    
            zsegfile=os.path.join(savedir,imgname+"-z-seg.tif")
            imsave(zimgfile,np.max(img,axis=0))
            imsave(zspinefile,np.max(label,axis=0))
            imsave(zmaskfile,np.max(mask,axis=0))
            imsave(zdenfile,np.max(den,axis=0))
            imsave(zsegfile,np.max(seg,axis=0))
            show_info("save mip file : "+"\n".join([zimgfile,zspinefile,zdenfile,zsegfile,zmaskfile]))
        
        # save point for train
        s=[str(b) for b in boxsize]
        filename=os.path.join(savedir,imgname+"-".join(s)+".npy")
        
        if filename:
            image = plugin.image.value.data
            ndim=image.ndim
            dps=plugin.denpoints.value.data[:,ndim-2:]
            #print(dps)
            sps=plugin.spinepoints.value.data[:,ndim-2:]
            # ambigus=plugin.ambiguspoints.value.data[:,ndim-2:]
            
            if ndim==3:
                image=np.max(image,axis=0)
            dpsimg=segment.cropfromseed(image,dps,boxsize=boxsize)
            spsimg=segment.cropfromseed(image,sps,boxsize=boxsize)
            # ambimg=segment.cropfromseed(image,ambigus,boxsize=(16,16))
            images=dpsimg+spsimg
            #t=np.array(images)
            #print(np.unique(t))
            #print(t.shape,dpsimg[0].shape)
            #print(t.shape[2])
            print("point number : ",len(images))
            lables=[0,]*len(dpsimg)+[1,]*len(spsimg)
            np.savez(filename,img=images,lab=lables,size=boxsize,allow_pickle=True)
            
            show_info("save point file : "+filename)
        
    @change_handler(plugin.bmethod, init=False)
    def _bmethod(model_name_star: str): 
        bmethod=get_value_plugin("bmethod") 
        #show_info("KKKKKKKKK") 
        if isinstance(bmethod,str)and bmethod == "custom":
            plugin.threshold_slide.show()
        else:
            plugin.threshold_slide.hide()  
    
    # @change_handler(plugin.image, init=False)
    # def _image(model_name_star: str): 
    #     # bmethod=get_value_plugin("bmethod") 
    #     # x = plugin.image.value.data
    #     clim=plugin.image.get_clim()
    #     limits_range=plugin.image.value.contrast_limits_range()
    #     limits=plugin.image.value.contrast_limits()
    #     #show_info("KKKKKKKKK") 
    #     plugin.threshold_slide.max=limits_range[1]
    #     plugin.threshold_slide.min=limits_range[0]
    #     plugin.threshold_slide.value=limits
          
        
    @change_handler(plugin.spotmethod, init=False)
    def _spot_method(model_name_star: str): 
        spotm=get_value_plugin("spotmethod") 
        #show_info("KKKKKKKKK") 
        if spotm in ["CNN","SVM"]:
            plugin.model_file.show() 
            if spotm=="CNN":
                plugin.model_file.mode="r"
            else:
                plugin.model_file.mode="r"
        else:
            plugin.model_file.hide()
    
    return plugin

#widget 2 : predict widget
def plugin_wrapper2():

    #-----------------------#
    #   call back  #
    #-----------------------#
    DEBUG = False

    def change_handler(*widgets, init=True, debug=DEBUG):
        def decorator_change_handler(handler):
            @functools.wraps(handler)
            def wrapper(*args):
                source = Signal.sender()
                emitter = Signal.current_emitter()
                if debug:

                    print(
                        f'{str(emitter.name).upper()}: {source.name} = {args!r}'
                    )
                return handler(*args)

            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(widget.value)
            return wrapper

        return decorator_change_handler

    #-----------------------#
    #   GUI  #
    #-----------------------#
    @magicgui(
        image=dict(label="gray Images"),
        axes=dict(
            widget_type="ComboBox",
            label="Image Axes",
            choices=["xy","zxy"],
            value="xy",
        ),       
        spinepr=dict(label="spine pr"),
        denpr=dict(label="dendrite pr"),
        bgpr=dict(label="background pr"),
        mask=dict(label="mask Image"),
        spine=dict(label="spine layer"),
        # segmask=dict(label="segment layer"),

        preprocess1=dict(widget_type="Label",
                         label="<br><b>1. predict</b>"),
        model_name=dict(
            widget_type="ComboBox",
            label="network",
            choices=net_method,
            value=net_method[0],
            tooltip="current only supprt unet2d"
        ),
        model_file=dict(widget_type="FileEdit",visible=True,label="model file"),
        run_predict= dict(widget_type="PushButton", text="run predict"),
        progressbar=dict(widget_type="ProgressBar",min=0,max=10,step=1,label="/"),
        
        preprocess2=dict(widget_type="Label",
                         label="<br><b>2. mask threshold</b>"),
        bmethod=dict(
            widget_type="ComboBox",
            label="Binary method",
            choices=Bmethod_name+["custom"],
            value=Bmethod_name[-3],
        ),
        threshold1=dict(widget_type="LineEdit",label="spine threshold",value="0.1"),  
        threshold2=dict(widget_type="LineEdit",label="den threshold",value="0.1"),  
        adth_button=dict(widget_type="PushButton", text="binarylize"),
      
        
        preprocess3=dict(widget_type="Label",
                         label="<br><b>3.instance segment</b>"),
        instance_method=dict(
            widget_type="ComboBox",
            label="Instance method",
            choices=Instance_funcname,
            value=Instance_funcname[0]),
        radiusrange=dict(widget_type="LineEdit", label="radius", value="5"),
       # bgth=dict(widget_type="LineEdit",label="spine threshold",value="0.1"),
        minspinesize=dict(
            widget_type="SpinBox",
            label="spine min pixels",
            min=0,
            max=10000,
            step=1,
            value=0,),  
        maxspinesize=dict(
            widget_type="SpinBox",
            label="spine max pixels",
            min=1,
            max=10000,
            step=1,
            value=100,),
        run_spine= dict(widget_type="PushButton", text="run  instance seg"),
        
        
        preprocess4=dict(widget_type="Label",
                         label="<br><b>4. manual modify spine</b>"),
        label_re=dict(label="Label Layer"),
        
        startlabel=dict(
            widget_type="SpinBox",
            label="start",
            min=1,
            max=300,
            step=1,
            value=1,),
        add_button=dict(widget_type="PushButton", text="add label",tooltip=" from 1"),
        
        save_button=dict(widget_type="PushButton", text="save result"),
        layout="vertical",
        persist=True,
        call_button=False,  # if True auto add "run" button
    )
    def plugin(
        viewer: napari.Viewer,
        image: napari.layers.Image,
        axes,
        
        spinepr: napari.layers.Image,
        denpr: napari.layers.Image,
        bgpr: napari.layers.Image,
        mask:napari.layers.Labels,
        spine:napari.layers.Labels,
        # segmask:napari.layers.Labels,

        preprocess1,
        model_name,
        model_file,
        run_predict,
        progressbar,
        
        preprocess2,
        bmethod,
        threshold1,
        threshold2,
        adth_button,
        
        preprocess3,
        instance_method,
        radiusrange,
        # bgth,
        minspinesize,
        maxspinesize,
        run_spine,
        preprocess4,
        label_re:napari.layers.Labels,
        startlabel,
        add_button,
        save_button
        # axes,
        # preprocess1,
        # area_button,
        # mmethod,
        # save_button
      
    ) -> List[napari.types.LayerDataTuple]:
        x = image.data
        axes = axes_check_and_normalize(axes, length=x.ndim)
        func=get_value_plugin("Binaty_func")
        #show_info("KKKKKKKKK") 
        if (isinstance(func,str)):
            plugin.threshold1.show() 
            plugin.threshold2.show() 
        else:
            plugin.threshold1.hide() 
            plugin.threshold2.hide() 
        instance_m=get_value_plugin("instance method") # str
        if (instance_m=="peak"):
            # plugin.bgth.hide()
            plugin.radiusrange.label="radius"
        else:
            plugin.radiusrange.label="bg threshold"
        #lambdav.hide() 
        # plugin.mips=[
        #             plugin.MIPimage,
        #             plugin.MIPspinepr,
        #             plugin.MIPdenpr,
        #             plugin.MIPmask,
        #             plugin.MIPspine,
        #         ]
    plugin.oris=[
        plugin.image,
        plugin.spinepr,
        plugin.denpr,
        plugin.mask,
        plugin.spine,
        # plugin.segmask,
    ]
    # plugin._mip_mask=None
    # plugin._all_mask=None
    plugin._model=None

    def get_value_plugin(keystr):
        if keystr=="Binaty_func":
            methodname=plugin.bmethod.value
            if methodname=="custom":
                return "custom"
            func=Bmethod_func[methodname]
            return func
        if keystr=="radiusrange":
            return float(plugin.radiusrange.value)
        if keystr=="Network":
            return plugin.model_name.value
        if keystr=="modelwieght":
            return plugin.model_file.value
        if keystr=="axes":
            axes=plugin.axes.value 
            axes=str.lower(axes)
            return axes
        if keystr=="run_mask":
            return int(plugin.run_mask.value)
        if keystr=="spinesize":
            return int(plugin.minspinesize.value), int(plugin.maxspinesize.value)
        if keystr=="track_method"    :
            return int(plugin.track_method.value)
        if keystr=="startlabel":
            return int(plugin.startlabel.value)
        if keystr=="instance method":
            return plugin.instance_method.value
    #-----------------------#
    #  utils for get value   #
    #-----------------------#
    @change_handler(plugin.adth_button, init=False)
    def _adth_button():
        show_info("plugin image value type"+str(type(plugin.image.value.data)))
        func=get_value_plugin("Binaty_func")
        spinepr=plugin.spinepr.value.data
        denpr=plugin.denpr.value.data
        #x=minimum_filter(x, size=(3,) * x.ndim)
        if (isinstance(func,str)):
            spineth=float(plugin.threshold1.value)
            denth=float(plugin.threshold2.value)
            spine=spinepr>spineth
            den=denpr>denth
            funcname="custom"
        elif "local_threshold" in func.__name__:
            den=func(denpr)
            spine=func(spinepr)
            
            funcname=func.__name__
        else:
            den=denpr>func(denpr)
            spine=spinepr>func(spinepr)
            
            funcname=func.__name__
        mask=den.copy()*1
        mask[spine>0]=2  
        image_layer=plugin.viewer.value.add_labels(mask, name="th_"+funcname)
        plugin.mask.value=image_layer
        
    @change_handler(plugin.run_predict, init=False)
    def _run_predict():
        
        network=get_value_plugin("Network")
        modelpath=get_value_plugin("modelwieght")
        axes=get_value_plugin("axes")
        imgs=plugin.image.value.data
        ndim=imgs.ndim
        if axes[0]=="t":
            timemodel=True
        else:
            timemodel=False
        if timemodel and imgs.ndim<=2:
            messg=QMessageBox(None)
            messg.setText("Plase check image: only "+str(imgs.ndim)+" dimension!")
            messg.exec_()
            return 
        if network=="unet2d":
            model=load_file_model(modelpath)
            if model is None: return
            all_mask,all_spine_pr,all_den_pr,bgpr=unetseg.predict_single_img(model,imgs)
            # model=UNet2d()
            # h,w=imgs.shape[-2:]
            # model.build(input_shape =(4,h,w,1))
            # model.load_weights(modelpath)
            #all_mask,all_spine_pr,all_den_pr=model.predict_single_img(imgs)
            im1=plugin.viewer.value.add_image(all_spine_pr, name="spine_pr",colormap="magenta",
                opacity=0.5, blending="additive")
            im2=plugin.viewer.value.add_image(all_den_pr, name="den_pr",colormap="green",
                opacity=0.5, blending="additive")
            all_mask=modify_mask(all_mask,sizeth=4)
            im3=plugin.viewer.value.add_labels(all_mask, name="seg_mask",opacity=0.5)
            im4=plugin.viewer.value.add_image(bgpr, name="bg_pr",colormap="gray_r",visible=False,
                opacity=0.5, blending="additive")
            plugin.spinepr.value=im1
            plugin.denpr.value=im2
            plugin.mask.value=im3
            plugin.bgpr.value=im4
            # plugin._all_mask=all_mask
            plugin._model=model
    @change_handler(plugin.bmethod, init=False)
    def _bmethod(model_name_star: str): 
        bmethod=get_value_plugin("Binaty_func")
        #show_info("KKKKKKKKK") 
        if isinstance(bmethod,str)and bmethod == "custom":
                    
            plugin.threshold1.show() 
            plugin.threshold2.show() 
        else:
            plugin.threshold1.hide() 
            plugin.threshold2.hide() 
    @change_handler(plugin.instance_method, init=False)
    def _instance_method(model_name_star: str): 
        instance_m=get_value_plugin("instance method") # str      
        if (instance_m=="peak"):
            # plugin.bgth.hide()
            plugin.radiusrange.label="radius"
        else:
            plugin.radiusrange.label="bg threshold"
            
    # @change_handler(plugin.axes, init=False)
    # def _axes(model_name_star: str): 
    #     change_visible()
    
    # def change_visible():
    #     axes=get_value_plugin("axes") 
    #     #show_info(axes+"kkkkkkkkkkkk")
    #     if "t" not in axes:
    #         plugin.track_method.value=0
    #         plugin.track_method.enabled=False
    #         plugin.run_track.enabled=False
    #     else:
    #         plugin.run_track.enabled=True
    #         plugin.track_method.enabled=True
    #     track_method=get_value_plugin("track_method") # MIP Btracker
        
    #     if "t" in axes and track_method==1: # MIP
    #         for wid in plugin.mips:
    #             wid.show()
    #         for wid in plugin.oris:
    #             wid.hide()

    #     else:# single
    #         for wid in plugin.mips:
    #             wid.hide()
    #         for wid in plugin.oris:
    #             wid.show()

           

    @change_handler(plugin.run_spine, init=False)
    def _run_spine():  
        axes=get_value_plugin("axes")
        imgs=plugin.image.value.data
        ndim=imgs.ndim
        if ndim==3:
            searchbox=[5,5,5]
        else:
            searchbox=[5,5]
        radius=get_value_plugin("radiusrange") # or bgth
        minspinesize,maxspinesize=get_value_plugin("spinesize")
        #print(minspinesize,maxspinesize)
        spinepl=plugin.spinepr
        denpl=plugin.denpr
        maskl=plugin.mask
        spine_l=plugin.spine
            
        masks=maskl.value.data
        spineprs=spinepl.value.data
        
        instance_m=get_value_plugin("instance method") 
        instance_func=Instance_funcdict[instance_m]
        show_info("run func: "+instance_func.__name__)
        if instance_m=="peak":
            spine_label=instance_func(spineprs,masks==2,searchbox,radius,spinesize_range=[minspinesize,maxspinesize])
        elif instance_m=="background":
            bgpr=plugin.bgpr.value.data
            spine_label=instance_func(spineprs,masks==2,bgpr,radius,spinesize_range=[minspinesize,maxspinesize])
            
        # pr_corner=peakfilter(spineprs,radius,0,use_gaussian=False)*(masks)#*adth
        # spine_label=segment.label_instance_water(spineprs,pr_corner,masks, 
        #                                 maxspinesize,searchbox=searchbox)
        # spine_label=remove_small_objects(spine_label,min_size=minspinesize)
        
        im=plugin.viewer.value.add_labels(spine_label, name="spine instance")  
        spine_l.value=im
    @change_handler(plugin.add_button, init=False)
    def _add_button():
        startlabel=get_value_plugin("startlabel")
        preseg=plugin.label_re.value.data
        preseg,startlab=segment.resortseg(preseg,startlabel)
        plugin.label_re.value.selected_label=startlab
    @change_handler(plugin.save_button, init=False)
    def _save_button():
       
        img = plugin.image.value.data
        img=np.array(img)
        imgname = plugin.image.value.name
        ndim=img.ndim
        
        spine = plugin.spine.value.data
        spine=spine.astype(np.uint16)
        spine,startlab=segment.resortseg(spine,2)
        
        mask = plugin.mask.value.data
        mask[mask==2]=0
        mask[spine>0]=2
        den=mask==1
        den[spine>0]=0
        seg=spine.copy()
        seg[den>0]=1
        spinepr=plugin.spinepr.value.data
        denpr=plugin.denpr.value.data

        
        savedir = QFileDialog.getExistingDirectory(None, "choose Save  as...", "")
        if not savedir: 
            show_info("no save")
            return
        savedir=os.path.join(savedir,imgname)
        create_dir(savedir)
        # 将会生成的image文件
        imgfile=os.path.join(savedir,imgname+".tif")
        maskfile=os.path.join(savedir,imgname+"-mask.tif")
        spineprfile=os.path.join(savedir,imgname+"-spinepr.tif")
        spinefile=os.path.join(savedir,imgname+"-spine.tif")
        denprfile=os.path.join(savedir,imgname+"-denpr.tif")
        denfile=os.path.join(savedir,imgname+"-den.tif")
        segfile=os.path.join(savedir,imgname+"-seg.tif")
        imsave(imgfile,img)
        imsave(spineprfile,spinepr)
        imsave(spinefile,spine)
        imsave(denprfile,denpr)
        imsave(denfile,den)
        imsave(segfile,seg)
        imsave(maskfile,mask)
        show_info("save file : "+"\n".join([imgfile,spinefile,denfile,segfile,maskfile]))
        if ndim==3:
            zimgfile=os.path.join(savedir,imgname+"-z.tif")
            zmaskfile=os.path.join(savedir,imgname+"-z-mask.tif")
            zspinefile=os.path.join(savedir,imgname+"-z-spine.tif")
            zdenfile=os.path.join(savedir,imgname+"-z-den.tif")    
            zsegfile=os.path.join(savedir,imgname+"-z-seg.tif")
            imsave(zimgfile,np.max(img,axis=0))
            imsave(zspinefile,np.max(spine,axis=0))
            imsave(zmaskfile,np.max(mask,axis=0))
            imsave(zdenfile,np.max(den,axis=0))
            imsave(zsegfile,np.max(seg,axis=0))
            show_info("save mip file : "+"\n".join([zimgfile,zspinefile,zdenfile,zsegfile,zmaskfile]))
        
    return plugin


# widget 2 : series label widget
def plugin_wrapper3():
    #-----------------------#
    #   call back  #
    #-----------------------#
    DEBUG = False
    def change_handler(*widgets, init=True, debug=DEBUG):
        def decorator_change_handler(handler):
            @functools.wraps(handler)
            def wrapper(*args):
                source = Signal.sender()
                emitter = Signal.current_emitter()
                if debug:

                    print(
                        f'{str(emitter.name).upper()}: {source.name} = {args!r}'
                    )
                return handler(*args)

            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(widget.value)
            return wrapper

        return decorator_change_handler
    #-----------------------#
    #   GUI  #
    #-----------------------#
    @magicgui(
        image=dict(label="Grayscale Image",tooltip="time-series"),
        binaryimage=dict(label="Binary Image",tooltip="neuron boundary"),
        spinepoints=dict(label="spine point",
                         tooltip="dendritic seed points for region growing\n \
                             that are outside the dendrite mask and inside the neuron mask\n\
                            show as a size of 1/3 radius size."),
        denmask=dict(label="dendrite Image",tooltip="dendrite boundary"),
        axes=dict(widget_type="LineEdit", label="Image Axes", value="tyx",tooltip="no use, current only support txy"),
        
        preprocess1=dict(widget_type="Label",
                         label="<br><b>1. threshold</b>"),
        bmethod=dict(
            widget_type="ComboBox",
            label="Binary method",
            choices=Bmethod_name,
            value=Bmethod_name[-1],
        ),
        adth_button=dict(widget_type="PushButton", text="binarylize"),
        preprocess2=dict(widget_type="Label",
                         label="<br><b>2. find seed point</b>"))
    def plugin(
        viewer: napari.Viewer,
        image: napari.layers.Image,
        axes,
        track_method,
        binary: napari.layers.Image,
        spinepr: napari.layers.Image,
        denpr: napari.layers.Image,
        mask:napari.layers.Labels,
        spine:napari.layers.Labels,
        MIPimage: napari.layers.Image,
        MIPspinepr: napari.layers.Image,
        MIPdenpr: napari.layers.Image,
        MIPmask:napari.layers.Labels,
        MIPspine:napari.layers.Labels,
        
        preprocess1,
        bmethod,
        adth_button,
        preprocess2,
        model_name,
        model_file,
        run_predict,
        progressbar,
        
        preprocess3,
        spineth,
        denth,
        spinesize,
        run_mask,
        
        preprocess4,
        run_spine,
        preprocess5,
        btrack_file,
        run_track,
        save_button
        # axes,
        # preprocess1,
        # area_button,
        # mmethod,
        # save_button
      
    ) :
        pass
        
#widget 4 : time series tracking widget
def plugin_wrapper4():

    #-----------------------#
    #   call back  #
    #-----------------------#
    DEBUG = False

    def change_handler(*widgets, init=True, debug=DEBUG):
        def decorator_change_handler(handler):
            @functools.wraps(handler)
            def wrapper(*args):
                source = Signal.sender()
                emitter = Signal.current_emitter()
                if debug:

                    print(
                        f'{str(emitter.name).upper()}: {source.name} = {args!r}'
                    )
                return handler(*args)

            for widget in widgets:
                widget.changed.connect(wrapper)
                if init:
                    widget.changed(widget.value)
            return wrapper

        return decorator_change_handler

    #-----------------------#
    #   GUI  #
    #-----------------------#
    @magicgui(
        image=dict(label="gray Images"),
        axes=dict(
            widget_type="ComboBox",
            label="Image Axes",
            choices=Axes_type,
            value=Axes_type[1],
        ),
        track_method=dict(
            widget_type="RadioButtons",
            #visible=False,
            label="stratagy",
            choices=strategy_mode,
            value=DEFAULTS["strategy_mode"],
            orientation="horizontal",
        ),
        
        binary=dict(label="binary Image"),
        spinepr=dict(label="spine pr"),
        denpr=dict(label="dendrite pr"),
        mask=dict(label="segment layer"),
        spine=dict(label="instance layer"),
        
        MIPimage=dict(label="tMIP gray Images",visible=False),
        MIPspinepr=dict(label="tMIP spine pr",visible=False),
        MIPdenpr=dict(label="tMIP den pr",visible=False),
        MIPmask=dict(label="tMIP segment layer",visible=False),
        MIPspine=dict(label="tMIP instance layer",visible=False),
        
        preprocess1=dict(widget_type="Label",
                         label="<br><b>1. threshold</b>"),
        bmethod=dict(
            widget_type="ComboBox",
            label="Binary method",
            choices=Bmethod_name,
            value=Bmethod_name[-1],
        ),
        adth_button=dict(widget_type="PushButton", text="binarylize"),
        
        
        preprocess2=dict(widget_type="Label",
                         label="<br><b>2. predict</b>"),
        model_name=dict(
            widget_type="ComboBox",
            label="network",
            choices=net_method,
            value=net_method[0],
            tooltip="current only supprt unet2d"
        ),
        model_file=dict(widget_type="FileEdit",visible=True,label="model file"),
        run_predict= dict(widget_type="PushButton", text="run predict"),
        progressbar=dict(widget_type="ProgressBar",min=0,max=10,step=1,label="/"),
        
        
        preprocess3=dict(widget_type="Label",
                         label="<br><b>3.modify semantic segment</b>"),
        
        spineth=dict(widget_type="FloatSpinBox",value=0,visible=True),
        denth=dict(widget_type="FloatSpinBox",value=0,visible=True),
        spinesize=dict(
            widget_type="SpinBox",
            label="spine max pixels",
            min=1,
            max=10000,
            step=1,
            value=100), 
        run_mask=dict(widget_type="RadioButtons",
            #visible=False,
            label="run",
            choices=modify_mode,
            value=1,
            orientation="horizontal"),

        
        preprocess4=dict(widget_type="Label",
                         label="<br><b>4.Instance segment</b>"),
        
        run_spine= dict(widget_type="PushButton", text="run  instance seg"),
        
        preprocess5=dict(widget_type="Label",
                         label="<br><b>5.track instance</b>"),
        btrack_file=dict(widget_type="FileEdit",visible=True,label="btrack  model"),
        run_track= dict(widget_type="PushButton", text="run  tracking"),
        # preprocess1=dict(widget_type="Label",label="<br><b>Measurement</b>"),
        # bmethod=dict(
        #     widget_type="ComboBox",
        #     label="Binary method",
        #     choices=Bmethod_name[:-1],
        #     value=Bmethod_name[0],
        # ),
        # area_button=dict(widget_type="PushButton", text="get area"),
        # mmethod=dict(
        #     widget_type="ComboBox",
        #     label="Measurement",
        #     choices=measurement_choices,
        #     value=measurement_choices[0],
        # ),
        save_button=dict(widget_type="PushButton", text="save result"),
        layout="vertical",
        persist=True,
        call_button=False,  # if True auto add "run" button
    )
    def plugin(
        viewer: napari.Viewer,
        image: napari.layers.Image,
        axes,
        track_method,
        binary: napari.layers.Image,
        spinepr: napari.layers.Image,
        denpr: napari.layers.Image,
        mask:napari.layers.Labels,
        spine:napari.layers.Labels,
        MIPimage: napari.layers.Image,
        MIPspinepr: napari.layers.Image,
        MIPdenpr: napari.layers.Image,
        MIPmask:napari.layers.Labels,
        MIPspine:napari.layers.Labels,
        
        preprocess1,
        bmethod,
        adth_button,
        preprocess2,
        model_name,
        model_file,
        run_predict,
        progressbar,
        
        preprocess3,
        spineth,
        denth,
        spinesize,
        run_mask,
        
        preprocess4,
        run_spine,
        preprocess5,
        btrack_file,
        run_track,
        save_button
        # axes,
        # preprocess1,
        # area_button,
        # mmethod,
        # save_button
      
    ) -> List[napari.types.LayerDataTuple]:
        x = image.data
        axes = axes_check_and_normalize(axes, length=x.ndim)
        
    plugin.mips=[
                plugin.MIPimage,
                plugin.MIPspinepr,
                plugin.MIPdenpr,
                plugin.MIPmask,
                plugin.MIPspine,
            ]
    plugin.oris=[
        plugin.binary,
        plugin.spinepr,
        plugin.denpr,
        plugin.mask,
        plugin.spine,
    ]
    plugin._mip_mask=None
    plugin._all_mask=None
    plugin._model=None

    def get_value_plugin(keystr):
        if keystr=="Binaty_func":
            methodname=plugin.bmethod.value
            func=Bmethod_func[methodname]
            return func
        if keystr=="Network":
            return plugin.model_name.value
        if keystr=="modelwieght":
            return plugin.model_file.value
        if keystr=="axes":
            axes=plugin.axes.value 
            axes=str.lower(axes)
            return axes
        if keystr=="run_mask":
            return int(plugin.run_mask.value)
        if keystr=="spinesize":
            return int(plugin.spinesize.value) 
        if keystr=="track_method"    :
            return int(plugin.track_method.value)
    #-----------------------#
    #  utils for get value   #
    #-----------------------#
    @change_handler(plugin.adth_button, init=False)
    def _adth_button():
        show_info("plugin image value type"+str(type(plugin.image.value.data)))
        x = plugin.image.value.data
        #x=minimum_filter(x, size=(3,) * x.ndim)
        func=get_value_plugin("Binaty_func")
        #plugin.viewer.value.add_image(x, name="minimum filter")
        if func.__name__=="local_threshold":
            adth=func(x)
        else:
            adth=x>func(x)
        image_layer=plugin.viewer.value.add_image(adth, name="th_"+func.__name__)
        plugin.binary.value=image_layer
        
    @change_handler(plugin.run_predict, init=False)
    def _run_predict():
        
        network=get_value_plugin("Network")
        modelpath=get_value_plugin("modelwieght")
        axes=get_value_plugin("axes")
        imgs=plugin.image.value.data
        if axes[0]=="t":
            timemodel=True
        else:
            timemodel=False
        if timemodel and imgs.ndim<=2:
            messg=QMessageBox(None)
            messg.setText("Plase check image: only "+str(imgs.ndim)+" dimension!")
            messg.exec_()
            return
        if network=="unet2d":
            model=unet.UNet2D()
            model.build(input_shape =(4,512,512,1))
            model.load_weights(modelpath)
            if timemodel:
                framenum=imgs.shape[0]
                p=plugin.progressbar
                p.min=0
                p.max=framenum
                p.value=0
                all_mask,all_spine_pr,all_den_pr= model.predict_time_imgs(imgs,p)
                maximg=np.max(imgs,axis=0)
                mip_mask,mip_spine_pr,mip_den_pr=model.predict_single_img(maximg) 
                im1=plugin.viewer.value.add_image(maximg, name="tMIP-"+plugin.image.value.name,
                    opacity=0.5, blending="additive")
                im2=plugin.viewer.value.add_image(mip_spine_pr, name="tMIP-spine_pr",colormap="magenta",
                    opacity=0.5, blending="additive")
                im3=plugin.viewer.value.add_image(mip_den_pr, name="tMIP-den_pr",colormap="green",
                    opacity=0.5, blending="additive")
                mip_mask=modify_mask(mip_mask,sizeth=4)
                im4=plugin.viewer.value.add_labels(mip_mask, name="tMIP-seg_mask",opacity=0.5)
                plugin.MIPimage.value=im1
                plugin.MIPspinepr.value=im2
                plugin.MIPdenpr.value=im3
                plugin.MIPmask.value=im4
                #plugin.MIPimage.value=maximg
                plugin._mip_mask=mip_mask
            else:
                all_mask,all_spine_pr,all_den_pr=model.predict_single_img(imgs)
            im1=plugin.viewer.value.add_image(all_spine_pr, name="spine_pr",colormap="magenta",
                opacity=0.5, blending="additive")
            im2=plugin.viewer.value.add_image(all_den_pr, name="den_pr",colormap="green",
                opacity=0.5, blending="additive")
            all_mask=modify_mask(all_mask,sizeth=4)
            im3=plugin.viewer.value.add_labels(all_mask, name="seg_mask",opacity=0.5)
            
            plugin.spinepr.value=im1
            plugin.denpr.value=im2
            plugin.mask.value=im3
            plugin._all_mask=all_mask
            plugin._model=model
            
            

    @change_handler(plugin.track_method, init=False)
    def _track_method(model_name_star: str): 
        change_visible()
    @change_handler(plugin.axes, init=False)
    def _axes(model_name_star: str): 
        change_visible()
    
    def change_visible():
        axes=get_value_plugin("axes") 
        #show_info(axes+"kkkkkkkkkkkk")
        if "t" not in axes:
            plugin.track_method.value=0
            plugin.track_method.enabled=False
            plugin.run_track.enabled=False
        else:
            plugin.run_track.enabled=True
            plugin.track_method.enabled=True
        track_method=get_value_plugin("track_method") # MIP Btracker
        
        if "t" in axes and track_method==1: # MIP
            for wid in plugin.mips:
                wid.show()
            for wid in plugin.oris:
                wid.hide()

        else:# single
            for wid in plugin.mips:
                wid.hide()
            for wid in plugin.oris:
                wid.show()
            
        
        
    
    @change_handler(plugin.run_mask, init=False)
    def _run_mask():  
        maskmode=get_value_plugin("run_mask")       
        track_method=get_value_plugin("track_method") #track_method==1:  MIP
        
        if track_method:
            spinepl=plugin.MIPspinepr
            denpl=plugin.MIPdenpr
            maskl=plugin.MIPmask
            orimask=plugin._mip_mask
        else:
            spinepl=plugin.spinepr
            denpl=plugin.denpr
            maskl=plugin.mask
            orimask=plugin._all_mask
            
        if maskmode==1: #rese to
            maskl.data=orimask
        else:
            spineth=plugin.spineth.value
            denth=plugin.denth.value
            m1=spinepl.data>spineth
            m2=denpl.data>denth
            m2[m1>0]=0
            mask=m1*2+m2*1
            maskl.data=mask
           

    @change_handler(plugin.run_spine, init=False)
    def _run_spine():  
        axes=get_value_plugin("axes")
        if axes[0]=="t":
            timemodel=True
        else:
            timemodel=False
        if "z" in axes:
            searchbox=[5,5,5]
        else:
            searchbox=[5,5]
        track_method=get_value_plugin("track_method") #track_method==1:  MIP
        
        if track_method: # 1 : MIP
            spinepl=plugin.MIPspinepr
            denpl=plugin.MIPdenpr
            maskl=plugin.MIPmask
            spine_l=plugin.MIPspine
        else:
            spinepl=plugin.spinepr
            denpl=plugin.denpr
            maskl=plugin.mask
            spine_l=plugin.spine
            
        masks=maskl.value.data==2
        spineprs=spinepl.value.data
        maxspinesize=get_value_plugin("spinesize")
        if timemodel==False or track_method: 
            pr_corner=peakfilter(spineprs,3,0,use_gaussian=True)*(masks)#*adth
            spine_label=segment.label_instance_water(spineprs,pr_corner,masks, 
                                            maxspinesize,searchbox=searchbox)
            spine_label=remove_small_objects(spine_label,min_size=4)
            mipst="tMIP-" if track_method else ""
            im=plugin.viewer.value.add_labels(spine_label, name=mipst+"spine instance")  
            spine_l.value=im
        else: # time series
            spine_labels=[]
            for mask,spinepr in zip(masks,spineprs):
                den=mask==1
                i+=1
                pr_corner=peakfilter(spinepr,3,0,use_gaussian=True)*(mask)
                proj_spine_label=segment.label_instance_water(spinepr,pr_corner,mask, 
                                                maxspinesize,searchbox=searchbox)
                spine_label=remove_small_objects(proj_spine_label,min_size=6)
                spine_labels.append(spine_label)
            spine_labels=np.array(spine_labels)
            im=plugin.viewer.value.add_labels(spine_labels, name="spine instance")  
            spine_l.value=im
    @change_handler(plugin.run_track, init=False)
    def _run_track():  
        axes=get_value_plugin("axes")
        track_method=get_value_plugin("track_method") #track_method==1:  MIP
        
        if track_method: # 1 : MIP
            spinepl=plugin.MIPspinepr
            denpl=plugin.MIPdenpr
            maskl=plugin.MIPmask
            spine_l=plugin.MIPspine
        else:
            spinepl=plugin.spinepr
            denpl=plugin.denpr
            maskl=plugin.mask
            spine_l=plugin.spine
            
        masks=maskl.value.data==2
        spineprs=spinepl.value.data
        spine_label=spine_l.value.data
        maxspinesize=get_value_plugin("spinesize")
        if track_method: 
            masks=plugin.mask.value.data==2 
            spine_labels=[spine_label*mask for mask in masks]
            spine_labels=np.array(spine_labels)
            im=plugin.viewer.value.add_labels(spine_labels, name="spine instance")  
            plugin.spine.value=im
        else:
            #btrack
            pass
    @change_handler(plugin.save_button, init=False)
    def _save_button():
        boxsize=get_value_plugin("savebox")
        
        axes=get_value_plugin("axes")
        track_method=get_value_plugin("track_method")
        if "z" in axes:
            pass
        imgname = plugin.image.value.name
        savedir = QFileDialog.getExistingDirectory(None, "choose Save  as...", "")
        if "t" in axes and track_method:
            img=plugin.MIPimage.value.data
            spinepr=plugin.MIPspinepr.value.data
            denpr=plugin.MIPdenpr.value.data
            mask=plugin.MIPmask.value.data
            spine=plugin.MIPspine.value.data
            spine=spine.astype(np.uint16)
            spine,startlab=segment.resortseg(spine,2)
            prefix="tMIP-"
            imsave(os.path.join(savedir,prefix+imgname+"-spinepr.tif"),spinepr)
            imsave(os.path.join(savedir,prefix+imgname+"-denpr.tif"),denpr)
            imsave(os.path.join(savedir,prefix+imgname+"-mask.tif"),mask)
            imsave(os.path.join(savedir,prefix+imgname+"-spine.tif"),spine)
            imsave(os.path.join(savedir,prefix+imgname+".tif"),img)
            
        spinepr=plugin.spinepr.value.data
        denpr=plugin.denpr.value.data
        mask=plugin.mask.value.data
        spine=plugin.spine.value.data
        img = plugin.image.value.data
        spine=spine.astype(np.uint16)
        spine,startlab=segment.resortseg(spine,2)
        prefix=""
        imsave(os.path.join(savedir,prefix+imgname+"-spinepr.tif"),spinepr)
        imsave(os.path.join(savedir,prefix+imgname+"-denpr.tif"),denpr)
        imsave(os.path.join(savedir,prefix+imgname+"-mask.tif"),mask)
        imsave(os.path.join(savedir,prefix+imgname+"-spine.tif"),spine)    
        imsave(os.path.join(savedir,prefix+imgname+".tif"),img)
        
    return plugin    

    
    # def get_value_plugin(keystr):
    #     if keystr=="measure":
    #         return plugin.mmethod.value
    #     if keystr=="Binaty_func":
    #         methodname=plugin.bmethod.value
    #         func=Bmethod_func[methodname]
    #         return func
    # @change_handler(plugin.area_button, init=False)
    # def _area_button():
    #     imgs=plugin.image.value.data
    #     labless=plugin.labelmask.value.data
    #     #mmethod=get_value_plugin("measure")
    #     #show_info("measurement : "+mmethod)
    #     #filename, _ = QFileDialog.getSaveFileName(None, "Save  as...", ".", "*.csv")
    #     #dirpath,shortname,suffix=split_filename(filename)
    #     # change imgs
    #     func=get_value_plugin("Binaty_func")
    #     labless=th_img(imgs,labless,func)
    #     #print(labless.shape,labless.dtype)
    #     plugin.viewer.value.add_labels(labless, name="area segment")
    # @change_handler(plugin.save_button, init=False)
    # def _save_button():
    #     imgs=plugin.image.value.data
    #     labless=plugin.labelmask.value.data
    #     mmethod=get_value_plugin("measure")
    #     show_info("measurement : "+mmethod)
    #     filename, _ = QFileDialog.getSaveFileName(None, "Save  as...", ".", "*.csv")
    #     dirpath,shortname,suffix=split_filename(filename)
    #     if mmethod=="all":
    #         for mm in measurement_choices[1:]:
    #             # if "area" in mm:
    #             #     func=get_value_plugin("Binaty_func")
    #             #     lss=th_img(imgs,labless,func)
    #             #     df=label_series_statics(imgs,lss,mmethod)
    #             #     outfile=os.path.join(dirpath,shortname+"_"+mmethod+suffix)
    #             #     df.to_csv(outfile)
    #             #     show_info(outfile)
    #             #     # plugin.viewer.value.add_labels(lss, name="area segment")
                    
    #             # else:
    #             df=label_series_statics(imgs,labless,mm)
    #             outfile=os.path.join(dirpath,shortname+"_"+mm+suffix)
    #             df.to_csv(outfile)
    #             show_info(outfile)             
    #     else :
    #         # if "area" in mmethod:
    #         #     # change imgs
    #         #     func=get_value_plugin("Binaty_func")
    #         #     labless=th_img(imgs,labless,func)
    #         #     print(labless.shape,labless.dtype)
    #         #     # plugin.viewer.value.add_labels(labless, name="area segment")
    #         df=label_series_statics(imgs,labless,mmethod)
    #         outfile=os.path.join(dirpath,shortname+"_"+mmethod+suffix)
    #         df.to_csv(outfile)
    #         show_info(outfile)
            
 
    return plugin

