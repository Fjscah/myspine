from pint import Measurement
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea,QFileDialog,QMessageBox
from qtpy.QtCore import Qt
import napari
import functools
import numbers
import os
import time
import scipy.ndimage
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

from .networks import get_network,model_dict
from .imgio import napari_base
from .seg import segment,unetseg,trace

from .strutensor.hessian import StructureEvi,enhance_ridges_2d

import time
import pickle
from skimage.morphology import remove_small_objects
import numpy as np
import os
from .utils.file_base import split_filename
from .utils.spine_struct import spines_distance,gwdt_enable
from .utils.npixel import array_slice

from .__variables__ import *

net_method=list(model_dict.keys())



def load_file_model(filepath):
    #lmodel=torch.jit.load(filepath,map_location=torch.device('cpu'))
    state=torch.load(filepath,map_location=torch.device('cpu'))
    
    lmodel=get_network(state["network_type"],state["kwargs"])
    #state = torch.load(checkpoint_save_path)
    lmodel.load_state_dict(state["model"])
    # except Exception as e:
    #     show_info(e)
    #     return None
    return lmodel,state["network_type"]


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
        
        model,modelname=load_file_model(modelpath)
        plugin.model_name.value=modelname
        if model is None: return
        ins,prob,mask=model.off_predict(imgs)
            
            # all_mask,all_spine_pr,all_den_pr,bgpr=unetseg.predict_single_img(model,imgs)
            # model=UNet2d()
            # h,w=imgs.shape[-2:]
            # model.build(input_shape =(4,h,w,1))
            # model.load_weights(modelpath)
            #all_mask,all_spine_pr,all_den_pr=model.predict_single_img(imgs)
            
        im1=plugin.viewer.value.add_image(prob[2], name="spine_pr",colormap="magenta",
                opacity=0.5, blending="additive")
        im2=plugin.viewer.value.add_image(prob[1], name="den_pr",colormap="green",
                opacity=0.5, blending="additive")
        im3=plugin.viewer.value.add_image(prob[0], name="bg_pr",colormap="gray_r",visible=False,
                opacity=0.5, blending="additive")
        # all_mask=modify_mask(all_mask,sizeth=4)
        im4=plugin.viewer.value.add_labels(mask, name="seg_mask",opacity=0.5)
        im5=plugin.viewer.value.add_labels(ins, name="seg_ins",opacity=0.5)
        
        
        plugin.spinepr.value=im1
        plugin.denpr.value=im2
        plugin.bgpr.value=im3
        plugin.mask.value=im4
        plugin.spine.value=im5
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



