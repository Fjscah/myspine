
from pint import Measurement
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea,QFileDialog,QMessageBox
from qtpy.QtCore import Qt
import napari
import functools

from typing import List, Union
from warnings import warn



from magicgui import magicgui
from magicgui import widgets as mw

from psygnal import Signal

from napari.utils.notifications import show_info
from scipy.ndimage import distance_transform_edt



from .seg import segment,unetseg,trace

from .strutensor.hessian import StructureEvi,enhance_ridges_2d

from skimage.morphology import remove_small_objects
import numpy as np
from .utils.spine_struct import spines_distance,gwdt_enable
from .utils.npixel import array_slice

from .__variables__ import *








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
  
        image=dict(label="gray Images"),
        axes=dict(
            widget_type="ComboBox",
            label="Image Axes",
            choices=["xy","zxy"],
            value="xy",
        ),      
        binaryimage=dict(label="binary Image",nullable=True),
        denmask=dict(label="den Image",nullable=True),
        # segmask=dict(label="segment layer",nullable=True),

        preprocess1=dict(widget_type="Label",
                         label="<br><b>1. seg dendrite</b>"),
        bmethod=dict(
            widget_type="ComboBox",
            label="Binary method",
            choices=Bmethod_name,
            value=Bmethod_name[-1],
        ),
        adth_button=dict(widget_type="PushButton", text="binarylize"),
        
        preprocess2=dict(widget_type="Label",
                         label="<br><b>2. run link</b>"),
        denpoints=dict(label="den point",nullable=True),
        weight_image=dict(label="weight image",nullable=True),
        run_link= dict(widget_type="PushButton", text="run link"),
       
        layout="vertical", #
        persist=True,
        call_button=False,  # if True auto add "run" button
    )
    def plugin(
        viewer: napari.Viewer,
        image: napari.layers.Image,
        axes,    
        binaryimage:napari.layers.Labels,
        denmask:napari.layers.Labels,
        # spinemask:napari.layers.Labels,
        # segmask:napari.layers.Labels,
        
        preprocess1,
        bmethod,
        adth_button,
        preprocess2,
        denpoints:napari.layers.Points,
        weight_image: napari.layers.Image,
        run_link,

      
    ) -> List[napari.types.LayerDataTuple]:
        x = image.data
        #axes = axes_check_and_normalize(axes, length=x.ndim)
        
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

    @change_handler(plugin.run_link, init=False)
    def _run_link(model_name_star: str): 
        if plugin.denpoints.value is None:
            messg=QMessageBox(None)
            messg.setText("Please set point layer !")
            messg.exec_()
            return
        points=plugin.denpoints.value.data
        mask=plugin.binaryimage.value.data
        img=plugin.weight_image.value.data
        if not points.any(): return
        if len(points)!=2 : 
            messg=QMessageBox(None)
            messg.setText("the point number must be two !")
            messg.exec_()
            return
        # dis_matrixt=distance_transform_edt(mask)
        # dis_matrix=np.max(dis_matrixt)-dis_matrixt+1
        # dis_matrix=mask.copy()
        # dis_matrix[mask==0]=1000
        	# Penalty weight for the edges
        # M = np.max(dis_matrixt)**1.01
        # p_v = 1000000*(1-dis_matrixt/M)**16
        # p_v = p_v.astype(np.float32)
        img=np.max(img)-img+1
        paths=trace.get_path_2(mask,img,points[0],points[1])
        denmask=np.zeros_like(mask)
        print(paths)
        for point in paths:
            # rad= int(dis_matrixt[tuple(point)]+1)
            # obj=array_slice(denmask,point,rad,center=True,obj_only=True)
            # print(obj,rad)
            # denmask[obj]=1
            denmask[tuple(point)]=1
        
      
        if plugin.denmask.value is not None:
            oldden=plugin.denmask.value.data
            maxn=np.max(oldden)+1
            oldden[denmask>0]=maxn
            #newden=np.bitwise_or(oldden,denmask)
            plugin.denmask.value.data=oldden#plugin.viewer.value.add_labels(denmask, name="denmask")
        else:
            plugin.denmask.value=plugin.viewer.value.add_labels(denmask, name="denmask")
        plugin.denpoints.value.data=[]
        # plugin.viewer.value.add_image(geodist, name="geodist")
    return plugin
           

