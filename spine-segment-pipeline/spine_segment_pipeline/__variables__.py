from .cflow.localthreadhold import all_threshold_func,local_threshold,local_threshold_23d
from .seg import segment,unetseg,trace
import os


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