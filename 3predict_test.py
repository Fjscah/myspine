

from utils.yaml_config import YAMLConfig

from train.trainers.predict import Predict

#%% set predict mode: sigle image or multi images(dir)
predictmode="single" # multi , sigle image or multi images(dir)
configfile=r"config\defalut_2d_seg_UnetL3.yaml"
test_flag=True  #False, wheather to use test folder ( load test label together)

# if premodel not specify will search from yaml profile
premodel=""#r"../test-dataset/model/2Dtorchunet-b/seg\ep003-loss0.415.pth"
# if datas not specify will load test data folder ,datas supprt imgfile,imgfiles,imgdolder
datas=r"myspine-dataset/2D-morph-seg/img/"
datas=[r"myspine-dataset/2D-morph-seg/img/",r"myspine-dataset/2D-morph-seg/label/","-seg"]
datas=None
datas=r"F:\data\Train\2015\actin-20151220-1\actin-20151220-1.tif"



# print(os.path.isdir(datas))
if configfile is not None:
    default_configuration=YAMLConfig(
        configfile
        )
    
   
    print("===== Train dataset ======")
    trainmodel = Predict()
    trainmodel.setting(default_configuration,use_gpu=False)
    trainmodel.predict(premodel=premodel,data=datas)


# elif predictmode=="signle":
#     pass
# elif predictmode=="multi":
#     if test_flag:
#        pass
#     else:
#         pass