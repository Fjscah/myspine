import gc
import logging
import os
import sys
import warnings
from venv import create
from collections import OrderedDict
from torch.optim import lr_scheduler
# import cv2
# import cv2
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from torch.utils.data import DataLoader

import numpy as np
import scipy.signal

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.optim as optim 
from torch.optim import Adam,AdamW,Adagrad,RMSprop,SGD
from ..networks import unetplusplus
sys.path.append(".")
from utils import file_base

from .metrics import  iou_score,miou_weight
from keras.metrics import categorical_accuracy
from ..dataset.dataloader import DataLoader, CustomDatasetUnet2D,augtransform,get_train_tranform

from .metrics import AverageMeter
from .device import show_cpu_gpu,set_use_gpu
from utils.file_base import create_dir
from utils.basic_wrap import Logger,logit
from utils.yaml_config import YAMLConfig
from utils.basic_wrap import timing
from .loss import *
from torchsummary import summary
# from torchinfo import summary
matplotlib.use('Agg')


optimizer_dict = {
    'Adam': Adam,
    'AdamW': AdamW,
    'Adagrad': Adagrad,
    'RMSProp': RMSprop,
    'SGD': SGD,
}

loss_dict={
    "FocalLoss":FocalLoss("multiclass",2),
    "b_cross_entropy": nn.BCELoss(),
}






class LossHistory:
    def __init__(self, log_dir):
        import datetime
        self.time_str = datetime.datetime.strftime(datetime.datetime.now(),
                                                   '%Y_%m_%d_%H_%M_%S')
        self.save_path = os.path.join(log_dir, "loss_" + str(self.time_str))
        

        self.losses = []
        self.metric = []
        self.val_loss = []
        self.val_metric = []

        os.makedirs(self.save_path)
        string="\t".join(["epoch","loss","iou","val_loss","val_iou"])
        with open(
                os.path.join(self.save_path,
                             "epoch_loss_" + str(self.time_str) + ".txt"),
                'a') as f:
            f.write(string)
            f.write("\n")

    def on_epoch_end(self, epoch,train_log,val_log, pltflag=0):
        self.losses.append(train_log['loss'])
        self.metric.append(train_log['iou'])
        self.val_loss.append(val_log['loss'])
        self.val_metric.append(val_log['iou'])
        
        lists=[str(v) for v in [epoch,train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']]]
        string=",".join(lists)
        with open(
                os.path.join(self.save_path,
                             "epoch_loss_" + str(self.time_str) + ".txt"),
                'a') as f:
            f.write(string)
            f.write("\n")
        pltflag=(epoch%50==0) &(epoch>0)
        if pltflag:
            self.loss_plot()
       

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()

        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        try:
            plt.plot(
                iters,
                scipy.signal.savgol_filter(
                    self.losses, 5 if len(self.losses) < 25 else 15, 3),
                'green',
                linestyle='--',
                linewidth=2,
                label='smooth train loss')
            plt.plot(
                    iters,
                    scipy.signal.savgol_filter(
                        self.val_loss, 5 if len(self.losses) < 25 else 15, 3),
                    '#8B4513',
                    linestyle='--',
                    linewidth=2,
                    label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(
            os.path.join(self.save_path,
                         "epoch_loss_" + str(self.time_str) + ".png"))
        #---------------------------------------------------------------------
        plt.figure()

        plt.plot(iters, self.metric, 'red', linewidth=2, label='train loss')
        try:
            plt.plot(
                iters,
                scipy.signal.savgol_filter(
                    self.metric, 5 if len(self.metric) < 25 else 15, 3),
                'green',
                linestyle='--',
                linewidth=2,
                label='smooth train loss')
            plt.plot(
                    iters,
                    scipy.signal.savgol_filter(
                        self.val_metric, 5 if len(self.metric) < 25 else 15, 3),
                    '#8B4513',
                    linestyle='--',
                    linewidth=2,
                    label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(
            os.path.join(self.save_path,
                         "epoch_miou_" + str(self.time_str) + ".png"))

        plt.cla()
        
        plt.close("all")




class Trainer:
    EPOCH_PHASE = 0
    ITERATION_PHASE = 1

    def setting(self, configuration: YAMLConfig = None,use_gpu=True):

        if configuration == None:
            self.configuration = configuration
            return

        self.configuration = configuration
        self.network_info=self.configuration.config["Network"]
        dict_a = self.configuration.config["Path"]
        dict_b = self.configuration.config["Training"]
        dict_c = self.configuration.config["Data"]
        #print(dict_b)
        self.__dict__.update(dict_a)
        self.__dict__.update(dict_b)
        self.__dict__.update(dict_c)
        self.__dict__.update(self.network_info)
        self.Train_path=os.path.abspath(self.Train_path)
        if "z" in self.axes:
            self.imgshape = (self.input_sizez, self.input_sizexy,
                             self.input_sizexy,1)
        else:
            self.imgshape = (self.input_sizexy, self.input_sizexy,1)
        file_base.create_dir(self.log_path)
        self.logger=Logger(os.path.join(os.path.abspath(self.log_path),"train_info.log"))
    
        self.initial_gpu(use_gpu)
        self.inital_model()
        

        self.initial_optimizer()
        self.initial_metric()
        self.initial_loss()
        self.show_train_info()
        
        # self.initial_loss()
    def show_train_info(self):
        dicts=dict(
            save_suffix=self.save_suffix,
            imgshape=self.imgshape,
            modelname=self.model.__class__.__name__,
            layer_num=self.layer_num,
            batch_size=self.batch_size,
            optimizer_name=self.optimizer_name,
            initial_learning_rate=self.learning_rate,
            loss=self.loss.__class__.__name__,
            metric=self.metric.__name__,
            out_layer=self.model.out.__class__.__name__,
            
            
        )
        # keys = ["modelname","layer_num" "batch_size", "cls_weight","label_suffix", 
        #         "initial_learning_rate","optimizer_name","loss_type","metrics","imgshape"]
        kvs = [f"{k:<15}" + "\t:\t" + v.__repr__() for k,v in dicts.items()]
        self.logger.logger.info(
            "\n=========YAML TRAIN INFO==============\n"+
            "\n".join(kvs)+
            "\n======================================\n"
        )
        pass
    def initial_gpu(self,use_gpu):
        device,ngpu,ncpu=set_use_gpu(use_gpu)
        self.use_gpu=use_gpu
        self.device=device
        self.ngpu=ngpu
        self.ncpu=ncpu
    def initial_loss(self):
        cls_weights = np.array([1,]*self.num_classes, np.float32)
        self.enhance_border=True if "border" in self.save_suffix else False
        if self.enhance_border:
            lossmode="multilabel"
        else:
            lossmode="multiclass"
            
        # self.loss = nn.BCEWithLogitsLoss()
        
        self.loss=loss_dict[self.loss_type].to(self.device)
        # self.cls_weight
       
        # #------------------------------------------------------------------#
        # dice_loss = False
        # #------------------------------------------------------------------#
        # #   是否使用focal loss来防止正负样本不平衡
        # #------------------------------------------------------------------#
       
        # self.loss = weighted_categorical_crossentropy(self.cls_weight)

    def initial_optimizer(self):

        self.optimizer = optimizer_dict[self.optimizer_name](self.model.parameters(),
            lr=self.learning_rate, weight_decay=1e-5)

    def inital_model(self):
        create_dir(self.model_path)
        network_type = self.configuration.get_entry(['Network', 'modelname'])
        num_classes=self.num_classes
        if "unet3d" == network_type:
            self.model = unet.UNet3D(self.configuration)
        elif "unet2d" == network_type:
            self.model =unetplusplus.UNet2d(num_classes,1)
        elif "unet++"==network_type:
            self.model=unetplusplus.NestedUNet(num_classes,1)
        self.model.load_network_set(self.network_info)
        self.model.to(self.device)
        
        summary(self.model,(1,self.input_sizexy,self.input_sizexy))
    def initial_metric(self):
        self.metric=miou_weight([0])    
            
    def load_weight(self,denovo,premodel):
        #-----------------------#
        #   Load model weights  #
        #-----------------------#
        checkpoint_save_path = premodel
        #checkpoint_save_path = ""  #r"models\M2d_seg\modelep100-loss0.011.h5"  # 模型参数保存路径
        if (not checkpoint_save_path) and (denovo is False):
            # find neweat weight file
            paths=file_base.file_list_bytime(self.model_path,".pth")
            checkpoint_save_path=paths[-1] if paths else ""   
        if checkpoint_save_path and os.path.exists(checkpoint_save_path):
            self.model.load_state_dict(torch.load(checkpoint_save_path))
            self.model.eval()
            self.logger.logger.info(
                "\n==============LOAD PRETRAINED model==============\n"+
                checkpoint_save_path+
                "\n=================================================\n"
            )
        

    def __init__(self, configuration: YAMLConfig = None):
        self.setting(configuration)
        #self.logger = logging.getLogger('info')
        # self.model.compile(optimizer=self.optimizer,loss='binary_crossentropy', metrics=['accuracy'])
        #self.log = log
        self.num_train_examples = 0
        self.best_val_f1_scores = [0, 0, 0]
        self.best_val_epoch = 0
       
    
    
    def train_epoch(self,train_dataloader,model,lossfunc,metricfunc,optimizer):
        avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
        model.train()
        pbar = tqdm(total=len(train_dataloader),desc="train",bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for image, label in train_dataloader:
                # 将模型的参数梯度初始化为0
            # img,lab=batch
            model.train()
            image = image.to(self.device)
            label = label.to(self.device)
            # print(image.shape)
            output = model(image)
            loss = lossfunc(output, label)
            iou = metricfunc(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            avg_meters['loss'].update(loss.item(), image.size(0))
            avg_meters['iou'].update(iou, image.size(0))
            
            postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
        
        return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])
    
    def valid_epoch(self,valid_dataloader,model,lossfunc,metricfunc):
        avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            pbar = tqdm(total=len(valid_dataloader),desc="valid",bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            
            for image, label in valid_dataloader:
                image = image.to(self.device)
                label = label.to(self.device)

                # compute output

                output = model(image)
                loss = lossfunc(output, label)
                iou = metricfunc(output, label)

                avg_meters['loss'].update(loss.item(), image.size(0))
                avg_meters['iou'].update(iou, image.size(0))

                postfix = OrderedDict([
                    ('loss', avg_meters['loss'].avg),
                    ('iou', avg_meters['iou'].avg),
                ])
                pbar.set_postfix(postfix)
               
                pbar.update(1)
            pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg)])
    
    @logit("error.log")
    def train(self, denovo=False,premodel=""):
        #trdatasets=CusDataset(self.configuration)
        #trdatasets,vadatasets,tedatasets=tfd.load_train_test(num_class=self.num_classes)

        if self.configuration is None:
            raise Exception('Please set Yaml configuration first!')
        #-----------------------#
        #  config load  #
        #-----------------------#
        Train_path=self.Train_path
        suffix=self.save_suffix # seg,spine,den,border
        num_classes=self.num_classes
        log_dir=self.log_path
        
        epochs=self.epochs # 迭代几个周期
        batch_size=self.batch_size #每批多少张图像
        step=min(30,30) #每个周期有几批数据
        iteration=step*batch_size*epochs # 总迭代次数
        epoch_iterration=step*batch_size
        
        #-----------------------#
        #   data load  #
        #-----------------------#
        train_trainform=get_train_tranform()
        self.enhance_border=True if "border" in self.save_suffix else False
        train_datast=CustomDatasetUnet2D(Train_path + "\\train",suffix,num_classes,transform=train_trainform,iteration=epoch_iterration,des="train",enhance_border=self.enhance_border)
        valid_datast=CustomDatasetUnet2D(Train_path + "\\valid",suffix,num_classes,iteration=epoch_iterration,des="valid",enhance_border=self.enhance_border)
        
        # dalaloader size equal to train_dataloader.__len__ and return data wich has been packaged with batch_size
        train_dataloader = DataLoader(train_datast,batch_size = batch_size,shuffle=True)
        valid_dataloader = DataLoader(valid_datast,batch_size = batch_size,shuffle=True)

        numwork=min(4,self.ncpu)
        
 
        
        #-----------------------#
        #   Log  #
        #-----------------------#
        log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ])
        history=LossHistory(self.log_path)
        #-----------------------#
        #   model load  #
        #-----------------------#
        # self.model =None
        # self.show_train_info()
        self.load_weight(denovo,premodel)
        self.model.train() # set model mode as train
        model=self.model
        
        lossfunc = self.loss
        optimizer =self.optimizer  
        metric=self.metric
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0.00001)
        # lambda1 = lambda epoch: 0.9 ** epoch                     
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        #-----------------------#
        #   Train  #
        #-----------------------#
        create_dir(os.path.join(self.model_path ,"checkpoint"))
        val_log=self.valid_epoch(valid_dataloader,model,lossfunc,metric)
        best_iou = val_log['iou']
        print("baestloss",val_log['loss'],"best_iou",best_iou)
        for epoch in range(self.epochs):
            if self.use_gpu:
                gpu_use_info=f", {torch.cuda.get_device_name(0)} Memory Usage : Allocated-{round(torch.cuda.memory_allocated(0)/1024**3,1)} GB , Cached-{round(torch.cuda.memory_reserved(0)/1024**3,1)} GB"
            else:
                gpu_use_info=""
            lr=scheduler.get_last_lr()[0]
            print(f"epoch : {epoch} / {self.epochs}, lr : {lr:.5f}"+gpu_use_info)
            # num_iters = 0
            # epoch_loss_1 = 0.0
            train_log=self.train_epoch(train_dataloader,model,lossfunc,metric,optimizer)
            val_log=self.valid_epoch(valid_dataloader,model,lossfunc,metric)
            
            history.on_epoch_end(epoch,train_log=train_log,val_log=val_log,pltflag=epoch+1)
            scheduler.step()

            print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

            log['epoch'].append(epoch)
            log['lr'].append(scheduler.get_last_lr()[0])
            log['loss'].append(train_log['loss'])
            log['iou'].append(train_log['iou'])
            log['val_loss'].append(val_log['loss'])
            log['val_iou'].append(val_log['iou'])
            
            #---- save weight
            
            if val_log['iou'] > best_iou:
                filepath=os.path.join(self.model_path , 'ep{0:03d}-loss{1:.3f}.pth'.format(epoch,val_log['loss']))
                torch.save(model.state_dict(), filepath)
                torch.save(model,os.path.join(self.model_path ,"checkpoint","best.pth"))
                best_iou = val_log['iou']
                print("=> saved best model",filepath)
                # delete old version weights
                paths=file_base.file_list_bytime(self.model_path,".pth")
                if len(paths)>self.keep_top:
                    for path in paths[:-self.keep_top]:
                        os.remove(path)
                        print("remove file:",path)
                trigger = 0
            if self.use_gpu:
                torch.cuda.empty_cache()
        # return metric, self.best_metric 
        print("best_iou",best_iou)



if __name__ == "__main__":
    trainmodel = Trainer()

    premodel =r"D:\code\myspine\models\M2d_seg\modelep200-loss0.133.h5"
    trainmodel.train(premodel)
