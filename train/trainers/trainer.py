
import os

from venv import create
from collections import OrderedDict,defaultdict
from torch.optim import lr_scheduler

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from torch.utils.data import DataLoader
import time
import numpy as np
import scipy.signal

import os
import torch
from torch import nn

from tqdm import tqdm
import torch.optim as optim 
from torch.optim import Adam,AdamW,Adagrad,RMSprop,SGD
from ..networks.unetplusplus import NestedUNet,UNet2d
from utils import file_base

from ..metrics.metrics import  iou_score,miou_weight
from ..dataset.dataloader import SpineDataset,augtransform

from ..metrics.metrics import AverageMeter
from .device import show_cpu_gpu,set_use_gpu
from utils.file_base import create_dir
from utils.basic_wrap import Logger,logit
from utils.yaml_config import YAMLConfig
from utils.basic_wrap import timing


from train.loss import get_lossfunc
from train.metrics import get_metricfunc
from train.networks import get_network



from torchsummary import summary

from ..metrics.metrics import matching,matching_dataset,print_matching_maps
from tensorboardX import SummaryWriter
# from torchinfo import summary
#matplotlib.use('Agg')
from .visual import Visualizer
optimizer_dict = {
    'Adam': Adam,
    'AdamW': AdamW,
    'Adagrad': Adagrad,
    'RMSProp': RMSprop,
    'SGD': SGD,
}

def savemodel(model,path,inn):
    # inn=torch.tensor(torch.rand(size=(1,1,256,256))).to("cuda")
    model = torch.jit.trace(model,inn)
    torch.jit.save(model,path)
# plt.ion()
# plt.ioff()
# plt.switch_backend("agg")


class History:
    def __init__(self, log_dir,keys=["loss","iou","val_loss","val_iou"]):
        import datetime
        self.time_str = datetime.datetime.strftime(datetime.datetime.now(),
                                                   '%Y_%m_%d_%H_%M_%S')
        self.save_path = os.path.join(log_dir, "loss_" + str(self.time_str))
        #self.writer = SummaryWriter(logdir=log_dir)
        self.datas=np.zeros((0,len(keys)))
        self.keys=keys

        os.makedirs(self.save_path)
        self.head="\t".join(self.keys)
        

    def on_epoch_end(self, epoch,updatadict, pltflag=True):
        epoch_result=[updatadict[k] for k in self.keys]
        self.datas=np.vstack((self.datas,epoch_result))
       
        np.savetxt(os.path.join(self.save_path,
                             "epoch_loss_" + str(self.time_str) + ".txt"),
                   self.datas,fmt="%.4f",delimiter="\t",header=self.head)
        # pltflag=(epoch%50==0) &(epoch>0)
        # pltflag=pltflag or (epoch%50==49) &(epoch>0)
        pltflag=True
        if pltflag:
            self._plot()
       
       

    def _plot(self):
        iters = self.datas.shape[0]

        fig=plt.figure()

      
        try:
            for n,key in enumerate(self.keys):
                data=self.datas[:,n]
                plt.plot(
                    list(range(len(data))),
                    data,
                    # scipy.signal.savgol_filter(
                    #     self.datas[:,n], 5 if len(self.datas[:,n]) < 25 else 15, 3),
                    #'green',
                    linestyle='--',
                    linewidth=2,
                    label=key)
           
        except Exception as e:
            print(e)
            raise

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.title('Histoty')
        plt.legend(loc="upper right")

        fig.savefig(
            os.path.join(self.save_path,
                         "epoch_loss_" + str(self.time_str) + ".png"))
        # #---------------------------------------------------------------------
        # plt.figure()

        # plt.plot(iters, self.metric, 'red', linewidth=2, label='train loss')
        # try:
        #     plt.plot(
        #         iters,
        #         scipy.signal.savgol_filter(
        #             self.metric, 5 if len(self.metric) < 25 else 15, 3),
        #         'green',
        #         linestyle='--',
        #         linewidth=2,
        #         label='smooth train MIoU')
        #     plt.plot(
        #             iters,
        #             scipy.signal.savgol_filter(
        #                 self.val_metric, 5 if len(self.metric) < 25 else 15, 3),
        #             '#8B4513',
        #             linestyle='--',
        #             linewidth=2,
        #             label='smooth val MIou')
        # except:
        #     pass

        # plt.grid(True)
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('A Loss Curve')
        # plt.legend(loc="upper right")

        # plt.savefig(
        #     os.path.join(self.save_path,
        #                  "epoch_miou_" + str(self.time_str) + ".png"))

        # plt.cla()
        
        plt.close()




class Trainer:
    EPOCH_PHASE = 0
    ITERATION_PHASE = 1

    def setting(self, configuration: YAMLConfig = None,use_gpu=True):
   
        if configuration == None:
            self.configuration = configuration
            return

        self.configuration = configuration
        # self.network_info=self.configuration.config["Network"]
        dict_a = self.configuration.config["Path"]
        dict_b = self.configuration.config["Training"]
        dict_c = self.configuration.config["Data"]
        #print(dict_b)
        self.__dict__.update(dict_a)
        self.__dict__.update(dict_b)
        self.__dict__.update(dict_c)
        # self.__dict__.update(self.network_info)
        self.ori_path=os.path.abspath(self.ori_path)
        self.exp_path=os.path.abspath(self.exp_path)
        
        if "z" in self.axes:
            self.imgshape = (self.input_sizez, self.input_sizexy,
                             self.input_sizexy,1)
        else:
            self.imgshape = (self.input_sizexy, self.input_sizexy,1)
        self.initial_log()
        self.initial_gpu(use_gpu)
        self.initial_allfunc()
        self.inital_model()
        self.start_epoch=0

        self.show_train_info()
        
        # self.initial_loss()
    def show_train_info(self):
        network_type = self.Network["name"]
        netkwarg=self.Network["kwargs"]
        lossname=self.loss_type["name"]
        losskwargs=self.loss_type["kwargs"]
        mname=self.Metrics["name"]
        mkwargs=self.Metrics["kwargs"]
        dicts=dict(
            # save_suffix=self.save_suffix,
            imgshape=self.imgshape,
            modelname=self.model.__class__.__name__,
            netk_warg=netkwarg,
            loss=self.loss.__class__.__name__,
            loss_kwargs=losskwargs,
            metric=self.metric.__name__,
            metrics_kwargs=mkwargs,
            batch_size=self.batch_size,
            # optimizer_name=self.optimizer_name,
            initial_learning_rate=self.learning_rate,
            # out_layer=self.model.out.__class__.__name__,
            
            
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

        

    def initial_allfunc(self):
        
        #--- Model---#
        create_dir(self.model_path)
        network_type = self.Network["name"]
        netkwarg=self.Network["kwargs"]
       
        self.model=get_network(network_type,netkwarg)
       # self.model.load_network_set(self.network_info)
        self.model.to(self.device)
        
        
        
        #--- LOSS---#
        # cls_weights = np.array([1,]*self.num_classes, np.float32)
        self.enhance_border=True if "border" in self.save_suffix else False
        if self.enhance_border:
            lossmode="multilabel"
        else:
            lossmode="multiclass"
        
        # self.loss = nn.BCEWithLogitsLoss()
        lossname=self.loss_type["name"]
        losskwargs=self.loss_type["kwargs"]
        self.loss=get_lossfunc(lossname,losskwargs).to(self.device)
        
        #--- OPTMIZER---#
        self.optimizer = optimizer_dict[self.optimizer_name](self.model.parameters(),
            lr=self.learning_rate, weight_decay=1e-5)
        
        #---scheduler---#
        #scheduler=lr_scheduler.LinearLR(optimizer,start_factor=0.5,total_iters=4)
        def lambda_(epoch):
            return pow((1 - ((epoch) / 100)), 0.9)
        self.scheduler =torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda_, )
        # scheduler = lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=100, eta_min=0.00001)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        #--- METRIC---#
        mname=self.Metrics["name"]
        mkwargs=self.Metrics["kwargs"]
        self.metric=get_metricfunc(mname,mkwargs)
        
        # #--- INSTANCE---#
        # iname=self.instance["name"]
        # ikwargs=self.instance["kwargs"]
        # self.ins=get_instancefunc(iname,ikwargs)
        
        #---- Visualizer---#
        self.visual=Visualizer(self.model.get_visual_keys())
        
        

    def inital_model(self):
    #     create_dir(self.model_path)
    #     network_type = self.Network["name"]
    #     netkwarg=self.Network["kwargs"]
       
    #     self.model=get_network(network_type,netkwarg)
    #    # self.model.load_network_set(self.network_info)
    #     self.model.to(self.device)
        
        summary(self.model,(1,self.input_sizexy,self.input_sizexy))

      
    def initial_log(self):
        file_base.create_dir(self.log_path)
        self.logger=Logger(os.path.join(os.path.abspath(self.log_path),"train_info.log"))
        history=History(self.log_path)
        self.history=history        
    def load_weight(self,denovo,premodel):
        """
        denove:        denovo
        not denovo and not premodel:    load newest weight
        not denovo and premodel:        load premodel
        """
        #-----------------------#
        #   Load model weights  #
        #-----------------------#

        if denovo is False:
            checkpoint_save_path = premodel
            # 模型参数保存路径
            if not checkpoint_save_path:
                # find neweat weight file
                paths=file_base.file_list_bytime(self.model_path,".pth")
                checkpoint_save_path=paths[-1] if paths else ""   
            if checkpoint_save_path and os.path.exists(checkpoint_save_path):
                state = torch.load(checkpoint_save_path)
                self.model.load_state_dict(state["model"])
                self.start_epoch=state["epoch"]
                self.history.datas=state["history"]
                self.model.eval()
                self.logger.logger.info(
                    "\n==============LOAD PRETRAINED model==============\n"+
                    checkpoint_save_path+
                    "\n=================================================\n"
                )
            else:
                self.logger.logger.info(
                    "\n==============NO PRETRAINED model==============\n"+
                    "DENOVO"
                    "\n=================================================\n"
                )
    def save_weight(self,epoch,loss):
        filepath=os.path.join(self.model_path , 'ep{0:03d}-loss{1:.3f}.pth'.format(epoch,loss))
        state={
            "model":self.model.state_dict(),
            "epoch":epoch,
            "history":self.history.datas,
            "kwargs":self.model.kwargs,
            "network_type":self.Network["name"],
        }
        torch.save(state, filepath)
        savemodel(self.model,self.best_pth,torch.zeros(1,1,self.input_sizexy,self.input_sizexy).to(self.device))

        print("=> saved best model",filepath)
        # delete old version weights
        paths=file_base.file_list_bytime(self.model_path,".pth")
        if len(paths)>self.keep_top:
            for path in paths[:-self.keep_top]:
                os.remove(path)
                print("remove file:",path)
       

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
        for image,ins, label in train_dataloader:
                # 将模型的参数梯度初始化为0
            # img,lab=batch
            model.train()
            image = image.to(self.device)
            label = label.to(self.device)
            ins = ins.to(self.device)
            # print(image.shape)
            output = model(image)
            loss = lossfunc(output, label,ins)
            #loss = lossfunc(output, label,ins,iou=True,iou_meter=avg_meters['iou'])
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
    
    def valid_epoch(self,valid_dataloader,model,lossfunc,metricfunc,num_classes):
        avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            pbar = tqdm(total=len(valid_dataloader),desc="valid",bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            
            for image,ins, label in valid_dataloader:
                image = image.to(self.device)
                label = label.to(self.device)
                ins=ins.to(self.device)
                # compute output
                output = model(image)
                # print(output.shape,label.shape,ins.shape)
                # loss = lossfunc(output, label,ins,iou=True,iou_meter=avg_meters['iou'])
                loss = lossfunc(output, label,ins)
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

        return OrderedDict([('val_loss', avg_meters['loss'].avg),
                            ('val_iou', avg_meters['iou'].avg)])
    
    def metric_epoch(self,valid_dataloader,model,lossfunc,metricfunc,epoch):
        # metrtics for instance segment
        model.eval()
        ins_s=[]
        label_s=[]
        if epoch%self.mAP_epoch==0 or epoch%self.visual_epoch==0:
            i=5 if epoch%self.mAP_epoch==0 else 3
            with torch.no_grad():
            # print("Caculating mAP.....")
              
                for images, inss,labels in valid_dataloader:
                    i-=1
                    for img,ins,label in zip(images,inss,labels):
                        # image = image.to(self.device)
                        # model.predict_2d_img(image)
                        img=img.squeeze()
                        spine_label,output=self.model.predict(img)
                        #m=matching(ins,spine_label,[0.5,0.6,0.8])
                        #ins[ins<2]=0
                        ins_s.append(ins.squeeze().cpu().detach().numpy())
                        label_s.append(spine_label)
                    if not i: break
                if "ins" in self.task_type:
                    self.model.show_result(self.visual,[img.cpu(),ins,spine_label,output])
                if "seg" in self.task_type:
                    self.model.show_result(self.visual,[img.cpu(),ins,spine_label,output])
                    
                if epoch%self.mAP_epoch==0:
                    m=matching_dataset(ins_s,label_s,[0.2,0.3,0.4,0.5,0.6,0.7,0.8])
                    print_matching_maps(m)
        
            
    @logit("error.log")
    def train(self, denovo=False,premodel=""):
        #trdatasets=CusDataset(self.configuration)
        #trdatasets,vadatasets,tedatasets=tfd.load_train_test(num_class=self.num_classes)

        if self.configuration is None:
            raise Exception('Please set Yaml configuration first!')
        #-----------------------#
        #  config load  #
        #-----------------------#
        exp_path=self.exp_path
        crop_path=self.crop_path
        suffix=self.save_suffix # seg,spine,den,border
        num_classes=self.num_class
        
        
        epochs=self.epochs # 迭代几个周期
        batch_size=self.batch_size #每批多少张图像
        step=min(30,30) #每个周期有几批数据
        iteration=step*batch_size*epochs # 总迭代次数
        epoch_iterration=step*batch_size
        

        #-----------------------#
        #   data load  #
        #-----------------------#
        t1=time.time()
        train_trainform=None
        self.enhance_border=True if "border" in self.save_suffix else False
        self.make_dis=True if "dis" in self.save_suffix else False
        train_datast=SpineDataset(crop_path + "/train",suffix,num_classes,transform=train_trainform,
                                         iteration=epoch_iterration,des="train",
                                         enhance_border=self.enhance_border,
                                         task_type=self.task_type,
                                         )
        valid_datast=SpineDataset(crop_path + "/valid",suffix,num_classes,iteration=epoch_iterration,des="valid",
                                         enhance_border=self.enhance_border,
                                        task_type=self.task_type,
                                         )
        
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
        history=self.history
    
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
        scheduler=self.scheduler
        
        # lambda1 = lambda epoch: 0.9 ** epoch                     


        #-----------------------#
        #   Train  #
        #-----------------------#
        create_dir(os.path.join(self.model_path ,"checkpoint"))
        val_log=self.valid_epoch(valid_dataloader,model,lossfunc,metric,num_classes)
        self.metric_epoch(valid_dataloader,model,lossfunc,metric,self.start_epoch)
        best_iou = val_log['val_iou']
        self.best_pth=os.path.join(self.model_path ,"checkpoint","best.pt")
        bestpath=self.best_pth
        #savemodel(self.model,bestpath,torch.zeros(1,1,self.input_sizexy,self.input_sizexy).to(self.device))
        print("best loss=",val_log['val_loss'],"best_iou=",best_iou)
        self.save_weight(self.start_epoch,val_log['val_loss'])
        print("=> saved inital best model",bestpath)
        #self.model=torch.nn.DataParallel(model)
        for epoch in range(self.start_epoch,self.start_epoch+self.epochs):
            if self.use_gpu:
                gpu_use_info=f", {torch.cuda.get_device_name(0)} Memory Usage : Allocated-{round(torch.cuda.memory_allocated(0)/1024**3,1)} GB , Cached-{round(torch.cuda.memory_reserved(0)/1024**3,1)} GB"
            else:
                gpu_use_info=""
            lr=scheduler.get_last_lr()[0]
         
            print(f"epoch : {epoch} / {(self.epochs+self.start_epoch)}, lr : {lr:.5f}"+gpu_use_info)
            # num_iters = 0
            # epoch_loss_1 = 0.0
            train_log=self.train_epoch(train_dataloader,model,lossfunc,metric,optimizer)
            val_log=self.valid_epoch(valid_dataloader,model,lossfunc,metric,num_classes)
            self.metric_epoch(valid_dataloader,model,lossfunc,metric,epoch)
            val_log.update(train_log)
            history.on_epoch_end(epoch,val_log,pltflag=epoch+1)
            scheduler.step()

            # print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
            #   % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

            log['epoch'].append(epoch)
            log['lr'].append(scheduler.get_last_lr()[0])
            log['loss'].append(train_log['loss'])
            log['iou'].append(train_log['iou'])
            log['val_loss'].append(val_log['loss'])
            log['val_iou'].append(val_log['iou'])
            
     
            #-----------------------#
            #   save weight  #
            #-----------------------#
            if val_log['val_iou'] > best_iou:
                self.save_weight(epoch,val_log['val_loss'])
                best_iou = val_log['val_iou']
            if self.use_gpu:
                torch.cuda.empty_cache()
        # return metric, self.best_metric 
        print("best_iou",best_iou)
        t2=time.time()
        run_time=t2-t1
        # 计算时分秒
        hour = run_time//3600
        minute = (run_time-3600*hour)//60
        second = run_time-3600*hour-60*minute
        # 输出
        print (f'Running time ：{hour} hour {minute} minute {second} second')


if __name__ == "__main__":
    trainmodel = Trainer()

    premodel =r"D:\code\myspine\models\M2d_seg\modelep200-loss0.133.h5"
    trainmodel.train(premodel)
