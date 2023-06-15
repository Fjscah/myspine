import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import torch
from skimage.measure import regionprops, label
from skimage.feature import peak_local_max
from skimage.morphology import dilation as im_dilation
from skimage.morphology import square as mor_square
from PIL import Image
import math
import torch.nn.functional as F
from train.trainers.visual import Visualizer
from .BaseNet import *
from .UNet import UNet
from matplotlib import pyplot as plt
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels,  out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class TrippleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels//2 , kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels//2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels//2 , out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# class UNet(nn.Module):
    
#     def __init__(self, n_channels,  bilinear=True):
#         #nchanel is input channel
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 128)
#         self.down3 = Down(128, 128)
#         # self.down4 = Down(512, 512)
#         # self.up1 = Up(1024, 256, bilinear)
#         self.up2 = Up(256, 128, bilinear)
#         self.up3 = Up(256, 64, bilinear)
#         self.up4 = Up(128, 64, bilinear)


#     def forward(self, x):
#         inc = self.inc(x)
#         d1 = self.down1(inc)
#         d2 = self.down2(d1)
#         d3 = self.down3(d2)
#         # d4 = self.down4(d3)
#         # u1 = self.up1(d4, d3)
       
#         u2 = self.up2(d3, d2)
#         u3 = self.up3(u2, d1)
#         x = self.up4(u3, inc)
#         return x

# class UNet_with_two_decoder(UNet):
#     def __init__(self, n_channels,  bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 512)

#         self.up1_1 = Up(1024, 256, bilinear)
#         self.up1_2 = Up(512, 128, bilinear)
#         self.up1_3 = Up(256, 64, bilinear)
#         self.up1_4 = Up(128, 64, bilinear)

#         self.up2_1 = Up(1024, 256, bilinear)
#         self.up2_2 = Up(512, 128, bilinear)
#         self.up2_3 = Up(256, 64, bilinear)
#         self.up2_4 = Up(128, 64, bilinear)


#     def forward(self, x):
#         inc = self.inc(x)
#         d1 = self.down1(inc)
#         d2 = self.down2(d1)
#         d3 = self.down3(d2)
#         d4 = self.down4(d3)

#         u1 = self.up1_1(d4, d3)
#         u2 = self.up1_2(u1, d2)
#         u3 = self.up1_3(u2, d1)
#         y1 = self.up1_4(u3, inc)

#         u1 = self.up2_1(d4, d3)
#         u2 = self.up2_2(u1, d2)
#         u3 = self.up2_3(u2, d1)
#         y2 = self.up2_4(u3, inc)

#         return y1, y2



    
    
class WNet(BaseNet):
    def __init__(self, **kwargs):
                 #input_channel=1,num_class=3,unet_dim=32, dis2emb_channels=8, emb_channels=8, # net paras
                 #seeds_thres=0.7,seeds_min_dis=5,similarity_thres=0.7, # cluster paras
                 #):
        super(WNet, self).__init__(**kwargs)
        # net paras
        self.input_dim = kwargs["input_channel"]
        self.num_class = kwargs["num_class"]
        self.unet_ch = kwargs["unet_dim"]
        self.dfeat_ch = kwargs["dis2emb_channels"]
        self.emb_ch = kwargs["emb_channels"]
        # cluster paras
        self.seeds_thres=kwargs["seeds_thres"]
        self.seeds_min_dis=kwargs["seeds_min_dis"]
        self.similarity_thres=kwargs["similarity_thres"]

        # backbone
        self.dnet = UNet(self.input_dim,self.num_class,3)# out channel   64
        
        # emb head
        self.efeat_head = nn.Sequential(
            nn.Conv2d(self.input_dim, self.dfeat_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # efeat
        self.enet = UNet(self.dfeat_ch + 1,self.emb_ch,3) # out ctorch.pernutehannel 64

        
    #-----------------------#
    #   Train/Valid  #
    #-----------------------#
    def forward(self, input, user_data=''):
       # x = self.dnet(input)
        
        seed_map = self.seed_forward(input,no_grad=False)#self.seed_head(x)
        # distmap = self.distmap_predictor(distmap)
        # sx = torch.cat((seed_map, input), dim=1)
        # seed_map = self.snet(sx)
   
        
        
        efeat = self.efeat_head(input)
        # efeat = F.normalize(efeat, p=2, dim=1)
        seed_map_copy =seed_map.detach()
        ex = torch.cat((efeat,seed_map_copy[:,-1:]), dim=1)
        # ex = torch.cat((efeat,input), dim=1)
        efeat = self.enet(ex)
        # efeat = self.emb_predictor(efeat)
        efeat = F.normalize(efeat, p=2, dim=1)
        # efeat=F.tanh(efeat)
 
        return seed_map, efeat # seed_map for find grow center, efeat for constain grow border

    def seed_forward(self,input,no_grad=False):
        if no_grad:
            with torch.no_grad():
                seed_map = self.dnet(input)
        else:
            seed_map = self.dnet(input)
        return seed_map    
    
    
    #-----------------------#
    #   Predict  #
    #-----------------------#
    def off_predict(self,im):
        # im=rescale(im,2)
        im=normalize(im,1,99.8)
        
      
      
        ins,pred=self.predict(im)
        prob, efeat=pred
        prob=prob.squeeze().cpu().detach().numpy()
        mask=np.argmax(prob,axis=0)
        ins=remove_small_objects(ins,10)
        
        return ins,prob,mask
        
    def predict(self,img):
        im=self.valid_im(img).to(self.cur_device)
        pd_distmap, efeat = self.forward(im)
        # pd_distmap=torch.softmax(pd_distmap,dim=1)
        pd_distmap=torch.sigmoid(pd_distmap)
        pd_seeds, pd_label = self.get_prediction(pd_distmap[0,-1].cpu().detach().numpy(),
                                                    efeat[0].cpu().detach())
        return pd_label,[pd_distmap, efeat]
              
    def get_prediction(self, distmap, efeat):
        efeat = efeat.permute((1,2,0)) #(h,w,c)
       
        seeds, seg=mask_from_seeds_v3(efeat, distmap,self.similarity_thres,self.seeds_thres, self.seeds_min_dis)
        #
        seg=remove_small_objects(seg,10)
        # seg = remove_noise(seg, distmap)

        return seeds, seg

    
    def get_visual_keys(self) -> List[str]:
        return ["image","seed1","seed2","label","GT","mask"]
    
    def show_result(self, visualizer: Visualizer, visual_result):
        img,GT,ins,output=visual_result
        pd_distmap, efeat=output
        pd_distmap=torch.sigmoid(pd_distmap)
        visualizer.display(img.cpu(), 'image')
        GT=label2rgb(GT[0].cpu().numpy(),image=img.cpu().numpy(),bg_label=0,alpha=0.4)
        visualizer.display(GT, 'GT')  # it shows prediction / GT-mask
        ins=label2rgb(ins,image=img.cpu().numpy(),bg_label=0,alpha=0.4)
        visualizer.display(ins, 'label')  # it shows prediction / GT-mask
        # efeat=torch.abs(efeat)
        efeat=(efeat+1)/2
        visualizer.display(efeat[0,:3].cpu(), 'seed1')  # seed map
        visualizer.display(efeat[0,3:6].cpu(), 'seed2')  # seed map
        visualizer.display(pd_distmap[0,:3].cpu().numpy(), 'mask')  # seed map
        visualizer.save()    

    def visual_cluster(self,img):# 3d plot
        fig,ax=get_ball_grid()
        im=self.valid_im(img).to(self.cur_device)
        pd_distmap, efeat = self.forward(im)
        pd_distmap=pd_distmap[0,-1].cpu().detach().numpy()
        efeat=efeat[0].cpu().detach()
        pd_distmap=torch.sigmoid(pd_distmap)
        pd_seeds, pd_label = self.get_prediction(pd_distmap,efeat)
       
        props = regionprops(pd_label)

        inverse_distmap = np.ones_like(pd_distmap)* pd_distmap.max() - pd_distmap
        inverse_distmap = np.expand_dims(inverse_distmap,0)

        mean = {}
        for p in props:
            row, col = p.coords[:, 0], p.coords[:, 1]

            key_feat = efeat[:,row, col] * inverse_distmap[:,row,col]
            key_feat = torch.mean(key_feat, dim=1)
            emb_mean = F.normalize(key_feat, p=2, dim=0)
            mean[p.label] = emb_mean
        scatter=np.array([ v.numpy() for v in mean.values()])
        print(scatter.shape)
        scatter_color=scatter[:,0:3]/2+0.5
        ax.scatter(scatter[:,0],scatter[:,1],scatter[:,2],c=scatter_color[:,:3])
       # plt.show()
        return (efeat[0:3].permute(1,2,0).numpy()+1)/2
        
from train.trainers.visual import get_ball_grid


def peakfilter(img,footprint=5,th=0.5,exborder=True):
    
    if footprint is None:
        footprint=np.ones((3,) * img.ndim, dtype=np.int8)
    elif isinstance(footprint,int):
        footprint=np.ones((footprint,) * img.ndim, dtype=np.int8)
        
    img=img.copy()
    if exborder:
        img[0,...]=0
        img[-1,...]=0
    image2=im_dilation(img,footprint)
    mask=img>=image2
    mask[img<th]=0
    if exborder:
        mask[0,...]=0
        mask[-1,...]=0
    return mask

def get_seeds(dist_map, seeds_thres=0.7, seeds_min_dis=5):
    dist_map = np.squeeze(dist_map)  # (h,w)
    coordinates  = peak_local_max(dist_map, min_distance=seeds_min_dis, threshold_abs=seeds_thres * dist_map.max())
    mask = np.zeros_like(dist_map)
    for coord in coordinates:
        mask[coord[0], coord[1]] = 1.0
    return mask

def remove_noise(l_map, d_map, min_size=10, min_intensity=0.1):
    max_instensity = d_map.max()
    props = regionprops(l_map, intensity_image=d_map)
    for p in props:
        if p.area < min_size:
            l_map[l_map==p.label] = 0
        if p.mean_intensity/max_instensity < min_intensity:
            l_map[l_map==p.label] = 0
    return label(l_map)

def mask_from_seeds(efeat, seeds, similarity_thres):
    """
    :param norm_1_embedding: (h, w, c)
    :param seeds:
    :param similarity_thres:
    :return:
    """
    seeds = label(seeds).astype('uint8')
    props = regionprops(seeds)

    mean = {}
    for p in props:
        row, col = p.coords[:, 0], p.coords[:, 1]

        emb_mean = torch.mean(efeat[row, col], dim=0)
        emb_mean = F.normalize(emb_mean, p=2, dim=0)
        mean[p.label] = emb_mean

    while True:
        dilated = im_dilation(seeds, mor_square(3))
        front_r, front_c = np.nonzero(seeds != dilated)
        #numpy
        if not front_c.any(): break
        similarity = [torch.dot(efeat[r, c, :], mean[dilated[r, c]]) for r, c in zip(front_r, front_c)]
        add_ind = torch.stack([s > similarity_thres for s in similarity], dim=0).cpu().detach().numpy()

        # print(add_ind.sum())
        if np.sum(add_ind) == 0: break

        seeds[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]

    return seeds

def mask_from_seeds_v2(efeat, seeds, similarity_thres, distmap):
    """
    :param norm_1_embedding: (h, w, c)
    :param seeds:
    :param similarity_thres:
    :return:
    """
    seeds = label(seeds).astype('uint8')
    props = regionprops(seeds)

    inverse_distmap = np.ones_like(distmap)* distmap.max() - distmap
    inverse_distmap = np.expand_dims(inverse_distmap,2)

    mean = {}
    for p in props:
        row, col = p.coords[:, 0], p.coords[:, 1]

        key_feat = efeat[row, col] * inverse_distmap[row,col]
        key_feat = torch.mean(key_feat, dim=0)
        emb_mean = F.normalize(key_feat, p=2, dim=0)
        mean[p.label] = emb_mean

    while True:
        dilated = im_dilation(seeds, mor_square(3))
        front_r, front_c = np.nonzero(seeds != dilated)
        #numpy
        similarity = [torch.dot(efeat[r, c, :], mean[dilated[r, c]]) for r, c in zip(front_r, front_c)]

        add_ind = torch.stack([s > similarity_thres for s in similarity], dim=0).cpu().detach().numpy()


        # print(add_ind.sum())
        if np.sum(add_ind) == 0: break

        seeds[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]

    return seeds

def mask_from_seeds_v3(efeat, distmap,similarity_thres,seeds_thres,seeds_min_dis):
    """
    :param norm_1_embedding: (h, w, c)
    :param seeds:
    :param similarity_thres:
    :return:
    """
    seed=peakfilter(distmap,footprint=seeds_min_dis,th=seeds_thres,exborder=True)
    
    # return seed,(distmap>seeds_thres)
    bg=distmap>0.5
    
    bg_emb=efeat[bg]
    bg_emb=torch.mean(bg_emb,dim=0)
    bg_emb=F.normalize(bg_emb, p=2, dim=0)
    
    
    seeds = label(seed).astype('uint8')
    props = regionprops(seeds)

    inverse_distmap = np.ones_like(distmap)* distmap.max() - distmap
    inverse_distmap = np.expand_dims(inverse_distmap,2)

    mean = {}
    for p in props:
        row, col = p.coords[:, 0], p.coords[:, 1]

        key_feat = efeat[row, col] #* inverse_distmap[row,col]
        key_feat = torch.mean(key_feat, dim=0)
        emb_mean = F.normalize(key_feat, p=2, dim=0)
        mean[p.label] = emb_mean
    cnt =15
    for i in range(cnt):
        dilated = im_dilation(seeds, mor_square(3))
        front_r, front_c = np.nonzero(seeds != dilated)
        if not front_c.any(): break
    
        #numpy
        
        #mean_f=torch.mean(efeat[front_r,front_c],dim=0)
        #si_bg=torch.dot(mean_f,bg_emb)
        
        #if si_bg>similarity_thres: brea 
        # bg_similarities = [torch.dot(efeat[r, c, :], bg_emb) for r, c in zip(front_r, front_c)]
        similarity = [torch.dot(efeat[r, c, :], mean[dilated[r, c]]) for r, c in zip(front_r, front_c)]
        bg_ind=[bg[ r, c] for r, c in zip(front_r, front_c)]

        add_ind = torch.stack([s > similarity_thres for s in similarity], dim=0).cpu().detach().numpy()
        # add_ind_bg = torch.stack([ s < 0.5 for s in bg_similarities], dim=0).cpu().detach().numpy()
        bg_ind=np.array(bg_ind)
        add_ind=add_ind*bg_ind
        #add_ind=add_ind*add_ind_bg
        # print(add_ind.sum())
        if np.sum(add_ind) == 0: break
        
         

        seeds[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]

    return seed,seeds







# u=UNet(3).to('cuda')
# summary(u,(3,256,256))