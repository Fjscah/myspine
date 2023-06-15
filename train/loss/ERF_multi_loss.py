"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import math

import numpy as np

import torch
import torch.nn as nn
from .lovasz_losses import lovasz_hinge
from .loss import FocalLoss

class multi_SpatialEmbLoss(nn.Module):

    def __init__(self, to_center=True, n_sigma=1, foreground_weight=1, width=256,height=256,num_class=3,resolusion=256):
        super().__init__()

        print('Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}'.format(
            to_center, n_sigma, foreground_weight))

        self.to_center = to_center
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight
        self.classnum=num_class
        self.resolusion=resolusion
        # pixel coordinate map for 512x512
        self.assign_xym((height,width))

        
        self.B_loss=FocalLoss("multiclass",2)
    def assign_xym(self,shape):
        print("============= assign loss xym =============",shape,self.resolusion)
        ym = torch.linspace(0, shape[0]//self.resolusion, shape[0]).view(1, -1, 1).expand(1, shape[0], shape[1])
        xm = torch.linspace(0, shape[1]//self.resolusion, shape[1]).view(1, 1, -1).expand(1, shape[0], shape[1])
        xym = torch.cat((xm, ym), 0)
        self.register_buffer("xym", xym)
    def forward(self, prediction, labels,instances,  w_inst=1, w_var=10, w_seed=1, iou=False, iou_meter=None):
        '''
        prediction : model output (N, 4, h, w)
        instances : GT instance-mask (N, 1,h, w)
        labels : GT semantic-mask (N,nc, h, w)
        cn is class_number
        '''
        #print(prediction.shape,instances.shape,labels.shape)
        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        xym_s = self.xym[:, 0:height, 0:width].contiguous()  # (2, h, w)

        loss = 0

        
        binary_class_map=prediction[:, 2 + self.n_sigma    :  2 + self.n_sigma + self.classnum]
        binary_closs=self.B_loss(binary_class_map,labels[:,:])
        # binary_class_map=torch.sigmoid(binary_class_map)
        # binary_closs=torch.mean(
        #         torch.pow(binary_class_map - labels[b,:-1], 2))
        # muiti class -loss -semantic
        
        for b in range(0, batch_size):

            ### Instance-branch ###
            # pixel-embeddings = offset vectors + coordinate vectors
            spatial_emb = torch.tanh(prediction[b, 0:2]) + xym_s  # (2, h, w)
            # sigma value for each pixels (large simga -> bigger margin)
            sigma = prediction[b, 2:2 + self.n_sigma]  # (n_sigma, h, w)

            ### Seed-branch ###
            # seed map for each semantic classes
            # pixel-embeddings who lay close to their intance center -> high seed score
            seed_map = prediction[b, 2 + self.n_sigma+ self.classnum-1    :  2 + self.n_sigma + self.classnum]  # (cn, h, w), cn is class_number

            #seed_map=seed_maps[-1:]
            # binary_class_map=seed_maps[:-1]
            
            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0
            seed_map=torch.sigmoid(seed_map)
            
            
            
            

            instance = instances[b]  # (1, h, w)
            label = labels[b,-1:]  # (1, h, w)
            

            # instance lists except bg
            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # seed loss 1 - bg pixels are regressed to 0
            no_single_mask = (label ==0)
            if no_single_mask.sum() > 0:
                seed_loss += torch.sum(
                    torch.pow(seed_map[no_single_mask] - 0, 2))

            # for each instance
            for id in instance_ids:

                # mask of specific instance k (RoI pixels of instance k)
                # spatial_emb, sigma, seed_map below only consider the pixels of RoI k
                in_mask = instance.eq(id)  # (1, h, w)

                # calculate center of attraction
                # if to_center=True -> mean of x, y coordinate maps which belong to instance pixels
                if self.to_center:
                    xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
                    center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1

                # defining center as the mean over the pixel-embeddings of instance k
                else:
                    center = spatial_emb[in_mask.expand_as(spatial_emb)].view(
                        2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1
                #print(center)
                # calculate sigma belonging to specific instance
                sigma_in = sigma[in_mask.expand_as(
                    sigma)].view(self.n_sigma, -1) # n_sigma=1 x (h x w)

                # sigma_k = avg of all sigma_i belongs to instance k
                s = sigma_in.mean(1).view(
                    self.n_sigma, 1, 1)  # n_sigma x 1 x 1
        
                # calculate var loss before exp - e.q (5)
                var_loss = var_loss + \
                           torch.mean(
                               torch.pow(sigma_in - s[..., 0].detach(), 2))

                s = torch.exp(s * 10)  # https://github.com/davyneven/SpatialEmbeddings/issues/4
     
                # calculate gaussian output - e.q (9)
                # it outputs a fg/bg probability map
                # high value near the instance center, low value at the boundary area
                dist = torch.exp(-1 * torch.sum(
                    torch.pow(spatial_emb - center, 2) * s, 0, keepdim=True))  # (1, h, w), 1 for specific instance k

                # apply lovasz-hinge loss
                # it optimizes the i.o.u of each instance
                instance_loss = instance_loss + \
                                lovasz_hinge(dist * 2 - 1, (in_mask * 1).cuda())  # *1 to convert boolean to int

                # seed loss - e.q (10)
                # fg pixels of seed map are regressed to the gaussian output of pixel embedding
                seed_loss += self.foreground_weight * torch.sum(
                    torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2))

                # calculate instance iou
                if iou:
                    iou_meter.update(calculate_iou(dist > 0.5, in_mask))

                obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / (height * width)
            loss += w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss

 
       
        loss = loss / (b + 1)+binary_closs

        return (loss + prediction.sum() * 0).mean()
#F.binary_cross_entropy_with_logits(output, target, reduction="none")
def calculate_iou(pred, label):
    pred=pred.cuda()
    label=label.cuda()
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou
