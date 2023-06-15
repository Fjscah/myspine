import math



import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import FocalLoss
from skimage.morphology import dilation, disk,square,binary_dilation
from torchvision.ops import masks_to_boxes



class W_SpatialEmbLoss(nn.Module):
    def __init__(self, radius=10,w_inst=1, w_var=10, w_seed=1,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.B_loss=FocalLoss("multiclass",2)
        self.mse = torch.nn.MSELoss(reduction='sum')
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        self.footprint=disk(radius)
        self.radius=radius
        self.w_inst=w_inst
        self.w_var=w_var
        self.w_seed=w_seed
    def forward(self, prediction, labels,instances):
        """retrun wnet loss, includer dismap loss+ inter loss+intra loss+seed loss

        Args:
            prediction (Tensor,Tensor): dfeat(seed) :B,nc,H,W ; efeat :B,emb_c,H,W
            labels (_type_): _description_
            instances (_type_): _description_
            w_inst (int, optional): _description_. Defaults to 1.
            w_var (int, optional): _description_. Defaults to 10.
            w_seed (int, optional): _description_. Defaults to 1.
        """
        seed_maps,emb_maps=prediction
        #print(seed_maps.shape,emb_maps.shape,labels.shape,instances.shape)
        binary_closs=self.B_loss(seed_maps,labels)
        # instances=instances.detach()
        #return binary_closs
        batch_size, num_class,height, width = seed_maps.shape
        
        #-----------------------#
        #   get intra and inter loss  #
        #-----------------------#
        loss_intras=0
        loss_inters=0
        
        for b in range(batch_size):
            instance=instances[b,0]
            emd_map=emb_maps[b]
            loss_intra, loss_inter=self.losseach(emd_map,instance)
            
            loss_inters+=loss_inter
            loss_intras+=loss_intra
        
        loss_intras=loss_intras/batch_size
        loss_inters=loss_inters/batch_size
        #print(loss_intras,loss_inters,binary_closs)
        return binary_closs*self.w_seed+loss_inters*self.w_inst+loss_intras*self.w_var
        
    
    
    def losseach(self,emb_map,instance):
      
        c, h, w = emb_map.shape
        efeat = emb_map.permute(1,2,0).reshape(-1, c) #(h*w, c)
        n_ins = instance.max() + 1 
        key_features = []

        
        neighbor=torch.zeros(n_ins, 10, dtype=torch.int,device=emb_map.device) # include bg
        for l in range(0, n_ins):
            #mean_feat
            fg = (instance == l).reshape(-1)
            mean_feat = efeat[fg, :]
            # label's each pixel avg embedd vector
            mean_feat = torch.sum(mean_feat, dim=0) / fg.sum()
            key_features.append(mean_feat)
            
            if l==0:continue
            y, x = torch.where(instance == l)
            bounding_boxes = torch.zeros((4,), dtype=torch.int)
            bounding_boxes[ 0] = torch.min(x)
            bounding_boxes[1] = torch.min(y)
            bounding_boxes[2] = torch.max(x)
            bounding_boxes[3] = torch.max(y)
            
        
            x1, y1, x2, y2=bounding_boxes
            x1= x1-self.radius if x1>self.radius else 0
            x2= x2+self.radius if x2<w-self.radius else w
            y1= y1-self.radius if y1>self.radius else 0
            y2= y2+self.radius if y2<h-self.radius else h
        
            nei_labs,_=torch.unique(instance[y1:y2,x1:x2]).sort(descending=True) 
      
            len_nei=len(nei_labs)
            neighbor[l][:min(len_nei,10)] = nei_labs[:10]

        #Intra Loss
        key_features = torch.stack(key_features, dim=0) #(n_ins, c)
        key_features = F.normalize(key_features, p=2, dim=1) # normalize channel

        # labelmap_flat = label.flatten().long()
        # key_features_expand = key_features[labelmap_flat]
        # loss_intra_a = 1 - self.cosine_similarity(key_features_expand, efeat)
        # loss_intra_a_m = loss_intra_a.mean()


        instance = instance.reshape(h*w, 1).expand(h*w, c)
        key_feat_each_pixel = torch.gather(key_features, dim=0, index=instance) #(h*w, c)
        loss_intra = 1-self.cosine_similarity(key_feat_each_pixel, efeat)
        #print(loss_intra.min(),loss_intra.max())
        loss_intra=torch.exp(loss_intra)-1
        loss_intra = loss_intra.mean()



        #Inter Loss
        key_feat_1 = key_features.unsqueeze(1).expand(n_ins, n_ins, c)
        key_feat_2 = key_feat_1.clone().permute(1,0,2)
        key_feat_1 = key_feat_1.reshape(-1, c)
        key_feat_2 = key_feat_2.reshape(-1, c)

        loss_inter = self.cosine_similarity(key_feat_1, key_feat_2).reshape(n_ins, n_ins).abs() #(n_key, n_key)
        # loss_inter=torch.exp(loss_inter)-1
        neighbor_mask = torch.zeros(n_ins, n_ins, dtype=torch.float).to(emb_map.device)
        for label_main in range(1, n_ins):
            for label_neighbor in neighbor[label_main]:
                if label_neighbor == 0: break

                neighbor_mask[label_main][label_neighbor] = 1.0

        neighbor_mask[0, :] = .5
        neighbor_mask[:, 0] = .5
        neighbor_mask[0,0] = 0.
        if neighbor_mask.sum()>0:
            loss_inter=torch.exp(loss_inter)-1
            loss_inter = (loss_inter * neighbor_mask).sum() / neighbor_mask.sum()
            # loss_inter = (loss_inter * neighbor_mask).sum(dim=1) / neighbor_mask.sum(dim=1)
            loss_inter = loss_inter.sum()
            
        else:
            loss_inter=0

        return loss_intra, loss_inter
            
                
        
        
        
        
        
        
        