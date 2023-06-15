from typing import Any, List
from .base_cluster import BaseCluster
import numpy as np
import torch
from skimage.measure import regionprops, label
from skimage.feature import peak_local_max
from skimage.morphology import dilation as im_dilation
from skimage.morphology import square as mor_square
from PIL import Image
import math
import torch.nn.functional as F

class WnetCluster(BaseCluster):
    def __init__(self,seeds_thres=0.7,seeds_min_dis=5,similarity_thres=0.7) -> None:
        super().__init__()
        self.seeds_thres=seeds_thres
        self.seeds_min_dis=seeds_min_dis
        self.similarity_thres=similarity_thres
        
    def get_prediction(self, distmap, efeat):
        efeat = efeat.permute((1,2,0)) #(h,w,c)
        seeds, seg=mask_from_seeds_v3(efeat, distmap,self.similarity_thres,self.seeds_thres, self.seeds_min_dis)
        # seeds=peakfilter(distmap,footprint=self.seeds_min_dis,th=self.seeds_thres,exborder=True)
        # labels=distmap>self.seeds_thres
        # seeds = get_seeds(distmap, self.seeds_thres, self.seeds_min_dis)
        #labels,num=label(seeds,connectivity=2,return_num=True)
        #print(num,"++++++++++++++++++++++")
        #return seeds, labels
        # seg = mask_from_seeds_v3(efeat, seeds, self.similarity_thres,self.seeds_thres, self.seeds_min_dis)
     
        seg = remove_noise(seg, distmap)

        return seeds, seg

    def get_prediction_v2(self, distmap, efeat):
        
        seeds = get_seeds(distmap, self.seeds_thres, self.seeds_min_dis)

        efeat = efeat.permute((1,2,0)) #(h,w,c)
        seg = mask_from_seeds_v2(efeat, seeds, self.similarity_thres, distmap)
        seg = remove_noise(seg, distmap)

        return seeds, seg
    
    def get_visual_keys(self) -> List[str]:
        return ["image","seed1","seed2","mask","label","GT"]
    def ins_cluster(self, model, img) -> Any:
        im=self.valid_im(img).to(model.cur_device)
        pd_distmap, efeat = model(im)
        pd_distmap=torch.softmax(pd_distmap,dim=1)
        pd_seeds, pd_label = self.get_prediction(pd_distmap[0,-1].cpu().detach().numpy(),
                                                    efeat[0].cpu().detach())
        return pd_label,[pd_distmap, efeat]
    def show_result(self, visualizer, visual_result):
        img,GT,ins,output=visual_result
        pd_distmap, efeat=output
        visualizer.display(img.cpu(), 'image')
        visualizer.display(GT, 'GT')  # it shows prediction / GT-mask
        visualizer.display(ins, 'label')  # it shows prediction / GT-mask
        # efeat=torch.abs(efeat)
        efeat=(efeat+1)/2
        visualizer.display(efeat[0,:3].cpu(), 'seed1')  # seed map
        visualizer.display(efeat[0,3:6].cpu(), 'seed2')  # seed map
        visualizer.display(pd_distmap[0,:3].cpu(), 'mask')  # seed map
        visualizer.save()
    
    
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

        key_feat = efeat[row, col] * inverse_distmap[row,col]
        key_feat = torch.mean(key_feat, dim=0)
        emb_mean = F.normalize(key_feat, p=2, dim=0)
        mean[p.label] = emb_mean
    cnt =20
    for i in range(cnt):
        dilated = im_dilation(seeds, mor_square(3))
        front_r, front_c = np.nonzero(seeds != dilated)
        if not front_c.any(): break
    
        #numpy
        
        #mean_f=torch.mean(efeat[front_r,front_c],dim=0)
        #si_bg=torch.dot(mean_f,bg_emb)
        
        #if si_bg>similarity_thres: brea 
        bg_similarities = [torch.dot(efeat[r, c, :], bg_emb) for r, c in zip(front_r, front_c)]
        similarity = [torch.dot(efeat[r, c, :], mean[dilated[r, c]]) for r, c in zip(front_r, front_c)]
        bg_ind=[bg[ r, c] for r, c in zip(front_r, front_c)]

        add_ind = torch.stack([s > similarity_thres for s in similarity], dim=0).cpu().detach().numpy()
        add_ind_bg = torch.stack([ s < 0.5 for s in bg_similarities], dim=0).cpu().detach().numpy()
        bg_ind=np.array(bg_ind)
        add_ind=add_ind*bg_ind
        #add_ind=add_ind*add_ind_bg
        # print(add_ind.sum())
        if np.sum(add_ind) == 0: break
        
         

        seeds[front_r[add_ind], front_c[add_ind]] = dilated[front_r[add_ind], front_c[add_ind]]

    return seed,seeds