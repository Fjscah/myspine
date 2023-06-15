
from csbdeep.utils import normalize
import sympy
import torch
import numpy as np
from skimage.transform import resize,rescale
from skimage.util.shape import view_as_windows,view_as_blocks
from skimage.util import montage
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects
import colorcet as cc

from spinelib.seg.seg_base import resortseg
from .base_cluster import BaseCluster
from train.trainers.visual import Visualizer
class ERFbase_Cluster(BaseCluster):
    def __init__(self,device="cuda",resolusion=256,th=0.2,size_th=30) -> None:
        self.device=device
        self.resolusion=resolusion
        self.th=th 
        self.size_th=size_th
        pass

    def valid_im(self,im):
        if isinstance(im,np.ndarray):
            im=torch.from_numpy(im)
        if im.dim()==2:
            im=im.unsqueeze(0)
        if im.dim()==3:
            im=im.unsqueeze(0)
        return im
    
    def assign_xym(self,shape):
        print("============= assign xym =============")
        ym = torch.linspace(0, shape[0]//self.resolusion, shape[0]).view(1, -1, 1).expand(1, shape[0], shape[1])
        xm = torch.linspace(0, shape[1]//self.resolusion, shape[1]).view(1, 1, -1).expand(1, shape[0], shape[1])
        xym = torch.cat((xm, ym), 0)
        self.xym = xym.to(self.device)
    
    def check_shape_and_assign_xym(self,shape):
        if shape[0]>self.xym.shape[1] or shape[1]>self.xym.shape[2]:
            self.assign_xym(shape)
    def ins_cluster(self,model,im):
        #image=image.astype("float32") # H W,[C]
        #im=ToTensor()(image) # C H W
        # im=im.expand(1,256,256,1)
        
        im=self.valid_im(im)
       

        im=im.to(model.cur_device)
       
        ypred=model.forward(im)
        
        
        instance=self.cluster(ypred[0], n_sigma=1, threshold=self.th,size_th=self.size_th)[0]
        return instance.squeeze().cpu().detach().numpy(),ypred

    def ins_cluster_with_gt(self,model, im, instance, n_sigma=1):
        
        im=self.valid_im(im)
        prediction=model.forward(im)
        instance=self.cluster_with_gt(prediction[0], instance, n_sigma=n_sigma)
        return instance.squeeze().cpu().detach().numpy(),prediction
    def predict(self,model,im):
        # im=rescale(im,2)
        im=normalize(im,1,99.8)
        
        self.assign_xym(im.shape)
      
        mask,_=self.ins_cluster(model,im)
        mask=remove_small_objects(mask,10)
        # patch_shape=256,256
        # step=256-32
        # patch_shape2=im.shape[0]//(step),im.shape[1]//(step)
        # # -- filterbank1 on original image
        # print(patch_shape2)
        # mask=np.zeros(im.shape[:2],dtype="int16")
        # st=1
        # for i in range(patch_shape2[0]):
        #     for j in range(patch_shape2[1]):
        #         patchi=im[i*step:i*step+256,j*step:j*step+256]
        #         patch_ins,_=self.ins_cluster(model,patchi,th=0.6,size_th=50)
        #         patch_ins=clear_border(patch_ins)
        #         patch_ins,st=resortseg(patch_ins,st)
        #         Boldmask=mask[i*step:i*step+256,j*step:j*step+256]>0
        #         Nmaske=patch_ins>0
        #         intermask=np.logical_and(Boldmask,Nmaske)
        #         intermask=remove_small_objects(intermask,10)
        #         labs=np.unique(patch_ins[intermask])
        #         for lab in labs:
        #             if lab>1:
        #                 patch_ins[patch_ins==lab]=0
        #         # patch_ins[0:10]+=1
        #         # patch_ins[-10:]+=1
        #         # # patch_ins[:,0:10]+=1
        #         # patch_ins[:,-10:]+=1
        #         # print(patch_ins.shape)
        #         mask[i*step:i*step+256,j*step:j*step+256][patch_ins>0]=patch_ins[patch_ins>0]
         
       
        
        #pred_ins,_=self.ins_cluster(model,im,th=0.5,size_th=30)
        
       # pred_ins=rescale(pred_ins,0.5,preserve_range=True,anti_aliasing=False)
        return im,mask
    def get_visual_keys(self):
        return ["image","feat","seed","label","GT"]
    def show_result(self, visualizer:Visualizer, visual_result):
        img,GT,ins,output=visual_result
        print(visualizer.keys)
        visualizer.display(img.cpu(), 'image')
        visualizer.display(GT, 'GT')  # it shows prediction / GT-mask
        visualizer.display(ins, 'label')  # it shows prediction / GT-mask
        
        feat = torch.abs(output[0][0:3]).cpu()
        sigma = output[0][2].cpu()
        sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
        # feat[-1]=sigma
        visualizer.display(feat, 'feat')  # sigma map
        seed = torch.sigmoid(output[0][3:]).cpu()
        visualizer.display(seed, 'seed')  # seed map
        #plt.show()
        visualizer.save()
       
class single_Cluster(ERFbase_Cluster):

    def __init__(self, width=512,height=512,resolusion=256,device="cuda"):
        super().__init__(device,resolusion)
        
        self.assign_xym((height,width))


    def cluster_with_gt(self, prediction, instance, n_sigma=1, ):
        '''cluster the pixel embeddings into instance (training)'''

        height, width = prediction.size(1), prediction.size(2)
        self.check_shape_and_assign_xym((height,width))
        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w

        # spatial_emb = offset_vectors (first 2 channels of model output) + coordinate maps
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(height, width).byte().to(self.device)

        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]

        # for each specific instance
        for id in unique_instances:
            # mask of specific instance (RoI area)
            # spatial_emb, sigma below consider only pixels of RoI area
            mask = instance.eq(id).view(1, height, width)  # (1, h, w)

            # center of instance (mean embedding)
            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

            # define sigma_k - e.q (7)
            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)
            s = torch.exp(s * 10)  # n_sigma x 1 x 1

            # calculate gaussian score (distance of each pixel embedding from the center)
            # high value -> pixel embedding is close to the center
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))  # (h, w)

            # e.q (11)
            proposal = (dist > 0.5)  # (h, w)
            instance_map[proposal] = id  # (h, w)

        return instance_map

    def cluster(self, prediction, n_sigma=1, threshold=0.5,size_th=128):
        '''for inference'''
   
        height, width = prediction.size(1), prediction.size(2)
        self.check_shape_and_assign_xym((height,width))
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w

        # sigmoid is applied to seed_map
        # as the regression loss is used for training, b.g pixels become zero
        # it is similar to semantic mask
        seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + 1])  # 1 x h x w

        instance_map = torch.zeros(height, width).byte()
        instances = []

        count = 1

        # RoI pixels - set of pixels belonging to instances
        mask = seed_map > 0.5  # (1, h, w)
        #print("mask sum",mask.sum() )
        if mask.sum() > size_th:

            # only consider the pixels which belong to mask (RoI pixels)
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().to(self.device)
            instance_map_masked = torch.zeros(mask.sum()).byte().to(self.device)

            while (unclustered.sum() > size_th):
                #print("uncluster sum",unclustered.sum())
                # At inference time, we select a pixel embedding with a high seed score as a center
                # embedding with the highest seed score is close to instance's center
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()

                # In order for the embedding of a particular pixel to be a center,
                # the seed score at that pixel must be greater than 0.9 (to prevent all pixels from becoming centers)
                if seed_score < threshold:  # ths=0.9
                    break

                # define instance center (embedding with the highest seed score & > ths)
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)  # accompanying sigma (instance specific margin)

                # calculate gaussian output - e.q (5)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                          center, 2) * s, 0, keepdim=True))

                # e.q (11)
                proposal = (dist > 0.5).squeeze()

                # mask out all clustered pixels in the seed map, until all seeds are masked
                if proposal.sum() > size_th:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).byte()
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu().byte()
                        instances.append(
                            {'mask': instance_mask.squeeze() * 255, 'score': seed_score})
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map, instances, mask


class multi_Cluster(ERFbase_Cluster):

    def __init__(self, width=256,height=256,num_class=3,resolusion=256,device="cuda"):
        super().__init__(device,resolusion)

        self.assign_xym((height,width))
        self.num_class=num_class
        
        

    def cluster_with_gt(self, prediction, instance, n_sigma=1, ):
        """_summary_

        Args:
            prediction (_type_): (nc1+nc2),H,W , nc2=self.num_class
            instance (_type_): H,W
            n_sigma (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        '''cluster the pixel embeddings into instance (training)'''


        height, width = prediction.size(1), prediction.size(2)
        self.check_shape_and_assign_xym((height,width))
        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w

        # spatial_emb = offset_vectors (first 2 channels of model output) + coordinate maps
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(height, width).byte().to(self.device)

        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]

        # for each specific instance
        for id in unique_instances:
            # mask of specific instance (RoI area)
            # spatial_emb, sigma below consider only pixels of RoI area
            mask = instance.eq(id).view(1, height, width)  # (1, h, w)

            # center of instance (mean embedding)
            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

            # define sigma_k - e.q (7)
            s = sigma[mask.expand_as(sigma)].view(n_sigma, -1).mean(1).view(n_sigma, 1, 1)
            s = torch.exp(s * 10)  # n_sigma x 1 x 1

            # calculate gaussian score (distance of each pixel embedding from the center)
            # high value -> pixel embedding is close to the center
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))  # (h, w)

            # e.q (11)
            proposal = (dist > 0.5)  # (h, w)
            instance_map[proposal] = id  # (h, w)

        return instance_map

    def cluster(self, prediction, n_sigma=1, threshold=0.5,size_th=128):
        '''for inference, prediction (_type_): (nc1+nc2),H,W , nc2=self.num_class'''
   
        height, width = prediction.size(1), prediction.size(2)
        self.check_shape_and_assign_xym((height,width))
        xym_s = self.xym[:, 0:height, 0:width]
       
        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + n_sigma]  # n_sigma x h x w

        # sigmoid is applied to seed_map
        # as the regression loss is used for training, b.g pixels become zero
        # it is similar to semantic mask
        seed_maps = torch.sigmoid(
                prediction[2 + n_sigma:2 + n_sigma + self.num_class])  # (cn, h, w), cn is class_number

        seed_map=seed_maps[-1:]
        binary_class_map=seed_maps[0:-1]
        bg_mask=(binary_class_map[0]>0.5)
        # seed_map = torch.sigmoid(prediction[2 + n_sigma:2 + n_sigma + self.num_class])  # 1 x h x w

        instance_map = torch.zeros(height, width).byte()
        instances = []

        count = 1

        # RoI pixels - set of pixels belonging to instances
        mask = seed_map > 0.5  # (1, h, w)
        #print("mask sum",mask.sum() )
        if mask.sum() > size_th:

            # only consider the pixels which belong to mask (RoI pixels)
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).byte().to(self.device)
            instance_map_masked = torch.zeros(mask.sum()).byte().to(self.device)
            
            while (unclustered.sum() > size_th):
                #print("uncluster sum",unclustered.sum())
                # At inference time, we select a pixel embedding with a high seed score as a center
                # embedding with the highest seed score is close to instance's center
               
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()

                # In order for the embedding of a particular pixel to be a center,
                # the seed score at that pixel must be greater than 0.9 (to prevent all pixels from becoming centers)
                if seed_score < threshold:  # ths=0.9
                    break

                # define instance center (embedding with the highest seed score & > ths)
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)  # accompanying sigma (instance specific margin)

                # calculate gaussian output - e.q (5)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                          center, 2) * s, 0, keepdim=True))

                # e.q (11)
                proposal = (dist > 0.5).squeeze()

                # mask out all clustered pixels in the seed map, until all seeds are masked
                if proposal.sum() > size_th:
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).byte()
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu().byte()
                        instances.append(
                            {'mask': instance_mask.squeeze() * 255, 'score': seed_score})
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()
        #instance_map[bg_mask]=0
        return instance_map, instances, mask












