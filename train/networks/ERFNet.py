"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
forked from : https://github.com/davyneven/SpatialEmbeddings
"""
# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

from .BaseNet import *

class DownsamplerBlock (nn.Module):
    '''Downsampling by concatenating parallel output of
    3x3 conv(stride=2) & max-pooling'''
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput,
                              (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d (nn.Module):
    '''Factorized residual layer
    dilation can gather more context (a.k.a. atrous conv)'''
    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        
        # factorize 3x3 conv to (3x1) x (1x3) x (3x1) x (1x3)
        # non-linearity can be added to each decomposed 1D filters
        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)  # non-linearity

        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)

        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes,inputchannel=3):
        # num_classes = output channels
        super().__init__()
        self.initial_block = DownsamplerBlock(inputchannel, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))  # no dilation

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times (with dilation)
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(
            128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:  # not used in this code
            output = self.output_conv(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        # do not use max-unpooling operation
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder (nn.Module):
    '''small (not symmetric) decoder that upsamples encoder's output
    by fine-tuning the details'''
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)
        output = self.output_conv(output)

        return output



class Net(nn.Module):
    '''ERFNet'''
    def __init__(self, num_classes, encoder=None):  # use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)  # Encoder(3+1) in this code
        else:
            self.encoder = encoder

        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)  # predict=False by default
            return self.decoder.forward(output)





class BranchedERFNet(BaseNet):
    '''shared encoder + 2 branched decoders'''
    def __init__(self, **kwargs):
                 #num_classes=3, encoder=None,inputchannel=3,# net paras
                 #resolusion=256,th=0.2,size_th=10,  width=512,height=512, #  cluster pares, 
                 #width and height is just for ganerate small coords, which should be large than traing img size
                 #):
        super(BranchedERFNet, self).__init__(**kwargs)
    
        self.resolusion=kwargs["resolusion"]
        self.th=kwargs["th"]
        self.size_th=kwargs["size_th"]
        self.num_classes = kwargs["num_classes"]
        self.num_class=self.num_classes[1]
        self.inputchannel=kwargs["inputchannel"]
        height=kwargs["height"]
        width=kwargs["width"]
        encoder=None
        print('Creating branched erfnet with {} classes'.format(self.num_classes))
        # num_classes=[3, n], 3 for instance-branch (1 sigma) & n for seed branch, 
        # num_classes=[4, n], 4 for instance-branch (2 sigma) & n for seed branch
        # if only spne n=1, if spine+den+bg n=3
        # encoder is downsample network
        # decoder is updample network
        
        # shared encoder
        if (encoder is None):
            self.encoder = Encoder(sum(self.num_classes),inputchannel=self.inputchannel)  # Encoder(3+1)
        else:
            self.encoder = encoder
        
        # decoder for 2 branches (instance & seed)
        # Decoder(3) for instance branch & Decoder(1) for seed branch
        self.decoders = nn.ModuleList()
        for n in self.num_classes:
            self.decoders.append(Decoder(n))
        self.init_output()
        self.assign_xym((height,width))
        # self.cluster=Cluster()
    def assign_xym(self,shape):
        print("============= assign xym =============")
        ym = torch.linspace(0, shape[0]//self.resolusion, shape[0]).view(1, -1, 1).expand(1, shape[0], shape[1])
        xm = torch.linspace(0, shape[1]//self.resolusion, shape[1]).view(1, 1, -1).expand(1, shape[0], shape[1])
        xym = torch.cat((xm, ym), 0)
        self.xym = xym.to(self.cur_device)
    def check_shape_and_assign_xym(self,shape):
        if shape[0]>self.xym.shape[1] or shape[1]>self.xym.shape[2]:
            self.assign_xym(shape)
    def init_output(self, n_sigma=1):
        # if sum(self.num_classes) == 4:
        #     n_sigma = 1  # 1 sigma for circular margin
        # else:
        #     n_sigma = 2  # 2 sigma for elliptical margin

        with torch.no_grad():
            output_conv = self.decoders[0].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2+n_sigma, :, :].fill_(0)
            output_conv.bias[2:2+n_sigma].fill_(1)

    def forward(self, input, only_encode=False):
        if only_encode:  # not used in this code
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)  # (N, 128, h/8, w/8),L3
        
        # concat (N, 3, h, w) & (N, n, h, w)
        return torch.cat([decoder.forward(output) for decoder in self.decoders], 1)

    
    #-----------------------#
    #   Predict  #
    #-----------------------#
    
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

        instance_map = torch.zeros(height, width).byte().to(self.cur_device)

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

        instance_map = torch.zeros(height, width,dtype=torch.int16)
        instances = []

        count = 1

        # RoI pixels - set of pixels belonging to instances
        mask = seed_map > 0.5  # (1, h, w)
        peak=seed_map>=peakfilter(seed_map)
        #print("mask sum",mask.sum() )
        if mask.sum() > size_th:

            # only consider the pixels which belong to mask (RoI pixels)
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)
            peak_masked=peak[mask].view(1,-1).float()
            unclustered = torch.ones(mask.sum()).byte().to(self.cur_device)
            
            instance_map_masked = torch.zeros(mask.sum(),dtype=torch.int16).to(self.cur_device)
            
            while (unclustered.sum() > size_th):
                #print("uncluster sum",unclustered.sum())
                # At inference time, we select a pixel embedding with a high seed score as a center
                # embedding with the highest seed score is close to instance's center
               
                seed = (seed_map_masked * unclustered.float()*peak_masked).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()*peak_masked).max().item()

                # In order for the embedding of a particular pixel to be a center,
                # the seed score at that pixel must be greater than 0.9 (to prevent all pixels from becoming centers)
                if seed_score < threshold:  # ths=0.9
                    break

                # define instance center (embedding with the highest seed score & > ths)
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                # birger object has smaller sigma, whch will get larger dist score
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
    
    def predict(self,im):
        #image=image.astype("float32") # H W,[C]
        #im=ToTensor()(image) # C H W
        # im=im.expand(1,256,256,1)
        
        im=self.valid_im(im)
        
        ypred=self.forward(im)
        
        
        instance=self.cluster(ypred[0], n_sigma=1, threshold=self.th,size_th=self.size_th)[0]
        return instance.squeeze().cpu().detach().numpy(),ypred

    def ins_cluster_with_gt(self, im, instance, n_sigma=1):
        
        im=self.valid_im(im)
        prediction=self.forward(im)
        instance=self.cluster_with_gt(prediction[0], instance, n_sigma=n_sigma)
        return instance.squeeze().cpu().detach().numpy(),prediction
    def off_predict(self,im):
        # im=rescale(im,2)
        im=normalize(im,1,99.8)
        
        self.assign_xym(im.shape)
      
        ins,ypred=self.predict(im)
        ins=remove_small_objects(ins,10)
        prob=ypred[:,-3:].squeeze().cpu().detach().numpy() # 3 H W
        mask=np.argmax(prob,axis=0)
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
        return ins,prob,mask
    def get_visual_keys(self):
        return ["image","angle","label","GT","feat0","feat1","sigma","mask"]
    def show_result(self, visualizer:Visualizer, visual_result):
        img,GT,ins,output=visual_result
        #print(visualizer.keys)
        visualizer.display(img.cpu(), 'image')
        GT=label2rgb(GT[0].cpu().numpy(),image=img.cpu().numpy(),bg_label=0,alpha=0.4)
        ins=label2rgb(ins,image=img.cpu().numpy(),bg_label=0,alpha=0.4)
        visualizer.display(GT, 'GT')  # it shows prediction / GT-mask
        
        visualizer.display(ins, 'label')  # it shows prediction / GT-mask
        
        feat = output[0][0:2].cpu().numpy()
        
        from csbdeep.utils import normalize
        from matplotlib.colors import hsv_to_rgb
        
        feat0,feat1=feat
        feat0=normalize(feat0,1,99.8)-0.5
        feat1=normalize(feat1,1,99.8)-0.5
        
        height, width = feat.shape[1], feat.shape[2]
        xym_s = self.xym[:, 0:height, 0:width]
        # feat = torch.tanh(output[0][0:2])   # 2 x h x w
        # feat=feat.cpu().numpy()
        sigma = output[0][2].cpu()
        sigma=torch.sigmoid(sigma).cpu().numpy()
        sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
        # feat[-1]=sigma
        seed = torch.sigmoid(output[0][3:]).cpu()
        
        ang=np.angle(feat0 + (feat1) * 1j,deg=True)
        # feat[2]=0
        # feat=(feat-feat.min())/(feat.max()-feat.min())
        m=seed[2].numpy()
        print(np.max(feat0),np.max(feat1))
        print(np.min(feat0),np.min(feat1))
        print(np.min(ang),np.max(ang))
        
        hue=(ang+180)/360
        satuation=seed[2]
        value=seed[2]
        hsv=np.stack([hue,satuation,value],axis=-1)
        rgb=hsv_to_rgb(hsv)
        
        
        ang[m<0.5]=np.nan
        #rgb[m<0.5,:]=np.nan
        visualizer.display(seed, 'mask')  # seed map
        visualizer.display(rgb, 'angle')  # sigma map
        visualizer.display(feat0, 'feat0')  # sigma map
        feat=np.array([feat0,feat1,sigma-0.5])+0.5
        # feat[2]=0
        visualizer.display(feat1, 'feat1')  # sigma map
        sigma[m<0.5]=np.nan
        visualizer.display(sigma, 'sigma')  # sigma map
        
        #plt.show()
        visualizer.save()
    # def ins_cluster(self,im):
    #     #image=image.astype("float32") # H W,[C]
    #     #im=ToTensor()(image) # C H W
    #     # im=im.expand(1,256,256,1)

    #     im=im.to(self.cur_device)
    #     im=im.unsqueeze(0)
    #     im=im.unsqueeze(0)
    #     ypred=self.forward(im)
        
        
    #     instance=self.cluster.cluster(ypred[0], n_sigma=1, threshold=0.5)[0]
    #     return instance.squeeze().cpu().detach().numpy(),ypred

    # def ins_cluster_with_gt(self, im, instance, n_sigma=1):
    #     im=im.to(self.cur_device)
    #     im=im.unsqueeze(0)
    #     im=im.unsqueeze(0)
    #     prediction=self.forward(im)
    #     instance=self.cluster.cluster_with_gt(prediction[0], instance, n_sigma=n_sigma)
    #     return instance.squeeze().cpu().detach().numpy(),prediction
    

def peakfilter(seed):
    m = nn.MaxPool2d(kernel_size=5, stride=1,padding=2)
    return m(seed)



