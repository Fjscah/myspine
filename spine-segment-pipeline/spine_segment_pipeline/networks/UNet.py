
from .BaseNet import *
from ..seg import unetseg

import numpy as np
from skimage.morphology import remove_small_objects
 
 
class UnetBase(BaseNet):
    def predict_2d_img(self,image):
        image=image.astype("float32") # H W,[C]
        im=ToTensor()(image) # C H W
        # im=im.expand(1,256,256,1)
        im=im.to(self.cur_device)
        im=im.unsqueeze(1)
        ypred=self.forward(im)
        return ypred[0].cpu().detach().numpy()   
    def load_network_set(self,layer_num=3,out_layer="softmax"):
        self.layer_num=layer_num
        self.out_layer=out_layer
        if "sigmoid"==self.out_layer:
            self.out=torch.nn.Sigmoid() 
        elif "tanh"==self.out_layer:
            self.out=torch.nn.Tanh()
        else:
            self.out=torch.nn.Softmax(dim=1)   

    
    def predict(self,img):
        im=self.valid_im(img)
        pred=self.forward(im)[0].cpu().detach().numpy()
        # unetseg.predict_single_img(model,img)
        bgpr,denpr,spinepr=pred
        mask = np.argmax(pred[:3], axis=0)
        if self.out_layer: #sigmiod
            maskseed=(bgpr<0.009)*spinepr>0.7
            spine_label=unetseg.instance_unetmask_by_border(spinepr,mask==2,maskseed,spinesize_range=[10,800])
        else: #softmax
            spine_label=unetseg.instance_unetmask_bypeak(spinepr,mask,searchbox=[15,15],min_radius=5,spinesize_range=[4,800])
        spine_label=remove_small_objects(spine_label,10)
        return spine_label,[bgpr,denpr,spinepr,mask]
    
    def get_visual_keys(self):
        return ["image","seed","mask","label","GT"]
    
    def show_result(self,visualizer,visual_result):
        img,GT,ins,output=visual_result
        #print(visualizer.keys)
        # print(GT.shape,img.shape,ins.shape)
        visualizer.display(img.cpu(), 'image')
        GT=label2rgb(GT[0].cpu().numpy(),image=img.cpu().numpy(),bg_label=0,alpha=0.4)
        visualizer.display(GT, 'GT')  # it shows prediction / GT-mask
        
        ins=label2rgb(ins,image=img.cpu().numpy(),bg_label=0,alpha=0.4)
        
        visualizer.display(ins, 'label')  # it shows prediction / GT-mask
        

        visualizer.display(np.array(output[:3]), 'mask')  # seed map
        seeds=unetseg.peakfilter(output[2],5,0,use_gaussian=False)
        visualizer.display((output[0]<0.009)*(output[-2]>0.7), 'seed')  
        # visualizer.display(seeds, 'mask')  
        #plt.show()
        visualizer.save()    
    def off_predict(self,im):
        # im=rescale(im,2)
        im=normalize(im,1,99.8)
        
      
      
        ins,_=self.predict(im)
        bgpr,denpr,spinepr,mask=_
        ins=remove_small_objects(ins,10)    
        return ins,[bgpr,denpr,spinepr],mask
        
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

             
class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, layer_num=3,**kwargs):
        super().__init__()
        # self.load_network_set(**kwargs)
        self.layer_num=layer_num
        nb_filter = [32, 64, 128, 256, 512]
        self.norm=nn.InstanceNorm2d(input_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        if self.layer_num>2:
            self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
            self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        if self.layer_num>3:
            self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

            self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.final = nn.Conv2d(nb_filter[0], output_channels, kernel_size=1)
        # self.soft=nn.Softmax(dim=1)
    
    def forward(self, input):
        input=self.norm(input)
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        if self.layer_num==2:
            x1_1 = self.conv1_3(torch.cat([x1_0, self.up(x2_0)], 1))   
            x0_2 = self.conv0_4(torch.cat([x0_0, self.up(x1_1)], 1))   
            output = self.final(x0_2)
        elif self.layer_num==3:
            x3_0 = self.conv3_0(self.pool(x2_0))
            x2_1 = self.conv2_2(torch.cat([x2_0, self.up(x3_0)], 1))
            x1_2 = self.conv1_3(torch.cat([x1_0, self.up(x2_1)], 1))
            x0_3 = self.conv0_4(torch.cat([x0_0, self.up(x1_2)], 1))
            output = self.final(x0_3)
        if self.layer_num==4:
            x3_0 = self.conv3_0(self.pool(x2_0))
            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))   
            x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
            output = self.final(x0_4)
        # output=self.out(output)
        return output


class UNet2d(UnetBase):
    def __init__(self, **kwargs):
        super(UNet2d, self).__init__( **kwargs)
        layer_num=kwargs["layer_num"]
        out_layer=kwargs["out_layer"]
        self.load_network_set(layer_num,out_layer)
        
        
        self.num_classes = kwargs["num_classes"]
        input_channels=kwargs["input_channels"]
       
        
        nb_filter = [32, 64, 128, 256, 512]
        self.norm=nn.InstanceNorm2d(input_channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
        # self.soft=nn.Softmax(dim=1)
    
    def forward(self, input):
        input=self.norm(input)
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        if self.layer_num==2:
            x1_1 = self.conv1_3(torch.cat([x1_0, self.up(x2_0)], 1))   
            x0_2 = self.conv0_4(torch.cat([x0_0, self.up(x1_1)], 1))   
            output = self.final(x0_2)
        elif self.layer_num==3:
            x3_0 = self.conv3_0(self.pool(x2_0))
            x2_1 = self.conv2_2(torch.cat([x2_0, self.up(x3_0)], 1))
            x1_2 = self.conv1_3(torch.cat([x1_0, self.up(x2_1)], 1))
            x0_3 = self.conv0_4(torch.cat([x0_0, self.up(x1_2)], 1))
            output = self.final(x0_3)
        if self.layer_num==4:
            x3_0 = self.conv3_0(self.pool(x2_0))
            x4_0 = self.conv4_0(self.pool(x3_0))
            x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))   
            x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
            x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
            x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
            output = self.final(x0_4)
        output=self.out(output)
        return output



class CNN(nn.Module):
    def __init__(self, num_class, input_channels=3,xysize=16, **kwargs):
        super().__init__()
        self.numclass=num_class
        self.layer_num=4
        self.xysize=xysize
        featurelen=(self.xysize//2)**2*16
        self.model=nn.Sequential(
            
                nn.Conv2d(1,16,[7,7],padding=3),
                nn.ReLU(),
                nn.MaxPool2d([2,2],2),
                nn.Conv2d(16,16,[3,3],padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(featurelen,16),
                nn.ReLU(),
                nn.Linear(16,self.numclass)
            
        )
        

    def forward(self, input):

        output=self.model(input)
        return output

    # torch.save(model,path)
if __name__=="__main__":
    # model=CNN(2,1,16).cuda()
    # summary(model,(1,16,16))
    model=UNet2d(3,1)
    savemodel(model,"kk.pth")