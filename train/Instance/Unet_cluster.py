
from spinelib.seg import unetseg
from .base_cluster import BaseCluster
import numpy as np
from skimage.morphology import remove_small_objects
class Uner_cluster(BaseCluster):
    def __init__(self,**kwarg) -> None:
        pass
    
    def ins_cluster(self,model,img):
        mask,spinepr,denpr,bgpr=unetseg.predict_single_img(model,img)
        
        if model.out_layer:
            spine_label=unetseg.instance_unetmask_by_border(spinepr,mask==2,bgpr=bgpr,th=0.009,spinesize_range=[4,800])
        else:
            spine_label=unetseg.instance_unetmask_bypeak(spinepr,mask,searchbox=[15,15],min_radius=5,spinesize_range=[4,800])
        spine_label=remove_small_objects(spine_label,10)
        return spine_label,[bgpr,denpr,spinepr,mask]
    
    def get_visual_keys(self):
        return ["image","seed","mask","label","GT"]
    
    def show_result(self,visualizer,visual_result):
        img,GT,ins,output=visual_result
        #print(visualizer.keys)
        visualizer.display(img.cpu(), 'image')
        visualizer.display(GT, 'GT')  # it shows prediction / GT-mask
        visualizer.display(ins, 'label')  # it shows prediction / GT-mask
        

        visualizer.display(np.array(output[:3]), 'seed')  # seed map
        visualizer.display(output[-1], 'mask')  
        #plt.show()
        visualizer.save()
    
    # elif "dis" in self.task_type:
                            
    #         spine_label=unetseg.instance_unetmask_by_dis(model,img,th=0.15,spinesize_range=[4,800])