# fork from deepspinenet
import yaml

import os
from .file_base import create_dir,split_filename
class YAMLConfig:
    def __init__(self, path):
        if not path: return
        dirpath,shorname,suffix=split_filename(path)
        self.config_path=os.path.abspath(dirpath)
        self.path=os.path.abspath(path)
        print("== Configure file dir",self.config_path)
        print("== Configure file path",self.path)
        with open(str(path), 'r') as f:
            self.config = yaml.safe_load(f)
        self.init_default()
    def init_default(self):
        #dict_a = self.config["Path"]
        cong=self.config
        
        dataroot= self.get_entry(['Path', 'ori_path'])      
        trainroot= self.get_entry(['Path', 'exp_path']) 
        dataroot=self.get_abs_path(dataroot)
        print("== Data root",dataroot)
        trainroot=self.get_abs_path(trainroot)
        print("== Train root",trainroot)

        self.set_entry(['Path', 'exp_path'],trainroot,overlap=True,isdir=True)
        self.set_entry(['Path', 'ori_path'],dataroot,overlap=True,isdir=True)
        self.set_entry(['Path', 'crop_path'],os.path.join(trainroot,"data"),isdir=True)
        crop_path=self.get_entry(['Path', 'crop_path'])   
        crop_path=self.get_abs_path(crop_path)
        self.set_entry(['Path', 'crop_path'],crop_path,overlap=True,isdir=True)
        print("== crop_path",crop_path)
        
           
        
        self.set_entry(['Path', 'label_path'],os.path.join(crop_path,"labelcrop"),isdir=True)
        self.set_entry(['Path', 'img_path'],os.path.join(crop_path,"imgcrop"),isdir=True)
        
        self.set_entry(['Path', 'orilabel_path'],os.path.join(dataroot,"label"),isdir=True)
        self.set_entry(['Path', 'oriimg_path'],os.path.join(dataroot,"img"),isdir=True)
        
        self.set_entry(['Path', 'log_path'],os.path.join(trainroot,"log"),isdir=True)
        self.set_entry(['Path', 'model_path'],os.path.join(trainroot,"model"),isdir=True)
        self.trainroot=trainroot
        self.dataroot=dataroot # ori
        self.crop_path=crop_path# train
    
     
    def get_abs_path(self,path):
        if not path: return path
        isabs=os.path.isabs(path)
        if isabs:
            # print("absolute rootdir",":\t",path)
            return path
            
        else:    
            # print("relative rootdir",":\t",path)
            abspath=os.path.join(self.config_path,path)
            # print("absolute rootdir",":\t",abspath)
            return abspath
             
    def set_entry(self,entry_path,value,overlap=False,isdir=False):
        temp_value = self.config
        for key in entry_path[:-1]:
            if key not in temp_value :
                raise ValueError('Parameter "{}" with path "{}" '
                                 'not found in configuration file.'.format(key, entry_path))
            elif key not in temp_value:
                return None
            else:
                temp_value = temp_value[key]
        if temp_value[entry_path[-1]] and overlap:
            if isdir:
                create_dir(value)
            temp_value[entry_path[-1]]=value
        elif not temp_value[entry_path[-1]]:
            if isdir:
                create_dir(value)
            temp_value[entry_path[-1]]=value
        
    def get_entry(self, entry_path, required=True):
        temp_value = self.config
        for key in entry_path:
            if key not in temp_value and required:
                raise ValueError('Parameter "{}" with path "{}" '
                                 'not found in configuration file.'.format(key, entry_path))
            elif key not in temp_value:
                return None
            else:
                temp_value = temp_value[key]

        return temp_value
    
    
