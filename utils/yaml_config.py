# fork from deepspinenet
import yaml

import os
from .file_base import create_dir,split_filename
class YAMLConfig:
    def __init__(self, path):
        if not path: return
        with open(str(path), 'r') as f:
            self.config = yaml.safe_load(f)
        self.init_default()
    def init_default(self):
        #dict_a = self.config["Path"]
        cong=self.config
        
        trainroot= self.get_entry(['Path', 'Train_path']) 
        print("relative rootdir",":\t",trainroot)
        trainroot=os.path.abspath(trainroot)
        print("absolute rootdir",":\t",trainroot)
        self.set_entry(['Path', 'Train_path'],trainroot,overlap=True,isdir=True)
        
        self.set_entry(['Path', 'label_path'],os.path.join(trainroot,"labelcrop"),isdir=True)
        self.set_entry(['Path', 'data_path'],os.path.join(trainroot,"imgcrop"),isdir=True)
        self.set_entry(['Path', 'orilabel_path'],os.path.join(trainroot,"label"),isdir=True)
        self.set_entry(['Path', 'oridata_path'],os.path.join(trainroot,"img"),isdir=True)
        self.set_entry(['Path', 'log_path'],os.path.join(trainroot,"log"),isdir=True)
        self.set_entry(['Path', 'model_path'],os.path.join(trainroot,"model"),isdir=True)
                
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
    
    
