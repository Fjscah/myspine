

import torch
import os

def show_cpu_gpu():
    """check gpu/cpu device 
    """
    print("================device available============")
    use_cuda=torch.cuda.is_available()
    print("cuda.is_available:\t",torch.cuda.is_available()) # cuda是否可用
    n_gpu=torch.cuda.device_count()
    print("cuda.device_count:\t",n_gpu) # 查看GPU数量
    print("cuda.device_name:\t",torch.cuda.get_device_name()) # 查看DEVICE（GPU）名
    print("current_device_id:\t",torch.cuda.current_device())  #检查目前使用GPU的序号
    n_cpu = os.cpu_count()
    print("cpu.worker_count:\t",n_cpu) # 查看GPU数量
    
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    
    if "cu" not in torch.__version__:
        print("=========Please install GPU version of Torch if you want to use gpu to train model=========")
    print("============================================")
    return device,n_gpu,n_cpu

def set_use_gpu(f_gpu):
    
    if f_gpu:
        device,n_gpu,n_cpu=show_cpu_gpu()
    else:
        n_cpu = os.cpu_count()
        device=torch.device('cpu')
        n_gpu=0
    print("device:",device)
    print("================device enable state============")
    print("use gpu\t: \t",device.type=="cuda")
    print("gpu number\t\t: \t",n_gpu)
    print("use cpu\t: \t",device.type=="cpu")
    print("cpu number\t\t: \t",n_cpu)
    
    return  device,n_gpu,n_cpu
   
if __name__=="__main__":
    show_cpu_gpu()


    