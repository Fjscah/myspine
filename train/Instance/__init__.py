

from .ERF_cluster import single_Cluster,multi_Cluster
from .base_cluster import BaseCluster
from .Unet_cluster import Uner_cluster
from .wnet_cluster import WnetCluster
instance_dict={
    "ERF_single_cluster":single_Cluster,
    "ERF_multi_cluster":multi_Cluster,
    "unet_cluster":Uner_cluster,
    "WnetCluster":WnetCluster,
}

def get_instancefunc(name, instance_opts={}) -> BaseCluster:
    if name in instance_dict:
        instance_class=instance_dict[name]
        return(instance_class(**instance_opts))
    
        
    else:
        
        raise RuntimeError("Dataset {} not available".format(name))