
import gwdt
import torch
import time
import numpy as np
from ..utils.npixel import valid_surround_pixel,valid_connect_pixel,track_back,get_move_sclices
from scipy.ndimage import generate_binary_structure
from .seg_base import get_point_from_mask,get_mask_from_point,get_mask_box
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
def get_path(image,start,end,space=1,v=1e10,lamb=1.0,iterations=2):
    #lamb = 1.0  # <-- Geodesic distance transform
    image=image.copy()

    start=[int(p) for p in start]
    end=[int(p) for p in end]
    image[tuple(start)]=0

    footprint=generate_binary_structure(image.ndim,3)
    
        
    geodesic_dist=gwdt.gwdt(image,footprint)
    
    
    paths=track_back(geodesic_dist,end)
    return paths,geodesic_dist




# Downsample points
def downsample_points(points, dsmp_resolution):
    """
    [INPUT]
    points : n x 2,3 point coordinates
    dsmp_resolution : [x, y, z] downsample resolution

    [OUTPUT]
    point_downsample : n x 3 downsampled point coordinates
    """

    if len(points.shape) == 1:
            points = np.reshape(points,(1,len(dsmp_resolution)))

    dsmp_resolution = np.array(dsmp_resolution, dtype=np.float)

    point_downsample = points/dsmp_resolution

    point_downsample = np.round(point_downsample)
    point_downsample = np.unique(point_downsample, axis=0)

    return point_downsample.astype(np.int32)

# Upsample points
def upsample_points(points, dsmp_resolution):
    """
    [INPUT]
    points : n x 2,3 point coordinates
    dsmp_resolution : [x, y, z] downsampled resolution

    [OUTPUT]
    point_upsample : n x 3 upsampled point coordinates
    """

    dsmp_resolution = np.array(dsmp_resolution)
    
    point_upsample = points*dsmp_resolution
    
    return point_upsample.astype(np.int32)


def _exclude_border(label, border_width=None):
    """Set label border values to 0.

    """
    if border_width is None:
        border_width=[1,]*label.ndim
    elif isinstance(border_width, int):
        border_width = [border_width,] * label.ndim
    # zero out label borders
    for i, width in enumerate(border_width):
        if width == 0:
            continue
        label[(slice(None),) * i + (slice(None, width),)] = 0
        label[(slice(None),) * i + (slice(-width, None),)] = 0
    return label
def make_graph(mask,weight_matrix=None):
    # Don't run skeletonization if the input is an empty array
    mask=_exclude_border(mask,border_width=1)
    #print(mask)
    ndim=mask.ndim
    
    if weight_matrix is None:
        weight_matrix=np.ones_like(mask)
    
    # get left_top indexs
    movec=np.array([1,]*ndim)
    objc=get_move_sclices(movec)
    obj_points = np.argwhere(mask[objc]>0)
    indexs=np.where(mask[objc]>0)
    
    # index mask
    index_mask=np.zeros_like(mask,dtype=np.uint64)
    number=len(indexs[0])
    index_mask[objc][indexs]=np.arange(1,number+1)
   # print(number)
    
    
    # make graph
    # graph=np.zeros((number+1,number+1))
    linespace=[np.arange(0,3) for p in range(ndim)]
    moves = np.meshgrid(*linespace,indexing="ij")
    moves=[m.ravel() for m in moves]
    #print(moves)
    rows,cols,vals=[],[],[]
    for move in zip(*moves):
        #print(move)
        if (move==movec).all():continue
        obj=get_move_sclices(move)
        nexti=index_mask[obj][indexs]
        weight=weight_matrix[obj][indexs]
        #print(nexti)
        # graph[np.arange(1,number+1),nexti]=weight
        rows.append(np.arange(1,number+1))
        cols.append(nexti)
        vals.append(weight)
        #graph[np.arange(1,number+1),nexti]=1
    rows=np.concatenate(rows)
    cols=np.concatenate(cols)
    vals=np.concatenate(vals)
    graph=csr_matrix((vals, (rows, cols)), shape=(number+1, number+1))
    return graph[1:,1:],obj_points+1,index_mask
    # print(graph[1:,1:])
    # print(index_mask)
# Skeletonization
def get_path_2(object_input, weight_matrix,start,end,object_id = 1, dsmp_resolution = [1,1], parameters = [6,6]):
    """
    [INPUT]
    object_input : object to skeletonize (N x 3 point cloud or 3D labeled array)
    object_id : object ID to skeletonize (Don't need this if object_input is in point cloud format)
    dsmp_resolution : downsample resolution
    parameters : list of "scale" and "constant" parameters (first: "scale", second: "constant")
                 larger values mean less senstive to detecting small branches
    init_roots : N x 3 array of initial root coordinates

    [OUTPUT]
    skeleton : skeleton object
    """
    start=[int(p) for p in start]
    end=[int(p) for p in end]
    ndim=len(start)
    # Don't run skeletonization if the input is an empty array
    object_input=_exclude_border(object_input,border_width=1)
    

    graph,points,indexmask=make_graph(object_input,weight_matrix)
    id_start=indexmask[tuple(start)]
    id_end=indexmask[tuple(end)]
    graph=csr_matrix(graph)
    # Graph shortest path in the weighted graph
    # return_predecessors return pre node idx other wise -9999
    dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=id_start, return_predecessors=True)
    length=dist_matrix[id_end]
    
    id_p=id_end
    paths=[]
    while id_p>=0:
        pre_id=predecessors[id_p]
        paths.append(points[id_p])
        id_p=pre_id
    print(length,len(paths))
    return paths
    
    
        
    

    
    # Downsample points
    # if sum(dsmp_resolution) > len(dsmp_resolution):
    #     print(">>>>> Downsample...")
    #     obj_points = downsample_points(obj_points, dsmp_resolution)
    #     init_root = downsample_points(init_root, dsmp_resolution)
    #     start = downsample_points(start, dsmp_resolution)
    #     end = downsample_points(end, dsmp_resolution)
        
    #     if init_dest.shape[0] != 0:
    #         init_dest = downsample_points(init_dest, dsmp_resolution)

    # Convert coordinates back into original coordinates
    # if skeleton.nodes.shape[0] != 0:
    #     skeleton.nodes = upsample_points(skeleton.nodes + min_bound - 1, dsmp_resolution)

    return skeleton


if __name__=="__main__":
    mask=np.arange(27).reshape((3,3,3))
    make_graph(mask)

# def get_path(image,start,end,space=1,v=1e10,lamb=1.0,iterations=2):
#     #lamb = 1.0  # <-- Geodesic distance transform
#     device = "cpu"
#     ndim=image.ndim
#     mask=np.zeros_like(image)
#     start=[int(p) for p in start]
#     end=[int(p) for p in end]
#     #print(start,end)
#     mask[tuple(start)]=1
#     if image.ndim == 3:
#         image = np.moveaxis(image, -1, 0)
#     else:
#         image = np.expand_dims(image, 0)
    
    
#     image = torch.from_numpy(image).unsqueeze_(0).to(device)
#     mask = (
#         torch.from_numpy(1 - mask.astype(np.float32))
#         .unsqueeze_(0)
#         .unsqueeze_(0)
#         .to(device)
#     )
#     if isinstance(space,int) or isinstance(space,float):
#         space=[space for i in image.shape]
#     if ndim==2:
#         geodesic_dist = FastGeodis.generalised_geodesic2d(
#         image, mask,v, lamb, iterations
#         )
#     elif ndim==3:
#         geodesic_dist = FastGeodis.generalised_geodesic3d(
#         image, mask, space, v, lamb, iterations
#         )
#     paths=track_back(geodesic_dist,end)
#     return paths
    
    