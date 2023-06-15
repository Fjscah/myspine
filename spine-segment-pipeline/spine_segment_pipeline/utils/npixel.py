import numpy as np

#=======================#
#   neighbour pixel  #
#=======================#
def IsValid(p,length):
    return p>=0 and p<length
def outOfbound(index,size):
    index=np.array(index)
    size=np.array(size)
    if np.min(index)<0:
        return True

    if np.min(size-index)<=0:
        return True
    return False
def connect_8(y,x):
    coords=[
        [y-1,x-1],
        [y-1,x],
        [y-1,x+1],
        [y,x-1],
        [y,x+1],
        [y+1,x-1],
        [y+1,x],
        [y+1,x+1],
    ]
    return coords

def connect_26(x,y,z):
    gg,hh,jj=np.mgrid[x-1:x+2,y-1:y+2,z-1:z+2]
    points=[ [x,y,z] for x,y,z in zip(gg.ravel(),hh.ravel(),jj.ravel())]
    points.pop(13) #remove center
    return points
def valid_connect_26(x,y,z,xx,yy,zz):
    points=connect_26(x,y,z)
    for n,(x,y,z) in enumerate(points):
        if IsValid(x,xx) and IsValid(y,yy) and IsValid(z,zz):
            continue
        points[n]=None
    return points

def connect_pixel(point):
    # arbitary dim neighbour pixel
    point=list(point)
    linespace=[np.arange(p-1,p+2) for p in point]
    grids = np.meshgrid(*linespace,indexing="ij")
    grids=[grid.ravel() for grid in grids]
    points=[ index for index in zip(*grids)]
    points.pop(3**len(point)//2)
    return points

def valid_connect_pixel(point,size):
    points=connect_pixel(point)
    #print(points)
    for n,p in enumerate(points):
        if outOfbound(p,size):
            points[n]=None
    return points

def get_move_sclices(move=[1,1,1]):
    
    obj=[slice(m,None) if m>0 else slice(None,m) if m<0 else slice(None) for m in move]
    return tuple(obj)

def valid_border_pixel(point,radius,shape):     
    point=list(point)
    linespace=[np.arange(p-radius,p+radius+1) for p in point]
    grids = np.meshgrid(*linespace,indexing="ij")
    grids=[grid.ravel() for grid in grids]
    points=[ index for index in zip(*grids)]
    borderpoints=[]
    for p in points:
        if np.max(np.abs(p-point))==radius:
            borderpoints.append(p)
    for n,p in enumerate(borderpoints):
        if outOfbound(p,shape):
            points[n]=None
    return points

def valid_surround_pixel(point,radius,shape):     
    point=list(point)
    linespace=[np.arange(p-radius,p+radius+1) for p in point]
    grids = np.meshgrid(*linespace,indexing="ij")
    grids=[grid.ravel() for grid in grids]
    points=[ index for index in zip(*grids)]
    for n,p in enumerate(points):
        if outOfbound(p,shape):
            points[n]=None
    return points
def valid_array_slice(arr,start,size,center=False,pad=None,rad=False):
    if not rad: # to rad size
        size=[s//2 for s in size]
    if center==True and pad==None:
        obj = (slice(np.max(int(st-s),0), int(st+s),1) for st,s in zip(start,size))
    elif pad==None and center==False: # left up
        # print(start,size)
        obj = (slice(np.max(int(st),0), int(st+s*2),1) for st,s in zip(start,size))
        # obj=tuple(obj)
    offset=[int(st-s) for st,s in zip(start,size)]
    # leftup=[o.start for o in obj]
    # cropbox=arr[obj]
    # if (croptobbox.shape-size).any():
    #     pass
        #todo
    # for o in obj:
    #     print(o.start,o.step)
    obj=tuple(obj)
    return arr[obj],obj,offset
def array_slice(arr,start,size,center=False,pad=None,obj_only=False):
    if isinstance(size,int):
        size=[size for i in start]
    if center==True and pad==None:
        obj = [slice(np.max(int(st-s//2),0), int(st+s//2),1) for st,s in zip(start,size)]
        obj=tuple(obj)
       
    elif pad==None and center==False:
        obj = [slice(np.max(int(st),0), int(st+s),1) for st,s in zip(start,size)]
        obj=tuple(obj)
       
    #cropbox=arr[obj]
    # if (croptobbox.shape-size).any():
    #     pass
    #     #todo
    if obj_only:
        return obj
    return arr[obj] 


def track_back(distance_matrix,end):
    shape=distance_matrix.shape
   # print(end)
    points=valid_connect_pixel(end,shape)
    paths=[end]
    old_point=end
    while True :
        dis_values=[distance_matrix[tuple(p)] if p is  not None else np.inf for p in points]
        # dis_value=np.min(dis_values)
        min_idx=np.argmin(dis_values)
        min_dis,min_point=dis_values[min_idx],points[min_idx]
        if min_point is None or min_dis>=distance_matrix[tuple(old_point)]:
            break
        #print(min_point)
        paths.append(min_point)
        points=valid_connect_pixel(min_point,shape) 
        old_point=min_point
    paths.reverse()
    return paths
           