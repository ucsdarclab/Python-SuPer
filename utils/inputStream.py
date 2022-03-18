import os
import cv2
import open3d as o3d
import numpy as np
import torch
from torch_geometric.data import Data

from utils.config import *
from utils.utils import *

from utils.pcd2normal import *

def read_imgs(data_dir, img_id, img_format):
    
    if dataset_class == 'RGBD_mask':
        depth_file = os.path.join(data_dir,"{0:06d}-depth.{img_format}".format(img_id,img_format=img_format))
        image_file = depth_file.replace("depth","left") # Reading data from folder 'grasp1_2_psmnet'.

        rgb = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_file, -1).astype(np.float32)

        return rgb, depth

# TODO: Check how this function works in fusion-master.
# def noise3D(x,y,z):
    
#     out = np.sin(x*112.9898 + y*179.233 + z*237.212) * 43758.5453
#     return out - np.floor(out)

# Calculate the normal in camera space.
def getN(points):

    hL = points[1:-1, :-2]
    hR = points[1:-1, 2:]
    hD = points[:-2, 1:-1]
    hU = points[2:, 1:-1]

    N = torch.cross(hL-hR, hU-hD)
    N = torch.nn.functional.normalize(N, dim=-1)
    
    # Pad N to the same size as points
    out_N = torch.ones_like(points) * float('nan')
    out_N[1:-1,1:-1] = N
    
    return out_N, torch.isnan(out_N[:,:,0])
    
# Depth --> 3d point cloud in camera frame.
# def depthProcessing(rgb, Z, depth_ID=0):
def depthProcessing(rgb, Z, time, ID):

    Z[Z==0] = np.nan # Convert invalid depth values (0) to nan.
    Z = cv2.bilateralFilter(Z,15,80,80)
    Z *= depth_scale

    points, norms, invalid = CentralDiff.forward(Z)
    # points, norms, invalid = SNE.forward(Z)

    # calculate the radius, confidence and the time stamp
    rad = numpy_to_torch(Z) / (SQRT2 * fx * torch.clamp(torch.abs(norms[:,:,2]), 0.26, 1.0))
    scale_u, scale_v = U / WIDTH, V / HEIGHT
    dc2 = (2.*scale_u-1.)**2 + (2.*scale_v-1.)**2
    conf = torch.exp(-dc2*DIVTERM)

    isED = torch.zeros((HEIGHT,WIDTH), dtype=bool, device=dev)
    if ED_sample_method == 'grid': dnode = 18
    elif ED_sample_method == 'uniform': dnode = 4
    start_node = int(dnode/2.)
    isED[start_node::dnode, start_node::dnode] = True

    if vis_depth_preprocessing:
        # normal_ element range: [-1,1] & nan
        vis_normal(str(ID)+".jpg", depth=Z, \
            spoint=torch_to_numpy(points[isED]), snormal=torch_to_numpy(norms[isED]), \
            normal=torch_to_numpy(norms))

    points = points.view(-1,3)
    norms = norms.view(-1,3)
    colors = torch.as_tensor(rgb, dtype=fl32_, device=dev).view(-1,3)
    isED = isED.view(-1)
    rad = rad.view(-1)
    conf = conf.view(-1)
    valid = ~invalid.view(-1)

    # return points, norms, rad, conf,isED, valid
    return Data(points=points, norms=norms, colors=colors, \
        rad=rad, conf=conf, isED=isED, valid=valid, \
        rgb=rgb, time=time, ID=ID)
