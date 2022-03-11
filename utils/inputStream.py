import os
import cv2
import open3d as o3d
import numpy as np
import torch

from utils.config import *
from utils.utils import *

from utils.pcd2normal import *

# References: 
# [1] Real-time 3D reconstruction in dynamic scenes using point-based fusion.

def read_imgs(data_dir, img_id, img_format, use_mask=False):
    
    if dataset_class == 'RGBD_mask':
        depth_file = os.path.join(data_dir,"{0:06d}-depth.{img_format}".format(img_id,img_format=img_format))
        # image_file = depth_file.replace("depth","color")
        image_file = depth_file.replace("depth","left") # Reading data from folder 'grasp1_2_psmnet'.

        if os.path.exists(image_file) and os.path.exists(depth_file):

            rgb = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(depth_file, -1).astype(np.float32)
            
            if use_mask:
                mask_file = depth_file.replace("depth","mask") 
                mask = cv2.imread(mask_file,0)
                
                kernel = np.ones((9, 9), np.uint8)
                mask = cv2.erode(mask, kernel)
                depth[mask > 0] = 0.0

            return rgb, depth

        else:
            
            return None, None

# TODO: Check how this function works in fusion-master.
# def noise3D(x,y,z):
    
#     out = np.sin(x*112.9898 + y*179.233 + z*237.212) * 43758.5453
#     return out - np.floor(out)

# # Convert depth map to point cloud.
# def depth2pcd(Z):
#     X = (U - cx) * Z / fx
#     Y = (V - cy) * Z / fy
#     return X, -Y, -Z

# # Calculate the normal in camera space.
# def getN(points):

#     # Estimate normals from central differences (Ref[1]-Section3, 
#     # link: https://en.wikipedia.org/wiki/Finite_difference), 
#     # i.e., f(x+h/2)-f(x-h/2), of the vertext map.
#     hL = points[1:-1, :-2]
#     hR = points[1:-1, 2:]
#     hD = points[:-2, 1:-1]
#     hU = points[2:, 1:-1]
#     N = torch.cross(hR-hL, hD-hU)

#     N = torch.nn.functional.normalize(N, dim=-1)
    
#     # Pad N to the same size as points
#     out_N = torch.ones_like(points) * float('nan')
#     out_N[1:-1,1:-1] = N
    
#     return out_N, torch.isnan(out_N[:,:,0])
    
# Depth --> 3d point cloud in camera frame.
def depthProcessing(rgb, Z, depth_ID=0):

    Z[Z==0] = np.nan # Convert masked out depth values (0) to nan.
    Z *= depth_scale

    points, norms, invalid = CentralDiff.forward(Z)
    # points, norms, invalid = SNE.forward(Z)

    # Calculate the radius, confidence and the time stamp.
    rad = numpy_to_torch(Z) / (SQRT2 * fx * torch.clamp(torch.abs(norms[:,:,2]), 0.26, 1.0))
    scale_u, scale_v = U / WIDTH, V / HEIGHT
    dc2 = (2.*scale_u-1.)**2 + (2.*scale_v-1.)**2
    conf = torch.exp(-dc2*DIVTERM)

    # TODO: Need better sampling method.
    # isED = torch.rand((HEIGHT,WIDTH), device=dev) > 0.997
    isED = torch.zeros((HEIGHT,WIDTH), dtype=bool, device=dev)
    dnode = 18
    start_node = int(dnode/2.)
    isED[start_node::dnode, start_node::dnode] = True

    if vis_depth_preprocessing:
        # normal_ element range: [-1,1] & nan
        vis_normal(str(depth_ID)+".jpg", depth=Z, \
            spoint=torch_to_numpy(points[isED]), snormal=torch_to_numpy(norms[isED]), \
            normal=torch_to_numpy(norms))

    points = points.view(-1,3)
    norms = norms.view(-1,3)
    isED = isED.view(-1)
    rad = rad.view(-1)
    conf = conf.view(-1)
    valid = ~invalid.view(-1)

    return points, norms, isED, rad, conf, valid