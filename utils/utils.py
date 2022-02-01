import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import os
import shutil
import cv2

from utils.config import *

# Input (n_samples, n_features)
# Output: Normalize each sample
def normalization(inputs):

    inputs_shape = inputs.shape
    inputs = np.reshape(inputs,(-1,3))
    inputs_norm = preprocessing.normalize(inputs, norm='l2')

    return np.reshape(inputs_norm, inputs_shape)

# Find n_neighbors of inputs in targets
def KNN(inputs, targets, n_neighbors=4):

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(targets)
    weights, indexs = nbrs.kneighbors(inputs)

    if n_neighbors == 1:
        weights = np.squeeze(weights)
        indexs = np.squeeze(indexs)

    return weights, indexs

# Inner product
def inner_prod(a,b):
    return np.sum(a*b, axis=-1)

# Distance
def distance(a,b):
    return np.linalg.norm(a-b, axis=-1)

# def evaluate_projection_quality(real_img, wrap_img, mask):

#     compare_psnr(real_img[mask], wrap_img[mask])

# Reset contents in folder "foldername"
def reset_folder(foldername):

    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(foldername)

def put_text(image, text):
  
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
  
    # org
    org = (50, 50)
  
    # fontScale
    fontScale = 1
   
    # Blue color in BGR
    color = (255, 255, 255)
  
    # Line thickness of 2 px
    thickness = 2
   
    # Using cv2.putText() method
    image = cv2.putText(image, text, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    
    return image

########## PyTorch Functions ##########
def torch_delete(inputs, indexs):
    mask = torch.ones(len(inputs), dtype=bool, device=cuda0)
    mask[indexs] = False
    return inputs[mask]

# Distance
def torch_distance(a,b):
    return torch.linalg.norm(a-b, dim=-1)

# Inner product
def torch_inner_prod(a,b):
    return torch.sum(a*b, dim=-1)
    
# point cloud to coordinates on image plane (y*HEIGHT+x)
# if vis_only==True, only keeps the points that are visable after projection
def pcd2depth(pcd, vis_only=False, round_coords=True):

    X = pcd[:,0]
    Y = - pcd[:,1]
    Z = - (pcd[:,2] + 1e-8)
    # u_ = X * K[0, 0] / Z + K[0, 2]
    u_ = X * fx / Z + cx
    u = torch.round(u_)
    # v_ = Y * K[1, 1] / Z + K[1, 2]
    v_ = Y * fy / Z + cy
    v = torch.round(v_)
    coords = v * WIDTH + u
    coords = coords.long()
    valid_proj = (u>=0) & (u<WIDTH) & (v>=0) & (v<HEIGHT)

    if vis_only:

        Z = Z[valid_proj]
        _coords = coords[valid_proj]
        index = torch.arange(len(pcd), dtype=int, device=cuda0)[valid_proj]

        sort_idx = torch.argsort(Z) # Sort Z from small to large
        _coords = _coords[sort_idx]
        index = index[sort_idx]

        _coords, sort_idx = torch.sort(_coords, dim=0, stable=True)
        index = index[sort_idx]

        _, valid_index = torch.unique_consecutive(_coords, return_counts=True)
        if len(valid_index) > 0:
            valid_index = torch.cumsum(valid_index, dim=0) - valid_index[0]
            valid_index = index[valid_index]

            coords = coords[valid_index]

            if round_coords:
                return v.long(), u.long(), coords, valid_index
            else:
                return v_, u_, coords, valid_index
    
    else:

        invalid = ~valid_proj

        if round_coords:
            v[invalid] = -1.0
            u[invalid] = -1.0
            return v.long(), u.long(), coords[valid_proj], valid_proj.nonzero().squeeze(1)
        else:
            v_[invalid] = -1.0
            u_[invalid] = -1.0
            return v_, u_, coords[valid_proj], valid_proj.nonzero().squeeze(1)