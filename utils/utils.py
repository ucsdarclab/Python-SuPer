from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pyplot as plt
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

# # Find n_neighbors of inputs in targets
# def KNN(inputs, targets, n_neighbors=4):

#     nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(targets)
#     weights, indexs = nbrs.kneighbors(inputs)

#     if n_neighbors == 1:
#         weights = np.squeeze(weights)
#         indexs = np.squeeze(indexs)

#     return weights, indexs

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

def name_file(method, data_cost, depth_cost, arap_cost, \
    rot_cost, corr_cost, \
    data_lambda, depth_lambda, arap_lambda, rot_lambda, \
    corr_lambda, \
    use_lambda=False):

    def str_lambda(lambda_):
        lambda_ = str(np.round(lambda_,1))
        num1, num2 = lambda_.split(".")
        if num2 == "0": return num1
        else: return num1 + "-" + num2

    filename = method
    if use_lambda:
        if data_cost: filename += "_"+str_lambda(data_lambda)+"data"
        if depth_cost: filename += "_"+str_lambda(depth_lambda)+"depth"
        if arap_cost: filename += "_"+str_lambda(arap_lambda)+"ARAP"
        if rot_cost: filename += "_"+str_lambda(rot_lambda)+"rot"
        if corr_cost: filename += "_"+str_lambda(corr_lambda)+"corr"

    else:
        if data_cost: filename += "_data"
        if depth_cost: filename += "_depth"
        if arap_cost: filename += "_ARAP"
        if rot_cost: filename += "_rot"
        if corr_cost: filename += "_corr"

    return filename

def get_PWCNet_model(model_iteration):
    model_module_to_load = "full_model" # A: "only_flow_net", B: "full_model"
    model_name           = "model_A"    # your model's name, e.g., "model_A", "chairs_things"

    return os.path.join("./experiments", "models", model_name, f"{model_name}_{model_iteration}.pt")

# Update the moving average (and min, max) of a function/step. 
def runtime_moving_average(inputs=None, new_time=None, init=False):
    # 'inputs' is a list of 1) times that this function/step 
    # has been implemented, 2) average, 3) min, and 4) max running time.
    # 'new_time' is the latest running time.
    # 'init', if True, init the 'inputs' vector.

    if init:
        return [0.0, 0.0, 1e8, 0.0]
    
    else:
        inputs[1] = (inputs[0]/(inputs[0]+1)) * inputs[1] + new_time / (inputs[0]+1)
        inputs[0] += 1
        inputs[2] = min(inputs[2], new_time)
        inputs[3] = max(inputs[3], new_time)
        return inputs

##### Visualization Functions

# Visualize (point and) normal map.
def vis_normal(filename, depth=None, spoint=None, snormal=None, normal=None):
    # Inputs: 2) "normal": Normal map of size HEIGHTxWIDTHx3 (numpy),
    # input element values: [-1,1] & nan; 3) "spoint": Sampled points 
    # of size ...x3 (numpy); 4) "snormal": Sampled normals of size 
    # ...x3 (numpy), input element values: [-1,1].

    # Cutout the white margin.
    def cut_white_margin(img):

        h, w, _ = img.shape
        margin_yx = np.argwhere(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) < 250)
        margin_y, margin_x = margin_yx[:,0], margin_yx[:,1]
        img = img[max(np.min(margin_y)-10, 0) : min(np.max(margin_y)+10, h), \
            max(np.min(margin_x)-10, 0) : min(np.max(margin_x)+10, w)]
        return img

    # Concatenate images on x-axis.
    def concat_imgs(img1, img2):
        th = img1.shape[0]
        sh, sw, _ = img2.shape
        return np.concatenate([img1, cv2.resize(img2, (int(sw/sh*th), th))], axis=1)

    img = None

    if depth is not None:
        invalid_map = np.isnan(depth)
        valid_depth = depth[~invalid_map]
        min_depth, max_depth = np.min(valid_depth), np.max(valid_depth)*1.2
        depth[invalid_map] = max_depth
        depth = (255 * (depth-min_depth) / (max_depth-min_depth)).astype('uint8')
        img = cut_white_margin(cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB))
    
    # Point-arrow map.
    if spoint is not None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_axis_off()

        spoint = spoint[::5]
        snormal = snormal[::5]

        x, y, z = spoint[...,0], spoint[...,1], spoint[...,2]
        u, v, w = snormal[...,0], snormal[...,1], snormal[...,2]
        color_array = np.reshape((snormal + 1) / 2, (-1,3))
        ax.scatter(x, y, z, color='r')
        ax.quiver(x, y, z, u, v, w, length=0.5, normalize=True) # color=color_array

        ax.quiver(0, 0, 0, 1, 0, 0, arrow_length_ratio=0.5, color='r') # x
        ax.text(1.1, 0, 0, r'$x$')
        ax.quiver(0, 0, 0, 0, 1, 0, arrow_length_ratio=0.5, color='lime') # y
        ax.text(0, 1.1, 0, r'$y$')
        ax.quiver(0, 0, 0, 0, 0, 1, arrow_length_ratio=0.5, color='b') # z
        ax.text(0, 0, 1.1, r'$z$')

        # fig.tight_layout()
        ax.view_init(azim=-80, elev=10)
        view1 = cut_white_margin(mplfig_to_npimage(fig).astype('uint8'))
        ax.view_init(azim=-80, elev=50)
        view2 = cut_white_margin(mplfig_to_npimage(fig).astype('uint8'))
        
        if img is None:
            img = concat_imgs(view1, view2)
        else:
            img = concat_imgs(concat_imgs(img, view1), view2)

    if normal is not None:
        # Visualize normals with colors.
        normal[np.isnan(normal)] = 1.
        normal = (255 * (normal + 1) / 2).astype('uint8')
        normal = cut_white_margin(normal)
        if img is None:
            img = normal
        else:
            img = concat_imgs(img, normal)

    cv2.imwrite(os.path.join(F_depth_prepro, filename), img[:,:,::-1])

########################################

# PyTorch-related Functions

########################################
def torch_to_numpy(inputs):
    return inputs.detach().cpu().numpy()

def numpy_to_torch(inputs, dtype=tfdtype_):
    return torch.as_tensor(inputs, dtype=dtype, device=dev)

def torch_delete(inputs, indexs):
    mask = torch.ones(len(inputs), dtype=bool, device=dev)
    mask[indexs] = False
    return inputs[mask]

# Distance
def torch_distance(a,b,keepdim=False):
    return torch.linalg.norm(a-b, dim=-1, keepdim=keepdim)

# Inner product
def torch_inner_prod(a,b):
    return torch.sum(a*b, dim=-1)

def get_skew(inputs):

    a1, a2, a3 = torch.split(inputs, 1, dim=-1)
    zeros = torch.zeros_like(a1)

    return torch.stack([torch.cat([zeros, a3, -a2], dim=-1), \
        torch.cat([-a3, zeros, a1], dim=-1), \
        torch.cat([a2, -a1, zeros], dim=-1)], dim=3)

# point cloud to coordinates on image plane (y*HEIGHT+x)
# if vis_only==True, only keeps the points that are visable after projection
def pcd2depth(pcd, round_coords=True, depth_sort=False, conf_sort=False, conf=None):

    X = pcd[:,0]
    Y = - pcd[:,1]
    Z = - (pcd[:,2] + 1e-8) # Z > 0
    
    u_ = X * fx / Z + cx
    v_ = Y * fy / Z + cy
    u = torch.round(u_).long()
    v = torch.round(v_).long()
    coords = v * WIDTH + u
    valid_proj = (u>=0) & (u<WIDTH) & (v>=0) & (v<HEIGHT)

    if depth_sort or conf_sort:

        # Keep only the valid projections.
        Z = Z[valid_proj]
        coords = coords[valid_proj]
        indicies = valid_proj.nonzero(as_tuple=True)[0]

        # Sort based on depth or confidence.
        if depth_sort:
            _, sort_indices = torch.sort(Z) # Z(depth) from small to large.
        elif conf_sort:
            _, sort_indices = torch.sort(conf[valid_proj])
        coords, indicies = coords[sort_indices], indicies[sort_indices]


        # Sort based on coordinate.
        coords, sort_indices = torch.sort(coords, stable=True)
        indicies = indicies[sort_indices]
        coords, counts = torch.unique_consecutive(coords, return_counts=True)
        counts = torch.cat([torch.tensor([0], device=dev), \
            torch.cumsum(counts[:-1], dim=0)])
        indicies = indicies[counts]

        if round_coords:
            # return v[indicies].long(), u[indicies].long(), coords, indicies
            return v[indicies], u[indicies], coords, indicies
        else:
            return v_[indicies], u_[indicies], coords, indicies

        # if len(valid_index) > 0:
        #     valid_index = torch.cumsum(valid_index, dim=0) - valid_index[0]
        #     valid_index = index[valid_index]

        #     coords = coords[valid_index]

            

        # Z = Z[valid_proj]
        # _coords = coords[valid_proj]
        # index = torch.arange(len(pcd), dtype=int, device=dev)[valid_proj]

        # sort_idx = torch.argsort(Z) # Sort Z from small to large
        # _coords = _coords[sort_idx]
        # index = index[sort_idx]

        # _coords, sort_idx = torch.sort(_coords, dim=0, stable=True)
        # index = index[sort_idx]

        # _, valid_index = torch.unique_consecutive(_coords, return_counts=True)
        # if len(valid_index) > 0:
        #     valid_index = torch.cumsum(valid_index, dim=0) - valid_index[0]
        #     valid_index = index[valid_index]

        #     coords = coords[valid_index]

        #     if round_coords:
        #         return v.long(), u.long(), coords, valid_index
        #     else:
        #         return v_, u_, coords, valid_index
    
    else:

        # Outputs: 1) & 2) The y- and x-coordinates of all points;
        # 3) The y*Height+x coordinates of all valid projections;
        # 4) The index of all points with valid projections.
        # if round_coords:
        #     return v.long(), u.long(), coords[valid_proj], valid_proj.nonzero().squeeze(1)
        # else:
        #     return v_, u_, coords[valid_proj], valid_proj.nonzero().squeeze(1)
        if round_coords:
            return v, u, coords[valid_proj], valid_proj.nonzero(as_tuple=True)[0]
        else:
            return v_, u_, coords[valid_proj], valid_proj.nonzero(as_tuple=True)[0]

# Visualize the norm map.
# Coordinate system: x: left-to-right, y: bottom-to-up, z: points to the image
# White: Norms who have angles less than 30 degrees with vector (0,0,-1);
# Green: Norms who have angles between 90 to 60 with (0,0,=1); Red: nan norms.
def vis_norm_map(norms, valid, filename, coords=None):
    angs = torch_inner_prod(norms, \
        torch.tensor([[0,0,-1]], device=dev))
    angs = angs.view(HEIGHT,WIDTH).detach().cpu().numpy()
    angs[~valid.view(HEIGHT,WIDTH).detach().cpu().numpy()] = np.nan
    arrow_map = np.zeros((HEIGHT,WIDTH,3))

    arrow_map[angs > 1/2] = np.array([[255,255,255]])
    arrow_map[np.isnan(angs)] = np.array([[0,0,255]])
    arrow_map[(angs < 1/2) & (angs > 0)] = np.array([[0,255,0]])
    cv2.imwrite(filename+".jpg",arrow_map)

# Assign a color to each point in a point cloud to visualize the
# deformation tracking results for qualitative evaluation.
def init_qual_color(points, margin=0.):
    # 'points' should be a Nx3 torch tensor
    
    max_dist, _ = torch.max(points, dim = 0, keepdim=True)
    min_dist, _ = torch.min(points, dim = 0, keepdim=True)

    return (255. - margin * 2) * (max_dist - points) / (max_dist - min_dist) + margin

# Save frames with SuPer 20-point tracking results.
def vis_track_rst(labelPts, track_rst, evaluate_id, rgb, filename):
            
    gt_ = labelPts['gt'][evaluate_id]
    super_cpp_ = labelPts['super_cpp'][evaluate_id]
    SURF_ = labelPts['SURF'][evaluate_id]
            
    rgb = rgb[:,:,::-1].astype(np.uint8) # RGB2BGR
    for k, coord in enumerate(track_rst):

        offset = 6
        gt_y = int(gt_[k][0])
        gt_x = int(gt_[k][1])
        if gt_y > 0 and gt_x > 0:
            rgb[gt_y-offset:gt_y+offset, \
                    gt_x-offset:gt_x+offset,:] = \
                    np.array([[0, 255, 0]])

            offset = 4
            super_cpp_y = int(super_cpp_[k][0])
            super_cpp_x = int(super_cpp_[k][1])
            if super_cpp_y > 0 and super_cpp_x > 0:
                rgb = cv2.line(rgb, (gt_x,gt_y), (super_cpp_x,super_cpp_y), (0,255,0), 1)
                rgb[super_cpp_y-offset:super_cpp_y+offset, \
                        super_cpp_x-offset:super_cpp_x+offset,:] = \
                        np.array([[0, 0, 255]])

            offset = 3
            SURF_y = int(SURF_[k][0])
            SURF_x = int(SURF_[k][1])
            if SURF_y > 0 and SURF_x > 0:
                rgb[SURF_y-offset:SURF_y+offset, \
                        SURF_x-offset:SURF_x+offset,:] = \
                        np.array([[255, 0, 0]])

            offset = 2
            y = int(coord[0])
            x = int(coord[1])
            rgb = cv2.line(rgb, (gt_x,gt_y), (x,y), (0,255,0), 1)
            rgb[y-offset:y+offset, \
                    x-offset:x+offset,:] = \
                    np.array([[255, 0, 255]])

        cv2.imwrite(filename,rgb)

# Transformation of surfels: eq (10) in SuPer paper.
def Trans_points(d_surfels, ednodes, beta, surfel_knn_weights, grad=False, skew_v=None):
    # Inputs: d_surfels: p - g_i; ednodes: g_i; 
    # beta: [q_i; b_i]; surfel_knn_weights: alpha_i.

    # 'trans_surfels': T(q_i,b_i)(p-g_i); 'Jacobian': d_out/d_q_i
    trans_surfels, Jacobian = transformQuatT(d_surfels, beta, grad=grad, skew_v=skew_v)
    
    trans_surfels += ednodes
    surfel_knn_weights = surfel_knn_weights.unsqueeze(-1)
    trans_surfels = torch.sum(surfel_knn_weights * trans_surfels, dim=-2)

    if grad:
        Jacobian *= surfel_knn_weights.unsqueeze(-1)
    
    return trans_surfels, Jacobian

# Output: T(q,b)v in eq (10)/(11) in SuPer paper. 
# Inputs: 'beta': [q;b] or [q] for b=0. 
def transformQuatT(v, beta, grad=False, skew_v=None):

    qw = beta[...,0:1]
    qv = beta[...,1:4]
    # chn = beta.size()[-1]
    # if chn == 7:
    #     t = beta[...,4:7]
    # else:
    #     t = torch.zeros((num, n_neighbors_, 3), layout=torch.sparse_coo, device=dev)

    cross_prod = torch.cross(qv, v, dim=-1)

    tv = v + 2.0 * qw * cross_prod + \
        2.0 * torch.cross(qv, cross_prod, dim=-1) 
        
    # tv = rv + t
    if beta.size()[-1] == 7:
        tv += beta[...,4:7]

    if grad:
        # eye_3 = torch.eye(3, dtype=fl32_, device=dev).unsqueeze(0).unsqueeze(0)
        eye_3 = torch.eye(3, dtype=fl32_, device=dev).view(1,1,3,3)

        d_qw = 2 * cross_prod.unsqueeze(-1)

        qv_v_inner = torch.sum(qv*v, dim=-1)
        qv = qv.unsqueeze(-1)
        v = v.unsqueeze(-2)
        qv_v_prod = torch.matmul(qv, v)
        d_qv = 2 * (qv_v_inner.unsqueeze(-1).unsqueeze(-1) * eye_3 + \
                qv_v_prod - 2 * torch.transpose(qv_v_prod,2,3) - \
                qw.unsqueeze(-1) * skew_v)
        return tv, torch.cat([d_qw, d_qv], dim=-1)
    else:
        return tv, 0

# Uniform Sampling
# Ref: Embedded Deformation for Shape Manipulation
# Inputs: 'cands': candidate points, 'EDs': existed ednodes.
def find_ED_nodes(cands, EDs=None):
    # Remove all candidates that are within the given radius.
    def update_cands(cands, cand_indices, ED):
        del_indices = (torch_distance(cands, ED) < ED_rad).nonzero(as_tuple=True)[0]
        cands = torch_delete(cands, del_indices)
        cand_indices = torch_delete(cand_indices, del_indices)
        return cands, cand_indices

    cand_indices = torch.arange(len(cands))

    if EDs is not None:
        for ED in EDs:
            cands, cand_indices = update_cands(cands, cand_indices, ED)
    
    ED_indices = []
    while len(cands) > 0:
        # Randomly select a new ED node and add it to 'EDs'.
        new_ED_index = torch.randint(len(cands), (1,))
        new_ED = cands[new_ED_index]
        if EDs is None:
            EDs = new_ED
        else:
            EDs = torch.cat([EDs, new_ED], dim=0)
        ED_indices.append(cand_indices[new_ED_index])
        
        # Remove all candidates that are within the given radius.
        cands, cand_indices = update_cands(cands, cand_indices, new_ED)

    if len(ED_indices) > 0:
        ED_indices = torch.cat(ED_indices)
    return EDs, ED_indices

def update_KNN_weights(points=None, targets=None, n_neighbors_=None, dists=None):
    def dist2weight(dists):
        weights = 1. - dists[:,:-1] / (dists[:,-1].unsqueeze(1)+1e-8)
        return torch.nn.functional.softmax(weights**1, dim=1)

    if dists is None:
        D = torch.cdist(points, targets)
        dists, sort_idx = D.topk(k=n_neighbors_+1, dim=-1, largest=False, sorted=True)
        return dist2weight(dists), sort_idx[:,:-1], sort_idx[:,-1]

    else:
        return dist2weight(dists)

# Convert depth map to point cloud.
def depth2pcd(Z):
    X = (U - cx) * Z / fx
    Y = (V - cy) * Z / fy
    return X, Y, Z