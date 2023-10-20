""" Utilities for logging, visualization, evaluation """

from tarfile import is_tarfile
import os, io, sys, random, glob, shutil, logging, numpy as np, pandas as pd
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from PIL import Image, ImageColor, ImageDraw, ImageFont

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms as T
import torch.backends.cudnn as cudnn


import matplotlib.pyplot as plt

from pytorch3d.ops import (
    ball_query,
    knn_points,
)
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

def get_grid_coords(h, w, dtype=torch.long, batch_size=None, stack_dim=None):
    """
    Get grids of coordinates from the 1D inputs.
    """
    u = torch.arange(w, dtype=dtype).cuda()
    v = torch.arange(h, dtype=dtype).cuda()
    u, v = torch.meshgrid(u, v, indexing='xy')

    if batch_size is not None:
        u = u.unsqueeze(0).repeat(batch_size,1,1)
        v = v.unsqueeze(0).repeat(batch_size,1,1)
    
    if stack_dim is None:
        return u, v
    else:
        return torch.stack([u, v], dim=stack_dim)

def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = None,
    radius: int = 3,
    width: int = 3,
) -> torch.Tensor:

    """
    Draws Keypoints on given RGB image.
    The values of the input image should be uint8 between 0 and 255.
    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoints location for each of the N instances,
            in the format [x, y].
        connectivity (List[Tuple[int, int]]]): A List of tuple where,
            each tuple contains pair of keypoints to be connected.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.
    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with keypoints drawn.
    """

    if keypoints.ndim != 3:
        raise ValueError("keypoints must be of shape (num_instances, K, 2)")

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()

    for kpt_id, kpt_inst in enumerate(img_kpts):
        for inst_id, kpt in enumerate(kpt_inst):
            x1 = kpt[0] - radius
            x2 = kpt[0] + radius
            y1 = kpt[1] - radius
            y2 = kpt[1] + radius
            draw.ellipse([x1, y1, x2, y2], fill=colors, outline=None, width=0)

        if connectivity:
            for connection in connectivity:
                start_pt_x = kpt_inst[connection[0]][0]
                start_pt_y = kpt_inst[connection[0]][1]

                end_pt_x = kpt_inst[connection[1]][0]
                end_pt_y = kpt_inst[connection[1]][1]

                draw.line(
                    ((start_pt_x, start_pt_y), (end_pt_x, end_pt_y)),
                    width=width,
                )

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)
        
def blur_image(imgs, method="pyt-gauss", kernel=15):
    """
    Smooth images.
    Input: imgs [H, W], [H, W, C], [B, H, W], or [B, H, W, C], B: batch size, C: channel.
    Output: It should be the same type (numpy/torch) as the input.
    """
    def blur_func(img):
        if method == "cv2-bilateral":
            return cv2.bilateralFilter(img.astype(np.float32),15,80,80)

    is_tensor = torch.is_tensor(imgs)
    if "cv2-" in method:
        if is_tensor: imgs = torch_to_numpy(imgs)

        if batch:
            imgs = np.stack([blur_func(img) for img in imgs], axis=0)
        else:
            imgs = blur_func(imgs)

        if is_tensor: return numpy_to_torch(imgs)
        else: return imgs

    elif method == "pyt-gauss":
        if not is_tensor: imgs = numpy_to_torch(imgs)

        # (C, H, W) or (B, C, H, W)
        dim = len(imgs.size())
        if dim == 2: # (h, w) --> (1, 1, h, w)
            imgs = imgs.unsqueeze(0).unsqueeze(0)
        elif dim == 3: # (c, h, w) -> (1, ch, h, w)
            imgs = imgs.unsqueeze(0)
        else:
            assert imgs.dim() == 4

        blur = T.Compose([T.GaussianBlur(kernel)])
        imgs = blur(imgs)

        if dim == 2:
            imgs = imgs[0,0]
        elif dim == 3 and batch:
            imgs = imgs[0,:,:,:]
        
        if not is_tensor:
            return torch_to_numpy(imgs)
        else:
            return imgs

def torch_inner_prod(a, b, dim=-1, keepdim=False):
    return torch.sum(a*b, dim=dim, keepdim=keepdim)

def torch_distance(a,b,keepdim=False):
    return torch.linalg.norm(a-b, dim=-1, keepdim=keepdim)

def torch_sq_distance(a, b, dim=-1, keepdim=False):
    return torch.pow(a - b, 2).sum(dim)

def torch_dilate(inputs, kernel=10, dtype=torch.bool):
    B, C, H, W = inputs.size() 
    kernel_tensor = torch.ones((C, 1, kernel, kernel)).to(inputs.device)
    outputs = F.conv2d(inputs, kernel_tensor, padding='same', groups=C)
    outputs = (outputs > 0).type(dtype)
    return outputs



def pcd2depth(inputs, pcd, round_coords=True, valid_margin=0):
    ''' point cloud to coordinates on image plane (y*HEIGHT+x) 
        Note: A screen coordinate in `coord `is an int. For example:
        ---------------------
        | 0 | 1 | 2 | 3 | 4 | 
        | 5 | 6 | 7 | 8 | 9 |
        |10 |11 |12 |13 |14 |
        ---------------------
    '''

    height, width = inputs[("color",0)][0,0].size()
    X, Y, Z = pcd[...,0], pcd[...,1], (pcd[...,2] + 1e-8)
    
    u_ = X * inputs["K"][0,0,0] / Z + inputs["K"][0,0,2]
    v_ = Y * inputs["K"][0,1,1] / Z + inputs["K"][0,1,2]
    u = torch.round(u_).long()      # (N,). Screen coordinate for each point
    v = torch.round(v_).long()      # (N,). Screen coordinate for each point

    coords = v * width + u
    valid_proj = (v >= valid_margin) & (v < height-1-valid_margin) & \
        (u >= valid_margin) & (u < width-1-valid_margin)    # 

    if round_coords:    return v, u, coords, valid_proj # (N,) 
    else:               return v_, u_, coords, valid_proj

def depth2pcd(inputs, Z):
    ''' Depth map to point cloud.
    Input:  - inputs: Either a dict or torch tensor, encoding camera intrinsics K
            - Z: Torch tensor. Depth map of shape (1, B, H, W)
    Output: X, Y, Z
    '''

    assert Z.shape[0] == 1, f"Depth map should be of shape (1, B, H, W), but got {Z.shape}"
    if not isinstance(inputs, torch.Tensor):
        cx, cy, fx, fy = inputs['cx'], inputs['cy'], inputs['fx'], inputs['fy']
    else:
        assert inputs.squeeze().shape == (4,4), f'Camera intrinsic cannot be squeezed to (4, 4). The intrinsic matrix is of shape {inputs.shape}'
        inputs = inputs.squeeze()
        cx, cy, fx, fy = inputs[0,2], inputs[1,2], inputs[0,0], inputs[1,1]
    
    Z = Z.squeeze(1)
    b, h, w = Z.size()
    u, v = get_grid_coords(h, w, batch_size=b)

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    # return X.type(fl32_), Y.type(fl32_), Z
    return X.to(torch.float32), Y.to(torch.float32), Z



def find_knn(points1, points2, num_classes=-1, seg1=None, seg2=None, k=20, radius=0.5, method="knn"):
    def group_knn(p1, p2):
        if method == "ball_query":
            dists, idx, _ = ball_query(p1.unsqueeze(0), p2.unsqueeze(0), K=k, radius=radius)
        elif method == "knn":
            dists, idx, _ = knn_points(p1.unsqueeze(0), p2.unsqueeze(0), K=k)
        return torch.sqrt(dists[0]), idx[0]

    if num_classes <= 0:
        return group_knn(points1, points2)
    else:
        N1, N2 = len(points1), len(points2)
        points1_ids = torch.arange(N1)
        points2_ids = torch.arange(N2)
        dists = 1e8 * torch.ones((N1, k), dtype=torch.float64).cuda()
        idx = - torch.ones((N1, k), dtype=torch.long).cuda()
        for class_id in range(num_classes):
            val1 = seg1==class_id
            p1 = points1[val1]

            val2 = (seg2==class_id).nonzero(as_tuple=True)[0]
            p2 = points2[val2]

            if len(p1) == 0 and len(p2) == 0:
                continue
            assert len(p1) > 0 and len(p2) >= k
            _dists_, _idx_ = group_knn(p1, p2)
            dists[val1] = _dists_
            idx[val1] = val2[_idx_]

        return dists, idx

def KLD(P, Q, eps=1e-13, dim=-1):
    """
    return KL(P||Q)
    P: target
    Q: input
    """
    return (P * (P / (Q + eps) + eps).log()).sum(dim)

def JSD(P, Q, eps=1e-13, dim=-1):
    M = 0.5 * (P + Q)
    return 0.5 * (KLD(P, M, eps=eps, dim=dim) + KLD(Q, M, eps=eps, dim=dim))

def pyt_erode(x, kernel=11):
    eroder = nn.MaxPool2d(kernel, stride=1, padding=int((kernel-1)/2))
    x = 1 - eroder(1 - x)
    return x

def pyt_dilate(x, kernel=11):
    dilater = nn.MaxPool2d(kernel, stride=1, padding=int((kernel-1)/2))
    x = dilater(x)
    return x

def erode_dilate_seg(seg, kernel=31):
    valid_seg = torch.ones(seg.size(), dtype=bool_).cuda()
    
    for class_id in torch.unique(seg):
        class_seg = torch.zeros_like(seg)
        class_seg[seg == class_id] = 1
        valid_seg &= class_seg == pyt_dilate(pyt_erode(class_seg.type(fl32_)))

    return valid_seg

def find_edge_region(x, num_classes=0, class_list=None, kernel=11, ignore_img_edge=True):
    assert x.dim()==4
    if x.size(1) > 1:
        x = torch.argmax(x, dim=1, keepdim=True)
    b, _, h, w = x.size()
    if class_list is None:
        class_list = torch.unique(x)
    N = len(class_list)

    label_map = torch.ones(num_classes).to(x.device) * N
    for k, class_id in enumerate(class_list):
        label_map[class_id] = k
    class_mask = F.one_hot(label_map[x].long())[...,0:N].permute(0, 4, 1, 2, 3)

    erode_class_mask = torch_dilate(1. - class_mask.reshape(-1, 1, h, w), 
                                    kernel=kernel)
    erode_class_mask = erode_class_mask.reshape(b, -1, 1, h, w)
    edge_mask = torch.any(erode_class_mask & class_mask.bool(), 1)

    if ignore_img_edge:
        edge_mask[:, :, 0:kernel] = False
        edge_mask[:, :, -kernel:] = False
        edge_mask[:, :, :, 0:kernel] = False
        edge_mask[:, :, :, -kernel:] = False

    return edge_mask

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def conf2color(confs):
    ''' Convert float-valued surfel confidence score to RGB color (for plotting heat map)'''
    assert len(confs.shape) == 1, f'Point condfidences should be of shape (N,), but got {confs.shape}'
    if torch.is_tensor(confs): confs = confs.cpu().numpy()
    cmap = plt.get_cmap('magma')
    colors = torch.as_tensor(cmap(confs))[:,:-1].cuda() # remove alpha
    return colors

def png_to_gif(png_dir = "./", duration = 500):
    ''' Find all png files in `png_dir`, aggregate them into a .gif file '''
    frames = []
    imgs = glob.glob(os.path.join(os.path.expanduser(png_dir), "*.png"))
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save('png_to_gif.gif', format='GIF', append_images=frames[1:], 
                   save_all=True, duration=duration, loop=0)

def plot_pcd(pcd, colors, filename=None, view_params=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the dense point cloud
    ax.scatter(pcd[:, 0], 
            pcd[:, 1], 
            -pcd[:, 2], 
            c=np.clip(colors,0,1), 
            s=1)

    if view_params is not None:
        ax.view_init(elev=view_params[0], azim=view_params[1])

    # Hide grid lines
    ax.grid(False)
    ax.axis('off')

    # Hide axis numbers
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if filename is None:
        fig.canvas.draw()
        pcd_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        pcd_image = pcd_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        return pcd_image
    
    else:
        fig.savefig(filename, bbox_inches='tight', pad_inches=0)

def get_gt(args) -> tuple:
    ''' Returns a list of all ground truth files
    Input:  args parsed from argparser
    Output: Tuple containing the following:
        - gt_supercpp_surf: Dict of dicts. {
            'gt':           {...}
            'super_cpp':    {...}
            'SURF':         {...}
        }
        - gt: dict. {
            '000010': array of shape (20, 3). Homogeneous screen coord. of 20 tracked points at time 000010
            '000020': array of shape (20, 3). Homogeneous screen coord. of 20 tracked points at time 000020
            ...
            '000520': array of shape (20, 3). Homogeneous screen coord. of 20 tracked points at time 000520
        }
        - gt_intkeys: [10, 20, ..., 520]
        - gt_strkeys: ['000010', '000020', ..., '000520']
        - gt_array: np array of shape (52, 20, 3). Homogeneous screen coord. of 20 tracked points at all times
    ''' 
    data_dir = os.path.expanduser(args.data_dir)
    if not os.path.exists(data_dir): 
        raise ValueError(f'Path {data_dir} does not exist. This is likely an error with args.data_dir configuration.')

    gt_path = os.path.join(data_dir, args.tracking_gt_file)
    if not os.path.exists(gt_path):
        raise ValueError(f'Ground truth file does not exist!')

    gt_supercpp_surf = np.array(np.load(gt_path, allow_pickle=True)).tolist()
    gt = gt_supercpp_surf['gt']  # keys: 'gt', 'super_cpp', 'SURF'. 
    gt_intkeys = sorted([int(k) for k in gt.keys()])                   # 04, 05, 06, ..., 10, ...
    gt_strkeys = sorted([f"{int(k):06d}" for k in gt.keys()])
    gt_array = np.stack([gt[k] for k in gt_strkeys], axis=0)
    return gt_supercpp_surf, gt, gt_intkeys, gt_strkeys, gt_array

def save_and_log_fig(fname, title, summary_writer, time):
    ''' Save the plt figure and log to tensorboard 
    Inputs:
    - fname: str. Path to save the figure.
    - title: str. Title of the figure.
    - summary_writer: Tensorboard summary writer.
    - time: int. Current time stamp.
    '''
    plt.savefig(fname, format='png'); plt.clf(); plt.close()
    summary_writer.add_image(title, T.ToTensor()(plt.imread(fname, format='png')), global_step=time)
    return

def log_trackpts_err(err_array, res_array, edge_ids, gt_array, logdir, summary_writer, time):
    ''' Log reprojection error of 20 tracked points across N time stamps.
    Inputs:
    - err_array: np array (N, 20). 
    - res_array: np array (N, 20, 3). Predicted point locations in homogeneous screen coord. 
    - gt_array: np array (N, 20, 3). Ground truth point locations in homogeneous screen coord.
    - logdir: str. Path to log directory.
    - summary_writer: Tensorboard summary writer.
    - time: int. Current time stamp.

    Produces 3 plots, all of which are logged to Tensorboard and saved to logdir:
    - Plot 1: Bar plot of reprojection error stats for each point.
    - Plot 2: Reprojection error of each point over each frame.
    - Plot 3: 20 plots, predicted and ground truth point trajectories.

    Also logs the following to Tensorboard:
    - Mean reprojection error across all points and all time steps.
    - Std dev of reprojection error across all points and all time steps.
    '''
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    ''' Draw plot 1: Bar plot of reprojection error stats for each point. '''
    if not os.path.exists(logdir): os.makedirs(custom_log_manager.logdir)
    fname = os.path.expanduser(f'{logdir}/reprojerr_avg0.png')
    title = f'plots/Reprojection error of 20 tracked points averaged over all time steps'
    valid = err_array >= 0       # (N, 20)
    numpoints = err_array.shape[1]
    std_ = np.array([np.std(err_array[:,i][valid[:,i]]) for i in range(numpoints)])
    mean_ = np.array([np.mean(err_array[:,i][valid[:,i]]) for i in range(numpoints)])

    plt.figure(figsize=(12,3)); bar_width = 0.25
    plt.scatter(np.arange(numpoints), mean_, s=40, color=colors[1])
    # plt.bar(np.arange(numpoints), mean_, bar_width, color=colors[1])  # Use bar plot if comparing different approaches
    plt.errorbar(np.arange(numpoints), mean_, std_, linestyle='None', color='k', elinewidth=.8, capsize=3)
    plt.xticks(np.arange(numpoints), np.arange(numpoints)+1)
    plt.xlabel('Tracked point ID'); plt.ylabel('Error (unit: pixel)')
    plt.grid(True); plt.title(title)
    save_and_log_fig(fname, title, summary_writer, time)


    ''' Draw plot 2: Reprojection error of each point over time. '''
    fname = os.path.expanduser(f'{logdir}/reprojerr_avg1.png')
    title = f'plots/Average reprojection error of 20 tracked points across all time steps'
    x = np.arange(err_array.shape[0])
    plt.figure(figsize=(24,6)); window_size = 5
    time_error = np.average(err_array, axis=1)  # (N,). Avg. error over all points for each time step.
    y = pd.Series(time_error) if len(x) < window_size else pd.Series(time_error).rolling(window_size).mean().tolist()
    plt.plot(x, time_error, '.', color=colors[0], linewidth=0.8)
    plt.plot(x, y, color=colors[0], linewidth=0.8)
    plt.xticks(x, (x+1)*10)
    plt.xlabel('Time step'); plt.ylabel('Error (unit: pixel)')
    plt.grid(True); plt.title(title)
    save_and_log_fig(fname, title, summary_writer, time)

    ''' Draw plot 3: predicted and ground truth point trajectories. '''
    fname = os.path.expanduser(f'{logdir}/reprojerr_trajectory.png')
    title = f'plots/Trajectory of 20 tracked points across all time steps'
    tensor_imgs = []
    for i in range(numpoints):
        plt.figure(figsize=(4,4))

        x_gt, y_gt = gt_array[:,i,0], gt_array[:,i,1]
        valid = (x_gt > 0) & (y_gt > 0)
        x_gt, y_gt = x_gt[valid], y_gt[valid]
        plt.plot(x_gt, y_gt, color=colors[0], label='Ground Truth')

        x_res, y_res = res_array[:,i,0], res_array[:,i,1]
        valid = (x_res > 0) & (y_res > 0)
        x_res, y_res = x_res[valid], y_res[valid]
        plt.plot(x_res, y_res, color=colors[1], label='Predicted'); plt.legend(loc='upper right'); 
        
        xmin, xmax = np.min(np.concatenate([x_gt, x_res])), np.max(np.concatenate([x_gt, x_res]))
        ymin, ymax = np.min(np.concatenate([y_gt, y_res])), np.max(np.concatenate([y_gt, y_res]))

        xtics, ytics = np.arange(xmin-10, xmax+10)[::20], np.arange(ymin-10, ymax+10)[::20]
        plt.xticks(xtics, [round(t) for t in xtics], rotation=60)
        plt.yticks(ytics, [round(t) for t in ytics])
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5); 
        plt.title(f'point {i+1}')
        
        buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
        tensor_imgs.append(T.ToTensor()(Image.open(buf))); 
        buf.close(); plt.clf(); plt.close()
    
    plt.figure(figsize=(30,24)); plt.box(False); plt.title(title, fontsize=24)
    plt.imshow(make_grid(tensor_imgs, nrow=5).permute(1,2,0).numpy())
    plt.xticks([]); plt.yticks([])
    plt.xlabel('x (unit: pixel)', fontsize=24); plt.ylabel('y (unit: pixel)',fontsize=24)
    save_and_log_fig(fname, title, summary_writer, time)

    summary_writer.add_scalar('reprojerr/pythonsuper_mean', np.mean(err_array), time)
    summary_writer.add_scalar('reprojerr/pythonsuper_std', np.std(err_array), time)

    if len(edge_ids) > 0:
        select = np.zeros(err_array.shape[1], dtype=np.bool)
        select[np.array(edge_ids) - 1] = True
        
        summary_writer.add_scalar('reprojerr/pythonsuper_edge_pts_mean', 
                                  np.mean(err_array[:, select]), 
                                  time)
        summary_writer.add_scalar('reprojerr/pythonsuper_edge_pts_std', 
                                  np.std(err_array[:, select]), 
                                  time)

    return

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

def merge_transformation(q1, q2):
    R1 = quaternion_to_matrix(q1[:, 0:4])
    R2 = quaternion_to_matrix(q2[:, 0:4])
    R = torch.matmul(R2, R1)
    q = matrix_to_quaternion(R)

    t = q2[:, 4:] + torch.matmul(R2, q1[:, 4:][:, :, None])[:, :, 0]

    return torch.concat([q, t], dim=1)