import os
import sys
import shutil
import logging
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from PIL import Image, ImageColor, ImageDraw, ImageFont

import torch
import torch.nn as nn

from pytorch3d.ops import (
    ball_query,
    knn_points,
)

from utils.config import *


def get_grid_coords(h, w, dtype=long_, batch_size=None, stack_dim=None):
    """
    Get grids of x,y coordinates.
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
    radius: int = 2,
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

    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(draw_keypoints)
    # if not isinstance(image, torch.Tensor):
    #     raise TypeError(f"The image must be a tensor, got {type(image)}")
    # elif image.dtype != torch.uint8:
    #     raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    # elif image.dim() != 3:
    #     raise ValueError("Pass individual images, not batches")
    # elif image.size()[0] != 3:
    #     raise ValueError("Pass an RGB image. Other Image formats are not supported")

    # if keypoints.ndim != 3:
    #     raise ValueError("keypoints must be of shape (num_instances, K, 2)")

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

def torch_to_numpy(inputs):
    return inputs.detach().cpu().numpy()

def numpy_to_torch(inputs, dtype=fl32_, to_gpu=True):
    if to_gpu:
        return torch.as_tensor(inputs, dtype=dtype).cuda()
    else:
        return torch.as_tensor(inputs, dtype=dtype)

def torch_inner_prod(a, b, dim=-1, keepdim=False):
    """
    Inner product.
    """
    return torch.sum(a*b, dim=dim, keepdim=keepdim)

def torch_distance(a,b,keepdim=False):
    return torch.linalg.norm(a-b, dim=-1, keepdim=keepdim)

def torch_sq_distance(a, b, dim=-1, keepdim=False):
    return torch.pow(a - b, 2).sum(dim)

# def torch_delete(inputs, indexs):
#     mask = torch.ones(len(inputs), dtype=bool).cuda()
#     mask[indexs] = False
#     return inputs[mask]

# # Reset contents in folder "foldername"
# def reset_folder(foldername):

#     if os.path.exists(foldername):
#         shutil.rmtree(foldername)
#     os.makedirs(foldername)

# point cloud to coordinates on image plane (y*HEIGHT+x)
# if vis_only==True, only keeps the points that are visable after projection
# def pcd2depth(inputs, pcd, round_coords=True, conf_sort=False, conf=None, valid_margin=0):
def pcd2depth(inputs, pcd, round_coords=True, valid_margin=0):

    X = pcd[...,0]
    Y = pcd[...,1]
    Z = (pcd[...,2] + 1e-8)
    
    u_ = X * inputs["fx"] / Z + inputs["cx"]
    v_ = Y * inputs["fy"] / Z + inputs["cy"]
    u = torch.round(u_).long()
    v = torch.round(v_).long()
    coords = v * inputs["width"] + u
    valid_proj = (v >= valid_margin) & (v < inputs["height"]-1-valid_margin) & \
        (u >= valid_margin) & (u < inputs["width"]-1-valid_margin)

    # if depth_sort or conf_sort:

    #     # Keep only the valid projections.
    #     Z = Z[valid_proj]
    #     coords = coords[valid_proj]
    #     indicies = valid_proj.nonzero(as_tuple=True)[0]

    #     # Sort based on depth or confidence.
    #     if depth_sort:
    #         _, sort_indices = torch.sort(Z) # Z(depth) from small to large.
    #     elif conf_sort:
    #         _, sort_indices = torch.sort(conf[valid_proj])
    #     coords, indicies = coords[sort_indices], indicies[sort_indices]

    #     # Sort based on coordinate.
    #     coords, sort_indices = torch.sort(coords, stable=True)
    #     indicies = indicies[sort_indices]
    #     coords, counts = torch.unique_consecutive(coords, return_counts=True)
    #     counts = torch.cat([torch.tensor([0]).cuda(), \
    #         torch.cumsum(counts[:-1], dim=0)])
    #     indicies = indicies[counts]

    #     if round_coords:
    #         return v[indicies], u[indicies], coords, indicies
    #     else:
    #         return v_[indicies], u_[indicies], coords, indicies
    # else:
    # if round_coords:
    #     return v, u, coords[valid_proj], valid_proj
    # else:
    #     return v_, u_, coords[valid_proj], valid_proj

    if round_coords:
        return v, u, coords, valid_proj
    else:
        return v_, u_, coords, valid_proj

# Convert depth map to point cloud.
def depth2pcd(inputs, Z):

    Z = Z.squeeze(1)
    b, h, w = Z.size()
    u, v = get_grid_coords(h, w, batch_size=b)
    
    X = (u - inputs["cx"]) * Z / inputs["fx"]
    Y = (v - inputs["cy"]) * Z / inputs["fy"]
    return X.type(fl32_), Y.type(fl32_), Z

def find_knn(points1, points2, num_classes=-1, seman1=None, seman2=None, k=20, radius=0.5, method="knn"):
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
        dists = 1e8 * torch.ones((N1, k), dtype=fl64_).cuda()
        idx = - torch.ones((N1, k), dtype=long_).cuda()
        for class_id in range(num_classes):
            val1 = seman1==class_id
            p1 = points1[val1]

            val2 = (seman2==class_id).nonzero(as_tuple=True)[0]
            p2 = points2[val2]

            if len(p1) == 0 and len(p2) == 0:
                continue
            assert len(p1) > 0 and len(p2) >= k
            _dists_, _idx_ = group_knn(p1, p2)
            dists[val1] = _dists_
            idx[val1] = val2[_idx_]

        return dists, idx

# def write_args(filename, args):
#     with open(filename, 'w') as f:
#         for arg in vars(args):
#             f.write(arg)
#             f.write(':  ')
#             f.write(str(getattr(args, arg)))
#             f.write('\n')

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

class CustomLogManager:
    '''
    To use this, 
    1. import the instance `custom_log_manager` in your main script and `setup`
    2. import the instance `custom_log_manager` in any `.py` file you would like logging,
    and `logger = custom_log_manager.get_logger(NAME, level=LEVEL)`, and use `logger.info`
    or `logger.warn` to replace all `print` statements.
    * do NOT setup your custom_log_manager more than once.
    '''
    def __init__(self):
        self._loggers = []
        self.file_handler = None
        self.console_handler = None
        pass

    def setup(self, to_file: bool=False, log_prefix: str=None, time_stamp=None, logdir: str='logs'):
        '''
        Setup handlers depending on the given parameters. 
        When `to_file==True` and `log_prefix==None`, a default prefix 'log' will be given.
        `log_prefix` does not do anything when `to_file` is False.
        `time_stamp` will be appended at the end of the file name if not None.
        Examples:
        ```
        import datetime
        from utils import custom_log_manager
        time_stamp = datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
        custom_log_manager.setup(to_file=True, time_stamp=time_stamp)
        logger = custom_log_manager.get_logger('main')
        ```
        or
        ```
        custom_log_manager.setup(to_file=True, log_prefix=f'train_exp{mod_id}', time_stamp=time_stamp)
        logger = custom_log_manager.get_logger('evaluate')
        ```
        '''
        if to_file and log_prefix is None: 
            log_prefix = 'log'
        self.logdir = logdir
        self.to_file = to_file
        self.time_stamp = time_stamp
        if self.to_file:
            if self.time_stamp is not None:
                log_fn = f"{log_prefix}-{self.time_stamp}.txt"
            else:
                log_fn = f"{log_prefix}.txt"
        self.fmt = logging.Formatter(
            fmt="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
            datefmt="%m-%d %H:%M:%S")
        if self.to_file:
            os.makedirs(self.logdir, exist_ok=True)
            log_full_fn = os.path.join(self.logdir, log_fn)
            self.file_handler = logging.FileHandler(log_full_fn)
            self.file_handler.setLevel(logging.DEBUG)
            self.file_handler.setFormatter(self.fmt)
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(logging.DEBUG)
        self.console_handler.setFormatter(self.fmt)

        # Some loggers may have been added before setup in the main file.
        for logger in self._loggers:
            if self.console_handler is not None:
                logger.addHandler(self.console_handler)
            if self.file_handler is not None:
                logger.addHandler(self.file_handler)
            
    
    def get_logger(self, name: str, level=logging.DEBUG):
        '''
        The `name` will be the second element in a log line. 
        Usually used to identify which function/namespace/file/etc. is logging.
        You can set it to any arbitrary string.
        '''
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if self.console_handler is not None:
            logger.addHandler(self.console_handler)
        if self.file_handler is not None:
            logger.addHandler(self.file_handler)
        self._loggers.append(logger)
        return logger

custom_log_manager = CustomLogManager()