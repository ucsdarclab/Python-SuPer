import os
import shutil
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from PIL import Image, ImageColor, ImageDraw, ImageFont

from pytorch3d.ops import (
    ball_query,
    knn_points,
)

from utils.config import *


def get_grid_coords(h, w, dtype=long_, batch_size=None, stack_dim=None):
    """
    Get grids of x,y coordinates.
    """
    u = torch.arange(w, dtype=dtype, device=dev)
    v = torch.arange(h, dtype=dtype, device=dev)
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
        
def blur_image(imgs, method="pyt-gauss"):
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

        blur = T.Compose([T.GaussianBlur(15)])
        imgs = blur(imgs).type(fl64_)

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

def numpy_to_torch(inputs, dtype=fl32_, device=dev):
    return torch.as_tensor(inputs, dtype=dtype, device=device)

def torch_inner_prod(a, b, dim=-1, keepdim=False):
    """
    Inner product.
    """
    return torch.sum(a*b, dim=dim, keepdim=keepdim)

def torch_distance(a,b,keepdim=False):
    return torch.linalg.norm(a-b, dim=-1, keepdim=keepdim)

def torch_delete(inputs, indexs):
    mask = torch.ones(len(inputs), dtype=bool, device=dev)
    mask[indexs] = False
    return inputs[mask]

# Reset contents in folder "foldername"
def reset_folder(foldername):

    if os.path.exists(foldername):
        shutil.rmtree(foldername)
    os.makedirs(foldername)

# point cloud to coordinates on image plane (y*HEIGHT+x)
# if vis_only==True, only keeps the points that are visable after projection
def pcd2depth(CamParams, pcd, round_coords=True, depth_sort=False, conf_sort=False, conf=None, valid_margin=0):

    X = pcd[...,0]
    Y = - pcd[...,1]
    Z = - (pcd[...,2] + 1e-8) # Z > 0
    
    u_ = X * CamParams.fx / Z + CamParams.cx
    v_ = Y * CamParams.fy / Z + CamParams.cy
    u = torch.round(u_).long()
    v = torch.round(v_).long()
    coords = v * CamParams.WIDTH + u
    valid_proj = (v >= valid_margin) & (v < CamParams.HEIGHT-1-valid_margin) & \
        (u >= valid_margin) & (u < CamParams.WIDTH-1-valid_margin)

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
    
    else:
        if round_coords:
            return v, u, coords[valid_proj], valid_proj#.nonzero(as_tuple=True)[0]
        else:
            return v_, u_, coords[valid_proj], valid_proj#.nonzero(as_tuple=True)[0]

# Convert depth map to point cloud.
def depth2pcd(CamParams, Z):

    Z = Z.squeeze(1)
    b, h, w = Z.size()
    u, v = get_grid_coords(h, w, batch_size=b)
    
    X = (u - CamParams.cx) * Z / CamParams.fx
    Y = (v - CamParams.cy) * Z / CamParams.fy
    return X, Y, Z

def find_knn(p1, p2, k=20, radius=0.5, method="knn"):
    if method == "ball_query":
        dists, idx, _ = ball_query(p1.unsqueeze(0), p2.unsqueeze(0), K=k, radius=radius)
        dists = dists[0]
        idx = idx[0]

    elif method == "knn":
        dists, idx, _ = knn_points(p1.unsqueeze(0), p2.unsqueeze(0), K=k)
        dists = dists[0]
        idx = idx[0]

    return dists, idx

def write_args(filename, args):
    with open(filename, 'w') as f:
        for arg in vars(args):
            f.write(arg)
            f.write(':  ')
            f.write(str(getattr(args, arg)))
            f.write('\n')