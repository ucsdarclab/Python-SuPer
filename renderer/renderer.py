import math
import numpy as np
import cv2
import torch
from torch import nn

from pytorch3d.renderer.points.pulsar import Renderer
from pytorch3d.transforms import matrix_to_rotation_6d

from utils.config import *
from utils.utils import *

class Pulsar(nn.Module):
    """
    Pulsar in PyTorch modules.
    Ref: Pulsar: Efficient Sphere-based Neural Rendering
    """
    def __init__(self) -> None:
        super(Pulsar, self).__init__()
        self.gamma = 1.0e-5

    def get_cam_params(self, inputs, view_scale, opencv=True):

        R = torch.tensor(
            [[-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0,  0.0, 1.0]], 
            dtype=fl32_).cuda()
        r = matrix_to_rotation_6d(R)
        
        focal_length = 0.01
        h, w = int(inputs["height"] * view_scale), int(inputs["width"] * view_scale)
        cx, cy = inputs["cx"] * view_scale, inputs["cy"] * view_scale
        fx = inputs["fx"] * view_scale

        cam_params = torch.tensor([
            0.0,
            0.0,
            0.0,  # Position 0, 0, 0 (x, y, z)
            -r[0], -r[1], -r[2],  
            -r[3], -r[4], -r[5],  # camera rotation
            focal_length,
            (focal_length * w) / fx,
            -math.ceil(cx - w / 2),
            -math.ceil(cy - h / 2),
        ], dtype=fl32_).cuda()
        return cam_params

    def forward(self, inputs, data, colors=None, view_scale=1.0, rad=0.01, bg_col=torch.tensor([0.0, 0.0, 0.0]).cuda()):

        if colors is None:
            colors = data.colors
        points = data.points
        surfel_num = len(points)
        
        # Points
        vert_pos = points.type(fl32_)

        # Colors
        vert_col = colors.type(fl32_)

        # Radii of points
        vert_rad = torch.ones(surfel_num, dtype=fl32_).cuda() * rad
        # vert_rad = data.radii.type(fl32_)
        
        # Camera params
        cam_params = self.get_cam_params(inputs, view_scale)

        # The volumetric optimization works better with a higher number of tracked
        # intersections per ray.
        renderer = Renderer(
            int(inputs["width"] * view_scale),
            int(inputs["height"] * view_scale),
            surfel_num, n_track=64, right_handed_system=True
        ).cuda()

        return renderer.forward(
            vert_pos,
            vert_col,
            vert_rad,
            cam_params,
            self.gamma,
            15.0,
            return_forward_info=False,
            bg_col=bg_col,
        )


# Project with rounded coordinates.
class Projector(nn.Module):
    def __init__(self, method="direct") -> None:
        super().__init__()

        self.method = method

    def forward(self, inputs, allModel, qual_color=False):

        if self.method == "direct":
            y, x, _, valid_indices = pcd2depth(inputs, allModel.points, depth_sort=True)
            out = torch.zeros((inputs["height"], inputs["width"], 3)).cuda()
            if qual_color:
                out[y,x] = allModel.eval_colors[valid_indices].float()
            else:
                out[y,x] = allModel.colors[valid_indices].float()

        elif self.method == "opencv":
            rod, _ = cv2.Rodrigues(np.eye(3)) # [R]otation matrix.
            # Flip the points upside down
            points = torch_to_numpy(allModel.points)
            img_quads, _ = cv2.projectPoints(points, rod, np.zeros((3,1)), inputs[("K", 0)][0, :3, :3], np.array([]))
            out = np.zeros((HEIGHT,WIDTH,3))
            y = np.round(img_quads[:,0,1]).astype(int)
            x = np.round(img_quads[:,0,0]).astype(int)
            valid_proj = (y>=0) & (y<HEIGHT) & (x>=0) & (x<WIDTH)
            if qual_color:
                out[y[valid_proj], x[valid_proj]] = torch_to_numpy(allModel.eval_colors[valid_proj])
            else:
                out[y[valid_proj], x[valid_proj]] = torch_to_numpy(allModel.colors[valid_proj])
            out = numpy_to_torch(out)

        return out