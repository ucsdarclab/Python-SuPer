import math
import numpy as np
import cv2
import open3d as o3d
import torch
from torch import nn

from pytorch3d.renderer.points.pulsar import Renderer
from pytorch3d.transforms import matrix_to_rotation_6d

# from models.loss import *
from utils.config import *
from utils.utils import *

# Ref: Pulsar: Efficient Sphere-based Neural Rendering
class Pulsar(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cam_params = self.get_cam_params()
        self.gamma = 1.0e-5

    def get_cam_params(self, opencv=True):

        R = torch.eye(3, dtype=fl32_)
        openCV2PyTorch3D_R = torch.tensor(
            [[-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0,  0.0, 1.0]], 
            dtype=fl32_)
        if(opencv==True):
            R *= openCV2PyTorch3D_R
        
        r = matrix_to_rotation_6d(R)
        T = torch.zeros((3,1))
        R = torch.cat([R, T], dim=1)
        R = torch.cat([R, torch.tensor([[0.,0.,0.,1]])], dim=0)
        focal_length = 0.01

        cam_params = torch.tensor([
            -R[0,3], -R[1,3], -R[2,3],  # camera position
            -r[0], -r[1], -r[2],  
            -r[3], -r[4], -r[5],  # camera rotation
            focal_length,
            (focal_length*WIDTH)/fx,
            -math.ceil(cx - WIDTH / 2),
            -math.ceil(cy - HEIGHT / 2),
        ], dtype=fl32_)
        return cam_params

    def forward(self, allModel, qual_color=False):

        if qual_color:
            valid = ~torch.isnan(allModel.eval_colors[:,0])
            points = allModel.points[valid]
            colors = allModel.eval_colors[valid]
        else:
            points = allModel.points
            colors = allModel.colors
        surfel_num = len(points)

        # Points
        self.register_parameter(
            "vert_pos", 
            nn.Parameter(
                points.float().cpu(), requires_grad=True
            ),
        )
        
        # Colors
        self.register_parameter(
            "vert_col",
            nn.Parameter(
                colors.float().cpu(), requires_grad=True
            ),
        )

        # Radii of points
        self.register_parameter(
            "vert_rad",
            nn.Parameter(
                torch.ones(surfel_num, 
                    dtype=fl32_) * 0.1, # 0.0001
                requires_grad=True
            ),
        )
        
        # # Camera params
        # self.register_buffer("cam_params", self.cam_params)

        # The volumetric optimization works better with a higher number of tracked
        # intersections per ray.
        renderer = Renderer(
            WIDTH, HEIGHT, surfel_num, n_track=64, right_handed_system=True
        )

        return renderer.forward(
            self.vert_pos,
            self.vert_col,
            self.vert_rad,
            self.cam_params,
            self.gamma,
            15.0,
            return_forward_info=True,
        )[0].to(device=dev)

# Project with rounded coordinates.
class Projector(nn.Module):
    def __init__(self, method="direct") -> None:
        super().__init__()

        self.method = method

    def forward(self, allModel, qual_color=False):

        if self.method == "direct":
            y, x, _, valid_indices = pcd2depth(allModel.points, depth_sort=True)
            out = torch.zeros((HEIGHT,WIDTH,3), device=dev)
            if qual_color:
                out[y,x] = allModel.eval_colors[valid_indices].float()
            else:
                out[y,x] = allModel.colors[valid_indices]

        elif self.method == "opencv":
            rod, _ = cv2.Rodrigues(np.eye(3)) # [R]otation matrix.
            # Flip the points upside down
            points = torch_to_numpy(allModel.points)
            points[:,1] = -points[:,1]
            points[:,2] = -points[:,2]
            img_quads, _ = cv2.projectPoints(points, rod, np.zeros((3,1)), K, np.array([]))
            out = np.zeros((HEIGHT,WIDTH,3))
            y = np.round(img_quads[:,0,1]).astype(int)
            x = np.round(img_quads[:,0,0]).astype(int)
            valid_proj = (y>=0) & (y<HEIGHT) & (x>=0) & (x<WIDTH)
            if qual_color:
                out[y[valid_proj], x[valid_proj]] = torch_to_numpy(allModel.eval_colors[valid_proj])
            else:
                out[y[valid_proj], x[valid_proj]] = torch_to_numpy(allModel.colors[valid_proj])
            out = numpy_to_torch(out)

        # elif self.method == "o3d":
        #     vis = o3d.visualization.Visualizer()
        #     vis.create_window()
        #     vis.remove_geometry(allModel.pcd)
        #     vis.add_geometry(allModel.pcd)
        #     vis.poll_events()
        #     vis.update_renderer()
        #     out = numpy_to_torch(np.asarray(vis.capture_screen_float_buffer(True)))

        return out