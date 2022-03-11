import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import *
from utils.utils import *

class CentralDiff(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    @staticmethod
    def forward(Z):
        def getN(points):
            # Estimate normals from central differences (Ref[1]-Section3, 
            # link: https://en.wikipedia.org/wiki/Finite_difference), 
            # i.e., f(x+h/2)-f(x-h/2), of the vertext map.
            hL = points[1:-1, :-2]
            hR = points[1:-1, 2:]
            hD = points[:-2, 1:-1]
            hU = points[2:, 1:-1]
            N = torch.cross(hR-hL, hD-hU)

            N = torch.nn.functional.normalize(N, dim=-1)
            
            # Pad N to the same size as points
            out_N = torch.ones_like(points) * float('nan')
            out_N[1:-1,1:-1] = N
            
            return out_N, torch.isnan(out_N[:,:,0])
        
        Z = cv2.bilateralFilter(Z,15,80,80)
        ZF = cv2.bilateralFilter(Z,15,80,80)
        
        X, Y, Z = depth2pcd(numpy_to_torch(Z))
        points = torch.stack([X,Y,Z], dim=-1)
        norms, invalid = getN(points)

        # Ref[1]-"A copy of the depth map (and hence associated vertices and normals) are also 
        # denoised using a bilateral filter (for camera pose estimation later)."
        XF, YF, ZF = depth2pcd(numpy_to_torch(ZF))
        pointsF = torch.stack([XF,YF,ZF], dim=-1)
        normsF, invalidF = getN(pointsF)

        # Update the valid map.
        norm_angs = torch_inner_prod(norms, normsF)
        pt_dists = torch_distance(points, pointsF)
        invalid = (torch.abs(Z) < 1.0) | (torch.abs(Z) > 10.0) | invalid | invalidF | \
            (norm_angs < THRESHOLD_COSINE_ANGLE) | (pt_dists > THRESHOLD_DISTANCE)

        points[invalid] = float('nan')
        norms[invalid] = float('nan')
        return points, norms, invalid

# Code copied from https://github.com/Charmve/SNE-RoadSeg2, ./models/sne_model.py
class SNE(nn.Module):
    """Our SNE takes depth and camera intrinsic parameters as input,
    and outputs normal estimations.
    """
    def __init__(self):
        super(SNE, self).__init__()

    @staticmethod
    def forward(Z):
        h,w = HEIGHT, WIDTH
        Z = cv2.bilateralFilter(Z,15,80,80)
        X, Y, Z = depth2pcd(numpy_to_torch(Z))
        points = torch.stack([X,Y,Z], dim=-1)
        Y = -Y
        Z = -Z
        
        invalid = torch.isnan(Z)
        Z[invalid] = 0
        D = torch.div(torch.ones((h, w), device=dev), Z)

        Gx = torch.tensor([[0,0,0],[-1,0,1],[0,0,0]], dtype=fl64_, device=dev)
        Gy = torch.tensor([[0,-1,0],[0,0,0],[0,1,0]], dtype=fl64_, device=dev)

        Gu = F.conv2d(D.view(1,1,h,w), Gx.view(1,1,3,3), padding=1)
        Gv = F.conv2d(D.view(1,1,h,w), Gy.view(1,1,3,3), padding=1)

        nx_t = Gu * fx # 1, 1, h, w
        ny_t = Gv * fy # 1, 1, h, w

        phi = torch.atan(torch.div(ny_t, nx_t)) + torch.ones([1,1,h,w], device=dev)*3.141592657
        a = torch.cos(phi)
        b = torch.sin(phi)

        diffKernelArray = torch.tensor([[-1, 0, 0, 0, 1, 0, 0, 0, 0],
                                        [ 0,-1, 0, 0, 1, 0, 0, 0, 0],
                                        [ 0, 0,-1, 0, 1, 0, 0, 0, 0],
                                        [ 0, 0, 0,-1, 1, 0, 0, 0, 0],
                                        [ 0, 0, 0, 0, 1,-1, 0, 0, 0],
                                        [ 0, 0, 0, 0, 1, 0,-1, 0, 0],
                                        [ 0, 0, 0, 0, 1, 0, 0,-1, 0],
                                        [ 0, 0, 0, 0, 1, 0, 0, 0,-1]], dtype=fl64_, device=dev)

        sum_nx = torch.zeros((1,1,h,w), dtype=fl64_, device=dev)
        sum_ny = torch.zeros((1,1,h,w), dtype=fl64_, device=dev)
        sum_nz = torch.zeros((1,1,h,w), dtype=fl64_, device=dev)

        for i in range(8):
            diffKernel = diffKernelArray[i].view(1,1,3,3)
            X_d = F.conv2d(X.view(1,1,h,w), diffKernel, padding=1)
            Y_d = F.conv2d(Y.view(1,1,h,w), diffKernel, padding=1)
            Z_d = F.conv2d(Z.view(1,1,h,w), diffKernel, padding=1)

            nz_i = torch.div((torch.mul(nx_t, X_d) + torch.mul(ny_t, Y_d)), Z_d)
            norm = torch.sqrt(torch.mul(nx_t, nx_t) + torch.mul(ny_t, ny_t) + torch.mul(nz_i, nz_i))
            nx_t_i = torch.div(nx_t, norm)
            ny_t_i = torch.div(ny_t, norm)
            nz_t_i = torch.div(nz_i, norm)

            nx_t_i[torch.isnan(nx_t_i)] = 0
            ny_t_i[torch.isnan(ny_t_i)] = 0
            nz_t_i[torch.isnan(nz_t_i)] = 0

            sum_nx = sum_nx + nx_t_i
            sum_ny = sum_ny + ny_t_i
            sum_nz = sum_nz + nz_t_i

        theta = -torch.atan(torch.div((torch.mul(sum_nx, a) + torch.mul(sum_ny, b)), sum_nz))
        nx = torch.mul(torch.sin(theta), torch.cos(phi))
        ny = torch.mul(torch.sin(theta), torch.sin(phi))
        nz = torch.cos(theta)

        sign = torch.ones((1,1,h,w), dtype=fl64_, device=dev)
        sign[ny > 0] = -1

        nx = torch.mul(nx, sign)[0,0,...]
        ny = torch.mul(ny, sign)[0,0,...]
        nz = torch.mul(nz, sign)[0,0,...]

        invalid = invalid | torch.isnan(nz) | (torch.abs(Z) < 1.0) | (torch.abs(Z) > 10.0)
        norms = torch.stack([nx, -ny, -nz], dim=-1)
        points[invalid] = float('nan')
        norms[invalid] = float('nan')

        # Left-hand ---> Right-hand
        return points, norms, invalid