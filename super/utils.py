import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.data import Data

from utils.utils import *

from depth.monodepth2.layers import disp_to_depth, BackprojectDepth, Project3D

def get_skew(inputs):

    a1, a2, a3 = torch.split(inputs, 1, dim=-1)
    zeros = torch.zeros_like(a1)

    return torch.stack([torch.cat([zeros, a3, -a2], dim=-1), \
        torch.cat([-a3, zeros, a1], dim=-1), \
        torch.cat([a2, -a1, zeros], dim=-1)], dim=3)

# Transformation of surfels: eq (10) in SuPer paper.
def Trans_points(d_surfels, ednodes, beta, surfel_knn_weights, grad=False, skew_v=None):
    # Inputs: d_surfels: p - g_i; ednodes: g_i; 
    # beta: [q_i; b_i]; surfel_knn_weights: alpha_i.

    # 'trans_surfels': T(q_i,b_i)(p-g_i); 'Jacobian': d_out/d_q_i
    trans_surfels, Jacobian = transformQuatT(d_surfels, beta, grad=grad, skew_v=skew_v)
    
    trans_surfels += ednodes
    if not np.isscalar(surfel_knn_weights):
        surfel_knn_weights = surfel_knn_weights.unsqueeze(-1)
    trans_surfels = torch.sum(surfel_knn_weights * trans_surfels, dim=-2)

    if grad:
        Jacobian *= surfel_knn_weights.unsqueeze(-1)
    
    return trans_surfels, Jacobian

# Output: T(q,b)v in eq (10)/(11) in SuPer paper. 
# Inputs: 'beta': [q;b] or [q] for b=0. 
def transformQuatT(v, beta, grad=False, skew_v=None):

    device = v.device

    qw = beta[...,0:1]
    qv = beta[...,1:4]

    cross_prod = torch.cross(qv, v, dim=-1)

    tv = v + 2.0 * qw * cross_prod + \
        2.0 * torch.cross(qv, cross_prod, dim=-1) 
        
    # tv = rv + t
    if beta.size()[-1] == 7:
        tv += beta[...,4:7]

    if grad:
        eye_3 = torch.eye(3, dtype=torch.float64, device=device).view(1,1,3,3)

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
    



def init_graph(opt, valid, step=1):
    """
    Init graph from a HxW valid map.
    """
    # device = valid.device

    # if not torch.is_tensor(valid):
    #     valid = numpy_to_torch(valid, device=device)
    # else:
    #     valid = valid.to(device) # TODO: Directly to GPU?
    h, w = valid.size()

    # start_x = int(opt.width * opt.depth_width_range[0])
    # end_x = int(opt.width * opt.depth_width_range[1])
    # start_u = start_x + int(((end_x - start_x) % step) / 2)
    # u = torch.arange(start_u, w-1, step).cuda()
    
    # start_v = int((opt.height % step) / 2)
    # v = torch.arange(start_v, h-1, step).cuda()

    u = torch.arange(0, w-1, step).cuda()
    v = torch.arange(0, h-1, step).cuda()
    
    u, v = torch.meshgrid(u, v, indexing='xy')
    anchor_valid = valid[v, u]
    u = u[anchor_valid] # size: N
    v = v[anchor_valid] # N

    index_map = -torch.ones((h,w), dtype=torch.long).cuda()
    index_map[v, u] = torch.arange(len(u)).cuda()

    # # s --- pt1
    # # | \
    # # |   \
    # # pt3  pt2
    # # N x 3(# edges) x 2(# verts for each edge) x 2(x,y)
    # edges = torch.stack([u, v], dim=1).view(-1,1,1,2).repeat(1,3,2,1)
    # edges[:,0,1,0] += step # pt1-x
    # edges[:,1,1,0] += step # pt2-x
    # edges[:,1,1,1] += step # pt2-y
    # edges[:,2,1,1] += step # pt3-y
    # s --- pt1
    # | \  /
    # | /  \
    # pt3  pt2
    # N x 3(# edges) x 2(# verts for each edge) x 2(x,y)
    edges = torch.stack([u, v], dim=1).view(-1,1,1,2).repeat(1,4,2,1)
    edges[:,0,1,0] += step # pt1-x
    edges[:,1,1,0] += step # pt2-x
    edges[:,1,1,1] += step # pt2-y
    edges[:,2,1,1] += step # pt3-y
    edges[:,3,0,0] += step # pt1-x
    edges[:,3,1,1] += step # pt3-y

    # N x 2(# faces) x 3(# verts for each face) x 2
    faces = torch.cat([ \
        torch.stack([u, v], dim=1).view(-1,1,1,2).repeat(1,2,1,1), \
        torch.stack([edges[:,0:2,1,:],edges[:,1:3,1,:]], dim=1)], \
        dim=2)
    
    valid = F.pad(valid, (0, step, 0, step), value=False)
    
    edges = edges[~torch.any(~valid[edges[...,1], edges[...,0]], dim=2)]
    edges = index_map[edges[...,1], edges[...,0]]
    edges = edges[~torch.any(edges < 0, dim=1)]
    
    faces = faces[~torch.any(~valid[faces[...,1], faces[...,0]], dim=2)]
    faces = index_map[faces[...,1], faces[...,0]]
    faces = faces[~torch.any(faces < 0, dim=1)]

    return index_map>=0, torch.t(edges), torch.t(faces) # Size: 2xN, 3xN
# TODO: Debug
# def init_graph(index_map, step=1):
#     """
#     Init graph from a HxW valid map.
#     """
#     ids = index_map[step:(index_map.size(0)-step):step, step:(index_map.size(1)-step):step]
#     val_ids = ids[ids >= 0]
    
#     # s --- pt1
#     # | \
#     # |   \
#     # pt3  pt2
#     ids[ids >= 0] = torch.arange(torch.count_nonzero(ids >= 0))
#     triangles = torch.cat([torch.stack([torch.flatten(ids[:-1, :-1]), 
#                                         torch.flatten(ids[1:, :-1]), 
#                                         torch.flatten(ids[1:, 1:])], dim=1),
#                            torch.stack([torch.flatten(ids[:-1, :-1]), 
#                                         torch.flatten(ids[:-1, 1:]), 
#                                         torch.flatten(ids[1:, 1:])], dim=1)]
#                             ,dim=0)
    
#     edge_index = torch.cat([torch.stack([torch.flatten(ids[:, :-1]), 
#                                         torch.flatten(ids[:, 1:])], dim=1),
#                             torch.stack([torch.flatten(ids[:-1, :]), 
#                                         torch.flatten(ids[1:, :])], dim=1),
#                             torch.stack([torch.flatten(ids[:-1, :-1]), 
#                                         torch.flatten(ids[1:, 1:])], dim=1)]
#                             ,dim=0)

#     triangles = triangles[~torch.any(triangles < 0, dim=1)]
#     edge_index = edge_index[~torch.any(edge_index < 0, dim=1)]

#     return val_ids, edge_index.permute(1, 0), triangles