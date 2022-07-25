import torch
from torch_geometric.data import Data

from utils.config import *
from utils.utils import *


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


def depth_preprocessing(CamParams, data, scale=0):
    """
    Convert depth map to 3D point cloud (x, y, z, norms) in the camera frame.
    """
    Z = data[("depth", scale)]
    Z[Z==0] = np.nan # Convert invalid depth values (0) to nan.
    Z = blur_image(Z)
    Z *= CamParams.depth_scale

    # Estimate (x,y,z) and norms from the depth map.
    points, norms, valid = pcd2norm(CamParams, Z)
    points = points[0]
    norms = norms[0]
    valid = valid[0]

    # # For visualizing the normals.
    # normal_map = torch.zeros_like(norms)
    # normal_map[valid] = norms[valid]

    # Init surfel graph, get the edges and faces of the graph.
    valid_verts, edge_index, face_index = init_graph(valid)
    points = points[valid_verts]
    norms = norms[valid_verts]

    # TODO
    index_map = - torch.ones((CamParams.HEIGHT, CamParams.WIDTH), dtype=long_)
    index_map[valid_verts] = torch.arange(valid_verts.count_nonzero())
    
    # Calculate the radii.
    radii = Z[0,0][valid_verts] / (CamParams.SQRT2 * CamParams.fx * \
        torch.clamp(torch.abs(norms[...,2]), 0.26, 1.0))

    # Calculate the confidence.
    U, V = get_grid_coords(CamParams.HEIGHT, CamParams.WIDTH)
    scale_u, scale_v = U / CamParams.WIDTH, V / CamParams.HEIGHT
    dc2 = (2.*scale_u-1.)**2 + (2.*scale_v-1.)**2
    confs = torch.exp(-dc2 * CamParams.DIVTERM).to(dev)

    rgb = data[("color", 0, scale)]
    colors = rgb[0].permute(1,2,0)[valid_verts]

    return Data(
        points=points,
        norms=norms,
        colors=colors,
        radii=radii, 
        confs=confs[valid_verts],
        edge_index=edge_index,
        face_index=face_index, 
        valid=valid_verts.view(-1),
        index_map=index_map,
        rgb=rgb, time=data["ID"], ID=data["ID"])
    
"""
getN() and pcd2norm() together are used to estimate the normal for
each point in the input point cloud.
Method: Estimate normals from central differences (Ref[1]-Section3, 
        link: https://en.wikipedia.org/wiki/Finite_difference), 
        i.e., f(x+h/2)-f(x-h/2), of the vertext map.
"""
def getN(points):
    b, h, w, _ = points.size()
    points = torch.nn.functional.pad(points, (0,0,1,1,1,1), value=float('nan'))
    hL = points[:, 1:-1, :-2, :].reshape(-1,3)
    hR = points[:, 1:-1, 2:, :].reshape(-1,3)
    hD = points[:, :-2, 1:-1, :].reshape(-1,3)
    hU = points[:, 2:, 1:-1, :].reshape(-1,3)

    N = torch.cross(hR-hL, hD-hU)
    N = torch.nn.functional.normalize(N, dim=-1).reshape(b,h,w,3)
    
    return N, ~torch.any(torch.isnan(N), -1)

def pcd2norm(CamParams, Z):
    ZF = blur_image(Z)
    
    X, Y, Z = depth2pcd(CamParams, Z)
    points = torch.stack([X,-Y,-Z], dim=-1)
    norms, valid = getN(points)

    # Ref[1]-"A copy of the depth map (and hence associated vertices and normals) are also 
    # denoised using a bilateral filter (for camera pose estimation later)."
    XF, YF, ZF = depth2pcd(CamParams, ZF)
    pointsF = torch.stack([XF,-YF,-ZF], dim=-1)
    normsF, validF = getN(pointsF)

    # Update the valid map.
    norm_angs = torch_inner_prod(norms, normsF)
    pt_dists = torch_distance(points, pointsF)
    valid = valid & validF & (norm_angs >= CamParams.THRESHOLD_COSINE_ANGLE) & \
        (pt_dists <= CamParams.THRESHOLD_DISTANCE)
    # (Z >= 1.0) & (Z <= 10.0)

    return points, norms, valid


def init_graph(valid, step=1):
    """
    Init graph from a HxW valid map.
    """
    device = valid.device

    if not torch.is_tensor(valid):
        valid = numpy_to_torch(valid, device=device)
    else:
        valid = valid.to(device) # TODO: Directly to GPU?
    h, w = valid.size()

    # u = torch.arange(int(step/2), w, step, device=device)
    # v = torch.arange(int(step/2), h, step, device=device)
    u = torch.arange(0, w-1-step, step, device=device)
    v = torch.arange(0, h-1-step, step, device=device)
    
    u, v = torch.meshgrid(u, v, indexing='xy')
    anchor_valid = valid[v, u]
    u = u[anchor_valid] # size: N
    v = v[anchor_valid] # N

    index_map = -torch.ones((h,w), dtype=long_, device=device)
    index_map[v, u] = torch.arange(len(u), device=device)

    # s --- pt1
    # | \
    # |   \
    # pt3  pt2
    # N x 3(# edges) x 2(# verts for each edge) x 2(x,y)
    edges = torch.stack([u, v], dim=1).view(-1,1,1,2).repeat(1,3,2,1)
    edges[:,0,1,0] += step # pt1
    edges[:,1,1,0] += step # pt2
    edges[:,1,1,1] += step
    edges[:,2,1,1] += step # pt3

    # N x 2(# faces) x 3(# verts for each face) x 2
    faces = torch.cat([ \
        torch.stack([u, v], dim=1).view(-1,1,1,2).repeat(1,2,1,1), \
        torch.stack([edges[:,0:2,1,:],edges[:,1:,1,:]], dim=1)], \
        dim=2)
    
    edges = edges[~torch.any(~valid[edges[...,1], edges[...,0]], dim=2)]
    edges = index_map[edges[...,1], edges[...,0]]
    
    faces = faces[~torch.any(~valid[faces[...,1], faces[...,0]], dim=2)]
    faces = index_map[faces[...,1], faces[...,0]]

    return index_map>=0, torch.t(edges), torch.t(faces) # Size: 2xN, 3xN