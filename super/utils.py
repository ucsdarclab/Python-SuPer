import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.data import Data

from utils.config import *
from utils.utils import *

from depth.monodepth2.layers import disp_to_depth, BackprojectDepth, Project3D

from bnmorph.layers import grad_computation_tools

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

def depth_preprocessing(opt, models, inputs, scale=0):
    """
    Convert depth map to 3D point cloud (x, y, z, norms) in the camera frame.
    """
    with torch.no_grad():
        if ("pcd", scale) in inputs:
            points = inputs[("pcd", scale)]
            if opt.normal_model == 'naive':
                norms, valid = getN(points)
            elif opt.normal_model == '8neighbors':
                norms, valid = getN(points, colors=inputs[("color", 0, 0)])
            valid &= ~torch.any(torch.isnan(points), dim=3)
            Z = - inputs[("depth", scale)]
        else:
            Z = inputs[("depth", scale)]
            Z[Z==0] = np.nan # Convert invalid depth values (0) to nan.
            Z = blur_image(Z)
            Z *= inputs["depth_scale"]

            # Estimate (x,y,z) and norms from the depth map.
            points, norms, valid = pcd2norm(opt, inputs, Z)
        inputs[("normal", 0)] = norms
        points = points[0].type(fl64_)
        norms = norms[0].type(fl64_)
        if ("seman", 0) in inputs:
            seman = inputs[("seman", 0)][0, 0]
            seman_conf = inputs[("seman_conf", 0)][0].permute(1, 2, 0)
        valid = valid[0]

        # # For visualizing the normals.
        # normal_map = torch.zeros_like(norms)
        # normal_map[valid] = norms[valid]

        # # Init surfel graph, get the edges and faces of the graph.
        # valid_verts, edge_index, face_index = init_graph(valid)
        valid_verts = valid
        points = points[valid_verts]
        norms = norms[valid_verts]

        index_map = - torch.ones((inputs["height"], inputs["width"]), dtype=long_)
        index_map[valid_verts] = torch.arange(valid_verts.count_nonzero())
        
        # Calculate the radii.
        radii = Z[0,0][valid_verts] / (np.sqrt(2) * inputs["fx"] * \
            torch.clamp(torch.abs(norms[...,2]), 0.26, 1.0))
        # radii = 0.002 * torch.ones(torch.count_nonzero(valid_verts), dtype=fl64_).cuda()

        # Calculate the confidence.
        if opt.use_ssim_conf:
            confs = torch.sigmoid(inputs[("disp_conf", 0)][0, 0])
        else:
            U, V = get_grid_coords(inputs["height"], inputs["width"])
            scale_u, scale_v = U / inputs["width"], V / inputs["height"]
            dc2 = (2.*scale_u-1.)**2 + (2.*scale_v-1.)**2
            confs = torch.exp(-dc2 * inputs["divterm"]).cuda()
        
        if opt.use_seman_conf:
            P = inputs[("warp_seman_conf", "s")][0].type(fl32_)
            Q = inputs[("seman_conf", 0)][0].type(fl32_)
            seman_confs = torch.exp(- 0.1 * (0.5 * (P * (P / (Q + 1e-13) + 1e-13).log()).sum(0) + \
                            0.5 * (Q * (Q / (P + 1e-13) + 1e-13).log()).sum(0))
                            )
            confs = 0.5 * (confs + seman_confs)

        rgb = inputs[("color", 0, scale)]
        colors = rgb[0].permute(1,2,0)[valid_verts]

    data = Data(points=points,
                norms=norms,
                colors=colors,
                radii=radii, 
                confs=confs[valid_verts],
                valid=valid_verts.view(-1),
                index_map=index_map,
                time=inputs["ID"])
    # Deleted parameters: edge_index, face_index, rgb, ID=inputs["ID"]
    
    if ("seman", 0) in inputs:
        data.seman = seman[valid_verts]
        data.seman_conf = seman_conf[valid_verts]

        # Calculate the (normalized) distance of project points to the semantic region edges.
        with torch.no_grad():
            tool = grad_computation_tools(batch_size=opt.batch_size, height=opt.height,
                                               width=opt.width).cuda()
            kernels = [3, 3, 3]
            edge_pts = []
            for class_id in range(opt.num_classes):
                seman_grad_bin = tool.get_semanticsEdge(
                    inputs[("seman", 0)], foregroundType=[class_id],
                    erode_foreground=True, kernel_size=kernels[class_id])
                edge_y, edge_x = seman_grad_bin[0,0].nonzero(as_tuple=True)
                edge_pts.append(torch.stack([edge_x/opt.width, edge_y/opt.height], dim=1).type(fl64_))

            sf_y, sf_x, _, _ = pcd2depth(inputs, data.points, round_coords=False)
            sf_coords = torch.stack([sf_x/opt.width, sf_y/opt.height], dim=1)
            dist2edge = torch.zeros_like(data.radii)
            for class_id in range(opt.num_classes):
                val_points = data.seman == class_id
                knn_dist2edge, knn_edge_ids = find_knn(sf_coords[val_points], 
                                                       edge_pts[class_id], k=1)
                dist2edge[val_points] = knn_dist2edge[:, 0]
            data.dist2edge = dist2edge
    
    return data, inputs
    
"""
getN() and pcd2norm() together are used to estimate the normal for
each point in the input point cloud.
Method: Estimate normals from central differences (Ref[1]-Section3, 
        link: https://en.wikipedia.org/wiki/Finite_difference), 
        i.e., f(x+h/2)-f(x-h/2), of the vertext map.
"""
def getN(points, colors=None):
    if colors is None:
        b, h, w, _ = points.size()
        points = torch.nn.functional.pad(points, (0,0,1,1,1,1), value=float('nan'))
        hL = points[:, 1:-1, :-2, :].reshape(-1,3)
        hR = points[:, 1:-1, 2:, :].reshape(-1,3)
        hD = points[:, :-2, 1:-1, :].reshape(-1,3)
        hU = points[:, 2:, 1:-1, :].reshape(-1,3)

        N = torch.cross(hR-hL, hD-hU)
        N = torch.nn.functional.normalize(N, dim=-1).reshape(b,h,w,3)
        
        return N, ~torch.any(torch.isnan(N), -1)
    
    else:
        b, h, w, _ = points.size()

        colors = colors.permute(0, 2, 3, 1)
        colors = torch.nn.functional.pad(colors, (0,0,1,1,1,1), value=float('nan'))
        col_cen = colors[:, 1:-1, 1:-1, :].reshape(-1,3)
        col_hL = torch.exp(-torch.mean(torch.abs(colors[:, 1:-1, :-2, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hLU = torch.exp(-torch.mean(torch.abs(colors[:, :-2, :-2, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hU = torch.exp(-torch.mean(torch.abs(colors[:, :-2, 1:-1, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hRU = torch.exp(-torch.mean(torch.abs(colors[:, :-2, 2:, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hR = torch.exp(-torch.mean(torch.abs(colors[:, 1:-1, 2:, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hRD = torch.exp(-torch.mean(torch.abs(colors[:, 2:, 2:, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hD = torch.exp(-torch.mean(torch.abs(colors[:, 2:, 1:-1, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hDL = torch.exp(-torch.mean(torch.abs(colors[:, 2:, :-2, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        
        points = torch.nn.functional.pad(points, (0,0,1,1,1,1), value=float('nan'))
        cen = points[:, 1:-1, 1:-1, :].reshape(-1,3)
        hL = (points[:, 1:-1, :-2, :].reshape(-1,3) - cen) * col_hL
        hLU = (points[:, :-2, :-2, :].reshape(-1,3) - cen) * col_hLU
        hU = (points[:, :-2, 1:-1, :].reshape(-1,3) - cen) * col_hU
        hRU = (points[:, :-2, 2:, :].reshape(-1,3) - cen) * col_hRU
        hR = (points[:, 1:-1, 2:, :].reshape(-1,3) - cen) * col_hR
        hRD = (points[:, 2:, 2:, :].reshape(-1,3) - cen) * col_hRD
        hD = (points[:, 2:, 1:-1, :].reshape(-1,3) - cen) * col_hD
        hDL = (points[:, 2:, :-2, :].reshape(-1,3) - cen) * col_hDL

        N = torch.stack([
            torch.cross(hL, hLU + hU + hRU + hR + hRD + hD + hDL),
            torch.cross(hLU, hU + hRU + hR + hRD + hD + hDL),
            torch.cross(hU, hRU + hR + hRD + hD + hDL),
            torch.cross(hRU, hR + hRD + hD + hDL),
            torch.cross(hR, hRD + hD + hDL),
            torch.cross(hRD, hD + hDL),
            torch.cross(hD, hDL)],
            dim=2).sum(2)
        N = torch.nn.functional.normalize(N, dim=-1).reshape(b,h,w,3)
        
        return N, ~torch.any(torch.isnan(N), -1)

def pcd2norm(opt, inputs, Z):
    ZF = blur_image(Z)
    
    X, Y, Z = depth2pcd(inputs, Z)
    points = torch.stack([X,Y,Z], dim=-1)
    if opt.normal_model == 'naive':
        norms, valid = getN(points)
    elif opt.normal_model == '8neighbors':
        norms, valid = getN(points, colors=inputs[("color", 0, 0)])

    # Ref[1]-"A copy of the depth map (and hence associated vertices and normals) are also 
    # denoised using a bilateral filter (for camera pose estimation later)."
    XF, YF, ZF = depth2pcd(inputs, ZF)
    pointsF = torch.stack([XF,YF,ZF], dim=-1)
    if opt.normal_model == 'naive':
        normsF, validF = getN(pointsF)
    elif opt.normal_model == '8neighbors':
        normsF, validF = getN(pointsF, colors=inputs[("color", 0, 0)])

    # Update the valid map.
    norm_angs = torch_inner_prod(norms, normsF)
    pt_dists = torch_distance(points, pointsF)
    valid = valid & validF & (norm_angs >= opt.th_cosine_ang) & \
        (pt_dists <= opt.th_dist)

    return points, norms, valid


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

    index_map = -torch.ones((h,w), dtype=long_).cuda()
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