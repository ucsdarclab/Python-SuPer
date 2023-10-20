import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from utils.utils import torch_distance


def init_graph(valid, step=1):
    """
    Init graph from a HxW valid map.
    """
    h, w = valid.size()

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

class DirectDeformGraph(nn.Module):
    '''
    Directly get the ED node graph through gridding or random (uniform) selection.
    '''

    def __init__(self, opt) -> None:
        super(DirectDeformGraph, self).__init__()
        self.opt = opt

    def init_ED_nodes(self, inputs, data, 
    candidates, candidates_norms, candidates_seg=None, candidates_seg_conf=None,
    downsample_params=None, ball_piv_radii = [0.08, 0.1, 0.15], edge_identify_method='grid_mesh'):

        if edge_identify_method == 'ball_pivoting':
            rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(ball_piv_radii))

            # Fliter isolated points.
            rec_mesh.compute_adjacency_list()
            val_points = torch.tensor([len(adjacency) >= 3 for adjacency in rec_mesh.adjacency_list])
            radii = []
            for k, adjacency in enumerate(rec_mesh.adjacency_list):
                if val_points[k]:
                    adjacency = np.array(list(adjacency))
                    adjacency = adjacency[val_points[adjacency]]
                    if len(adjacency) >= 3:
                        radii.append(torch.cdist(points[k][None, ...], points[adjacency]).mean())
                    else:
                        val_points[k] = False
            radii = torch.stack(radii)
            points = points[val_points]
            norms = norms[val_points]
            seg_conf = seg_conf[val_points]
        
            # edge_index: Map from old index to new index
            id_map = -torch.ones(len(val_points), dtype=long_)
            id_map[val_points] = torch.arange(torch.count_nonzero(val_points))
            triangles = numpy_to_torch(np.asarray(rec_mesh.triangles), dtype=long_)
            triangles = id_map[triangles]
            triangles = triangles[~torch.any(triangles == -1, dim=1)]
            edge_index = torch.cat([triangles[:,0:2], triangles[:,1:]], dim=0)
            edge_index = torch.unique(torch.sort(edge_index, dim=1)[0], dim=0).permute(1, 0)

            triangles_areas = torch.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                                    points[triangles[:, 2]] - points[triangles[:, 0]],
                                    dim=1)
            triangles_areas = 0.5 * torch.sqrt((triangles_areas**2).sum(1) + 1e-13)
            # triangles_areas = 0.25 * (triangles_areas**2).sum(1)
        
        elif edge_identify_method == 'knn':
            dists, edge_index = find_knn(points, points, k = self.opt.num_ED_neighbors + 1)
            radii = torch.sqrt(dists[:, 1:]).mean(1)
            edge_index = torch.stack([
                            torch.flatten(torch.arange(len(points))[..., None].repeat(1, self.opt.num_ED_neighbors)).cuda(), 
                            torch.flatten(edge_index[:, 1:])]
                            , dim=0)
            triangles = None
            triangles_areas = None
        
        elif edge_identify_method == 'grid_mesh':
            val_points, edge_index, triangles = init_graph(data.index_map >= 0, step=self.opt.mesh_step_size)
            val_points = data.index_map[val_points]

            points = candidates[val_points]
            norms = candidates_norms[val_points]
            if candidates_seg_conf is None:
                seg_conf = None
                seg = None
            else:
                seg_conf = candidates_seg_conf[val_points]
                seg = torch.argmax(seg_conf, dim=1)

            # If hard_seg, delete the boundary edges.
            if hasattr(self.opt, 'hard_seg'):
                if self.opt.hard_seg and self.opt.mesh_face:
                    inside_edges = seg[edge_index[0]] == seg[edge_index[1]]
                    edge_index = torch.stack([edge_index[0][inside_edges], edge_index[1][inside_edges]], dim=0)

                    inside_triangles = (seg[triangles[0]] == seg[triangles[1]]) & (seg[triangles[0]] == seg[triangles[2]])
                    triangles = torch.stack([triangles[0][inside_triangles], 
                                             triangles[1][inside_triangles], 
                                             triangles[2][inside_triangles]], dim=0)

            edges_lens = torch.norm(points[edge_index[0]] - points[edge_index[1]], dim=1)
            radii = []
            for k in range(len(points)):
                radii.append(edges_lens[torch.any(edge_index == k, dim=0)].mean()) # Old setup: 0.6 * mean
            radii = torch.stack(radii)

            triangles_areas = torch.cross(points[triangles[1]] - points[triangles[0]],
                                    points[triangles[2]] - points[triangles[0]],
                                    dim=1)
            triangles_areas = 0.5 * torch.sqrt((triangles_areas**2).sum(1) + 1e-13)
        
        invalid = torch.isnan(radii)
        if torch.any(invalid):
            radii[invalid] = radii[~invalid].mean()

        return points, norms, radii, edge_index, triangles, triangles_areas, seg, seg_conf

    def forward(self, inputs, data):

        points = data.points
        norms = data.norms
        valid = data.valid.view(self.opt.height, self.opt.width)
        downsample_params = [tuple(self.opt.downsample_params[i * 3: (i + 1) * 3]) for i in range(int(len(self.opt.downsample_params)/3))]
        if hasattr(data, 'seg'):
            ED_points, ED_norms, radii, edge_index, triangles, triangles_areas, seg, seg_conf = self.init_ED_nodes(inputs, data, points, norms,
                candidates_seg=data.seg, candidates_seg_conf=data.seg_conf,
                downsample_params=downsample_params, ball_piv_radii=self.opt.ball_piv_radii)
        else:
            ED_points, ED_norms, radii, edge_index, triangles, triangles_areas, _, _ = self.init_ED_nodes(inputs, data, points, norms, 
                downsample_params=downsample_params, ball_piv_radii=self.opt.ball_piv_radii)
        num = len(ED_points)
        edges_lens = torch_distance(ED_points[edge_index[0]], ED_points[edge_index[1]])

        graph = Data(points=ED_points, norms=ED_norms, radii=radii,
            edge_index=edge_index, edges_lens=edges_lens,
            triangles=triangles, triangles_areas=triangles_areas,
            num=num, param_num=num*7) 

        if self.opt.method == 'semantic-super':
            graph.seg = seg
            graph.seg_conf = seg_conf

        return graph