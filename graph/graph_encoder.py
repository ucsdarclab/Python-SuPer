from collections import namedtuple

import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_scatter import scatter_add, scatter_mean
from torch_sparse import coalesce

from torch_geometric.utils import softmax
from torch_geometric.nn import (
    GCNConv,
    TopKPooling,
    EdgePooling,
    ASAPooling,
    max_pool,
    fps,
    knn_interpolate,
    voxel_grid,
    graclus,
)
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    sort_edge_index,
)
from torch_geometric.utils.repeat import repeat

from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
import torchvision.transforms as T
from torchvision.transforms import (
    Normalize,
)

# from models.preprocessing import *
from super.loss import *

from utils.utils import *

class DirectDeformGraph(nn.Module):
    '''
    Directly get the ED node graph through gridding or random (uniform) selection.
    '''

    def __init__(self, opt) -> None:
        super(DirectDeformGraph, self).__init__()
        self.opt = opt

    def select_ED_nodes(self, points, norms, seman_conf=None, downsample_params=None, method='ball_pivoting'):
        if method=='ball_pivoting':
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
            pcd.normals = o3d.utility.Vector3dVector(norms.cpu().numpy())
            if seman_conf is not None:
                pcd.colors = o3d.utility.Vector3dVector(seman_conf.cpu().numpy())
            
            if downsample_params is not None:
                for _downsample_params_ in downsample_params:
                    voxel_size, nb_neighbors, std_ratio = _downsample_params_
                    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                    _, ind = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=std_ratio)
                    pcd = pcd.select_by_index(ind)

            points = numpy_to_torch(np.asarray(pcd.points), dtype=fl64_)
            norms = numpy_to_torch(np.asarray(pcd.normals), dtype=fl64_)
            return pcd, points, norms

    def init_ED_nodes(self, inputs, data, 
    candidates, candidates_norms, candidates_seman=None, candidates_seman_conf=None,
    boundary_pts=None, boundary_rad=30,
    downsample_params=None, ball_piv_radii = [0.08, 0.1, 0.15], edge_identify_method='grid_mesh'):

        # ### Alpha shapes ###
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(candidates.cpu().numpy())
        
        # pcd = pcd.voxel_down_sample(voxel_size=0.02)
        # _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.2)
        # pcd = pcd.select_by_index(ind)
        # pcd = pcd.voxel_down_sample(voxel_size=0.08)
        # _, ind = pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=0.5)
        # pcd = pcd.select_by_index(ind)

        # alpha = 0.1
        # rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        # rec_mesh.compute_vertex_normals()
        # ### Alpha shapes ###
        
        # ### Ball pivot ###
        # if candidates_seman is None:
        #     pcd, points, norms = self.select_ED_nodes(candidates, candidates_norms, downsample_params=downsample_params)
        #     seman_conf = numpy_to_torch(np.asarray(pcd.colors))
        # else:
        #     pcd = o3d.geometry.PointCloud()
        #     points = []
        #     norms = []
        #     for class_id in range(candidates_seman_conf.size(1)):
        #         candidates_ids = candidates_seman == class_id
        #         _pcd_, _points_, _norms_ = self.select_ED_nodes(candidates[candidates_ids], candidates_norms[candidates_ids], 
        #                                     seman_conf=candidates_seman_conf[candidates_ids], downsample_params=downsample_params)
        #         pcd += _pcd_
        #         points.append(_points_)
        #         norms.append(_norms_)
        #     points = torch.cat(points, dim=0)
        #     norms = torch.cat(norms, dim=0)
        #     seman_conf = numpy_to_torch(np.asarray(pcd.colors))
        # ### Ball pivot ###

        # ## grid ###
        # step = 30
        # ids = data.index_map[::step, ::step]
        # ids = ids[ids >= 0]
        # points = candidates[ids]
        # norms = candidates_norms[ids]
        # seman_conf = candidates_seman_conf[ids]

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        # pcd.normals = o3d.utility.Vector3dVector(norms.cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(seman_conf.cpu().numpy())
        # ## grid ###

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
            seman_conf = seman_conf[val_points]
        
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
            if self.opt.hard_seg and False:
                seman_map = - torch.ones_like(data.index_map)
                seman_map[data.index_map >= 0] = data.seman.to(data.index_map.device)
                val_points = []
                edge_index = []
                triangles = []
                id_offset = 0
                for class_id in range(self.opt.num_classes):
                    step_scale = 1.0
                    while True:
                        if torch.count_nonzero(seman_map == class_id) == 0:
                            break
                            
                        _val_points_, _edge_index_, _triangles_ = init_graph(self.opt, seman_map == class_id, 
                                                                             step=int(self.opt.mesh_step_size/step_scale))
                        if torch.count_nonzero(_val_points_) < 4 * self.opt.num_ED_neighbors:
                            step_scale += 1.
                        else:
                            class_id_map = data.index_map[_val_points_]
                            val_points.append(class_id_map)
                            edge_index.append(_edge_index_ + id_offset)
                            triangles.append(_triangles_ + id_offset)
                            id_offset += len(class_id_map)
                            break
                val_points = torch.cat(val_points)
                edge_index = torch.cat(edge_index, dim=1)
                triangles = torch.cat(triangles, dim=1)
            else:
                val_points, edge_index, triangles = init_graph(self.opt, data.index_map >= 0, step=self.opt.mesh_step_size)
                val_points = data.index_map[val_points]

            points = candidates[val_points]
            norms = candidates_norms[val_points]
            seman_conf = candidates_seman_conf[val_points]
            seman = torch.argmax(seman_conf, dim=1)

            # If hard_seman, delete the boundary edges.
            if self.opt.hard_seg and (self.opt.mesh_edge or self.opt.mesh_face):
                inside_edges = seman[edge_index[0]] == seman[edge_index[1]]
                edge_index = torch.stack([edge_index[0][inside_edges], edge_index[1][inside_edges]], dim=0)

                inside_triangles = (seman[triangles[0]] == seman[triangles[1]]) & (seman[triangles[0]] == seman[triangles[2]])
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

        if boundary_pts is not None:
            y, x, _, _ = pcd2depth(inputs, points, round_coords=False)
            pts = torch.stack([x, y], dim=1).type(fl64_)
            dists, _ = find_knn(pts, boundary_pts, k=1)
            isBoundary = dists[:, 0] <= boundary_rad
        else:
            isBoundary = None

        return points, norms, radii, edge_index, triangles, triangles_areas, isBoundary, seman, seman_conf

    def forward(self, inputs, data):

        points = data.points
        norms = data.norms
        valid = data.valid.view(self.opt.height, self.opt.width)

        # Find ED nodes near the boundary.
        # if self.opt.method == "seman-super" and False:
        #     ED_points = []
        #     ED_norms = []
        #     radii = []
        #     edge_index = []
        #     edge_index_offset = 0
        #     isBoundary = []
        #     seman = []
        #     seman_conf = []
        #     downsample_params=[
        #         [(0.02, 50, 0.5), (0.05, 20, 1.), (0.1, 20, 1.)],
        #         [(0.02, 50, 0.5), (0.05, 20, 1.), (0.1, 20, 1.)],
        #         [(0.02, 50, 0.5), (0.05, 20, 1.), (0.1, 20, 1.)]]
        #     ball_piv_radii = [
        #         [0.08, 0.1, 0.15],
        #         [0.08, 0.1, 0.15],
        #         [0.08, 0.1, 0.15]]
        #     # [(0.02, 40, 0.2), (0.05, 20, 1.)] [0.005, 0.02, 0.04, 0.06]
        #     kernels = [3, 3, 3]
        #     for class_id in range(self.opt.num_classes):
        #         seman_grad_bin = self.tool.get_semanticsEdge(
        #             inputs[("seman", 0)], foregroundType=[class_id],
        #             erode_foreground=True, kernel_size=kernels[class_id])
        #         edge_y, edge_x = seman_grad_bin[0,0].nonzero(as_tuple=True)
        #         edge_pts = torch.stack([edge_x, edge_y], dim=1).type(fl64_)

        #         candidates_ids = data.seman == class_id
        #         _ED_points_, _ED_norms_, _radii_, _edge_index_, _isBoundary_, _, _seman_conf_ = self.init_ED_nodes(inputs, data, 
        #             points[candidates_ids], norms[candidates_ids], 
        #             candidates_seman_conf=data.seman_conf[candidates_ids],
        #             boundary_pts=edge_pts, boundary_rad=40,
        #             downsample_params=downsample_params[class_id], ball_piv_radii=ball_piv_radii[class_id])
        #             # boundary_rad=rads[class_id][0], inside_rad=rads[class_id][1], #boundary_pts=(edge_x, edge_y), 
        #         ED_points.append(_ED_points_)
        #         ED_norms.append(_ED_norms_)
        #         radii.append(_radii_)
        #         edge_index.append(_edge_index_+edge_index_offset)
        #         edge_index_offset += len(_ED_points_)
        #         isBoundary.append(_isBoundary_)
        #         seman.append(class_id * torch.ones(len(_ED_points_), dtype=long_).cuda())
        #         seman_conf.append(_seman_conf_)

        #     ED_points = torch.cat(ED_points, dim=0)
        #     ED_norms = torch.cat(ED_norms, dim=0)
        #     radii = torch.cat(radii)
        #     edge_index = torch.cat(edge_index, dim=1)
        #     isBoundary = torch.cat(isBoundary)
        #     seman = torch.cat(seman)
        #     seman_conf = torch.cat(seman_conf, dim=0)
        # else:
        # downsample_params=[(0.02, 50, 0.5), (0.05, 20, 1.), (0.1, 20, 1.)]
        # ball_piv_radii = [0.08, 0.1, 0.15]
        # downsample_params=[(0.05, 40, 0.5)]
        # ball_piv_radii = [0.04, 0.08, 0.1]
        downsample_params = [tuple(self.opt.downsample_params[i * 3: (i + 1) * 3]) for i in range(int(len(self.opt.downsample_params)/3))]
        if hasattr(data, 'seg'):
            ED_points, ED_norms, radii, edge_index, triangles, triangles_areas, isBoundary, seg, seg_conf = self.init_ED_nodes(inputs, data, points, norms,
                candidates_seman=data.seg, candidates_seman_conf=data.seg_conf,
                downsample_params=downsample_params, ball_piv_radii=self.opt.ball_piv_radii)
        else:
            ED_points, ED_norms, radii, edge_index, triangles, triangles_areas, _, _, _ = self.init_ED_nodes(inputs, data, points, norms, 
                downsample_params=downsample_params, ball_piv_radii=self.opt.ball_piv_radii)
        num = len(ED_points)
        edges_lens = torch_distance(ED_points[edge_index[0]], ED_points[edge_index[1]])

        if self.opt.method == 'semantic-super':
            isBoundary = ~(seg[edge_index[0]] == seg[edge_index[1]])
            isBoundaryFace = ~((seg[triangles[0]] == seg[triangles[1]]) & (seg[triangles[0]] == seg[triangles[2]]))

            # if self.opt.use_edge_ssim_hints:
            #     if self.opt.mesh_edge:
            #         boundary_edge_type = torch.ones((torch.count_nonzero(isBoundary), 10), dtype=fl32_).cuda()
            #     if self.opt.mesh_face:
            #         boundary_face_type = torch.ones((torch.count_nonzero(isBoundaryFace), 10), dtype=fl32_).cuda()

            isTool = (seg[edge_index[0]] == 2) & (seg[edge_index[1]] == 2)

        graph = Data(points=ED_points, norms=ED_norms, radii=radii,
            edge_index=edge_index, edges_lens=edges_lens,
            triangles=triangles, triangles_areas=triangles_areas,
            num=num, param_num=num*7) 
            # TODO edges_weights=edges_weights
            # index_map=index_map, 
            # neighbor_radii=[neighbor_radii],
        if hasattr(data, 'seg'):
            graph.seg = seg
            graph.seg_conf = seg_conf
            graph.isBoundary = isBoundary
            if self.opt.method == 'semantic-super':
                graph.isBoundaryFace = isBoundaryFace
            
            graph.inside_edges = seg[edge_index[0]] == seg[edge_index[1]]
            graph.static_ed_nodes = torch.zeros(num, dtype=torch.bool).cuda()

        if self.opt.method == 'semantic-super':
            # if self.opt.use_edge_ssim_hints:
            #     if self.opt.mesh_edge:
            #         graph.boundary_edge_type = boundary_edge_type
            #     if self.opt.mesh_face:
            #         graph.boundary_face_type = boundary_face_type
            graph.isTool = isTool

        return graph


class CNNGraph(torch.nn.Module):
    def __init__(self):#, in_channels, hidden_channels, out_channels, depth,
                #  pool_ratios=0.5, sum_res=True, act=F.relu, use_dec=False):
        super().__init__()

        m = resnet50(pretrained = True)
        # # Get the available nodes for resnet50.
        # train_nodes, eval_nodes = get_graph_node_names(resnet50())
        # print(train_nodes, eval_nodes)
        return_nodes = {
            # node_name: user-specified key for output dict
            'layer1.2.relu_2': 'layer1', # 256, 1/4
            'layer2.3.relu_2': 'layer2', # 512, 1/8
            'layer3.5.relu_2': 'layer3', # 1024, 1/16
            # 'layer4.2.relu_2': 'layer4', # 2048, 1/32
        }
        self.enc = create_feature_extractor(m, return_nodes=return_nodes)

        # Build FPN
        in_channels_list = [256, 512, 1024] # [256, 512, 1024, 2048]
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels)

        # Image renderer
        self.renderer = Pulsar()
        self.color_enc = torch.nn.ModuleList() # TODO init parameter
        self.color_enc_depth = 4
        for i in range(self.color_enc_depth-1):
            self.color_enc.append(ResidualBlock(self.out_channels))
        self.color_enc.append(
            nn.Conv2d(self.out_channels, 3, kernel_size=3,
                      stride=1, padding=1, bias=False)
        )

        self.deNormalize = T.Compose([Normalize(mean = [ 0., 0., 0. ],
                                                std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                      Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                std = [ 1., 1., 1. ]),
                                    ])


    def forward(self, data):
        x = data.rgb
        if x.dim() == 3: x = x.unsqueeze(0)
        # TODO extract shape feature
        
        x = self.enc(x)
        x = self.fpn(x)

        ED_x = list(x.items())[-1][1]

        # Get the deform graph.
        # TODO
        v = V.type(long_)
        u = U.type(long_)
        for i in range(4):
            v = v[1::2, 1::2]
            u = u[1::2, 1::2]
        valid = data.valid.view(HEIGHT, WIDTH)[v,u]
        ED_idx = data.index_map[v,u][valid]
        ED_points = data.points[ED_idx]
        num = len(ED_points)
        _, ED_edge_index, _ = PreprocessingTools.init_graph(valid)
        edges_lens = torch_distance(ED_points[ED_edge_index[0]], ED_points[ED_edge_index[1]])
        
        ED_nodes = Data(points=ED_points, norms=data.norms[ED_idx],
            x=ED_x, valid=valid,
            colors=init_qual_color(ED_points, margin=50.),
            edge_index=ED_edge_index, edges_lens=edges_lens,
            num=num, param_num=num*7)

        return ED_nodes

    def loss(self, data, deform_graph):
        loss = 0.

        x = deform_graph.x
        for conv in self.color_enc:
            x = conv(x)
        x = x.view(3, -1).permute(1,0)[deform_graph.valid.view(-1)]
        render_img = self.renderer(deform_graph, colors=x, view_scale=1./16).permute(2,0,1)
        save_image(render_img, "test.jpg")
        target_img = self.deNormalize(T.Resize(render_img.size()[1:])(data.rgb))
        loss += ((render_img - target_img)**2).mean()
        
        return loss


class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu, use_dec=False):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList() # Convs to extract shape and visual features.
        self.down_vert_convs = torch.nn.ModuleList() # Convs to pred intermediate and last mesh.
        self.pools = torch.nn.ModuleList() # Feature and graph pooling layers.

        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        self.down_vert_convs.append(GCNConv(channels, out_channels, improved=True))
        for i in range(depth):
            # self.pools.append(TopKPooling(out_channels + channels, self.pool_ratios[i]))
            # self.pools.append(EdgePooling(out_channels + channels))
            self.pools.append(CorrEdgePooling(out_channels + channels))
            # self.pools.append(ASAPooling(out_channels + channels, ratio=self.pool_ratios[i], add_self_loops=True))
            
            self.down_convs.append(GCNConv(out_channels + channels, channels, improved=True))
            self.down_vert_convs.append(GCNConv(channels, out_channels, improved=True))

        self.use_dec = use_dec
        if use_dec:
            self.sum_res = sum_res
            
            in_channels = channels if sum_res else 2 * channels

            self.up_convs = torch.nn.ModuleList()
            for i in range(depth - 1):
                self.up_convs.append(GCNConv(in_channels, channels, improved=True))
            self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for conv in self.down_vert_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        if self.use_dec:
            for conv in self.up_convs:
                conv.reset_parameters()

    def forward(self, data, batch=None):
        """"""
        # x = torch.cat([data.points, data.norms], dim=-1)
        x = data.points
        edge_index = data.edge_index

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # edge_weight = x.new_ones(edge_index.size(1))
        
        x = self.down_convs[0](x, edge_index) # Optional 3rd input: edge_weight
        x = self.act(x)
        g = self.down_vert_convs[0](x, edge_index) # edge_weight

        if self.use_dec:
            xs = [x]
            edge_indices = [edge_index]
            edge_weights = [edge_weight]
            perms = []
        self.graphs = []
        self.clusters = []

        for i in range(1, self.depth + 1):
            # edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
            #                                            x.size(0))
            
            x = torch.cat([g, x], dim=-1)
            # x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
            #     x, edge_index, edge_weight, batch) # TopKPooling
            x, edge_index, batch, perm = self.pools[i - 1](
                x, edge_index, batch) # EdgePooling / CorrEdgePooling
            # x, edge_index, edge_weight, batch, perm = self.pools[i - 1](
            #     x, edge_index, edge_weight, batch) # ASAPooling
            
            if i < self.depth:
                self.graphs += [x[...,0:3]]
            
            x = self.down_convs[i](x, edge_index) # edge_weight
            x = self.act(x)
            g = self.down_vert_convs[i](x, edge_index) # edge_weight

            if i < self.depth and self.use_dec:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            self.clusters += [perm.cluster] # for EdgePooling / CorrEdgePooling
            # perms += [perm] # for TopKPooling
        self.graphs += [g]
        
        # Get the deform graph.
        ED_points = self.graphs[-1].clone()
        ED_edge_index = edge_index.clone()
        edges_lens = torch_distance(ED_points[ED_edge_index[0]], ED_points[ED_edge_index[1]])
        num = len(ED_points)
        ED_nodes = Data(points=ED_points, norms=g.clone(),
            colors=init_qual_color(ED_points, margin=50.),
            edge_index=ED_edge_index, edges_lens=edges_lens,
            num=num, param_num=num*7)
        # index_map=index_map, face_index=face_index, faces_areas=faces_areas,

        if self.use_dec:
            for i in range(self.depth):
                j = self.depth - 1 - i

                res = xs[j]
                edge_index = edge_indices[j]
                edge_weight = edge_weights[j]
                perm = perms[j]

                up = torch.zeros_like(res)
                up[perm] = x
                x = res + up if self.sum_res else torch.cat((res, up), dim=-1)
                x = self.up_convs[i](x, edge_index, edge_weight)
                x = self.act(x) if i < self.depth - 1 else x

            self.end = x

            return ED_nodes, Data(points=x[:,0:3], norms=x[:,3:], 
            colors=data.colors, edge_index=edge_index)

        else:
            return ED_nodes, None

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def loss(self, epoch, gt_data):
        l1_loss = nn.L1Loss()
        mse_loss = nn.MSELoss() # reduction='sum'

        gt_points = gt_data.points
        gt_norms = gt_data.norms
        # rad = gt_data.radii.mean()

        # _, gt2gt_idx = find_knn(gt_points, gt_points, k=5)

        loss = 0.
        graph_num = len(self.graphs)
        w = 1. / graph_num
        for i, graph in enumerate(self.graphs):
            N = len(graph)
            cluster = self.clusters[i]
            if i == 0:
                gt_points_ = scatter_mean(gt_points, cluster, dim=0, dim_size=N)
                gt_norms_ = scatter_mean(gt_norms, cluster, dim=0, dim_size=N)
            else:
                gt_points_ = scatter_mean(gt_points_, cluster, dim=0, dim_size=N)
                gt_norms_ = scatter_mean(gt_norms_, cluster, dim=0, dim_size=N)

            # # Chamfer loss.
            # pred2gt_dists, _ = find_knn(graph, gt_points, k=1)
            # gt2pred_dists, _ = find_knn(gt_points, graph, k=1)
            # loss += pred2gt_dists.mean() + gt2pred_dists.mean()

            # Normal loss.
            loss += w * (torch_inner_prod(graph - gt_points_, gt_norms_)**2).mean()

            # KNN loss?
            loss += w * (torch_sq_distance(graph, gt_points_)).mean()

            # # Affinity consistency loss.
            # if i == graph_num-1:
            #     # edge_index = torch.triu_indices(N, N, 1)
            #     # loss += w * l1_loss(
            #     #     torch_sq_distance(graph[edge_index[0]], graph[edge_index[1]]),
            #     #     torch_sq_distance(gt_points_[edge_index[0]], gt_points_[edge_index[1]])
            #     # )

            # # Edge length regularization.
            # pred2pred_dists, _ = find_knn(graph, graph, k=6)
            # loss += w * pred2pred_dists[:,1:].mean()
        
        # Graph coverage loss.
        # N = len(self.graphs[-1])
        # # rad = 2 * 0.8**epoch
        # rad = (1.75)**4 * rad
        # loss += mse_loss(
        #     torch.sigmoid(100. * (1./N * torch.exp(-torch.cdist(self.graphs[-1], gt_points)**2/rad**2).sum(0))),
        #     torch.ones((len(gt_points)), dtype=fl64_, device=gpu)
        #     )

        # Reconstruction loss. 
        # loss += torch_sq_distance(self.end, gt_points[gt2gt_idx].mean(1)).mean()

        return loss


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')


class CorrEdgePooling(torch.nn.Module):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers.

    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.

    To duplicate the configuration from the "Towards Graph Pooling by Edge
    Contraction" paper, use either
    :func:`EdgePooling.compute_edge_score_softmax`
    or :func:`EdgePooling.compute_edge_score_tanh`, and set
    :obj:`add_to_edge_score` to :obj:`0`.

    To duplicate the configuration from the "Edge Contraction Pooling for
    Graph Neural Networks" paper, set :obj:`dropout` to :obj:`0.2`.

    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (function, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0`)
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. (default: :obj:`0.5`)
    """

    unpool_description = namedtuple(
        "UnpoolDescription",
        ["edge_index", "cluster", "batch", "new_edge_score"])

    def __init__(self, in_channels, edge_score_method=None, dropout=0,
                 add_to_edge_score=0.5):
        super().__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout

        self.lin = torch.nn.Linear(2 * in_channels, 1)

        self.reset_parameters()


    def reset_parameters(self):
        self.lin.reset_parameters()


    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index, num_nodes):
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)


    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index, num_nodes):
        return torch.tanh(raw_edge_score)


    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index, num_nodes):
        return torch.sigmoid(raw_edge_score)


    def forward(self, x, edge_index, batch):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.

        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(Tensor)* - The pooled node features.
            * **edge_index** *(LongTensor)* - The coarsened edge indices.
            * **batch** *(LongTensor)* - The coarsened batch vector.
            * **unpool_info** *(unpool_description)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        e = self.lin(e).view(-1)
        # e = torch_inner_prod(x[edge_index[0]], x[edge_index[1]]) # Correlation.
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0))
        e = e + self.add_to_edge_score

        x, edge_index, batch, unpool_info = self.__merge_edges__(
            x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info


    def __merge_edges__(self, x, edge_index, batch, edge_score, edge_radius=3):
        nodes_remaining = set(range(x.size(0)))

        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        edge_argsort = torch.argsort(edge_score, descending=True)

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        i = 0
        new_edge_indices = []
        edge_index_cpu = edge_index.cpu()
        edge_valid = torch.ones(edge_index.size(1), dtype=torch.bool)
        for valid, edge_idx in zip(edge_valid, edge_argsort.tolist()):
            if not valid: continue

            source = edge_index_cpu[0, edge_idx].item()
            if source not in nodes_remaining:
                continue

            target = edge_index_cpu[1, edge_idx].item()
            if target not in nodes_remaining:
                continue

            new_edge_indices.append(edge_idx)

            edge_valid[edge_idx] = False
            del_list = set([source,target])
            new_del_list = set([])
            for del_item in del_list:
                neighbor_edge_idx = (edge_index == del_item).nonzero()
                edge_valid[neighbor_edge_idx[:,1]] = False
                new_del_list.update(
                    set(torch_to_numpy(
                        edge_index[1-neighbor_edge_idx[:,0], neighbor_edge_idx[:,1]]
                        ))
                )
            new_del_list.difference_update(del_list)
            nodes_remaining.difference_update(del_list)
            cluster[list(del_list)] = i
            # del_list = new_del_list # end for

            nodes_remaining.difference_update(new_del_list)
            cluster[list(new_del_list)] = i

            i += 1

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1
        cluster = cluster.to(x.device)

        # We compute the new features as an addition of the old ones.
        new_x = scatter_add(x, cluster, dim=0, dim_size=i)
        
        new_edge_score = edge_score[new_edge_indices]
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices), ))
            new_edge_score = torch.cat([new_edge_score, remaining_score])
        new_x = new_x * new_edge_score.view(-1, 1)

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster, batch=batch,
                                              new_edge_score=new_edge_score)

        return new_x, new_edge_index, new_batch, unpool_info

    def unpool(self, x, unpool_info):
        r"""Unpools a previous edge pooling step.

        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.

        Args:
            x (Tensor): The node features.
            unpool_info (unpool_description): Information that has
                been produced by :func:`EdgePooling.forward`.

        Return types:
            * **x** *(Tensor)* - The unpooled node features.
            * **edge_index** *(LongTensor)* - The new edge indices.
            * **batch** *(LongTensor)* - The new batch vector.
        """

        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'