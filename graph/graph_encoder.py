from collections import namedtuple

from pytorch3d.structures import Meshes

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

from utils.config import *
from utils.utils import *


class DirectDeformGraph(nn.Module):
    '''
    Directly get the ED node graph through gridding or random (uniform) selection.
    '''

    def __init__(self, model_args) -> None:
        super(DirectDeformGraph, self).__init__()
        self.model_args = model_args

    def forward(self, data, sample_method='grid'):

        points = data.points
        norms = data.norms
        valid = data.valid.view(self.model_args['CamParams'].HEIGHT, self.model_args['CamParams'].WIDTH)

        if sample_method: step = 28
        elif sample_method == 'uniform': step = 4
        isED, edge_index, face_index = init_graph(valid, step=step)
        index_map_valid = isED & valid # TODO
        isED = isED[valid]
        ED_points = points[isED]
        edges_lens = torch_distance(ED_points[edge_index[:,0]], ED_points[edge_index[:,1]])
        deform_mesh = Meshes(verts=[ED_points], faces=[torch.t(face_index)])
        faces_areas = deform_mesh.faces_areas_packed()

        num = len(ED_points)
        # TODO: Better way to estimate the radii.
        radii = edges_lens.mean() * torch.ones((num,), device=dev)
        neighbor_radii = 2 * edges_lens.mean() * torch.ones((num,), device=dev)

        # TODO
        u = torch.arange(0, self.model_args['CamParams'].WIDTH, step, device=dev)
        v = torch.arange(0, self.model_args['CamParams'].HEIGHT, step, device=dev)
        index_map_valid = index_map_valid[v][:,u]
        index_map = - torch.ones_like(index_map_valid, dtype=long_)
        index_map[index_map_valid] = torch.arange(num, device=dev)
        graph = Data(points=ED_points, norms=norms[isED], radii=[radii], neighbor_radii=[neighbor_radii],
            edge_index=edge_index, edges_lens=edges_lens,
            face_index=face_index, faces_areas=faces_areas,
            index_map=index_map, num=num, param_num=num*7)
        # colors=init_qual_color(ED_points, margin=50.),

        # if data.seg_conf is not None:
        #     graph.seg_conf = data.seg_conf[isED]

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