import os
import numpy as np
import cv2
from skimage import io

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import torchvision.transforms as T

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from models.preprocessing import *
from models.loss import *

from seg.seg_models import *

from utils.config import *

def edge_weights(edges, features):
    """
    Calculate edge weights (correlation of vertices features).
    edges: 2xN
    """
    weights = torch_inner_prod(
        features[edges[0]], features[edges[1]]
    ).type(fl32_)

    d = torch.sparse_coo_tensor(
        torch.stack([
            torch.arange(2*edges.size(1), device=dev),
            edges.view(-1)
        ], dim=0),
        weights.repeat(2),
        (2*edges.size(1), torch.max(edges)+1))
    d = torch.sparse.sum(d, dim=0).to_dense()

    return weights, d

def map2graph(valid, device=dev, z=None):
    """
    Init graph from a HxW valid map.
    """
    if not torch.is_tensor(valid):
        valid = numpy_to_torch(valid, device=device)
    else:
        valid = valid.to(device)
    h, w = valid.size()

    u = torch.arange(w, dtype=long_, device=device)
    v = torch.arange(h, dtype=long_, device=device)
    v, u = torch.meshgrid(v, u)
    v = v[valid]
    u = u[valid]

    # edge_end_a = torch.stack([u,v], dim=1).unsqueeze(1).repeat(1,8,1)
    edge_end_a = torch.stack([u,v], dim=1).unsqueeze(1).repeat(1,4,1)

    # edge_end_b = torch.stack([
    #     torch.stack([u-1, u, u+1, u-1, u+1, u-1, u, u+1], dim=1),
    #     torch.stack([v-1, v-1, v-1, v, v, v+1, v+1, v+1], dim=1)
    # ], dim=2)
    edge_end_b = torch.stack([
        torch.stack([u+1, u+1, u, u-1], dim=1),
        torch.stack([v, v+1, v+1, v+1], dim=1)
    ], dim=2)
    
    valid_edge = (edge_end_b[...,0] >= 0) & (edge_end_b[...,0] < w) & (edge_end_b[...,1] >= 0) & (edge_end_b[...,1] < h)

    edge_end_a = edge_end_a[...,0] + w * edge_end_a[...,1]
    edge_end_b = edge_end_b[...,0] + w * edge_end_b[...,1]
    edges = torch.stack([
        edge_end_a[valid_edge], edge_end_b[valid_edge]
    ], dim=0)

    return edges # edges: 2xN
# def map2graph(valid, device=dev, z=None):
#     """
#     Init graph from a HxW valid map.
#     """
#     if not torch.is_tensor(valid):
#         valid = numpy_to_torch(valid, device=device)
#     else:
#         valid = valid.to(device)
#     h, w = valid.size()
#     z = z.view(h,w,-1)

#     u = torch.arange(w, dtype=long_, device=device)
#     v = torch.arange(h, dtype=long_, device=device)
#     v, u = torch.meshgrid(v, u)
#     v = v[valid]
#     u = u[valid]

#     edge_end_a = torch.stack([u,v], dim=1).unsqueeze(1).repeat(1,8,1)
#     # edge_end_a = torch.stack([u,v], dim=1).unsqueeze(1).repeat(1,4,1)

#     edge_end_b = torch.stack([
#         torch.stack([u-1, u, u+1, u-1, u+1, u-1, u, u+1], dim=1),
#         torch.stack([v-1, v-1, v-1, v, v, v+1, v+1, v+1], dim=1)
#     ], dim=2)
#     # edge_end_b = torch.stack([
#     #     torch.stack([u+1, u+1, u, u-1], dim=1),
#     #     torch.stack([v, v+1, v+1, v+1], dim=1)
#     # ], dim=2)
    
#     valid_edge = (edge_end_b[...,0] >= 0) & (edge_end_b[...,0] < w) & (edge_end_b[...,1] >= 0) & (edge_end_b[...,1] < h)
    
#     weights = torch_inner_prod(
#         z[edge_end_a[valid_edge][:,1], edge_end_a[valid_edge][:,0]],
#         z[edge_end_b[valid_edge][:,1], edge_end_b[valid_edge][:,0]]
#     ).type(fl32_)
#     d = torch.zeros(valid_edge.size(), dtype=fl32_, device=device)
#     d[valid_edge] = weights
#     d = d.sum(1)

#     edge_end_a = edge_end_a[...,0] + w * edge_end_a[...,1]
#     edge_end_b = edge_end_b[...,0] + w * edge_end_b[...,1]
#     edges = torch.stack([
#         edge_end_a[valid_edge], edge_end_b[valid_edge]
#     ], dim=0)
#     # print(edges.size())
#     # print(torch.unique(edges, dim=1).size())
#     return edges, weights, d # Size: 2xN, 3xN

"""
Graph correction.
"""
# def graph_correction_loss(ref_y, y, edges, w, d, alpha=0.5, eps=1e-12):
def graph_correction_loss(ref_y, data, alpha=0.5, eps=1e-12):
    y = data.x
    edges = data.edge_index
    w = data.edge_weight
    d = data.x_weight
    
    normalize_ref_y = ref_y / (d+eps).unsqueeze(1)
    
    loss = 0.
    loss += alpha * (w * torch.norm(normalize_ref_y[edges[0]]-normalize_ref_y[edges[1]], dim=-1)).sum()
    # loss += alpha * l1_smoothness_loss(
    #     normalize_ref_y[edges[0]], normalize_ref_y[edges[1]], weights=w, reduction='sum')
    
    loss += (1-alpha) * torch.norm(ref_y-y, dim=-1).sum()
    # loss += (1-alpha) * mse_loss(ref_y, y, reduction='sum')
    # loss += (1-alpha) * l1_smoothness_loss(ref_y, y, reduction='sum', beta=0.1)

    return loss

class GCN(torch.nn.Module):
    def __init__(self, channel_in, channel_hidden=0, channel_out=0):
        super().__init__()
        if channel_hidden==0:
            channel_hidden = channel_in
        if channel_out == 0:
            channel_out = channel_in

        self.conv1 = GCNConv(channel_in, channel_hidden)
        self.conv2 = GCNConv(channel_hidden, channel_out)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        # return F.softmax(x, dim=1)
        return x