from abc import ABC, abstractmethod

from token import DEDENT
import numpy as np
import torch
import torch.nn.functional as F

from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import copy
import os

from typeguard import check_argument_types, check_return_type

from utils.config import *
from utils.utils import *
from utils.img_matching import *
from models.LM import *

# TODO Allow batch.
def bilinear_sample(inputs, v, u, index_map=None, fill='zero', grad=False, normalizations=None):
    """
    Given inputs (a list of feature maps) and target pixel locations (v, u) in the 
    feature maps, computes the output features through bilinear interpolation.

    * 'inputs': a list of features, each has size (N,channel,).
    * 'index_map': If not None, its size should be (HEIGHT, WIDTH), each non-negative 
    element indicates the location of the correponding feature vector. 
    * 'normalizations': A list of True/False to decide if the corresponding output 
    features will be normalized.
    """
    def list_normalization(xs, gradients=None):
        if normalizations is None: return xs, gradients

        for i, normalization in enumerate(normalizations):
            if normalization:
                scale = torch.norm(xs[i], dim=-1, keepdim=True) #(in_size,1,)
                xs[i] /= scale

                if gradients is not None:
                    gradients[i] /= scale.unsqueeze(-1)

        if gradients is None: return xs
        else: return xs, gradients

    inputs_channels = [input_.size()[-1] for input_ in inputs]
    features = torch.cat(inputs, dim=-1)

    fl_v = torch.floor(v)
    cel_v = torch.ceil(v)
    fl_u = torch.floor(u)
    cel_u = torch.ceil(u)

    n_block = torch.stack([fl_v, fl_v, cel_v, cel_v], dim=-1) # y: (in_size,4,)
    m_block = torch.stack([fl_u, cel_u, fl_u, cel_u], dim=-1) # x
    
    # U_nm: (in_size,4,cat_input_channel,)
    if index_map is None:
        # U_nm = features[n_block.long()*WIDTH+m_block.long()]
        U_nm = features[n_block.long(), m_block.long()]
    else:
        if fill == 'zero':
            U_nm = torch.zeros(n_block.size()+(features.size()[-1],), 
            device=gpu).type(features.type())
        elif fill == 'nan':
            U_nm = torch_nans(n_block.size()+(features.size()[-1],), dtype=features.type())
        
        U_nm_index_map = index_map[n_block.long(), m_block.long()]
        U_nm_valid = U_nm_index_map>=0
        U_nm[U_nm_valid] = features[U_nm_index_map[U_nm_valid]]
    
    n_block -= v.unsqueeze(-1)
    m_block -= u.unsqueeze(-1)
    n_block = n_block.unsqueeze(-1) # Change dim to (in_size,4,1,)
    m_block = m_block.unsqueeze(-1)
    if grad:
        d_n_block = torch.where(n_block >= 0, 1., -1.)
        d_m_block = torch.where(m_block >= 0, 1., -1.)
    n_block = torch.maximum(1-torch.abs(n_block), torch.tensor(0, device=dev))
    m_block = torch.maximum(1-torch.abs(m_block), torch.tensor(0, device=dev))

    outputs = torch.sum(U_nm * n_block * m_block, dim=-2) #(in_size,feature_channel,)
    outputs = torch.split(outputs, inputs_channels, dim=-1)

    if normalizations is not None:
        norms_scales = []
        for i, normalization in enumerate(normalizations):
            if normalization:
                norm_scale = torch.norm(outputs[i], dim=-1, keepdim=True) #(in_size,1,)
                outputs[i] /= norms_scale
            else:
                norm_scale

    if index_map is None:
        U_nm_valid = None
    else:
        U_nm_valid = U_nm_valid.view(len(U_nm_valid), -1)
        U_nm_valid = ~torch.any(~U_nm_valid, dim=-1)

    if grad:
        gradients = torch.stack([\
            torch.sum(U_nm * n_block * d_m_block, dim=1, dtype=dtype_),\
            torch.sum(U_nm * m_block * d_n_block, dim=1, dtype=dtype_)],\
            dim=2) # dOut-dv & dOut-du: (in_size,feature_channel,2)
        gradients = torch.split(gradients, inputs_channels, dim=-2)

        if normalizations is not None:
            outputs, gradients = list_normalization(outputs, gradients=gradients)
        return outputs, gradients, U_nm_valid
        
    else:
        if normalizations is not None:
            list_normalization(outputs)
        return outputs, None, U_nm_valid


def neighbor_sample(inputs, v, u):
    inputs_channels = [input_.size()[-1] for input_ in inputs]
    features = torch.cat(inputs, dim=-1)

    fl_v = torch.floor(v)
    cel_v = torch.ceil(v)
    fl_u = torch.floor(u)
    cel_u = torch.ceil(u)

    n_block = torch.stack([fl_v, fl_v, cel_v, cel_v], dim=0) # y: (4,in_size,)
    m_block = torch.stack([fl_u, cel_u, fl_u, cel_u], dim=0) # x

    outputs = features[n_block.long(), m_block.long()]
    return torch.split(outputs, inputs_channels, dim=-1)

class LossTool():

    @staticmethod
    def bilinear_intrpl_block(v, u, target_, index_map=None, grad=False, normalization=False):
        fl_v = torch.floor(v)
        cel_v = torch.ceil(v)
        fl_u = torch.floor(u)
        cel_u = torch.ceil(u)

        n_block = torch.stack([fl_v, fl_v, cel_v, cel_v], dim=1) # y
        m_block = torch.stack([fl_u, cel_u, fl_u, cel_u],dim=1) # x
        if index_map is None:
            U_nm = target_[n_block.long()*WIDTH+m_block.long()]
        else:
            U_nm = torch.ones(n_block.size()+(target_.size()[-1],), dtype=fl64_, device=dev) * float('nan')
            U_nm_index_map = index_map[n_block.long(), m_block.long()]
            U_nm_valid = U_nm_index_map>=0
            U_nm[U_nm_valid] = target_[U_nm_index_map[U_nm_valid]]
        n_block -= v.unsqueeze(1)
        m_block -= u.unsqueeze(1)
        n_block = n_block.unsqueeze(2)
        m_block = m_block.unsqueeze(2)
        if grad:
            d_n_block = torch.where(n_block >= 0, 1., -1.)
            d_m_block = torch.where(m_block >= 0, 1., -1.)
        n_block = torch.maximum(1-torch.abs(n_block), torch.tensor(0, device=dev)) # TODO
        m_block = torch.maximum(1-torch.abs(m_block), torch.tensor(0, device=dev))
        # if index_map is not None:
        #     n_block[~U_nm_valid] = 0
        #     n_block = 2. * n_block / torch.sum(n_block, dim=1, keepdim=True)
        #     m_block[~U_nm_valid] = 0
        #     m_block = 2. * m_block / torch.sum(m_block, dim=1, keepdim=True)
        target = torch.sum(U_nm * n_block * m_block, dim=1, dtype=dtype_)
        if normalization:
            norm_scale = torch.norm(target, dim=1, keepdim=True)
            target /= norms_scale

        if grad:
            # grad: dV_dx & dV_dy
            # if len(target_.size()) == 1:
            #     grad = torch.stack([\
            #         torch.sum(U_nm * n_block * d_m_block, dim=1, keepdim=True, dtype=dtype_),\
            #         torch.sum(U_nm * m_block * d_n_block, dim=1, keepdim=True, dtype=dtype_)],\
            #         dim=2) # dV: Nx1x2
            # else:
            grad = torch.stack([\
                torch.sum(U_nm * n_block * d_m_block, dim=1, dtype=dtype_),\
                torch.sum(U_nm * m_block * d_n_block, dim=1, dtype=dtype_)],\
                dim=2) # dV: Nxcx2
            
            if normalization:
                return target, grad/norms_scale.unsqueeze(2)
            else:
                return target, grad
        else:
            return target, None

    @staticmethod
    def dPi_block(trans_points):

        match_num = len(trans_points)
        Z = trans_points[:,2]
        sq_Z = torch.pow(Z,2)

        dPi = torch.zeros((match_num,2,3), dtype=dtype_, device=dev)
        dPi[:,0,0] = -fx/Z
        dPi[:,0,2] = fx*trans_points[:,0]/sq_Z
        dPi[:,1,1] = fy/Z
        dPi[:,1,2] = -fy*trans_points[:,1]/sq_Z

        return dPi

    # Find the indicies of all non-zero elements in the Jacobian matrix.
    # Output (2xN): row-column index of each element.
    @staticmethod
    def prepare_Jacobian_idx(cost_size, var_idxs, inc_idx):
        # Loss L can be calculated as L = \sum_i l(input_i)
        # Inputs: 1) 'cost_size': the number of l()'s outputs;
        # 2) 'var_idxs': for each l(), the column index of the first parameter.
        
        cost_num, neighbor_num = var_idxs.size()
        if cost_size == 1:
            var_num = len(inc_idx)
        else:
            var_num = inc_idx.size()[1]
        idx0 = torch.arange(cost_num*cost_size, device=dev).unsqueeze(1).repeat(1, neighbor_num*var_num)

        inc_idx = inc_idx.unsqueeze(0).unsqueeze(0)
        if cost_size == 1:
            idx1 = (var_idxs*7).unsqueeze(2) + inc_idx
        else:
            idx1 = (var_idxs*7).unsqueeze(2).unsqueeze(3) + inc_idx
            idx1 = idx1.permute(0,2,1,3)

        return torch.stack([idx0.flatten(), idx1.flatten()],dim=0)

    @staticmethod
    def prepare_jtj_jtl(Jacobian, loss):

        Jacobian_t = torch.transpose(Jacobian, 0, 1)
        if torch_version == "1.10.0":
            jtj = torch.sparse.mm(Jacobian_t, Jacobian)
        elif torch_version == "1.7.1":
            jtj = torch.sparse.mm(Jacobian_t, Jacobian.to_dense()).to_sparse()
        jtl = -torch.sparse.mm(Jacobian_t,loss)
        return jtj, jtl

class Loss(ABC):
    
    # Perform calculations that will be consistent 
    # throughout the optimization process.
    @abstractmethod
    def prepare(self, sfModel, new_data):
        pass

    @abstractmethod
    def forward(self, lambda_, beta, new_data, grad=False, dldT_only=False):
        # Target outputs: coordinates (y*HEIGHT+x, Nx2) of matched 
        # points in the two images, named match1 & match2.
        pass

class DataLoss(Loss):

    def __init__(self):
        self.inc_idx = torch.tensor([0,1,2,3,4,5,6], device=dev)

    def prepare(self, sfModel, new_data):
        self.sf_knn_weights = sfModel.sf_knn_weights
        self.sf_knn_indicies = sfModel.sf_knn_indices
        self.sf_knn = sfModel.ED_nodes.points[self.sf_knn_indicies] # All g_i in (10).
        self.sf_diff = sfModel.points.unsqueeze(1) - self.sf_knn # (p-g_i) in (10).
        self.skew_v = get_skew(self.sf_diff)

        self.J_size = sfModel.ED_nodes.param_num

    def forward(self, lambda_, beta, new_data, grad=False, dldT_only=False):

        ### Find correspondence based on projective ICP. Jacobian_elements: Nx4x3x4.
        trans_points, Jacobian_elements = Trans_points(self.sf_diff, self.sf_knn, \
            beta[self.sf_knn_indicies], self.sf_knn_weights, grad=grad, skew_v=self.skew_v)
        ## Project the updated points to the image plane.
        ## Only keep valid projections with valid new points.
        v, u, proj_coords, proj_valid_index = pcd2depth(trans_points, depth_sort=True, round_coords=False)
        valid_pair = new_data.valid[proj_coords] # Valid proj-new pairs.
        v, u = v[valid_pair], u[valid_pair]
        ## Find matched points & normal values, and calculate related grad if needed.
        # new_points, dpdPi = LossTool.bilinear_intrpl_block(v, u, new_data.points, grad=grad)
        # new_norms, dndPi = LossTool.bilinear_intrpl_block(v, u, new_data.norms, grad=grad)
        new_points, dpdPi = LossTool.bilinear_intrpl_block(v, u,
            new_data.points, index_map=new_data.index_map, grad=grad)
        new_norms, dndPi = LossTool.bilinear_intrpl_block(v, u,
            new_data.norms, index_map=new_data.index_map, grad=grad)
        intrpl_valid = ~torch.any(torch.isnan(new_points)|torch.isnan(new_norms), dim=1)
        new_points = new_points[intrpl_valid]
        new_norms = new_norms[intrpl_valid]

        sf_indicies = proj_valid_index[valid_pair][intrpl_valid]
        trans_points = trans_points[sf_indicies].type(fl32_)
        pt_diff = trans_points-new_points# T(p)-o in (13).
        loss = lambda_ * torch.sum(new_norms*pt_diff, dim=1, keepdim=True)
        
        if grad:
            dpdPi = dpdPi[intrpl_valid]
            dndPi = dndPi[intrpl_valid]

            Jacobian_elements = Jacobian_elements[sf_indicies].type(fl32_)
            knn_weights = self.sf_knn_weights[sf_indicies].unsqueeze(2).unsqueeze(3) # Nx4x1x1

            dPidT = LossTool.dPi_block(trans_points) # Nx2x3
            dpdT = torch.matmul(dpdPi,dPidT) # Nx3x3
            dndT = torch.matmul(dndPi,dPidT) # Nx3x3

            if dldT_only:
                dldT = torch.matmul(new_norms.unsqueeze(1), dpdT) + \
                    torch.matmul(pt_diff.unsqueeze(1), dndT)
                dldT *= lambda_
                return torch.block_diag(*dldT)

            dpdT = dpdT.unsqueeze(1) # Nx1x3x3
            dndT = dndT.unsqueeze(1) # Nx1x3x3

            dndq = torch.matmul(dndT, Jacobian_elements) # Nx4x3x4
            dndq = torch.cat([dndq, knn_weights*dndT.repeat(1,n_neighbors,1,1)], dim=3) # Nx4x3x7

            dpdq = Jacobian_elements - torch.matmul(dpdT, Jacobian_elements) # Nx4x3x4
            dpdq_vb = torch.eye(3, device=dev).unsqueeze(0).unsqueeze(0) - dpdT.repeat(1,n_neighbors,1,1)
            dpdq = torch.cat([dpdq, knn_weights*dpdq_vb], dim=3)

            J_idx = LossTool.prepare_Jacobian_idx(1, \
                self.sf_knn_indicies[sf_indicies], self.inc_idx)
            v = torch.matmul(new_norms.unsqueeze(1).unsqueeze(1),dpdq.float()).squeeze(2) + \
                torch.matmul(pt_diff.unsqueeze(1).unsqueeze(1),dndq.float()).squeeze(2) # Nx4x7
            
            v = v.flatten()
            valid = ~torch.isnan(v)
            Jacobian = torch.sparse_coo_tensor(J_idx[:,valid], \
                lambda_ * v[valid], (len(trans_points), self.J_size), dtype=fl32_)
            
            return LossTool.prepare_jtj_jtl(Jacobian, loss)
        else:
            return torch.pow(loss,2)

    @staticmethod
    def autograd_forward(model_args, src, trg, flow=None, loss_type='point-plane'):
        v_, u_, _, valid_idx = pcd2depth(src.points, round_coords=False, valid_margin=1)
        u = u_[valid_idx]
        v = v_[valid_idx]

        if flow is not None:
            trg_loc, _, _ = bilinear_sample([flow], v, u)
            trg_loc = trg_loc[0]
            v += trg_loc[...,0]
            u += trg_loc[...,1]

        if 'seman-super' in model_args['method']:
            src_seg_conf = neighbor_sample([torch.argmax(trg.seg_conf_map, dim=-1, keepdim=True)], v_, u_)
            src_seg_conf = src_seg_conf[0].squeeze(-1)
            valid_idx &= ~torch.any(~(src_seg_conf==torch.argmax(src.seg_conf, dim=-1).unsqueeze(0)), dim=0)

            v = v_[valid_idx]
            u = u_[valid_idx]

        if loss_type == 'point-point':
            sample_trg, _, sample_valid = bilinear_sample([trg.points], 
                v, u, index_map=trg.index_map)
            sample_trg_verts = sample_trg[0]
            return torch_sq_distance(
                src.points[valid_idx][sample_valid], sample_trg_verts[sample_valid]).sum()
        
        elif loss_type == 'point-plane':
            sample_trg, _, sample_valid = bilinear_sample([trg.points, trg.norms], 
                v, u, index_map=trg.index_map)
            sample_trg_verts, sample_trg_norms = sample_trg
            return (torch_inner_prod(
                sample_trg_norms[sample_valid],
                src.points[valid_idx][sample_valid] - sample_trg_verts[sample_valid])**2).sum()

class FeatLoss(Loss):

    def __init__(self, use_depth=False, use_point=False):
        if use_depth:
            self.inc_idx = torch.tensor([0,1,2,3,4,5,6], device=dev)
        elif use_point:
            self.inc_idx = torch.tensor([[0,1,2,3,4,5,6], \
                                         [0,1,2,3,4,5,6], \
                                         [0,1,2,3,4,5,6]], device=dev)

        self.use_depth = use_depth
        self.use_point = use_point

    def prepare(self, sfModel, new_data):
        self.sf_knn_weights = sfModel.sf_knn_weights
        self.sf_knn_indices = sfModel.sf_knn_indices
        self.sf_knn = sfModel.ED_nodes.points[self.sf_knn_indices] # All g_i in (10).
        self.sf_diff = sfModel.points.unsqueeze(1) - self.sf_knn # (p-g_i) in (10).
        self.skew_v = get_skew(self.sf_diff)

        self.J_size = sfModel.ED_nodes.param_num

    def forward(self, lambda_, beta, new_data, grad=False, dldT_only=False):
        # points, _, valid = new_data

        if self.use_depth: features_ = new_data.points[:,2:]
        elif self.use_point: features_ = new_data.points
        c_ = features_.size()[1]

        ### Find correspondence based on projective ICP.
        # Jacobian_elements: Nx4(neighbor_n)x3(x,y,z)x4(quaternion).
        trans_points, Jacobian_elements = Trans_points(self.sf_diff, self.sf_knn, \
            beta[self.sf_knn_indices], self.sf_knn_weights, grad=grad, skew_v=self.skew_v)
        ## Project the updated points to the image plane.
        ## Only keep valid projections with valid new points.
        v, u, _, proj_valid_index = pcd2depth(trans_points, round_coords=False)
        u = u[proj_valid_index]
        v = v[proj_valid_index]

        # target: Nxc, dVdPi: Nxcx2, Pi: project x,y
        target, dVdPi = LossTool.bilinear_intrpl_block(v, u, features_, grad=grad)
        valid_pair = ~torch.any(torch.isnan(target), 1)
        target = target[valid_pair]
        sf_indices = proj_valid_index[valid_pair]
        trans_points = trans_points[sf_indices].type(fl32_)

        # Loss: Feature of current model - Wrapped feature from the new frame.
        if self.use_depth:
            loss = lambda_ * (trans_points[:,2:] - target)
        elif self.use_point:
            loss = lambda_ * (trans_points - target).view(-1,1)
        
        if grad:
            Jacobian_elements = Jacobian_elements[sf_indices].type(fl32_)
            knn_weights = self.sf_knn_weights[sf_indices].unsqueeze(1).unsqueeze(-1) # Nx1x4x1

            dVdPi = dVdPi[valid_pair]

            ## Include bilinear term loss.
            dTdq = torch.flatten(torch.transpose(Jacobian_elements,1,2), start_dim=2) # Nx3x(4x4)
            dPidT = LossTool.dPi_block(trans_points) # Nx2x3(x,y,z)

            dVdT = torch.matmul(dVdPi, dPidT) # dVdPi: Nxcx2, dPidT: Nx2x3(x,y,z), dVdT: Nxcx3
            del dVdPi, dPidT
            if dldT_only:
                dVdT = -dVdT.squeeze(1)
                dVdT[:,2] += 1.
                dVdT *= lambda_
                return torch.block_diag(*dVdT)

            else:
                match_num = len(trans_points)

                # dVdT: Nxcx3, dTdq: Nx3x(n_neighborsx4), v: Nxcxn_neighborsx4
                v = - torch.matmul(dVdT,dTdq).view(match_num,c_,n_neighbors,4)
                vb = - dVdT.unsqueeze(2).repeat(1,1,4,1) # Nxcxn_neighborsx3

                if self.use_depth:
                    v += Jacobian_elements[:,:,2:,:].permute(0,2,1,3)
                    vb[...,2] += 1.

                elif self.use_point:
                    v += Jacobian_elements.permute(0,2,1,3)
                    vb[:,0,:,0] += 1.
                    vb[:,1,:,1] += 1.
                    vb[:,2,:,2] += 1.

                J_idx = LossTool.prepare_Jacobian_idx(c_, \
                    self.sf_knn_indices[sf_indices], self.inc_idx)
                
                vb *= knn_weights
                v = torch.cat([v,vb], dim=-1).flatten() #Nxcxn_neighborsx7

                Jacobian = torch.sparse_coo_tensor(J_idx, \
                    lambda_ * v.view(-1), (match_num*c_, self.J_size), dtype=fl32_)
                
                return LossTool.prepare_jtj_jtl(Jacobian, loss)

        else:
            return torch.pow(loss,2)

class ARAPLoss(Loss):

    def __init__(self):
        self.inc_idx_a = torch.tensor([[0,1,2,3,4],[0,1,2,3,5],[0,1,2,3,6]], device=dev)
        self.inc_idx_b = torch.tensor([[4],[5],[6]], device=dev)

    def prepare(self, sfModel, new_data):

        self.ED_knn_indices = sfModel.ED_nodes.ED_knn_indices

        self.d_EDs = sfModel.ED_nodes.points.unsqueeze(1) - \
                        sfModel.ED_nodes.points[self.ED_knn_indices]

        self.skew_EDv = get_skew(self.d_EDs)

        arap_idxa = LossTool.prepare_Jacobian_idx(3, \
            self.ED_knn_indices.view(-1,1), self.inc_idx_a)
        arap_idxb = LossTool.prepare_Jacobian_idx(3, \
            torch.arange(sfModel.ED_nodes.num, device=dev).unsqueeze(1).repeat(1,ED_n_neighbors).view(-1,1), \
            self.inc_idx_b)
        self.J_idx = torch.cat([arap_idxa, arap_idxb], dim=1)
        self.J_size = (sfModel.ED_nodes.num*ED_n_neighbors*3, sfModel.ED_nodes.param_num)

    def forward(self, lambda_, beta, new_data, grad=False, dldT_only=False):

        ED_t = beta[:,4:7].type(fl32_)
        beta = beta[self.ED_knn_indices]
        
        loss, Jacobian_elements = transformQuatT(self.d_EDs, beta, \
            grad=grad, skew_v=self.skew_EDv)

        loss = loss.type(dtype_)
        loss -= self.d_EDs.type(dtype_) + ED_t.unsqueeze(1)
        loss = lambda_ * loss.view(-1,1)
        
        if grad:
            match_num = len(self.d_EDs)
            Jacobian_elements = Jacobian_elements.type(dtype_)
            
            Jacobian_elements = torch.cat([Jacobian_elements, \
                torch.ones((match_num, ED_n_neighbors, 3, 1), dtype=dtype_, device=dev)], \
                dim=3).flatten()
            Jacobian_elements = torch.cat([Jacobian_elements, \
                -torch.ones(match_num * ED_n_neighbors * 3, dtype=dtype_, device=dev)])
            Jacobian_elements *= lambda_

            Jacobian = torch.sparse_coo_tensor(self.J_idx, \
                Jacobian_elements, self.J_size, dtype=dtype_)
                
            return LossTool.prepare_jtj_jtl(Jacobian, loss)
        else:
            return torch.pow(loss,2)

    @staticmethod
    def autograd_forward(input, beta):
        nodes = input.points
        knn_indices = input.ED_knn_indices
        knn_w = input.ED_knn_w
        
        nodes_t = beta[:,4:7]
        beta = beta[knn_indices]
        
        d_nodes = (nodes.unsqueeze(1) - nodes[knn_indices])
        loss, _ = transformQuatT(d_nodes, beta, skew_v=get_skew(d_nodes))
        loss -= d_nodes.type(fl32_) + nodes_t.unsqueeze(1)

        loss = knn_w * torch.pow(loss, 2).sum(-1)
        loss = loss.sum(-1)
        
        return loss.sum()

class RotLoss(Loss):

    def __init__(self):
        self.inc_idx = torch.tensor([0,1,2,3], device=dev)

    def prepare(self, sfModel, new_data):
        self.J_idx = LossTool.prepare_Jacobian_idx(1, \
            torch.arange(sfModel.ED_nodes.num, device=dev).unsqueeze(1), \
            self.inc_idx)
        self.J_size = (sfModel.ED_nodes.num, sfModel.ED_nodes.param_num)
        
    def forward(self, lambda_, beta, new_data, grad=False):
        beta = beta[:, 0:4].type(fl32_)

        loss = lambda_ * (1.-torch.sum(torch.pow(beta,2), dim=1, keepdim=True))

        if grad:
            v = (-lambda_*2*beta).view(-1)
            Jacobian = torch.sparse_coo_tensor(\
                self.J_idx, v, self.J_size, dtype=fl32_)

            return LossTool.prepare_jtj_jtl(Jacobian, loss)
        else:
            return torch.pow(loss, 2)

    @staticmethod
    def autograd_forward(beta):
        beta = beta[:, 0:4].type(fl32_)
        loss = 1. - torch.sum(torch.pow(beta,2), dim=1)
        return torch.pow(loss, 2)

class CorrLoss(Loss):

    def __init__(self, point_loss=False, point_plane_loss=False):
        if point_loss:
            self.inc_idx = torch.tensor([[0,1,2,3,4,5,6]], device=dev).repeat(3,1)

        if corr_method == 'opencv':
            self.matcher = cv2Matcher()
        elif corr_method == 'kornia':
            self.matcher = LoFTR()

        self.point_loss = point_loss
        self.point_plane_loss = point_plane_loss

    def prepare(self, sfModel, new_data):
        # Find correspondence.
        m1, m2 = self.matcher.match_features(sfModel.renderImg, new_data.rgb, new_data.ID)

        if len(m1) == 0:
            self.match_num = 0
            return
        
        ## Find the indices of matched surfels.
        valid_indices = sfModel.projdata[:,0].long()
        indices_map = - torch.ones((HEIGHT,WIDTH), dtype=long_, device=dev)
        indices_map[sfModel.projdata[:,1].long(), sfModel.projdata[:,2].long()] = valid_indices
        sf_indices = indices_map[m1[:,1].long(),m1[:,0].long()] # Indices of matched surfels.
        ## Estimate (interpolate) the 3D positions of the matched new points,
        ## and estimate the gradient related to these new points.
        # target: Nxc, dVdPi: Nxcx2, Pi: project x,y
        self.new_points, self.dVdPi = LossTool.bilinear_intrpl_block(m2[:,1], m2[:,0], \
            new_data.points, grad=True)
        
        new_valid = new_data.valid.type(fl32_).view(-1,1)
        new_valid[new_valid == 0] = float('nan')
        new_valid, _ = LossTool.bilinear_intrpl_block(m2[:,1], m2[:,0], new_valid)
        valid_pair = (sf_indices >= 0) & (~torch.isnan(new_valid[:,0]))
        self.match_num = torch.count_nonzero(valid_pair)
        if self.match_num > 0:
            sf_indices = sf_indices[valid_pair]
            self.new_points = self.new_points[valid_pair].type(fl32_)
            self.dVdPi = self.dVdPi[valid_pair]
        
            self.sf_knn_weights = sfModel.sf_knn_weights[sf_indices]
            self.sf_knn_indices = sfModel.sf_knn_indices[sf_indices]
            self.sf_knn = sfModel.ED_nodes.points[self.sf_knn_indices]
            self.sf_diff = sfModel.points[sf_indices].view(-1,1,3) - self.sf_knn
            self.skew_v = get_skew(self.sf_diff)

            self.J_size = (self.match_num * 3, sfModel.ED_nodes.param_num)
            if self.point_loss:
                self.J_idx = LossTool.prepare_Jacobian_idx(3, \
                    self.sf_knn_indices, self.inc_idx)

    def forward(self, lambda_, beta, new_data, grad=False, dldT_only=False):

        if self.match_num > 0:

            trans_points, Jacobian_elements = Trans_points(self.sf_diff, \
                    self.sf_knn, beta[self.sf_knn_indices], self.sf_knn_weights, \
                    grad=grad, skew_v=self.skew_v)

            trans_points = trans_points.type(fl32_)
            loss = lambda_ * (trans_points - self.new_points).view(-1,1)
        
            if grad:
                Jacobian_elements = Jacobian_elements.type(fl32_)
                dTdq = torch.flatten(torch.transpose(Jacobian_elements,1,2), start_dim=2) # Nx3x(4x4)
                dPidT = LossTool.dPi_block(trans_points) # Nx2x3(x,y,z)

                dVdT = torch.matmul(self.dVdPi, dPidT) # dVdPi: Nxcx2, dPidT: Nx2x3(x,y,z), dVdT: Nxcx3
                del dPidT
                
                if dldT_only:
                    dVdT = -dVdT.squeeze(1)
                    dVdT[:,2] += 1.
                    dVdT *= lambda_
                    return torch.block_diag(*dVdT)

                else:
                    match_num = len(trans_points)

                    # dVdT: Nxcx3, dTdq: Nx3x(n_neighborsx4), v: Nxcxn_neighborsx4
                    v = Jacobian_elements.permute(0,2,1,3) - \
                        torch.matmul(dVdT,dTdq).view(match_num,3,n_neighbors,4)
                    vb = - dVdT.unsqueeze(2).repeat(1,1,4,1) # Nxcxn_neighborsx3
                    vb[:,0,:,0] += 1.
                    vb[:,1,:,1] += 1.
                    vb[:,2,:,2] += 1.
                    vb *= self.sf_knn_weights.unsqueeze(1).unsqueeze(-1) # weights: Nx1x4x1
                    
                    v = torch.cat([v,vb], dim=-1).flatten() #Nxcxn_neighborsx7
                
                    Jacobian = torch.sparse_coo_tensor(self.J_idx, \
                        lambda_ * v.flatten(), self.J_size, dtype=fl32_)

                    return LossTool.prepare_jtj_jtl(Jacobian, loss)
                
            else:
                return torch.pow(loss,2)

        elif grad:
            return None, None
        
        else:
            return None

######################################
"""
Pytorch3d losses.
"""
######################################
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

def chamfer_loss(src, trg, sample_num=0):
    """
    Chamfer distance between pointclouds or meshes (must be ).
    """
    if sample_num == 0:
        if src.dim() == 2: src = src.unsqueeze(0)
        if trg.dim() == 2: trg = trg.unsqueeze(0)

        # Compare the two sets of pointclouds by computing (a) the chamfer loss.
        loss_chamfer, _ = chamfer_distance(trg, src)
    elif sample_num > 0:
        # Convert mesh to pytorch3d.structures.Meshes.
        if not isinstance(trg, Meshes):
            trg_mesh = get_pyt3d_mesh(trg)
        if not isinstance(src, Meshes):
            src_mesh = get_pyt3d_mesh(src)
        # Sample 5k points from the surface of each mesh.
        sample_trg = sample_points_from_meshes(trg_mesh, sample_num)
        sample_src = sample_points_from_meshes(src_mesh, sample_num)
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
    
    return loss_chamfer

######################################
"""
Triangle mesh related losses.
Modified from pytorch3d functions.
Link (v0.6.0): https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/point_mesh_distance.html#point_mesh_face_distance
"""
######################################

# PointFaceDistance
class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(ctx, points, points_first_idx, tris, tris_first_idx, max_points):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_points
        )
        ctx.save_for_backward(points, tris, idxs)
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists
        )
        return grad_points, None, grad_tris, None, None


# pyre-fixme[16]: `_PointFaceDistance` has no attribute `apply`.
point_face_distance = _PointFaceDistance.apply

# FacePointDistance
class _FacePointDistance(Function):
    """
    Torch autograd Function wrapper FacePointDistance Cuda implementation
    """

    @staticmethod
    def forward(ctx, points, points_first_idx, tris, tris_first_idx, max_tris):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_tris: Scalar equal to maximum number of faces in the batch
        Returns:
            dists: FloatTensor of shape `(T,)`, where `dists[t]` is the squared
                euclidean distance of `t`-th triangular face to the closest point in the
                corresponding example in the batch
            idxs: LongTensor of shape `(T,)` indicating the closest point in the
                corresponding example in the batch.

            `dists[t] = d(points[idxs[t]], tris[t, 0], tris[t, 1], tris[t, 2])`,
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`.
        """
        dists, idxs = _C.face_point_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_tris
        )
        ctx.save_for_backward(points, tris, idxs)
        return dists

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        grad_points, grad_tris = _C.face_point_dist_backward(
            points, tris, idxs, grad_dists
        )
        return grad_points, None, grad_tris, None, None


# pyre-fixme[16]: `_FacePointDistance` has no attribute `apply`.
face_point_distance = _FacePointDistance.apply

def point_mesh_face_distance(meshes: Meshes, pcls: Pointclouds):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    norms_packed = meshes.faces_normals_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # Find intersection of the mesh and point cloud. TODO
    _, face_idxs = _C.point_face_dist_forward(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    valid_point_to_face, face_valid = is_over_face(points, face_idxs, tris, norms_packed)

    # point to face distance: shape (P,) and indicies of the paired faces (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )

    # weight each example by the inverse of number of points in the example
    point_dist = point_to_face[valid_point_to_face].mean() / N
    # weights_p = 1.0 / valid_point_to_face.count_nonzero()
    # point_to_face = point_to_face[valid_point_to_face] * weights_p
    # point_dist = point_to_face.sum() / N

    # face to point distance: shape (T,)
    face_to_point = face_point_distance(
        points, points_first_idx, tris, tris_first_idx, max_tris
    )
    # face_to_point, _ = _C.face_point_dist_forward(
    #     points, points_first_idx, tris, tris_first_idx, max_tris
    # )

    # weight each example by the inverse of number of faces in the example
    face_dist = face_to_point[face_valid].mean() / N
    # weights_t = 1.0 / len(face_valid)
    # face_to_point = face_to_point[face_valid] * weights_t
    # face_dist = face_to_point.sum() / N
    
    return point_dist + face_dist

def is_over_face(points, idxs, tris, tri_norms):
    """
    Determine if points lie over the corresponding triangles in 3D.
    TODO: What if the point is in the face?
    """
    tris = tris[idxs]
    tri_norms = tri_norms[idxs]

    # # Decide if points lie on the different side of the faces as their
    # # normals, i.e. ax + by + cz + d < 0. If so, flip the normals.
    # ds = - torch_inner_prod(tris[:,0], tri_norms)
    # flip_norms = torch_inner_prod(points, tri_norms) + ds < 0
    # print((torch_inner_prod(points, tri_norms) + ds < 1e-8).count_nonzero())
    # tri_norms[flip_norms] = - tri_norms[flip_norms]

    n1 = torch.cross(
        F.normalize(tris[:,1]-tris[:,0], dim=-1), 
        F.normalize(points-tris[:,0], dim=-1), 
        dim=-1)
    n2 = torch.cross(
        F.normalize(tris[:,2]-tris[:,1], dim=-1), 
        F.normalize(points-tris[:,1], dim=-1), 
        dim=-1)
    n3 = torch.cross(
        F.normalize(tris[:,0]-tris[:,2], dim=-1), 
        F.normalize(points-tris[:,2], dim=-1), 
        dim=-1)

    valid_pairs = (torch_inner_prod(n1, tri_norms) >= 0.2) & \
        (torch_inner_prod(n2, tri_norms) >= 0.2) & \
        (torch_inner_prod(n3, tri_norms) >= 0.2)

    # print(valid_pairs.size(),valid_pairs.count_nonzero())
    
    return valid_pairs, torch.unique(idxs[valid_pairs])