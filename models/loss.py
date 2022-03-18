from abc import ABC, abstractmethod

from token import DEDENT
import numpy as np
import torch
import copy
import os

from typeguard import check_argument_types, check_return_type

from utils.config import *
from utils.utils import *
from utils.img_matching import *
from models.LM import *

class LossTool():

    @staticmethod
    def bilinear_intrpl_block(v, u, target_, grad=False, normalization=False):

        fl_v = torch.floor(v)
        cel_v = torch.ceil(v)
        fl_u = torch.floor(u)
        cel_u = torch.ceil(u)

        n_block = torch.stack([fl_v, fl_v, cel_v, cel_v], dim=1) # y
        m_block = torch.stack([fl_u, cel_u, fl_u, cel_u],dim=1) # x
        U_nm = target_[n_block.long()*WIDTH+m_block.long()]
        n_block -= v.unsqueeze(1)
        m_block -= u.unsqueeze(1)
        # if len(target_.size()) == 2:
        #     n_block = n_block.unsqueeze(2)
        #     m_block = m_block.unsqueeze(2)
        n_block = n_block.unsqueeze(2)
        m_block = m_block.unsqueeze(2)
        if grad:
            d_n_block = torch.where(n_block >= 0, 1., -1.)
            d_m_block = torch.where(m_block >= 0, 1., -1.)
        n_block = torch.maximum(1-torch.abs(n_block), torch.tensor(0, device=dev)) # TODO
        m_block = torch.maximum(1-torch.abs(m_block), torch.tensor(0, device=dev))
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
        # points, norms, valid = new_data

        ### Find correspondence based on projective ICP. Jacobian_elements: Nx4x3x4.
        trans_points, Jacobian_elements = Trans_points(self.sf_diff, self.sf_knn, \
            beta[self.sf_knn_indicies], self.sf_knn_weights, grad=grad, skew_v=self.skew_v)
        ## Project the updated points to the image plane.
        ## Only keep valid projections with valid new points.
        v, u, proj_coords, proj_valid_index = pcd2depth(trans_points, depth_sort=True, round_coords=False)
        valid_pair = new_data.valid[proj_coords] # Valid proj-new pairs.
        v, u = v[valid_pair], u[valid_pair]
        ## Find matched points & normal values, and calculate related grad if needed.
        new_points, dpdPi = LossTool.bilinear_intrpl_block(v, u, new_data.points, grad=grad)
        new_norms, dndPi = LossTool.bilinear_intrpl_block(v, u, new_data.norms, grad=grad)

        sf_indicies = proj_valid_index[valid_pair]
        trans_points = trans_points[sf_indicies].type(fl32_)
        pt_diff = trans_points-new_points# T(p)-o in (13).
        loss = lambda_ * torch.sum(new_norms*pt_diff, dim=1, keepdim=True)
        
        if grad:
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

        self.ED_knn_indices = sfModel.ED_knn_indices

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
            return torch.pow(loss,2)

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

# def CoordLoss(Loss):

#     def __init__(self):

#     def prepare(self, sfModel):
#         self.sf_knn = sfModel.ED_points[sfModel.surfel_knn_indexs]
#         self.sf_diff = sfModel.points.unsqueeze(1) - self.sf_knn # p-g_i
#         self.skew_v = get_skew(self.sf_diff)

#     def forward(self, lambda_, beta, new_data, grad=False, dldT_only=False):
#     # def forward(lambda_, trans_points, target, \
#         # grad=False, dldT_only=False, Jacobian_size=None, idx=None, \
#         # Jacobian_elements=None, knn_weights=None):
#         # Jacobian_elements: Nx4x3x4

#         v, u, _, _ = pcd2depth(trans_points, vis_only=False, round_coords=False)
#         # u /= (WIDTH - 1)
#         # v /= (HEIGHT - 1)
#         wrap_source = torch.stack([u,v], dim=1)
#         # wrap_source = wrap_source * 2 - 1

#         loss = lambda_ * (wrap_source - target).type(dtype_).view(-1,1)
#         -----> target - wrap_source
#         del v, u, wrap_source, target
        
#         if grad:

#             trans_points = trans_points.type(dtype_)
#             Jacobian_elements = Jacobian_elements.type(dtype_)

#             dPidT = DeformLoss.dPi_block(trans_points) # Nx2x3
#             # Pi_scale = torch.tensor([[[2/(WIDTH-1)],[2/(HEIGHT-1)]]], dtype=dtype_, device=dev)
#             # dPidT *= Pi_scale

#             if dldT_only:
#                 dPidT *= lambda_
#                 return torch.block_diag(*dPidT)

#             # dTdq = torch.flatten(torch.transpose(Jacobian_elements,1,2), start_dim=2) # Nx3x(4x4)

#             match_num = len(trans_points)

#             v = torch.matmul(dPidT.unsqueeze(1),Jacobian_elements) # Nx4x2x4
#             vb = dPidT.unsqueeze(1).repeat(1,4,1,1) * knn_weights.unsqueeze(2).unsqueeze(3) # Nx4x2x3
#             v = torch.cat([v,vb], dim=3).permute(0,2,1,3).flatten()

#             # v = torch.matmul(dPidT,dTdq).view(match_num,2,n_neighbors,4)
#             # vb = dPidT.unsqueeze(2).repeat(1,1,4,1) * knn_weights.unsqueeze(1).unsqueeze(3)
#             # v = torch.cat([v,vb], dim=-1).flatten()

#             Jacobian = torch.sparse_coo_tensor(idx, \
#                 lambda_ * v, Jacobian_size, dtype=dtype_)

#             return DeformLoss.prepare_jtj_jtl(Jacobian, loss)

#         else:

#             return torch.pow(loss,2)