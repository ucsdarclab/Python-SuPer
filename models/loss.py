from token import DEDENT
import numpy as np
import torch
import copy
import os

from utils import pt_matching

from utils.config import *
from utils.utils import *

class DeformLoss():
    # def __init__(self, corr_cost):
        
    #     if corr_cost:
    #         self.matcher = pt_matching.Matcher()

    #     self.increase_id = torch.tensor([[0,1,2,3,4,5,6]], device=cuda0)

    #     self.vec_to_skew_mat = torch.tensor([
    #         [[0, 0, 0],[0, 0, -1],[0, 1, 0]],
    #         [[0, 0, 1],[0, 0, 0],[-1, 0, 0]],
    #         [[0, -1, 0],[1, 0, 0],[0, 0, 0]]
    #     ], dtype=float, device=cuda0)
    #     self.eye_3 = torch.eye(3, dtype=float, device=cuda0).unsqueeze(0).unsqueeze(0)

    increase_id = torch.tensor([[0,1,2,3,4,5,6]], device=cuda0)
    eye_3 = torch.eye(3, dtype=float, device=cuda0).unsqueeze(0).unsqueeze(0)

    # eq (10) & (11)
    @staticmethod
    def trans_points(d_surfels, ednodes, quat_, surfel_knn_weights, est_grad=False, skew_v=None):#, d_t=None):

        qw = quat_[...,0:1]
        qv = quat_[...,1:4]
        t = quat_[...,4:7]

        if est_grad:
            trans_surfels, Jacobian = DeformLoss.transformQuatT(d_surfels, qw, qv, t, est_grad=est_grad, skew_v=skew_v)#, d_t = d_t)
        else:
            trans_surfels, Jacobian = DeformLoss.transformQuatT(d_surfels, qw, qv, t)
        trans_surfels += ednodes
        surfel_knn_weights = surfel_knn_weights.unsqueeze(-1)
        trans_surfels = torch.sum(surfel_knn_weights * trans_surfels, dim=-2)

        if est_grad:
            Jacobian *= surfel_knn_weights.unsqueeze(-1)
        
        return trans_surfels, Jacobian

    # q: quarternion; t: translation
    @staticmethod
    def transformQuatT(v, qw, qv, t, est_grad=False, skew_v=None):
        
        cross_prod = torch.cross(qv, v, dim=-1)

        rv = v + 2.0 * qw * cross_prod + \
            2.0 * torch.cross(qv, cross_prod, dim=-1) 
            
        tv = rv + t

        if est_grad:
            d_qw = 2 * cross_prod.unsqueeze(-1)

            qv_v_inner = torch.sum(qv*v, dim=-1)
            qv = qv.unsqueeze(-1)
            v = v.unsqueeze(-2)
            qv_v_prod = torch.matmul(qv, v)
            d_qv = 2 * (qv_v_inner.unsqueeze(-1).unsqueeze(-1) * DeformLoss.eye_3 + \
                    qv_v_prod - 2 * torch.transpose(qv_v_prod,2,3) - \
                    qw.unsqueeze(-1) * skew_v)
            return tv, torch.concat([d_qw, d_qv], dim=-1)
        else:
            return tv, 0

    @staticmethod
    def data_term(trans_points, quat_, new_norms, new_points, \
        est_grad=False, Jacobian_size=None, idx0=None, idx1=None, Jacobian_elements=None):
            
        loss = torch.sum(new_norms*(trans_points-new_points), dim=1, keepdim=True).float()

        if est_grad:

            new_norms = new_norms.unsqueeze(1).unsqueeze(1)
            v = torch.matmul(new_norms,Jacobian_elements[...,0:4])
            v = torch.concat([v,torch.tile(new_norms, (1,n_neighbors,1,1))], dim=-1).flatten()
            Jacobian = torch.sparse_coo_tensor(torch.stack([idx0,idx1],dim=0), \
                v, Jacobian_size, dtype=torch.float32)

            Jacobian_t = torch.transpose(Jacobian, 0, 1)
            jtj = torch.sparse.mm(Jacobian_t, Jacobian)
            jtl = -torch.sparse.mm(Jacobian_t,loss)
            
            return loss, jtj, jtl
        else:
            return loss

    @staticmethod
    def arap_term(d_ednodes, quat_, ednodes_t, est_grad=False, skew_v=None, \
        Jacobian_size=None, idx0=None, idx1=None):
        
        qw = quat_[...,0:1]
        qv = quat_[...,1:4]
        t = quat_[...,4:7]
        
        if est_grad:
            loss, Jacobian_elements = DeformLoss.transformQuatT(d_ednodes, qw, qv, t, \
                est_grad=est_grad, skew_v=skew_v)
            Jacobian_elements = Jacobian_elements.view(-1,4)
            Jacobian_elements = torch.concat([Jacobian_elements, \
                torch.ones((Jacobian_size[0],1), device=cuda0), \
                -torch.ones((Jacobian_size[0],1), device=cuda0)], dim=1)
            Jacobian_elements *= arap_lambda
        else:
            loss, _ = DeformLoss.transformQuatT(d_ednodes, qw, qv, t)
        
        loss -= d_ednodes + ednodes_t
        loss = loss.view(-1,1).float()

        if est_grad:
            Jacobian = torch.sparse_coo_tensor(torch.stack([idx0,idx1],dim=0), \
                Jacobian_elements.view(-1), Jacobian_size, dtype=torch.float32)

            Jacobian_t = torch.transpose(Jacobian, 0, 1)
            jtj = torch.sparse.mm(Jacobian_t, Jacobian)
            jtl = -torch.sparse.mm(Jacobian_t,loss)
                
            return loss, jtj, jtl
        else:
            return loss
        
    @staticmethod
    def rot_term(quat_, est_grad=False, Jacobian_size=None, idx0=None, idx1=None):

        loss = rot_lambda * (1.-torch.sum(torch.pow(quat_[:,0:4],2), dim=1, keepdim=True)).float()

        if est_grad:

            v = quat_[:,0:4].float() * -2. * rot_lambda
            Jacobian = torch.sparse_coo_tensor(torch.stack([idx0,idx1],dim=0), \
                v.view(-1), Jacobian_size, dtype=torch.float32)

            Jacobian_t = torch.transpose(Jacobian, 0, 1)
            jtj = torch.sparse.mm(Jacobian_t, Jacobian)
            jtl = -torch.sparse.mm(Jacobian_t,loss)

            return loss, jtj, jtl
        else:
            return loss
                
    @staticmethod
    def corr_term(d_surfels, ednodes, quat_, knn_weights, new_points, \
        est_grad=False, Jacobian_size=None, idx0=None, idx1=None, skew_v=None):

        if est_grad:
            trans_surfels, Jacobian_elements = DeformLoss.trans_points(d_surfels, \
                ednodes, quat_, knn_weights, est_grad=est_grad, skew_v=skew_v)
        else:
            trans_surfels, _ = DeformLoss.trans_points(d_surfels, \
                ednodes, quat_, knn_weights)

        loss = corr_lambda * (trans_surfels-new_points).view(-1,1).float()
        
        if est_grad:

            match_num = len(d_surfels)
            v = torch.concat([Jacobian_elements,torch.ones((match_num,n_neighbors,3,1), device=cuda0)], dim=-1)
            Jacobian = torch.sparse_coo_tensor(torch.stack([idx0,idx1],dim=0), \
                corr_lambda * v.flatten().float(), Jacobian_size, dtype=torch.float32)

            Jacobian_t = torch.transpose(Jacobian, 0, 1)
            jtj = torch.sparse.mm(Jacobian_t, Jacobian)
            jtl = -torch.sparse.mm(Jacobian_t,loss)

            return loss, jtj, jtl
        else:
            return loss
    
    @staticmethod
    def bilinear_intrpl_block(v, u, depth, est_grad=False):

        fl_v = torch.floor(v)
        cel_v = torch.ceil(v)
        fl_u = torch.floor(u)
        cel_u = torch.ceil(u)

        n_block = torch.stack([fl_v, fl_v, cel_v, cel_v], dim=1) # y
        m_block = torch.stack([fl_u, cel_u, fl_u, cel_u],dim=1) # x
        U_nm = depth[n_block.long()*WIDTH+m_block.long()]
        n_block -= v.unsqueeze(1)
        m_block -= u.unsqueeze(1)
        if est_grad:
            d_n_block = torch.where(n_block >= 0, 1., -1.)
            d_m_block = torch.where(m_block >= 0, 1., -1.)
        n_block = torch.maximum(1-torch.abs(n_block), torch.tensor(0))
        m_block = torch.maximum(1-torch.abs(m_block), torch.tensor(0))

        target = torch.sum(U_nm * n_block * m_block, dim=1)

        if est_grad:
            dV_dx = torch.sum(U_nm * n_block * d_m_block, dim=1)
            dV_dy = torch.sum(U_nm * m_block * d_n_block, dim=1)
            
            return target, torch.stack([dV_dx, dV_dy], dim=1).unsqueeze(1).float() # dV: Nx1x2
        else:
            return target

    @staticmethod
    def depth_term(trans_points, y, x, depth, \
        est_grad=False, Jacobian_size=None, idx0=None, idx1=None, Jacobian_elements=None):
        # Jacobian_elements: Nx4x3x7

        Z = trans_points[:,2]
        sq_Z = torch.pow(Z,2)
        wrap_source = Z
        
        if est_grad:

            target, dV = DeformLoss.bilinear_intrpl_block(y, x, depth, est_grad=est_grad) # dV: Nx1x2
            loss = (wrap_source - target).unsqueeze(1).float()

            match_num = trans_points.size()[0]

            ## Include bilinear term loss.
            dT = torch.flatten(torch.transpose(Jacobian_elements,1,2), start_dim=2).float() # Nx3x(4x4)

            dPi = torch.zeros((match_num,2,3), device=cuda0)
            dPi[:,0,0] = -fx/Z
            dPi[:,0,2] = fx*trans_points[:,0]/sq_Z
            dPi[:,1,1] = fy/Z
            dPi[:,1,2] = -fy*trans_points[:,1]/sq_Z

            dV_dPi = torch.matmul(dV, dPi)
            del dV, dPi
            v = Jacobian_elements[:,:,2,:] - torch.matmul(dV_dPi,dT).view(match_num,n_neighbors,4)
            vb = -torch.tile(dV_dPi, (1,4,1))
            vb[...,2] += 1
            v = torch.concat([v,vb], dim=-1).flatten()

            # ## Exclude bilinear term loss.
            # v = torch.concat([Jacobian_elements[:,:,2,:], \
            #     torch.zeros((match_num,n_neighbors,3), device=cuda0)], dim=-1).flatten()

            Jacobian = torch.sparse_coo_tensor(torch.stack([idx0,idx1],dim=0), \
                v.view(-1), Jacobian_size, dtype=torch.float32)

            Jacobian_t = torch.transpose(Jacobian, 0, 1)
            jtj = torch.sparse.mm(Jacobian_t, Jacobian)
            jtl = -torch.sparse.mm(Jacobian_t,loss)
            
            return loss, jtj, jtl

        else:

            target = DeformLoss.bilinear_intrpl_block(y, x, depth)
            loss = (wrap_source - target).unsqueeze(1).float()
            return loss