import numpy as np
import torch

def get_skew(inputs):
    ''' Convert `inputs` to skew-symmetric matrices.
    Input:  (...,3)     
    Return: (...,3,3)   Skew-symmetric matrix.
    '''
    a1, a2, a3 = torch.split(inputs, 1, dim=-1)
    zeros = torch.zeros_like(a1)

    return torch.stack([torch.cat([zeros, a3, -a2], dim=-1), \
        torch.cat([-a3, zeros, a1], dim=-1), \
        torch.cat([a2, -a1, zeros], dim=-1)], dim=3)


def Trans_points(d_surfels, ednodes, beta, surfel_knn_weights, grad=False, skew_v=None):
    '''  Transformation of surfels: eq (10) in SuPer paper. https://arxiv.org/pdf/1909.05405.pdf 
    Inputs: 
        - d_surfels:        (N,4,3)    Displacements from 4 NN ED nodes     p - g_i
        - ednodes:          (N,4,3)    4 NN ED nodes for each surfel        g_i
        - beta:             (N,4,7)    Transformations of ED nodes          [q_i; b_i]
        - surfel_knn_weights: (N,4)    NN weights                           \alpha_i.
        - grad:
        - skew_v:           (N,4,3,3) 
    '''
    # 'trans_surfels': T(q_i,b_i)(p-g_i); 'Jacobian': d_out/d_q_i
    trans_surfels, Jacobian = transformQuatT(d_surfels, beta, grad=grad, skew_v=skew_v)
    
    trans_surfels += ednodes
    if not np.isscalar(surfel_knn_weights):
        surfel_knn_weights = surfel_knn_weights.unsqueeze(-1)
    trans_surfels = torch.sum(surfel_knn_weights * trans_surfels, dim=-2)

    if grad:
        Jacobian *= surfel_knn_weights.unsqueeze(-1)
    
    return trans_surfels, Jacobian


def transformQuatT(v, beta, grad=False, skew_v=None):
    ''' Calculate T(q,b)v in eq (10)/(11) in SuPer paper. https://arxiv.org/pdf/1909.05405.pdf
    Inputs:
        - v:    (J,...,3)           Vertices to be transformed
        - beta: (J or 1, ... ,4 or 7) Rotation q and translation b. [q] for b=0 or [q; b] for b!=0
            - q: R^4 rotation quaternion 
            - b: R^3 translation
    '''
    device = v.device
    qw = beta[...,0:1]
    qv = beta[...,1:4]

    cross_prod = torch.cross(qv, v, dim=-1)
    tv = v + 2.0 * qw * cross_prod + 2.0 * torch.cross(qv, cross_prod, dim=-1) 
        
    # tv = rv + t
    if beta.size()[-1] == 7: tv += beta[...,4:7]

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
    

