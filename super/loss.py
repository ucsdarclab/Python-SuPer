import torch
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
import torch.nn.functional as F

from utils.utils import pcd2depth, torch_inner_prod, torch_sq_distance, JSD
from super.utils import *

# Bilinear sampling for autograd
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
        U_nm = features[n_block.long(), m_block.long()]
    else:
        if fill == 'zero':
            U_nm = torch.zeros(n_block.size()+(features.size()[-1],)).type(features.type()).cuda()
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
    n_block = torch.maximum(1-torch.abs(n_block), torch.tensor(0).cuda())
    m_block = torch.maximum(1-torch.abs(m_block), torch.tensor(0).cuda())

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

# Bilinear sampling for derived gradient
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
            U_nm = torch.ones(n_block.size()+(target_.size()[-1],), dtype=torch.float64).cuda() * float('nan')
            
            h, w = index_map.size()
            U_nm_index_map = index_map[torch.minimum(torch.maximum(n_block.long(), 
                                                                    torch.tensor(0).cuda()),
                                                    torch.tensor(h-1).cuda()), 
                                       torch.minimum(torch.maximum(m_block.long(), 
                                                                    torch.tensor(0).cuda()),
                                                    torch.tensor(w-1).cuda())
                                       ].cuda()
            U_nm_valid = (U_nm_index_map>=0) & (n_block.long()>=0) & (n_block.long()<h) & (m_block.long()>=0) & (m_block.long()<w)

            U_nm[U_nm_valid] = target_[U_nm_index_map[U_nm_valid]]
        n_block -= v.unsqueeze(1)
        m_block -= u.unsqueeze(1)
        n_block = n_block.unsqueeze(2)
        m_block = m_block.unsqueeze(2)
        if grad:
            d_n_block = torch.where(n_block >= 0, 1., -1.)
            d_m_block = torch.where(m_block >= 0, 1., -1.)
        n_block = torch.maximum(1-torch.abs(n_block), torch.tensor(0).cuda())
        m_block = torch.maximum(1-torch.abs(m_block), torch.tensor(0).cuda())
        target = torch.sum(U_nm * n_block * m_block, dim=1, dtype=torch.float64)
        if normalization:
            norm_scale = torch.norm(target, dim=1, keepdim=True)
            target /= norms_scale

        # d(bilinear sampling results) / d(coordinate to do the sampling)
        # Reference: eq.(6-7) of Spatial Transformer Networks, Max Jaderberg et al.
        if grad:
            grad = torch.stack([\
                torch.sum(U_nm * n_block * d_m_block, dim=1),\
                torch.sum(U_nm * m_block * d_n_block, dim=1)],\
                dim=2) # dV: Nxcx2
            
            if normalization:
                return target, grad/norms_scale.unsqueeze(2)
            else:
                return target, grad
        else:
            return target, None

    # d(projection in the image plane) / d(position in the 3D space) for pin hole camera
    @staticmethod
    def dPi_block(trans_points, fx, fy):

        match_num = len(trans_points)
        Z = trans_points[:,2]
        sq_Z = torch.pow(Z,2)

        dPi = torch.zeros((match_num,2,3), dtype=torch.float64).cuda()
        dPi[:,0,0] = fx/Z
        dPi[:,0,2] = -fx*trans_points[:,0]/sq_Z
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
        idx0 = torch.arange(cost_num*cost_size, device=var_idxs.device).unsqueeze(1).repeat(1, neighbor_num*var_num)

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
        jtj = torch.sparse.mm(Jacobian_t, Jacobian)
        jtl = -torch.sparse.mm(Jacobian_t,loss)
        return jtj, jtl

class DataLoss(): # Loss

    def __init__(self):
        return

    def prepare(self, sf, new_data):
        self.n_neighbors = sf.knn_indices.size(1)
        self.J_size = sf.ED_nodes.param_num

        self.sf_knn_w = sf.knn_w
        self.sf_knn_indices = sf.knn_indices
        self.sf_knn = sf.ED_nodes.points[self.sf_knn_indices] # All g_i in (10).
        self.sf_diff = sf.points.unsqueeze(1) - self.sf_knn # (p-g_i) in (10).
        self.skew_v = get_skew(self.sf_diff)

    def forward(self, lambda_, beta, inputs, new_data, grad=False, dldT_only=False):
        ### Find correspondence for ICP based on 3D-to-2D projection. Jacobian_elements: Nx4x3x4.
        trans_points, Jacobian_elements = Trans_points(self.sf_diff, self.sf_knn, \
            beta[self.sf_knn_indices], self.sf_knn_w, grad=grad, skew_v=self.skew_v)
        ## Project the updated points to the image plane.
        ## Only keep valid projections with valid new points.
        v, u, proj_coords, proj_valid_index = pcd2depth(inputs, trans_points, round_coords=False)
        valid_pair = new_data.valid[torch.minimum(torch.maximum(
                                                    proj_coords,
                                                    torch.tensor(0).cuda()
                                                ),
                                                torch.tensor(len(new_data.valid)-1).cuda())] # Valid proj-new pairs.
        valid_pair &= (proj_coords >= 0) & (proj_coords < len(new_data.valid))
        v, u = v[valid_pair], u[valid_pair]
        ## Find matched points & normal values, and calculate related grad if needed.
        new_points, dpdPi = LossTool.bilinear_intrpl_block(v, u,
            new_data.points, index_map=new_data.index_map, grad=grad)
        new_norms, dndPi = LossTool.bilinear_intrpl_block(v, u,
            new_data.norms, index_map=new_data.index_map, grad=grad)
        intrpl_valid = ~torch.any(torch.isnan(new_points)|torch.isnan(new_norms), dim=1) # Invalid new surfels.
        new_points = new_points[intrpl_valid]
        new_norms = new_norms[intrpl_valid]

        sf_indicies = proj_valid_index[valid_pair][intrpl_valid]
        trans_points = trans_points[valid_pair][intrpl_valid][sf_indicies]
        pt_diff = trans_points-new_points# T(p)-o in (13).
        loss = lambda_ * torch.sum(new_norms*pt_diff, dim=1, keepdim=True)
        
        if grad:
            dpdPi = dpdPi[intrpl_valid]
            dndPi = dndPi[intrpl_valid]

            Jacobian_elements = Jacobian_elements[valid_pair][intrpl_valid][sf_indicies]
            knn_w = self.sf_knn_w[valid_pair][intrpl_valid][sf_indicies].unsqueeze(2).unsqueeze(3) # Nx4x1x1

            dPidT = LossTool.dPi_block(trans_points, inputs["K"][0,0,0], inputs["K"][0,1,1]) # Nx2x3
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
            dndq = torch.cat([dndq, knn_w*dndT.repeat(1,self.n_neighbors,1,1)], dim=3) # Nx4x3x7

            dpdq = Jacobian_elements - torch.matmul(dpdT, Jacobian_elements) # Nx4x3x4
            dpdq_vb = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0) - dpdT.repeat(1,self.n_neighbors,1,1)
            dpdq = torch.cat([dpdq, knn_w*dpdq_vb], dim=3)

            J_idx = LossTool.prepare_Jacobian_idx(1, \
                self.sf_knn_indices[valid_pair][intrpl_valid][sf_indicies], 
                torch.arange(7, device=device))
            v = torch.matmul(new_norms.unsqueeze(1).unsqueeze(1),dpdq).squeeze(2) + \
                torch.matmul(pt_diff.unsqueeze(1).unsqueeze(1),dndq).squeeze(2) # Nx4x7
            
            v = v.flatten()
            valid = ~torch.isnan(v)
            Jacobian = torch.sparse_coo_tensor(J_idx[:,valid], \
                lambda_ * v[valid], (len(trans_points), self.J_size))
            
            return LossTool.prepare_jtj_jtl(Jacobian, loss)
        else:
            return torch.pow(loss,2)

    @staticmethod
    def autograd_forward(opt, inputs, src, trg, 
    correspts=None, correspts_valid=None,
    flow=None, 
    loss_type='point-plane', 
    max=None,
    huber_th=-1,
    src_seg=None, src_seg_conf=None, soft_seg=None):
        
        if correspts is not None:
            valid_idx = correspts_valid
            u = correspts[0][valid_idx]
            v = correspts[1][valid_idx]
        else:
            v_, u_, _, valid_idx = pcd2depth(inputs, 
                                             src.points, 
                                             round_coords=False, 
                                             valid_margin=1)
            u = u_[valid_idx]
            v = v_[valid_idx]

        if flow is not None:
            assert correspts is None

            grid = torch.stack([u_ * 2 / opt.width - 1, 
                                v_ * 2 / opt.height - 1
                               ], dim=1).view(1, -1, 1, 2).float()
            trg_loc = F.grid_sample(flow, grid)[0,:,:,0]
            u_ += trg_loc[0]
            v_ += trg_loc[1]

            valid_margin = 1
            valid_idx = (v_ >= valid_margin) & (v_ < opt.height - 1 - valid_margin) & \
                        (u_ >= valid_margin) & (u_ < opt.width - 1 - valid_margin)

            u = u_[valid_idx]
            v = v_[valid_idx]
        
        if loss_type == 'point-point':
            if src_seg is not None:
                sample_trg, _, sample_valid = bilinear_sample(
                    [trg.points, F.one_hot(trg.seg, num_classes=opt.num_classes), trg.seg_conf], 
                    v, u, index_map=trg.index_map)
                sample_trg_seg = sample_trg[1].argmax(-1)
                sample_trg_seg_conf = sample_trg[2]
                sample_trg_seg_conf = sample_trg_seg_conf[torch.arange(len(sample_trg_seg)), sample_trg_seman]
            else:
                sample_trg, _, sample_valid = bilinear_sample([trg.points], 
                    v, u, index_map=trg.index_map)
            sample_trg_verts = sample_trg[0]
            
            losses = torch_sq_distance(src.points[valid_idx][sample_valid], 
                                       sample_trg_verts[sample_valid]
                                      )
        elif loss_type == 'point-plane':
            if src_seg is not None:
                sample_trg, _, sample_valid = bilinear_sample([trg.points, 
                                                               trg.norms, 
                                                               trg.seg_conf
                                                              ], 
                                                              v, 
                                                              u, 
                                                              index_map=trg.index_map
                                                             )
                sample_trg_seg_conf = sample_trg[2]
                sample_trg_seg_conf = sample_trg_seg_conf.softmax(1)
            else:
                # ALL valid are false
                sample_trg, _, sample_valid = bilinear_sample([trg.points, trg.norms], 
                    v, u, index_map=trg.index_map)
            sample_trg_verts, sample_trg_norms = sample_trg[0], sample_trg[1]

            losses = torch_inner_prod(sample_trg_norms[sample_valid],
                                      src.points[valid_idx][sample_valid] - sample_trg_verts[sample_valid]
                                     )**2

        if max is not None:
            losses = losses[losses < max]

        if (huber_th > 0) or (src_seg is not None):
            weights_list = []

            if huber_th > 0:
                with torch.no_grad():
                    weights = torch.minimum(huber_th / torch.exp(torch.abs(losses) + 1e-20), torch.tensor(1).cuda())
                weights_list.append(weights.detach())

            if src_seg is not None:
                src_seg = src_seg[valid_idx]
                src_seg_conf = src_seg_conf[valid_idx]

                assert soft_seg is not None
                if soft_seg:
                    weights = torch.exp(- 0.1 * JSD(src_seg_conf, sample_trg_seg_conf))
                else:
                    sample_trg_seg = torch.argmax(sample_trg_seg_conf, dim=1).long()
                    weights = torch.where(src_seg == sample_trg_seg, 
                                          torch.tensor(1, dtype=torch.float64).cuda(), 
                                          torch.tensor(0, dtype=torch.float64).cuda())
                
                weights_list.append(weights[sample_valid].detach())

            if len(weights_list) == 1:
                weights = weights_list[0]
            else:
                power = 1 / len(weights_list)
                weights = torch.prod(torch.pow(torch.stack(weights_list, dim=1), power), dim=1)
            losses *= weights

        return losses.sum()

class ARAPLoss():

    def __init__(self):
        return

    def prepare(self, sfModel, new_data):

        self.ED_knn_indices = sfModel.ED_nodes.knn_indices
        self.ED_n_neighbors = self.ED_knn_indices.size(1)

        self.d_EDs = sfModel.ED_nodes.points.unsqueeze(1) - \
                        sfModel.ED_nodes.points[self.ED_knn_indices]

        self.skew_EDv = get_skew(self.d_EDs)

        inc_idx_a = torch.tensor([[0,1,2,3,4],[0,1,2,3,5],[0,1,2,3,6]], device=device)
        arap_idxa = LossTool.prepare_Jacobian_idx(3, \
            self.ED_knn_indices.reshape(-1,1), inc_idx_a)
        inc_idx_b = torch.tensor([[4],[5],[6]], device=device)
        arap_idxb = LossTool.prepare_Jacobian_idx(3, \
            torch.arange(sfModel.ED_nodes.num, device=device).unsqueeze(1).repeat(1,self.ED_n_neighbors).view(-1,1), \
            inc_idx_b)
        self.J_idx = torch.cat([arap_idxa, arap_idxb], dim=1)
        self.J_size = (sfModel.ED_nodes.num*self.ED_n_neighbors*3, sfModel.ED_nodes.param_num)

    def forward(self, lambda_, beta, inputs, new_data, grad=False, dldT_only=False):

        ED_t = beta[:,4:7]
        beta = beta[self.ED_knn_indices]
        
        loss, Jacobian_elements = transformQuatT(self.d_EDs, beta, \
            grad=grad, skew_v=self.skew_EDv)

        loss -= self.d_EDs + ED_t.unsqueeze(1)
        loss = lambda_ * loss.view(-1,1)
        
        if grad:
            match_num = len(self.d_EDs)
            Jacobian_elements = Jacobian_elements
            
            Jacobian_elements = torch.cat([Jacobian_elements, \
                torch.ones((match_num, self.ED_n_neighbors, 3, 1), dtype=torch.float64, device=device)], \
                dim=3).flatten()
            Jacobian_elements = torch.cat([Jacobian_elements, \
                -torch.ones(match_num * self.ED_n_neighbors * 3, dtype=torch.float64, device=device)])
            Jacobian_elements *= lambda_

            Jacobian = torch.sparse_coo_tensor(self.J_idx, \
                Jacobian_elements, self.J_size)
                
            return LossTool.prepare_jtj_jtl(Jacobian, loss)
        else:
            return torch.pow(loss,2)

    @staticmethod
    def autograd_forward(input, beta):
        nodes = input.points
        knn_indices = input.knn_indices
        knn_w = input.knn_w
        
        nodes_t = beta[:,4:7]
        beta = beta[knn_indices]
        
        d_nodes = (nodes.unsqueeze(1) - nodes[knn_indices])
        loss, _ = transformQuatT(d_nodes, beta, skew_v=get_skew(d_nodes))
        loss -= d_nodes.type(torch.float32) + nodes_t.unsqueeze(1)

        loss = knn_w * torch.pow(loss, 2).sum(-1)
        loss = loss.sum(-1)
        
        return loss.sum()

class RotLoss():

    def __init__(self):
        return

    def prepare(self, sfModel, new_data):

        self.J_idx = LossTool.prepare_Jacobian_idx(1, \
            torch.arange(sfModel.ED_nodes.num, device=device).unsqueeze(1), \
            torch.arange(4, device=device))
        self.J_size = (sfModel.ED_nodes.num, sfModel.ED_nodes.param_num)
        
    def forward(self, lambda_, beta, inputs, new_data, grad=False):
        beta = beta[:, 0:4].type(torch.float32)

        loss = lambda_ * (1.-torch.sum(torch.pow(beta,2), dim=1, keepdim=True))

        if grad:
            v = (-lambda_*2*beta).view(-1)
            Jacobian = torch.sparse_coo_tensor(\
                self.J_idx, v, self.J_size, dtype=torch.float32)

            return LossTool.prepare_jtj_jtl(Jacobian, loss)
        else:
            return torch.pow(loss, 2)

    @staticmethod
    def autograd_forward(beta):
        loss = 1. - torch.matmul(beta[:, 0:4][:, None, :], beta[:, 0:4][:, :, None])[:, 0, 0]
        loss = torch.pow(loss, 2)
        return loss.sum()