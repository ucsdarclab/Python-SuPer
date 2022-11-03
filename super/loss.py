from abc import ABC, abstractmethod

import torch.nn.functional as F

from utils.utils import *
from super.utils import *

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
            U_nm = torch.ones(n_block.size()+(target_.size()[-1],), dtype=fl64_).cuda() * float('nan')
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
        n_block = torch.maximum(1-torch.abs(n_block), torch.tensor(0).cuda()) # TODO
        m_block = torch.maximum(1-torch.abs(m_block), torch.tensor(0).cuda())
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

        dPi = torch.zeros((match_num,2,3), dtype=dtype_).cuda()
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
    def autograd_forward(opt, inputs, src, trg, 
    correspts=None, correspts_valid=None,
    flow=None, 
    loss_type='point-plane', reduction='mean', square_loss=True, output_ids=None,
    color_hint=False, huber_th=-1, max_losses=None, min_losses=None, 
    src_seman=None, src_seman_conf=None, soft_seman=None):
        
        if correspts is not None:
            valid_idx = correspts_valid
            u = correspts[0][valid_idx]
            v = correspts[1][valid_idx]

            # import cv2
            # ed_v, ed_u, _, _ = pcd2depth(inputs, src.points, round_coords=False)
            # img = 255 * torch_to_numpy(inputs[("color", 0, 0)][0].permute(1,2,0))[...,::-1]
            # img = img.astype(np.uint8).copy()
            # colors = [(255,0,0),
            #           (0,255,0),
            #           (0,0,255)]
            # ed_colors = [(150,0,0),
            #              (0,150,0),
            #              (0,0,150)]
            # for _u_, _v_, _ed_u_, _ed_v_, class_id in zip(u, v, ed_u[valid_idx], ed_v[valid_idx], src_seman[valid_idx]):
            #     img = cv2.rectangle(img, 
            #         (int(_u_ - 5), int(_v_ - 5)), (int(_u_ + 5), int(_v_ + 5)),
            #         colors[class_id], -1)

            #     img = cv2.arrowedLine(img, 
            #         (int(_ed_u_+0.3*(_u_-_ed_u_)), int(_ed_v_+0.3*(_v_-_ed_v_))), 
            #         (int(_ed_u_+0.6*(_u_-_ed_u_)), int(_ed_v_+0.6*(_v_-_ed_v_))),
            #         colors[class_id], 1)

            #     img = cv2.circle(img, (int(_ed_u_), int(_ed_v_)), 4, (255, 255, 255), -1)
            #     img = cv2.circle(img, (int(_ed_u_), int(_ed_v_)), 3, ed_colors[class_id], -1)

            # for _u_, _v_, class_id, knn_id in zip(ed_u[~valid_idx], ed_v[~valid_idx], src_seman[~valid_idx], src.knn_indices[~valid_idx]):
            #     img = cv2.circle(img, (int(_u_), int(_v_)), 4, ed_colors[class_id], -1)

            # #     # if class_id == 2:
            # #     #     for _knn_id_ in knn_id:
            # #     #         img = img.astype(np.uint8).copy()
            # #     #         img = cv2.line(img, 
            # #     #             (int(_u_.item()), int(_v_.item())), 
            # #     #             (int(ed_u[_knn_id_].item()), int(ed_v[_knn_id_].item())), 
            # #     #             (255, 255, 255), 1)

            # cv2.imwrite("bnmorph.jpg", img)
            
        else:
            v_, u_, _, valid_idx = pcd2depth(inputs, src.points, round_coords=False, valid_margin=1)
            u = u_[valid_idx]
            v = v_[valid_idx]

        if flow is not None:
            assert correspts is None

            if isinstance(flow, tuple):
                matches1, matches2 = flow
                if len(matches1) == 0:
                    return 0

                v, u = matches2[:, 1], matches2[:, 0]

                src_index_map = - torch.ones((inputs["height"], inputs["width"]), dtype=long_)
                src_index_map[v_[valid_idx].type(long_), u_[valid_idx].type(long_)] = torch.arange(len(v_))[valid_idx]
                src_indexs = src_index_map[matches1[:, 1].type(long_), matches1[:, 0].type(long_)]
                sample_src_verts = src.points[src_indexs]
                # sample_src_norms = src.norms[src_indexs]
                src_sample_valid = src_indexs >= 0
                # sample_src, _, src_sample_valid = bilinear_sample([src.points], 
                #     matches1[:, 1], matches1[:, 0], index_map=src_index_map)
                # sample_src, _, src_sample_valid = bilinear_sample([trg.points], 
                #     matches1[:, 1]+0.1, matches1[:, 0]+0.1, index_map=trg.index_map)
                # sample_src_verts = sample_src[0]
            else:
                grid = torch.stack(
                    [u_ * 2 / inputs["width"] - 1, 
                    v_ * 2 / inputs["height"] - 1], dim=1).view(1, -1, 1, 2).type(fl32_)
                trg_loc = F.grid_sample(flow, grid)[0,:,:,0]
                u_ += trg_loc[0]
                v_ += trg_loc[1]

                valid_margin = 1
                valid_idx = (v_ >= valid_margin) & (v_ < inputs["height"]-1-valid_margin) & \
                    (u_ >= valid_margin) & (u_ < inputs["width"]-1-valid_margin)

                u = u_[valid_idx]
                v = v_[valid_idx]
        
        if loss_type == 'point-point':
            if src_seman is not None:
                sample_trg, _, sample_valid = bilinear_sample(
                    [trg.points, F.one_hot(trg.seman, num_classes=opt.num_classes), trg.seman_conf], 
                    v, u, index_map=trg.index_map)
                sample_trg_seman = sample_trg[1].argmax(-1)
                sample_trg_seman_conf = sample_trg[2]
                sample_trg_seman_conf = sample_trg_seman_conf[torch.arange(len(sample_trg_seman)), sample_trg_seman]
            else:
                sample_trg, _, sample_valid = bilinear_sample([trg.points], 
                    v, u, index_map=trg.index_map)
            sample_trg_verts = sample_trg[0]
            
            if 'sample_src_verts' in locals():
                sample_valid = sample_valid & src_sample_valid
                losses = torch_sq_distance(
                    sample_src_verts[sample_valid], sample_trg_verts[sample_valid])
            else:
                losses = torch_sq_distance(
                    src.points[valid_idx][sample_valid], sample_trg_verts[sample_valid])
        
        elif loss_type == 'point-plane':
            if src_seman is not None:
                sample_trg, _, sample_valid = bilinear_sample([trg.points, trg.norms, trg.seman_conf], 
                    v, u, index_map=trg.index_map)
                sample_trg_seman_conf = sample_trg[2]
                sample_trg_seman_conf = sample_trg_seman_conf / sample_trg_seman_conf.sum(1, keepdim=True) # Normalize
            else:
                sample_trg, _, sample_valid = bilinear_sample([trg.points, trg.norms], 
                    v, u, index_map=trg.index_map)

                # sample_grid = torch.stack([u / (opt.width-1) * 2 - 1, v / (opt.height-1) * 2 - 1], dim=1)[None, :, None, :].type(fl32_)
                # sample_trg_verts = F.grid_sample(inputs[("pcd", 0)].permute(0, 3, 1, 2), 
                #                                  sample_grid,
                #                                  align_corners=True
                #                                 )[0, :, :, 0].permute(1, 0)
                # sample_trg_norms = F.grid_sample(inputs[("normal", 0)].permute(0, 3, 1, 2), 
                #                                  sample_grid
                #                                 )[0, :, :, 0].permute(1, 0)
                # sample_valid = ~ torch.any(
                #                            torch.isnan(torch.cat([sample_trg_verts, sample_trg_norms], dim=1))
                #                            , dim=1
                #                           )
            sample_trg_verts, sample_trg_norms = sample_trg[0], sample_trg[1]

            if 'sample_src_verts' in locals():
                sample_valid = sample_valid & src_sample_valid
                losses = torch_inner_prod(
                    sample_trg_norms[sample_valid],
                    sample_src_verts[sample_valid] - sample_trg_verts[sample_valid])
            else:
                losses = torch_inner_prod(
                    sample_trg_norms[sample_valid],
                    src.points[valid_idx][sample_valid] - sample_trg_verts[sample_valid])
                # losses = (torch_inner_prod(
                #     src.norms[valid_idx][sample_valid],
                #     src.points[valid_idx][sample_valid] - sample_trg_verts[sample_valid])**2)

            if square_loss:
                losses = losses ** 2

        if max_losses is not None:
            losses = torch.where(losses < max_losses, losses, 0.)

        if min_losses is not None:
            losses = torch.where(losses > min_losses[valid_idx], losses, 0.)

        if (huber_th > 0) or (color_hint) or (src_seman is not None):
            weights_list = []

            if color_hint:
                src_colors = src.colors[valid_idx][sample_valid]

                trg_colors, _, sample_valid = bilinear_sample([trg.colors], v, u, index_map=trg.index_map)
                trg_colors = trg_colors[0][sample_valid]
                
                weights_list.append(torch.exp(- 0.1 * torch.abs(src_colors - trg_colors).mean(1)).detach())

            if huber_th > 0:
                with torch.no_grad():
                    weights = torch.minimum(huber_th / torch.exp(torch.abs(losses) + 1e-20), torch.tensor(1).cuda())
                weights_list.append(weights.detach())

            if src_seman is not None:
                src_seman = src_seman[valid_idx]
                src_seman_conf = src_seman_conf[valid_idx]
                sample_trg_seman = torch.argmax(sample_trg_seman_conf, dim=1).type(long_)

                # src_counter_seman_conf = src_seman_conf[torch.arange(len(sample_trg_seman), dtype=long_), sample_trg_seman]
                # src_seman_conf = src_seman_conf[torch.arange(len(src_seman), dtype=long_), src_seman]
                # sample_trg_seman_conf = sample_trg_seman_conf[torch.arange(len(sample_trg_seman), dtype=long_), sample_trg_seman]
                # if opt.sf_hard_seman_point_plane:
                #     weights = torch.where(src_seman == sample_trg_seman, 
                #                       src_seman_conf * sample_trg_seman_conf, 
                #                       torch.tensor(0, dtype=fl64_).cuda())
                # elif opt.sf_soft_seman_point_plane:
                #     weights = torch.where(src_seman == sample_trg_seman, 
                #                       src_seman_conf * sample_trg_seman_conf, 
                #                       src_counter_seman_conf * sample_trg_seman_conf)
                assert soft_seman is not None
                if soft_seman:
                    weights = torch.exp(- JSD(src_seman_conf, sample_trg_seman_conf))
                else:
                    weights = torch.where(src_seman == sample_trg_seman, 
                                          torch.tensor(1, dtype=fl64_).cuda(), 
                                          torch.tensor(0, dtype=fl64_).cuda())
                
                weights_list.append(weights[sample_valid].detach())
        
            if len(weights_list) == 1:
                weights = weights_list[0]
            else:
                power = 1 / len(weights_list)
                weights = torch.prod(torch.pow(torch.stack(weights_list, dim=1), power), dim=1)
            losses *= weights

        if output_ids is not None:
            with torch.no_grad():
                losses_ids = - torch.ones(len(src.points), dtype=long_).cuda()
                losses_valid = valid_idx.clone()
                losses_valid[valid_idx] = sample_valid.cuda()
                losses_ids[losses_valid] = torch.arange(len(losses)).cuda()
                losses_ids = losses_ids[output_ids]
                losses_ids = losses_ids[losses_ids >= 0]
            losses = torch.index_select(losses, 0, losses_ids)

        if not square_loss:
            return losses
        elif reduction == 'mean':
            return losses.mean()
        elif reduction == 'sum':
            return losses.sum()
        else:
            assert False

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
    def autograd_forward(input, beta, reduction='mean'):
        nodes = input.points
        knn_indices = input.knn_indices
        knn_w = input.knn_w
        
        nodes_t = beta[:,4:7]
        beta = beta[knn_indices]
        
        d_nodes = (nodes.unsqueeze(1) - nodes[knn_indices])
        loss, _ = transformQuatT(d_nodes, beta, skew_v=get_skew(d_nodes))
        loss -= d_nodes.type(fl32_) + nodes_t.unsqueeze(1)

        loss = knn_w * torch.pow(loss, 2).sum(-1)
        loss = loss.sum(-1)
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            assert False

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
    def autograd_forward(beta, reduction='mean', square_loss=True):
        # loss = 1. - torch.sum(torch.pow(beta[:, 0:4],2), dim=1)
        loss = 1. - torch.matmul(beta[:, 0:4][:, None, :], beta[:, 0:4][:, :, None])[:, 0, 0]
        # loss = torch.cat([loss, torch.norm(beta[:, 4:], dim=1)])

        if square_loss:
            loss = torch.pow(loss, 2)

        if not square_loss:
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            assert False

# class CorrLoss(Loss):

#     def __init__(self, point_loss=False, point_plane_loss=False):
#         if point_loss:
#             self.inc_idx = torch.tensor([[0,1,2,3,4,5,6]], device=dev).repeat(3,1)

#         if corr_method == 'opencv':
#             self.matcher = cv2Matcher()
#         elif corr_method == 'kornia':
#             self.matcher = LoFTR()

#         self.point_loss = point_loss
#         self.point_plane_loss = point_plane_loss

#     def prepare(self, sfModel, new_data):
#         # Find correspondence.
#         m1, m2 = self.matcher.match_features(sfModel.renderImg, new_data.rgb, new_data.ID)

#         if len(m1) == 0:
#             self.match_num = 0
#             return
        
#         ## Find the indices of matched surfels.
#         valid_indices = sfModel.projdata[:,0].long()
#         indices_map = - torch.ones((HEIGHT,WIDTH), dtype=long_, device=dev)
#         indices_map[sfModel.projdata[:,1].long(), sfModel.projdata[:,2].long()] = valid_indices
#         sf_indices = indices_map[m1[:,1].long(),m1[:,0].long()] # Indices of matched surfels.
#         ## Estimate (interpolate) the 3D positions of the matched new points,
#         ## and estimate the gradient related to these new points.
#         # target: Nxc, dVdPi: Nxcx2, Pi: project x,y
#         self.new_points, self.dVdPi = LossTool.bilinear_intrpl_block(m2[:,1], m2[:,0], \
#             new_data.points, grad=True)
        
#         new_valid = new_data.valid.type(fl32_).view(-1,1)
#         new_valid[new_valid == 0] = float('nan')
#         new_valid, _ = LossTool.bilinear_intrpl_block(m2[:,1], m2[:,0], new_valid)
#         valid_pair = (sf_indices >= 0) & (~torch.isnan(new_valid[:,0]))
#         self.match_num = torch.count_nonzero(valid_pair)
#         if self.match_num > 0:
#             sf_indices = sf_indices[valid_pair]
#             self.new_points = self.new_points[valid_pair].type(fl32_)
#             self.dVdPi = self.dVdPi[valid_pair]
        
#             self.sf_knn_weights = sfModel.sf_knn_weights[sf_indices]
#             self.sf_knn_indices = sfModel.sf_knn_indices[sf_indices]
#             self.sf_knn = sfModel.ED_nodes.points[self.sf_knn_indices]
#             self.sf_diff = sfModel.points[sf_indices].view(-1,1,3) - self.sf_knn
#             self.skew_v = get_skew(self.sf_diff)

#             self.J_size = (self.match_num * 3, sfModel.ED_nodes.param_num)
#             if self.point_loss:
#                 self.J_idx = LossTool.prepare_Jacobian_idx(3, \
#                     self.sf_knn_indices, self.inc_idx)

#     def forward(self, lambda_, beta, new_data, grad=False, dldT_only=False):

#         if self.match_num > 0:

#             trans_points, Jacobian_elements = Trans_points(self.sf_diff, \
#                     self.sf_knn, beta[self.sf_knn_indices], self.sf_knn_weights, \
#                     grad=grad, skew_v=self.skew_v)

#             trans_points = trans_points.type(fl32_)
#             loss = lambda_ * (trans_points - self.new_points).view(-1,1)
        
#             if grad:
#                 Jacobian_elements = Jacobian_elements.type(fl32_)
#                 dTdq = torch.flatten(torch.transpose(Jacobian_elements,1,2), start_dim=2) # Nx3x(4x4)
#                 dPidT = LossTool.dPi_block(trans_points) # Nx2x3(x,y,z)

#                 dVdT = torch.matmul(self.dVdPi, dPidT) # dVdPi: Nxcx2, dPidT: Nx2x3(x,y,z), dVdT: Nxcx3
#                 del dPidT
                
#                 if dldT_only:
#                     dVdT = -dVdT.squeeze(1)
#                     dVdT[:,2] += 1.
#                     dVdT *= lambda_
#                     return torch.block_diag(*dVdT)

#                 else:
#                     match_num = len(trans_points)

#                     # dVdT: Nxcx3, dTdq: Nx3x(n_neighborsx4), v: Nxcxn_neighborsx4
#                     v = Jacobian_elements.permute(0,2,1,3) - \
#                         torch.matmul(dVdT,dTdq).view(match_num,3,n_neighbors,4)
#                     vb = - dVdT.unsqueeze(2).repeat(1,1,4,1) # Nxcxn_neighborsx3
#                     vb[:,0,:,0] += 1.
#                     vb[:,1,:,1] += 1.
#                     vb[:,2,:,2] += 1.
#                     vb *= self.sf_knn_weights.unsqueeze(1).unsqueeze(-1) # weights: Nx1x4x1
                    
#                     v = torch.cat([v,vb], dim=-1).flatten() #Nxcxn_neighborsx7
                
#                     Jacobian = torch.sparse_coo_tensor(self.J_idx, \
#                         lambda_ * v.flatten(), self.J_size, dtype=fl32_)

#                     return LossTool.prepare_jtj_jtl(Jacobian, loss)
                
#             else:
#                 return torch.pow(loss,2)

#         elif grad:
#             return None, None
        
#         else:
#             return None