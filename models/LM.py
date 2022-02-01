# modified from: https://github.com/MengHao666/Minimal-Hand-pytorch
import time
import numpy as np
import torch
import cv2
import copy
import os

from utils import pt_matching

from utils.config import *
from utils.utils import *
from models.loss import *

# reference: https://github.com/DeformableFriends/NeuralTracking/blob/main/model/model.py
class LinearSolverLU(torch.autograd.Function):
    """
    LU linear solver.
    """

    @staticmethod #### TODO
    def forward(self, A, b):
        A_LU, pivots = torch.lu(A)
        x = torch.lu_solve(b, A_LU, pivots)
        
        return x

class LM_Solver():
    def __init__(self, method, data_cost, depth_cost, arap_cost, rot_cost, corr_cost, th_stop=1e-30):
        
        self.data_cost = data_cost
        self.depth_cost = depth_cost
        self.arap_cost = arap_cost
        self.rot_cost = rot_cost
        self.corr_cost = corr_cost
        if corr_cost:
            if method == 'super': 
                self.matcher = pt_matching.Matcher()
            elif method == 'dlsuper': 
                self.matcher = pt_matching.DeepMatcher()

        self.beta_init_ = torch.tensor([[1.,0.,0.,0.,0.,0.,0.]], dtype=float, device=cuda0)
        # self.increase_id = torch.tensor([[0,1,2,3,4,5,6]], device=cuda0)

        self.vec_to_skew_mat = torch.tensor([
            [[0, 0, 0],[0, 0, -1],[0, 1, 0]],
            [[0, 0, 1],[0, 0, 0],[-1, 0, 0]],
            [[0, -1, 0],[1, 0, 0],[0, 0, 0]]
        ], dtype=float, device=cuda0)
        self.eye_3 = torch.eye(3, dtype=float, device=cuda0).unsqueeze(0).unsqueeze(0)

    ## losses
    def prepareCostTerm(self, quat_, depth, trans_points=None, y=None, x=None, est_grad=False, Jacobian_elements=None):

        # calculate cost functions
        error = []
        jtj = torch.zeros((self.param_num, self.param_num), device=cuda0)
        jtl = torch.zeros((self.param_num, 1), device=cuda0)
        
        if self.data_cost:

            if est_grad:
                data_loss, data_jtj, data_jtl = DeformLoss.data_term(trans_points, \
                    quat_[self.match_surfel_knn_idxs], \
                    self.match_new_norms, self.match_new_points, \
                    est_grad=est_grad, Jacobian_size=(self.match_num, self.param_num), \
                    idx0=self.data_idx0, idx1=self.data_idx1, Jacobian_elements=Jacobian_elements)

                jtj += data_jtj
                jtl += data_jtl
                del data_jtj, data_jtl
            else:
                data_loss = DeformLoss.data_term(trans_points, \
                    quat_[self.match_surfel_knn_idxs], \
                    self.match_new_norms, self.match_new_points)
            
            error.append(data_loss)
            del data_loss

        if self.depth_cost:
            
            if est_grad:
                depth_loss, depth_jtj, depth_jtl = DeformLoss.depth_term(trans_points, y, x, depth, \
                    est_grad=est_grad, Jacobian_size=(self.match_num, self.param_num), \
                    idx0=self.data_idx0, idx1=self.data_idx1, Jacobian_elements=Jacobian_elements)
                
                jtj += depth_jtj
                jtl += depth_jtl
            else:
                depth_loss = DeformLoss.depth_term(trans_points, y, x, depth)

            error.append(depth_loss)
            del depth_loss
                
        if self.arap_cost:

            if est_grad:
                arap_loss, arap_jtj, arap_jtl = DeformLoss.arap_term(self.d_EDs, quat_[self.ED_knn_idxs], quat_[:,4:7].unsqueeze(1), \
                    est_grad=est_grad, skew_v=self.skew_EDv, Jacobian_size=(self.arap_loss_num, self.param_num),\
                    idx0=self.arap_idx0, idx1=self.arap_idx1)
                
                jtj += arap_jtj
                jtl += arap_jtl
            else:
                arap_loss = DeformLoss.arap_term(self.d_EDs, quat_[self.ED_knn_idxs], quat_[:,4:7].unsqueeze(1))
            
            error.append(arap_loss)
            del arap_loss
        
        if self.rot_cost:

            if est_grad:
                rot_loss, rot_jtj, rot_jtl = DeformLoss.rot_term(quat_, est_grad=est_grad, \
                    Jacobian_size=(self.ED_num, self.param_num), idx0=self.rot_idx0, idx1=self.rot_idx1)

                jtj += rot_jtj
                jtl += rot_jtl
            else:
                rot_loss = DeformLoss.rot_term(quat_)

            error.append(rot_loss)
            del rot_loss

        if self.corr_cost and self.corr_cost_num:

            if est_grad:
                corr_loss, corr_jtj, corr_jtl = DeformLoss.corr_term(self.corr_d_surfels, self.corr_anchor_ednodes, \
                    quat_[self.corr_surfel_knn_indexs], self.corr_surfel_knn_weights, self.corr_new, \
                    est_grad=est_grad, Jacobian_size=(self.corr_cost_num,self.param_num), \
                    idx0=self.corr_idx0, idx1=self.corr_idx1, skew_v=self.skew_corr_v)

                jtj += corr_jtj
                jtl += corr_jtl
            else:
                corr_loss = DeformLoss.corr_term(self.corr_d_surfels, self.corr_anchor_ednodes, \
                    quat_[self.corr_surfel_knn_indexs], self.corr_surfel_knn_weights, self.corr_new)

            error.append(corr_loss)
            del corr_loss

        if est_grad:
            return torch.concat(error), jtj, jtl


        else:
            return torch.concat(error)#.float()

    def update_correspondence(self, allsurfels, valid, beta, est_grad=False, skew_v=None):#, d_t=None):

        ednodes = allsurfels.ED_points[allsurfels.surfel_knn_indexs]
        d_surfels = allsurfels.points.unsqueeze(1) - ednodes
        trans_surfels, Jacobian_elements = DeformLoss.trans_points(d_surfels, ednodes, \
            beta[allsurfels.surfel_knn_indexs], allsurfels.surfel_knn_weights, \
            est_grad=est_grad, skew_v=skew_v) #, d_t=d_t

        try:
            # project the updated points & norms to the image plane
            v, u, proj_coords, proj_valid_index = pcd2depth(trans_surfels, vis_only=True, round_coords=False)
            
            # Find correspondence based on projective ICP
            # Reference: Real-time Geometry, Albedo and Motion Reconstruction 
            # Using a Single RGBD Camera
            pair_valid = valid[proj_coords]

            if est_grad:
                return proj_coords[pair_valid], proj_valid_index[pair_valid], \
                    trans_surfels[proj_valid_index][pair_valid], \
                    Jacobian_elements[proj_valid_index][pair_valid], \
                    v[proj_valid_index][pair_valid], u[proj_valid_index][pair_valid]
            else:
                return proj_coords[pair_valid], proj_valid_index[pair_valid], \
                    trans_surfels[proj_valid_index][pair_valid], \
                    v[proj_valid_index][pair_valid], u[proj_valid_index][pair_valid]
        except:
            pass

    # LM algorithm
    def LM(self, allsurfels, points, norms, rgb, time_, valid, inputImg, ID, num_Iter=10, matches=None):

        start = timeit.default_timer()

        surfels_num = allsurfels.surfel_num
        self.ED_num = allsurfels.ED_num
        
        inc_idx = torch.tensor([[0,1,2,3,4]], device=cuda0)
        increase_id = torch.tensor([[0,1,2,3,4,5,6]], device=cuda0)
        linear_solver = LinearSolverLU.apply

        minimal_loss = 1e10
        best_beta = torch.tile(self.beta_init_, (self.ED_num,1))

        # prepare fixed tensors
        self.param_num = self.ED_num * 7
        jtj_idx = torch.arange(self.param_num, device=cuda0)

        u = 50.
        v = 7.5
        beta = copy.deepcopy(best_beta)

        if self.data_cost or self.depth_cost or self.corr_cost:

            d_surfels = allsurfels.points.unsqueeze(1) - allsurfels.ED_points[allsurfels.surfel_knn_indexs]
            skew_v = torch.inner(d_surfels, self.vec_to_skew_mat)
            
        if self.arap_cost:

            self.ED_knn_idxs = allsurfels.ednode_knn_indexs

            self.d_EDs = allsurfels.ED_points.unsqueeze(1) - \
                            allsurfels.ED_points[self.ED_knn_idxs]

            self.skew_EDv = torch.inner(self.d_EDs, self.vec_to_skew_mat)
            self.d_EDt = torch.tile(self.eye_3, (self.ED_num,ED_n_neighbors,1,1))

            self.arap_loss_num = self.ED_num * ED_n_neighbors * 3

            self.arap_idx0 = torch.arange(self.arap_loss_num, device=cuda0).unsqueeze(1)
            self.arap_idx0 = torch.tile(self.arap_idx0, (1,6)).view(-1)
            self.arap_idx1 = torch.tile(self.ED_knn_idxs.view(-1,1)*7, (1,5)) + inc_idx
            identity_idx = torch.arange(self.ED_num, device=cuda0).unsqueeze(1) * 7 + 4
            identity_idx = torch.tile(identity_idx, (1,ED_n_neighbors)).view(-1,1)
            self.arap_idx1 = torch.concat([self.arap_idx1, identity_idx], dim=1)
            self.arap_idx1 = torch.tile(self.arap_idx1, (1,3))
            self.arap_idx1[:,10:12] += 1
            self.arap_idx1[:,16:18] += 2
            self.arap_idx1 = self.arap_idx1.view(-1)

        if self.rot_cost:

            self.rot_idx0 = torch.tile(torch.arange(self.ED_num, device=cuda0).unsqueeze(1), (1,4)).flatten()
            self.rot_idx1 = torch.tile((torch.arange(self.ED_num, device=cuda0)*7).unsqueeze(1), (1,4)) + inc_idx[:, 0:4]
            self.rot_idx1 = self.rot_idx1.flatten()

        if self.corr_cost:

            matches, self.corr_new = self.matcher.match_features(allsurfels.renderImg, rgb, \
                allsurfels.projdata, points, valid, ID)

            corr_match_num = len(matches)
            self.corr_cost_num = corr_match_num * 3

            if corr_match_num > 0:
                self.corr_surfel_knn_indexs = allsurfels.surfel_knn_indexs[matches]
                self.corr_surfel_knn_weights = allsurfels.surfel_knn_weights[matches]
                self.corr_anchor_ednodes = allsurfels.ED_points[self.corr_surfel_knn_indexs]
                self.corr_d_surfels = allsurfels.points[matches].unsqueeze(1) - self.corr_anchor_ednodes
                self.skew_corr_v = torch.inner(self.corr_d_surfels, self.vec_to_skew_mat)

                self.corr_idx0 = torch.tile(torch.arange(self.corr_cost_num, device=cuda0).unsqueeze(1), (1,5)).view(-1,3*5)
                self.corr_idx0 = torch.tile(self.corr_idx0,(1,n_neighbors)).flatten()
                self.corr_idx1 = torch.tile(self.corr_surfel_knn_indexs.view(-1,1)*7, (1,5)) + inc_idx
                self.corr_idx1 = torch.tile(self.corr_idx1, (1,3))
                self.corr_idx1[:,9] += 1
                self.corr_idx1[:,14] += 2
                self.corr_idx1 = self.corr_idx1.flatten()
        
        # last_loss = 0
        for i in range(num_Iter):

            if debug_mode:
                start_ = timeit.default_timer()
            
            if self.data_cost or self.depth_cost:

                proj_coords, match_surfels, trans_points, Jacobian_elements, y, x = \
                    self.update_correspondence(allsurfels, valid, beta, \
                    est_grad=True, skew_v=skew_v)

                # corr_map = torch.zeros(PIXEL_NUM)
                # corr_map[proj_coords] = 255
                # cv2.imwrite(os.path.join(output_folder, "debug","{}_{}.jpg".format(ID,i)), corr_map.view(HEIGHT,WIDTH).detach().cpu().numpy())

                self.match_new_points = points[proj_coords]
                self.match_new_norms = norms[proj_coords]

                self.match_surfel_knn_idxs = allsurfels.surfel_knn_indexs[match_surfels]

                self.match_num = len(trans_points)  
                self.data_idx0 = torch.arange(self.match_num, device=cuda0).unsqueeze(1).expand(self.match_num,n_neighbors*7).flatten()
                self.data_idx1 = torch.tile(self.match_surfel_knn_idxs.view(-1,1)*7, (1,7)) + DeformLoss.increase_id
                self.data_idx1 = self.data_idx1.flatten()

                loss, jtj, jtl = self.prepareCostTerm(beta, points[:,2], \
                    trans_points=trans_points, y=y, x=x, \
                    est_grad=True, Jacobian_elements=Jacobian_elements)
            else:
                loss, jtj, jtl = self.prepareCostTerm(beta, points[:,2], \
                    est_grad=True)
            jtj[jtj_idx,jtj_idx] += u

            try:
                delta = linear_solver(jtj, jtl).view(-1,7)
            except RuntimeError as e:
                print("\t\tSolver failed: Ill-posed system!", e)
                break
            
            beta += delta

            if self.data_cost or self.depth_cost:
                try:
                    proj_coords, match_surfels, trans_points, y, x = self.update_correspondence(allsurfels, valid, beta)
                except: # Reject the step.
                    u *= v
                    beta = copy.deepcopy(best_beta)
                self.match_new_points = points[proj_coords]
                self.match_new_norms = norms[proj_coords]
                loss = self.prepareCostTerm(beta, points[:,2], trans_points=trans_points, y=y, x=x)
            else:
                loss = self.prepareCostTerm(beta, points[:,2])

            loss = torch.sum(torch.pow(loss,2))
            
            if loss < minimal_loss: # Accept the step
                minimal_loss = loss
                u /= v

                best_beta = copy.deepcopy(beta)
            else: # Reject the step.
                u *= v
                beta = copy.deepcopy(best_beta)

            if debug_mode:
                stop_ = timeit.default_timer()
                print("***Debug*** Iter: {}; time: {}s; loss: {}".format(i,np.round(stop_-start_,3),loss) )

            # self.residual_memory.append(loss)

        stop = timeit.default_timer()
        print("Exceed maximum iter and end optimization. Loss: {}, time: {}s".format(minimal_loss, stop - start))

        return beta

    # def get_result(self):
    #     return self.residual_memory[-1]