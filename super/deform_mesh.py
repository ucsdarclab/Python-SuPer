import time, copy, torch, numpy as np, torch.nn as nn
from torch_geometric.data import Data
from super.loss import *
from super.nodes import Surfels

from depth.monodepth2.layers import SSIM

from utils.utils import merge_transformation, find_edge_region, find_knn

class GraphFit(nn.Module):
    def __init__(self, opt):
        super(GraphFit, self).__init__()

        self.opt = opt
        self.valid_margin = 1
        self.optim = opt.optimizer
        self.Niter= opt.num_optimize_iterations

    def infer_flow(self, models, source_img, target_img):
        flow = models.optical_flow(source_img, target_img) # x, y
        if isinstance(flow, list):
            flow = flow[-1]
        return flow.detach()

    def get_losses(self,
                   deform_verts, 
                   inputs, 
                   trg, 
                   src,
                   new_verts, 
                   models, 
                   flow=None,
                   renderImg=None,
                   init_iter=False):
        '''
        deform_verts: the transformation variables to estimate
        
        -- target --
        inputs, trg: new input data
        
        -- source --
        src: source data
        new_verts: source data updated with previously estimated deform_verts
        '''
        loss = 0.
        losses = {}

        ''' Regularization terms '''

        # Face loss.
        if self.opt.mesh_face:
            new_triangles_areas = torch.cross(new_verts[src.ED_nodes.triangles[1]] - new_verts[src.ED_nodes.triangles[0]],
                                    new_verts[src.ED_nodes.triangles[2]] - new_verts[src.ED_nodes.triangles[0]],
                                    dim=1)
            new_triangles_areas = 0.5 * torch.sqrt((new_triangles_areas**2).sum(1) + 1e-13)
            
            face_losses = self.opt.mesh_face_weight * (new_triangles_areas - src.ED_nodes.triangles_areas)**2
            face_losses = face_losses.sum()
            loss += face_losses
            losses["face_losses"] = face_losses

        # As-rigid-as-possible regularizer.
        if self.opt.mesh_arap:
            arap_loss = self.opt.mesh_arap_weight * \
                ARAPLoss.autograd_forward(src.ED_nodes, deform_verts[:-1, :])
            loss += arap_loss
            losses["arap_loss"] = arap_loss

        if self.opt.mesh_rot:
            rot_loss = RotLoss.autograd_forward(deform_verts)

            rot_loss = self.opt.mesh_rot_weight * rot_loss
            loss += rot_loss
            losses["rot_loss"] = rot_loss

        """ registration losses. """
        
        seg_icp = False
        if hasattr(self.opt, 'sf_hard_seg_point_plane') or hasattr(self.opt, 'sf_soft_seg_point_plane'):
            seg_icp = self.opt.sf_hard_seg_point_plane or self.opt.sf_soft_seg_point_plane
        if self.opt.sf_point_plane or seg_icp:
            if seg_icp:
                point_plane_loss =  DataLoss.autograd_forward(self.opt, 
                                                              inputs, 
                                                              self.new_data, 
                                                              trg,
                                                              src_seg=self.sf_seg, 
                                                              src_seg_conf=self.sf_seg_conf, 
                                                              soft_seg=self.opt.sf_soft_seg_point_plane)
            else:
                point_plane_loss = DataLoss.autograd_forward(self.opt, 
                                                             inputs, 
                                                             self.new_data, 
                                                             trg,
                                                             max=2e-5 if self.opt.depth_model == 'raft_stereo' else None)

            point_plane_loss = self.opt.sf_point_plane_weight * point_plane_loss
            loss += point_plane_loss
            losses["point_plane_loss"] = point_plane_loss

        if self.opt.sf_corr:
            loss_corr = self.opt.sf_corr_weight * \
                DataLoss.autograd_forward(self.opt, 
                                          inputs, 
                                          self.new_data, 
                                          trg, 
                                          flow=flow, 
                                          loss_type=self.opt.sf_corr_loss_type)
            loss += loss_corr
            losses["corr_loss"] = loss_corr

        """ Appearance loss. """
        if hasattr(self.opt, "render_loss"):
            if self.opt.render_loss:
                diff_render_loss = SSIM(kernel=11).cuda()(renderImg, inputs[("color", 0)]).mean(1, True)**2

                max_pool = nn.MaxPool2d(11, stride=1, padding=5)
                valid_render_loss = max_pool(-torch.min(renderImg, dim=1, keepdim=True).values) < 0
                diff_render_loss = diff_render_loss[valid_render_loss]
                diff_render_loss = diff_render_loss[diff_render_loss < 0.1]
                diff_render_loss = self.opt.render_loss_weight * diff_render_loss.sum()
                loss += diff_render_loss
                losses["render_loss"] = diff_render_loss

        ''' Semantic-Super losses '''
        if hasattr(self.opt, "sf_bn_morph"):
            if self.opt.sf_bn_morph:
                new_sf_y, new_sf_x, _, _ = pcd2depth(inputs, self.new_data.points, round_coords=False)
                sf_grid = torch.stack([new_sf_x, new_sf_y], dim=1)
                
                scaled_sf_grid = torch.stack([new_sf_x / self.opt.width * 2 - 1, 
                                              new_sf_y / self.opt.height * 2 - 1
                                             ],
                                             dim=1
                                            ).type(torch.float64)
                new_sf_seg = F.grid_sample(inputs[("seg_conf", 0)], 
                                           scaled_sf_grid[None, :, None, :],
                                          )[0, :, :, 0].argmax(0)
                    
                sf_morph_val = (~ (new_sf_seg == self.sf_seg)) & \
                               (scaled_sf_grid[:,0] > -1) & (scaled_sf_grid[:,0] < 1) & \
                               (scaled_sf_grid[:,1] > -1) & (scaled_sf_grid[:,1] < 1)

                if init_iter:
                    with torch.no_grad():
                        kernels = [3, 3, 3]
                        self.edge_pts = []
                        for class_id in range(self.opt.num_classes):
                            seg_grad_bin = find_edge_region(inputs[("seg", 0)], 
                                                            num_classes=self.opt.num_classes,
                                                            class_list=[class_id],
                                                            kernel=kernels[class_id])
                            edge_y, edge_x = seg_grad_bin[0,0].nonzero(as_tuple=True)
                            valid_edge = (edge_x >= self.valid_margin) & \
                                         (edge_x < self.opt.width-1-self.valid_margin) & \
                                         (edge_y >= self.valid_margin) & \
                                         (edge_y < self.opt.height-1-self.valid_margin)
                            self.edge_pts.append(torch.stack([edge_x[valid_edge], 
                                                              edge_y[valid_edge]], 
                                                             dim=1
                                                            ).type(torch.float64)
                                                )

                sf_bn_morph_loss = []
                img = inputs[("color", 0)][0].permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
                for class_id in range(self.opt.num_classes):
                    class_mask = (self.sf_seg == class_id) & sf_morph_val
                    if torch.any(class_mask) and len(self.edge_pts[class_id]) > 0:
                        knn_dists, knn_edge_ids = find_knn(sf_grid[class_mask], 
                                                           self.edge_pts[class_id], 
                                                           k=2)

                        # Filter out projected points that are closer to the image edge than the semantic edge.
                        dists_to_edge = torch.minimum(torch.minimum(sf_grid[class_mask].min(1).values,
                                                                    self.opt.width - sf_grid[class_mask, 0]),
                                                      self.opt.height - sf_grid[class_mask, 1]
                                                     )
                        valid_match = ~ torch.any(knn_dists > dists_to_edge[:, None], dim=1)

                        sf_morph_gt = self.edge_pts[class_id][knn_edge_ids]
                        sf_bn_morph_loss_temp = ((sf_morph_gt[valid_match] - \
                                                  sf_grid[class_mask][valid_match][:, None, :]
                                                 )**2
                                                ).sum(2).mean(1)
                        sf_bn_morph_loss.append(
                                                sf_bn_morph_loss_temp[sf_bn_morph_loss_temp > 15]
                                               )
                
                if len(sf_bn_morph_loss) > 0:
                    sf_bn_morph_loss = torch.concat(sf_bn_morph_loss)
                    sf_bn_morph_loss = self.opt.sf_bn_morph_weight * \
                                    sf_bn_morph_loss.mean()
                    loss += sf_bn_morph_loss
                    losses["sf_bn_morph_loss"] = sf_bn_morph_loss

        return loss, losses

    def deform_source(self, src, deform_verts):
        sf_knn_indices = src.knn_indices[src.isStable]              # (N,4)     4 NNs per surfel
        sf_knn_w = src.knn_w[src.isStable]                          # (N,4)
        sf_knn = src.ED_nodes.points[sf_knn_indices]                # (N,4,3)   KNN coordinates for each surfel
        sf_diff = src.points[src.isStable].unsqueeze(1) - sf_knn    # (N,4,3)   Dists from 4 NNs.
        skew_v = get_skew(sf_diff)                                  # (N,4,3,3) 

        # Deform J ED nodes (J,3) with J deformations (J,3). Apply both trans and rot.
        new_verts = src.ED_nodes.points + deform_verts[:-1,4:]
        new_norms, _ = transformQuatT(src.ED_nodes.norms, deform_verts[:-1,0:4])

        # Deform N surfels (N,3), each with 4 ED nodes (N,4,3).
        new_sf, _ = Trans_points(sf_diff, sf_knn, deform_verts[sf_knn_indices], sf_knn_w, skew_v=skew_v)                           
        new_sfnorms = torch.sum(new_norms[sf_knn_indices] * sf_knn_w.unsqueeze(-1), dim=1)                                              

        # Deform J ED nodes (J,3) with global transformation T_g (1,4)
        new_verts, _ = transformQuatT(new_verts, deform_verts[-1:, 0:4]) 
        new_verts = new_verts + deform_verts[-1:, 4:]
        # new_norms, _ = transformQuatT(new_norms, deform_verts[-1:, 0:4])
        # new_norms = F.normalize(new_norms, dim=-1)
        
        # Deform N surfels (N,3) with global transformation T_g (1,4)
        new_sf, _ = transformQuatT(new_sf, deform_verts[-1:, 0:4]) 
        new_sf = new_sf + deform_verts[-1:, 4:]
        new_sfnorms, _ = transformQuatT(new_sfnorms, deform_verts[-1:, 0:4])
        new_sfnorms = F.normalize(new_sfnorms, dim=-1)

        new_data = Data(points=new_sf, 
                        norms=new_sfnorms, 
                        colors=src.colors[src.isStable]
                       )

        return new_verts, new_data

    def forward(self, inputs: dict, src: Surfels, trg: Data, models):
        ''' 
        Call stack:
            super.fusion(models, inputs, sfdata) -> 
            GraphFit.forward(inputs, super.sf, sfdata, models)

        Input:
            - inputs: dict      item from dataloader.
            - src: Surfels      source data
            - trg: Data         target data (points,norms,colors,radii,etc., derived from `inputs`)
            - models: models from InitNets()

        Returns: deform_verts
        '''
        if self.opt.deform_udpate_method == 'super_edg':
            return self.deform_superedg(inputs, src, trg, models)
        else: 
            raise NotImplementedError(f'opt[\'deformation_update_method\'] {method} is not specified correctly')

    def deform_superedg(self, inputs: dict, src: Surfels, trg: Data, models):
        ''' Deform surfels and ED nodes using SuPer eq. (10) and (11).
        Input:
            - inputs: dict      item from dataloader.
            - src: Surfels      source data
            - trg: Data         target data (points,norms,colors,radii,etc., derived from `inputs`)
            - models: models from InitNets()
        Output:
            deform_verts
        '''

        if hasattr(src, 'seg'):
            self.sf_seg = src.seg[src.isStable]             # (N)
            self.sf_seg_conf = src.seg_conf[src.isStable]   # (N,2)

        """ Optimization loop. """
        # Init deformation (J+1, 7), J = src.ED_nodes.num = number of ED nodes
        deform_verts = torch.tensor(  
            np.repeat(np.array([[1.,0.,0.,0.,0.,0.,0.]]), src.ED_nodes.num+1, axis=0),
            dtype=torch.float64, device=torch.device('cuda'), requires_grad=True)

        if self.optim == "SGD":
            optimizer = torch.optim.SGD([deform_verts], lr=self.opt.learning_rate, momentum=0.9)
        elif self.optim == "Adam":
            optimizer = torch.optim.Adam([deform_verts], lr=self.opt.learning_rate)
        elif self.optim == "LM":
            u, v = 1.5, 5.
            minimal_loss = 1e10
            best_deform_verts = copy.deepcopy(deform_verts)

        start_time = time.time()
        for i in range(self.Niter):

            if self.optim in ["SGD", "Adam"]: optimizer.zero_grad()

            new_verts, self.new_data = self.deform_source(src, deform_verts)

            get_render_image = False
            if hasattr(self.opt, "render_loss"):
                if hasattr(self.opt, "render_loss"):
                    get_render_image = True
            if self.opt.sf_corr and self.opt.sf_corr_match_renderimg:
                get_render_image = True
            if True:
                renderImg = models.renderer(inputs, 
                                            self.new_data, 
                                            rad=self.opt.renderer_rad
                                           ).permute(2,0,1).unsqueeze(0)
            else:
                renderImg = None

            if self.opt.sf_corr:
                if hasattr(models, "optical_flow"):
                    if self.opt.sf_corr_match_renderimg:
                        self.flow = self.infer_flow(models, renderImg, inputs[("color", 0)])
                    elif i == 0:
                        self.flow = self.infer_flow(models, src.rgb, inputs[("color", 0)])
                else:
                    assert False
            else:
                self.flow = None
                    
            loss, losses = self.get_losses(deform_verts, 
                                           inputs, 
                                           trg, 
                                           src, 
                                           new_verts,  
                                           models,
                                           flow=self.flow, 
                                           renderImg=renderImg,
                                           init_iter=i==0)
                
            # Optimization step.
            if self.optim in ["SGD", "Adam"]:
                loss.backward()
                deform_verts.grad[-1] = deform_verts.grad[-1] / src.ED_nodes.num
                optimizer.step()
            
            elif self.optim == "LM":
                jtj = torch.zeros((src.ED_nodes.param_num+7, src.ED_nodes.param_num+7), layout=torch.sparse_coo, dtype=fl32_).cuda()
                jtl = torch.zeros((src.ED_nodes.param_num+7, 1), dtype=fl32_).cuda()
                for _loss_ in loss:
                    _loss_ = torch.abs(_loss_).sum()
                    _loss_.backward(retain_graph=True)
                    j = deform_verts.grad.reshape(1, -1)
                    jtj = jtj + torch.sparse.mm(j.T.to_sparse(), j.to_sparse())
                    jtl = jtl + j.T * _loss_
                jtj = jtj.to_dense()
                delta = torch.matmul(
                            torch.inverse(jtj + u * torch.eye(jtj.size(0)).cuda()),
                            jtl).reshape(deform_verts.size())
                
                delta = torch.matmul(
                            torch.inverse(jtj + u * torch.eye(jtj.size(0)).cuda()),
                            jtl).reshape(deform_verts.size())
                deform_verts = deform_verts + delta
                
                with torch.no_grad():
                    loss, _, _, _ = self.get_losses(deform_verts, 
                                                    inputs, 
                                                    trg, 
                                                    src, 
                                                    new_verts, 
                                                    nets)
                    loss = (torch.cat(loss)**2).sum()

                deform_verts = deform_verts.detach().cpu().numpy()
                if loss < minimal_loss: # Accept the step.
                    minimal_loss = loss
                    u /= v
                    best_deform_verts = deform_verts
                    deform_verts = torch.tensor(deform_verts, dtype=fl64_, 
                                            device=torch.device('cuda'), requires_grad=True)

                else: # Reject the step.
                    u *= v
                    deform_verts = torch.tensor(best_deform_verts, dtype=fl64_, 
                                                device=torch.device('cuda'), requires_grad=True)

            torch.cuda.empty_cache()

        total_time = time.time() - start_time

        if (src.time + 1) % self.opt.save_sample_freq == 0:
            src.summary_writer.add_scalar(f"optimization_record/optim_time_per_frame", total_time, trg.time)
            for key in losses.keys(): 
                src.summary_writer.add_scalar(f"optimization_record/{key}", losses[key], trg.time)

        return deform_verts