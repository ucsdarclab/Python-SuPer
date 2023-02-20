import numpy as np
import cv2
import copy
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch_geometric.data import Data

from super.loss import *

from utils.utils import *

from depth.monodepth2.layers import SSIM, AdaSSIM, compute_reprojection_loss, get_smooth_loss

class GraphFit(nn.Module):
    def __init__(self, opt):
        super(GraphFit, self).__init__()

        self.opt = opt

        # if self.opt.use_edge_ssim_hints:
        #     self.ssim = AdaSSIM().cuda()

        self.reduction="sum"
        self.valid_margin = 1
        self.optim = "SGD"
        self.Niter=10

    def get_losses(self, i, deform_verts, inputs, trg, src, src_graph, new_verts, src_edge_index, models, init_iter=False):
        square_loss = self.optim in ["SGD", "Adam"]

        boundary_edge_type = None
        boundary_face_type = None
        # if self.opt.use_edge_ssim_hints:
        #     if self.opt.mesh_edge:
        #         boundary_edge_type = src_graph.boundary_edge_type
        #     if self.opt.mesh_face:
        #         boundary_face_type = src_graph.boundary_face_type

        if self.opt.render_loss or self.opt.mesh_edge or self.opt.mesh_face:
            renderImg = models.renderer(inputs, 
                                            self.new_data, 
                                            rad=self.opt.renderer_rad
                                        ).permute(2,0,1).unsqueeze(0)

        losses = {}
        if square_loss:
            loss_mesh = 0.0
            loss_surfels = 0.0
            render_loss = 0.0
            feature_loss = 0.0
        else:
            loss_mesh = []
            loss_surfels = []
            render_loss = []

        """
        Mesh losses.
        """

        # if (self.opt.method == 'semantic-super') and (self.opt.mesh_edge or self.opt.mesh_face) and self.opt.use_edge_ssim_hints:
        #     if self.opt.mesh_edge:
        #         src_boundary_edge_index = src_edge_index.permute(1, 0)[src_graph.isBoundary]
        #     if self.opt.mesh_face:
        #         src_boundary_face_index = src_graph.triangles.permute(1, 0)[src_graph.isBoundaryFace]
            
        #     if i == self.Niter - 1:
        #         # Find the init edge end points project location in the image plane.
        #         init_ed_y, init_ed_x, _, _ = pcd2depth(inputs, src_graph.points, round_coords=False)

        #         # Find the updated edge end points project location in the image plane.
        #         current_ed_y, current_ed_x, _, _ = pcd2depth(inputs, new_verts, round_coords=False)

        #         if self.opt.mesh_edge:
        #             moving_boundary_edge = []
        #             margin = 10
        #             for edge_id1, edge_id2 in src_boundary_edge_index:
        #                 init_ct_x = 0.5 * (init_ed_x[edge_id1] + init_ed_x[edge_id2])
        #                 init_ct_y = 0.5 * (init_ed_y[edge_id1] + init_ed_y[edge_id2])
        #                 current_ct_x = 0.5 * (current_ed_x[edge_id1] + current_ed_x[edge_id2])
        #                 current_ct_y = 0.5 * (current_ed_y[edge_id1] + current_ed_y[edge_id2])
                        
        #                 win_h = torch.abs(init_ed_y[edge_id1] - init_ed_y[edge_id2]) + margin
        #                 win_w = torch.abs(init_ed_x[edge_id1] - init_ed_x[edge_id2]) + margin
        #                 x1, x2 = int(init_ct_x - win_w / 2), int(init_ct_x + win_w / 2)
        #                 y1, y2 = int(init_ct_y - win_h / 2), int(init_ct_y + win_h / 2)
        #                 prev_patch = F.pad(inputs[("prev_color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), 
        #                                                                 min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
        #                                 (min(max(0, -x1), win_w), min(max(0, x2-self.opt.width), win_w), 
        #                                     min(max(0, -y1), win_h), min(max(0, y2-self.opt.height), win_h))
        #                                 ) # Patch extracted from the previous frame with init projection.
        #                 cr_patch = F.pad(inputs[("color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), 
        #                                                         min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
        #                                 (min(max(0, -x1), win_w), min(max(0, x2-self.opt.width), win_w), 
        #                                 min(max(0, -y1), win_h), min(max(0, y2-self.opt.height), win_h))
        #                                 ) # Patch extracted from the current frame with init projection.

        #                 win_h = torch.abs(current_ed_y[edge_id1] - current_ed_y[edge_id2]) + margin
        #                 win_w = torch.abs(current_ed_x[edge_id1] - current_ed_x[edge_id2]) + margin
        #                 x1, x2 = int(current_ct_x - win_w / 2), int(current_ct_x + win_w / 2)
        #                 y1, y2 = int(current_ct_y - win_h / 2), int(current_ct_y + win_h / 2)
        #                 moving_patch = F.pad(inputs[("color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), 
        #                                                             min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
        #                                     (min(max(0, -x1), win_w), min(max(0, x2-self.opt.width), win_w), 
        #                                     min(max(0, -y1), win_h), min(max(0, y2-self.opt.height), win_h))
        #                                     ) # Patches extracted from the current frame with new projection.
        #                 moving_boundary_edge.append(torch.any(self.ssim(prev_patch, cr_patch) < \
        #                                             self.ssim(F.interpolate(prev_patch[None, ...], moving_patch.size()[1:3], mode='bilinear')[0], moving_patch)))
                    
        #             moving_boundary_edge = torch.tensor(moving_boundary_edge).cuda()

        #             new_edges_lens = torch.norm(new_verts[src_edge_index[0]][src_graph.isBoundary] - 
        #                                         new_verts[src_edge_index[1]][src_graph.isBoundary], dim=1)
        #             moving_boundary_edge &= (new_edges_lens > 0.2)

        #             moving_boundary_edge = moving_boundary_edge.type(fl32_)
        #             boundary_edge_type = torch.cat([boundary_edge_type[:, 1:], moving_boundary_edge[:, None]], dim=1)

        #         if self.opt.mesh_face:
        #             moving_boundary_face = []
        #             margin = 10
        #             for face_id1, face_id2, face_id3 in src_boundary_face_index:
        #                 init_ct_x = 1./3. * (init_ed_x[face_id1] + init_ed_x[face_id2] + init_ed_x[face_id3])
        #                 init_ct_y = 1./3. * (init_ed_y[face_id1] + init_ed_y[face_id2] + init_ed_y[face_id3])
        #                 current_ct_x = 1./3. * (current_ed_x[face_id1] + current_ed_x[face_id2] + current_ed_x[face_id3])
        #                 current_ct_y = 1./3. * (current_ed_y[face_id1] + current_ed_y[face_id2] + current_ed_y[face_id3])
                        
        #                 win_h = int(torch.max(torch.tensor([init_ed_y[face_id1], init_ed_y[face_id2], init_ed_y[face_id3]])) - \
        #                         torch.min(torch.tensor([init_ed_y[face_id1], init_ed_y[face_id2], init_ed_y[face_id3]])) + margin)
        #                 win_w = int(torch.max(torch.tensor([init_ed_x[face_id1], init_ed_x[face_id2], init_ed_x[face_id3]])) - \
        #                         torch.min(torch.tensor([init_ed_x[face_id1], init_ed_x[face_id2], init_ed_x[face_id3]])) + margin)
        #                 x1 = int(init_ct_x - win_w / 2)
        #                 x2 = x1 + win_w
        #                 y1 = int(init_ct_y - win_h / 2)
        #                 y2 = y1 + win_h
        #                 prev_patch = F.pad(inputs[("prev_color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), 
        #                                                                 min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
        #                                 (min(max(0, -x1), win_w), min(max(0, x2-self.opt.width), win_w), 
        #                                     min(max(0, -y1), win_h), min(max(0, y2-self.opt.height), win_h))
        #                                 ) # Patch extracted from the previous frame with init projection.
        #                 cr_patch = F.pad(inputs[("color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), 
        #                                                         min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
        #                                 (min(max(0, -x1), win_w), min(max(0, x2-self.opt.width), win_w), 
        #                                 min(max(0, -y1), win_h), min(max(0, y2-self.opt.height), win_h))
        #                                 ) # Patch extracted from the current frame with init projection.

        #                 cr_win_h = int(torch.max(torch.tensor([current_ed_y[face_id1], current_ed_y[face_id2], current_ed_y[face_id3]])) - \
        #                         torch.min(torch.tensor([current_ed_y[face_id1], current_ed_y[face_id2], current_ed_y[face_id3]])) + margin)
        #                 cr_win_w = int(torch.max(torch.tensor([current_ed_x[face_id1], current_ed_x[face_id2], current_ed_x[face_id3]])) - \
        #                         torch.min(torch.tensor([current_ed_x[face_id1], current_ed_x[face_id2], current_ed_x[face_id3]])) + margin)
        #                 x1 = int(init_ct_x - cr_win_w / 2)
        #                 x2 = x1 + cr_win_w
        #                 y1 = int(init_ct_y - cr_win_h / 2)
        #                 y2 = y1 + cr_win_h
        #                 cr_x1 = int(current_ct_x - cr_win_w / 2)
        #                 cr_x2 = cr_x1 + cr_win_w
        #                 cr_y1 = int(current_ct_y - cr_win_h / 2)
        #                 cr_y2 = cr_y1 + cr_win_h
        #                 prev_patch2 = F.pad(inputs[("prev_color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), 
        #                                                                  min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
        #                                     (min(max(0, -x1), cr_win_w), min(max(0, x2-self.opt.width), cr_win_w), 
        #                                     min(max(0, -y1), cr_win_h), min(max(0, y2-self.opt.height), cr_win_h))
        #                                     )
        #                 moving_patch = F.pad(inputs[("color", 0, 0)][0, :, min(max(cr_y1,0),self.opt.height): min(max(cr_y2,0),self.opt.height), 
        #                                                              min(max(cr_x1,0),self.opt.width): min(max(cr_x2,0),self.opt.width)],
        #                                      (min(max(0, -cr_x1), cr_win_w), min(max(0, cr_x2-self.opt.width), cr_win_w), 
        #                                      min(max(0, -cr_y1), cr_win_h), min(max(0, cr_y2-self.opt.height), cr_win_h))
        #                                     ) # Patches extracted from the current frame with new projection.
                                            
        #                 moving_boundary_face.append(torch.any(self.ssim(prev_patch, cr_patch) < self.ssim(prev_patch2, moving_patch)))
                    
        #             moving_boundary_face = torch.tensor(moving_boundary_face).cuda()
                    
        #             new_triangles_areas = torch.cross(new_verts[src_graph.triangles[1]][src_graph.isBoundaryFace] - new_verts[src_graph.triangles[0]][src_graph.isBoundaryFace],
        #                             new_verts[src_graph.triangles[2]][src_graph.isBoundaryFace] - new_verts[src_graph.triangles[0]][src_graph.isBoundaryFace],
        #                             dim=1)
        #             new_triangles_areas = 0.5 * torch.sqrt((new_triangles_areas**2).sum(1) + 1e-13)
        #             # print(torch.unique(new_triangles_areas))
        #             moving_boundary_face |= (new_triangles_areas < 0.025) # 0.025, 0.01

        #             moving_boundary_face = moving_boundary_face.type(fl32_)
        #             boundary_face_type = torch.cat([boundary_face_type[:, 1:], moving_boundary_face[:, None]], dim=1)

        # # Edge loss.
        # if self.opt.mesh_edge:
        #     new_edges_lens = torch.norm(new_verts[src_edge_index[0]] - new_verts[src_edge_index[1]], dim=1)
        #     edge_losses = new_edges_lens - src_graph.edges_lens

        #     if square_loss:
        #         edge_losses = self.opt.mesh_edge_weight * edge_losses ** 2
        #     else:
        #         edge_losses = np.sqrt(self.opt.mesh_edge_weight) * edge_losses
            
        #     if self.opt.method == 'semantic-super' and self.opt.use_edge_ssim_hints:
        #         connected_edges = torch.ones_like(src_graph.isBoundary)
        #         connected_edges[src_graph.isBoundary] = src_graph.boundary_edge_type[:, -1] == 1.
                
        #         edge_losses = edge_losses[connected_edges]

        #     if square_loss:
        #         if self.reduction == "mean":
        #             edge_losses = edge_losses.mean()
        #         elif self.reduction == "sum":
        #             edge_losses = edge_losses.sum()
        #         else:
        #             assert False
        #         loss_mesh += edge_losses
        #         losses["edge_loss"] = edge_losses
        #     else:
        #         loss_mesh.append(edge_losses)
        #         if self.reduction == "mean":
        #             losses["edge_loss"] = (edge_losses**2).mean()
        #         elif self.reduction == "sum":
        #             losses["edge_loss"] = (edge_losses**2).sum()
        #         else:
        #             assert False
                

        if self.opt.mesh_face:
            new_triangles_areas = torch.cross(new_verts[src_graph.triangles[1]] - new_verts[src_graph.triangles[0]],
                                    new_verts[src_graph.triangles[2]] - new_verts[src_graph.triangles[0]],
                                    dim=1)
            new_triangles_areas = 0.5 * torch.sqrt((new_triangles_areas**2).sum(1) + 1e-13)
            
            face_losses = new_triangles_areas - src_graph.triangles_areas

            if square_loss:
                face_losses = self.opt.mesh_face_weight * face_losses ** 2
            else:
                face_losses = np.sqrt(self.opt.mesh_face_weight) * face_losses

            # if self.opt.method == 'semantic-super' and self.opt.use_edge_ssim_hints:
            #     connected_faces = torch.ones_like(src_graph.isBoundaryFace)
            #     connected_faces[src_graph.isBoundaryFace] = src_graph.boundary_face_type[:, -1] == 1.
                
            #     face_losses = face_losses[connected_faces]

            if square_loss:
                if self.reduction == "mean":
                    face_losses = face_losses.mean()
                elif self.reduction == "sum":
                    face_losses = face_losses.sum()
                else:
                    assert False
                loss_mesh += face_losses
                losses["face_losses"] = face_losses

            else:
                loss_mesh.append(face_losses)
                if self.reduction == "mean":
                    losses["face_losses"] = (face_losses**2).mean()
                elif self.reduction == "sum":
                    losses["face_losses"] = (face_losses**2).sum()
                else:
                    assert False

        # Regularization terms.
        if self.opt.mesh_arap:
            arap_loss = self.opt.mesh_arap_weight * \
                ARAPLoss.autograd_forward(src_graph, deform_verts[:-1, :], reduction=self.reduction)
            loss_mesh += arap_loss

            losses["arap_loss"] = arap_loss

        if self.opt.mesh_rot:
            rot_loss = RotLoss.autograd_forward(deform_verts, reduction=self.reduction, square_loss=square_loss)

            if square_loss:
                rot_loss = self.opt.mesh_rot_weight * rot_loss
                loss_mesh += rot_loss
                losses["rot_loss"] = rot_loss
            else:
                rot_loss = np.sqrt(self.opt.mesh_rot_weight) * rot_loss
                loss_mesh.append(rot_loss)
                if self.reduction == "mean":
                    losses["rot_loss"] = (rot_loss**2).mean()
                elif self.reduction == "sum":
                    losses["rot_loss"] = (rot_loss**2).sum()

        if self.opt.sf_bn_morph:
            # Project surfels to the image plane.
            # sf_y, sf_x, _, _ = pcd2depth(inputs, self.new_sf, round_coords=False)
            # ori_new_sf_grid = torch.stack([sf_x, sf_y], dim=1)
            # new_sf_grid = ori_new_sf_grid.clone()
            # new_sf_grid[...,0] = new_sf_grid[...,0] / self.opt.width * 2 - 1
            # new_sf_grid[...,1] = new_sf_grid[...,1] / self.opt.height * 2 - 1
            # new_sf_seg = F.grid_sample(inputs[("seg_conf", 0)], 
            #                              new_sf_grid[None, :, None, :].type(torch.float32),
            #                             )[0, :, :, 0].argmax(0)
            new_sf_seg = F.grid_sample(inputs[("seg_conf", 0)], 
                                         self.new_sf_grid[None, :, None, :],
                                        )[0, :, :, 0].argmax(0)
            # new_sf_grid[...,0] = new_sf_grid[...,0] / (self.opt.width-1) * 2 - 1
            # new_sf_grid[...,1] = new_sf_grid[...,1] / (self.opt.height-1) * 2 - 1
            # new_sf_seg = nn.functional.grid_sample(inputs[("seg_conf", 0)], 
            #                                          new_sf_grid[None, :, None, :],
            #                                          align_corners=True
            #                                         )[0, :, :, 0].argmax(0)
                
            sf_morph_val = (~(new_sf_seg==self.sf_seg)) & \
                            (self.new_sf_grid[...,0] >= -1) & (self.new_sf_grid[...,0] <= 1) & \
                            (self.new_sf_grid[...,1] >= -1) & (self.new_sf_grid[...,1] <= 1)
            sf_inside_morph_val = (new_sf_seg==self.sf_seg) & \
                                  (self.new_sf_grid[...,0] >= -1) & (self.new_sf_grid[...,0] <= 1) & \
                                  (self.new_sf_grid[...,1] >= -1) & (self.new_sf_grid[...,1] <= 1)

            sorted_sf_seg, sf_sort_indices = torch.sort(self.sf_seg)
            sorted_ori_new_sf_grid = self.ori_new_sf_grid[sf_sort_indices]
            sorted_sf_morph_val = sf_morph_val[sf_sort_indices]
            sorted_sf_inside_morph_val = sf_inside_morph_val[sf_sort_indices]

            sf_morph_gt = []
            if init_iter:
                with torch.no_grad():
                    kernels = [3, 3, 3]
                    self.edge_pts = []
                    for class_id in range(self.opt.num_classes):
                        seg_grad_bin = find_edge_region(inputs[("seg", 0)], 
                                                        num_classes=self.opt.num_classes,
                                                        class_list=[class_id],
                                                        kernel=kernels[class_id]) # erode_foreground=True
                        edge_y, edge_x = seg_grad_bin[0,0].nonzero(as_tuple=True)
                        self.edge_pts.append(torch.stack([edge_x, edge_y], dim=1).type(torch.float64))

            for class_id in range(self.opt.num_classes):
                _, knn_edge_ids = find_knn(sorted_ori_new_sf_grid[sorted_sf_seg == class_id], 
                                                self.edge_pts[class_id], k=1)
                sf_morph_gt.append(self.edge_pts[class_id][knn_edge_ids])
            sf_morph_gt_seg = torch.cat([class_id * torch.ones(len(_sf_morph_gt_), dtype=torch.long) \
                                          for class_id, _sf_morph_gt_ in enumerate(sf_morph_gt)]
                                         )
            sf_morph_gt = torch.cat(sf_morph_gt, dim=0).mean(1)

            sorted_sf_morph_val &= (sf_morph_gt[:,0] >= self.valid_margin) & \
                                    (sf_morph_gt[:,0] < self.opt.width-1-self.valid_margin) & \
                                    (sf_morph_gt[:,1] >= self.valid_margin) & \
                                    (sf_morph_gt[:,1] < self.opt.height-1-self.valid_margin)
            sorted_sf_inside_morph_val &= (sf_morph_gt[:,0] >= self.valid_margin) & \
                                            (sf_morph_gt[:,0] < self.opt.width-1-self.valid_margin) & \
                                            (sf_morph_gt[:,1] >= self.valid_margin) & \
                                            (sf_morph_gt[:,1] < self.opt.height-1-self.valid_margin)

            # # if i == 0:
            # sample_img = torch_to_numpy(255 * inputs[("color", 0, 0)][0].permute(1,2,0))
            # if self.opt.data == "superv1":
            #     sf_morph_gt_win = 40
            # elif self.opt.data == "superv2":
            #     sf_morph_gt_win = 20
            # class_th = int(0.4 * 2 * sf_morph_gt_win**2)
            # sf_morph_val_ids = sorted_sf_morph_val.nonzero(as_tuple=True)[0]
            # self.sorted_sf_morph_val = torch.ones_like(sorted_sf_morph_val)
            # for sf_morph_val_id in sf_morph_val_ids:
            #     morph_gt_x, morph_gt_y =  sf_morph_gt[sf_morph_val_id]
            #     morph_gt_y1, morph_gt_y2 = int(morph_gt_y-sf_morph_gt_win), int(morph_gt_y+sf_morph_gt_win)
            #     morph_gt_x1, morph_gt_x2 = int(morph_gt_x-sf_morph_gt_win), int(morph_gt_x+sf_morph_gt_win)
                
            #     if ("full_depth", 0) in inputs:
            #         morph_gt_depth_win = inputs[("full_depth", 0)][0, 0, morph_gt_y1:morph_gt_y2, morph_gt_x1:morph_gt_x2] / 0.1 * 5
            #     else:
            #         morph_gt_depth_win = inputs[("depth", 0)][0, 0, morph_gt_y1:morph_gt_y2, morph_gt_x1:morph_gt_x2] / 0.1 * 5
            #     morph_gt_depth_win_val = ~ torch.isnan(morph_gt_depth_win)
                
            #     morph_gt_seg_win = inputs[("seg", 0)][0, 0, morph_gt_y1:morph_gt_y2, morph_gt_x1:morph_gt_x2]
            #     class_point_numbers = []
            #     class_depths = []
            #     for class_id in range(self.opt.num_classes):
            #         morph_gt_win_class_val = (morph_gt_seg_win == class_id) & morph_gt_depth_win_val
            #         class_point_numbers.append(torch.count_nonzero(morph_gt_win_class_val))
            #         if class_point_numbers[-1] == 0:
            #             class_depths.append(1e8)
            #         else:
            #             class_depths.append(torch.median(morph_gt_depth_win[morph_gt_win_class_val]))
            #     class_point_numbers = torch.tensor(class_point_numbers)
            #     class_depths = torch.tensor(class_depths)
            #     max_class_id = torch.argmin(class_depths)
                
            #     if self.opt.data == "superv1":
            #         delta_depth_th = 0.5
            #     elif self.opt.data == "superv2":
            #         delta_depth_th = 5
            #     if class_point_numbers[max_class_id] < class_th or \
            #     class_point_numbers[sf_morph_gt_seg[sf_morph_val_id]] < class_th or \
            #     torch.abs(class_depths[max_class_id] - class_depths[sf_morph_gt_seg[sf_morph_val_id]]) > delta_depth_th:
            #         sorted_sf_morph_val[sf_morph_val_id] = False
            #     elif self.opt.data == "superv1":
            #         # import random
            #         # if random.random() < 0.1:
            #         sample_img = cv2.rectangle(cv2.UMat(sample_img), 
            #                                 (int(morph_gt_x-3), int(morph_gt_y-3)), 
            #                                 (int(morph_gt_x-3), int(morph_gt_y-3)), 
            #                                 (0, 0, 255), -1)
                    
            #         morph_pred_x, morph_pred_y = sorted_ori_new_sf_grid[sf_morph_val_id]
            #         sample_img = cv2.circle(sample_img, 
            #                                 (int(morph_pred_x), int(morph_pred_y)), 
            #                                 2, (0, 255, 0), -1)

            #         sample_img = cv2.line(sample_img, 
            #                             (int(morph_gt_x), int(morph_gt_y)),
            #                             (int(morph_pred_x), int(morph_pred_y)), 
            #                             (0, 255, 0), 1)
            #         out_dir = "morph_sample"
            #         cv2.imwrite(os.path.join(out_dir, f"{inputs['filename'][0]}.png"), sample_img)
                    

            # sorted_sf_morph_val &= self.sorted_sf_morph_val

            # if torch.any(sf_morph_val):
            if torch.any(sorted_sf_morph_val):
                sf_morph_gt_scaled = sf_morph_gt.clone()
                # sorted_sf_morph_val &= inputs[("seg_valid", 0)][0, 0][sf_morph_gt_scaled[:, 1].type(long_), sf_morph_gt_scaled[:, 0].type(long_)]
                sf_morph_gt_scaled[:, 0] /= self.opt.width
                sf_morph_gt_scaled[:, 1] /= self.opt.height

                sf_morph_pred_scaled = sorted_ori_new_sf_grid.clone()
                sf_morph_pred_scaled[:, 0] /= self.opt.width
                sf_morph_pred_scaled[:, 1] /= self.opt.height

                if square_loss:
                    sf_bn_morph_loss = torch_sq_distance(sf_morph_pred_scaled[sorted_sf_morph_val], 
                                                         sf_morph_gt_scaled[sorted_sf_morph_val])
                    # with torch.no_grad():
                    #     morph_dists = torch.norm(sf_morph_pred_scaled[sorted_sf_morph_val] - \
                    #                             sf_morph_gt_scaled[sorted_sf_morph_val], dim=1, keepdim=True)
                    #     morph_targets = sf_morph_pred_scaled[sorted_sf_morph_val] + \
                    #                     (1. + src.dist2edge[src.isStable][sf_sort_indices][sorted_sf_morph_val][:, None] / morph_dists) * \
                    #                         (sf_morph_gt_scaled[sorted_sf_morph_val] - sf_morph_pred_scaled[sorted_sf_morph_val])
                    # sf_bn_morph_loss = torch_sq_distance(sf_morph_pred_scaled[sorted_sf_morph_val], morph_targets)
                    if self.reduction == "sum":
                        sf_bn_morph_loss = self.opt.sf_bn_morph_weight * sf_bn_morph_loss.sum()
                    elif self.reduction == "mean":
                        sf_bn_morph_loss = self.opt.sf_bn_morph_weight * sf_bn_morph_loss.mean()
                    else:
                        assert False

                    # with torch.no_grad():
                    #     inside_morph_dists = torch.norm(sf_morph_pred_scaled[sorted_sf_inside_morph_val] - \
                    #                                     sf_morph_gt_scaled[sorted_sf_inside_morph_val], 
                    #                                     dim=1, keepdim=True)
                    #     inside_morph_targets = sf_morph_gt_scaled[sorted_sf_inside_morph_val] + \
                    #                             (src.dist2edge[src.isStable][sf_sort_indices][sorted_sf_inside_morph_val][:, None] / inside_morph_dists) * \
                    #                                 (sf_morph_pred_scaled[sorted_sf_inside_morph_val] - sf_morph_gt_scaled[sorted_sf_inside_morph_val])
                    # sf_bn_inside_morph_loss = torch_sq_distance(sf_morph_pred_scaled[sorted_sf_inside_morph_val], inside_morph_targets)
                    # sf_bn_inside_morph_loss = torch.exp(-10.*inside_morph_dists[:, 0]) * sf_bn_inside_morph_loss
                    # sf_bn_inside_morph_loss = sf_bn_inside_morph_loss[src.dist2edge[src.isStable][sf_sort_indices][sorted_sf_inside_morph_val] < 5e-2]
                    # if self.reduction == "sum":
                    #     sf_bn_inside_morph_loss = self.opt.sf_bn_morph_weight * sf_bn_inside_morph_loss.sum()
                    # elif self.reduction == "mean":
                    #     sf_bn_inside_morph_loss = self.opt.sf_bn_morph_weight * sf_bn_inside_morph_loss.mean()
                    # else:
                    #     assert False
                    # sf_bn_morph_loss += sf_bn_inside_morph_loss

                    loss_surfels += sf_bn_morph_loss
                    losses["sf_bn_morph_loss"] = sf_bn_morph_loss
                else:
                    sf_bn_morph_loss = np.sqrt(self.opt.sf_bn_morph_weight) * \
                        torch.norm(sf_morph_pred_scaled[sorted_sf_morph_val]-sf_morph_gt_scaled[sorted_sf_morph_val], dim=1)
                    loss_surfels.append(sf_bn_morph_loss)
                    if self.reduction == "sum":
                        losses["sf_bn_morph_loss"] = (sf_bn_morph_loss**2).sum()
                    elif self.reduction == "mean":
                        losses["sf_bn_morph_loss"] = (sf_bn_morph_loss**2).mean()
                    else:
                        assert False
            
        """
        Surfel losses.
        """

        assert not (self.opt.sf_point_plane and self.opt.sf_hard_seg_point_plane and self.opt.sf_soft_seg_point_plane), \
            "Need to choose one of them: sf_point_plane, sf_hard_seg_point_plane, sf_soft_seg_point_plane."
        if self.opt.sf_point_plane or self.opt.sf_hard_seg_point_plane or self.opt.sf_soft_seg_point_plane:
            huber_th = -1.

            if self.opt.sf_hard_seg_point_plane or self.opt.sf_soft_seg_point_plane:
                point_plane_loss =  DataLoss.autograd_forward(self.opt, inputs, self.new_data, trg,
                        src_seg=self.sf_seg, src_seg_conf=self.sf_seg_conf, soft_seg=self.opt.sf_soft_seg_point_plane,
                        huber_th=huber_th, reduction=self.reduction, square_loss=square_loss)
            else:
                point_plane_loss = DataLoss.autograd_forward(self.opt, inputs, self.new_data, trg, 
                                                huber_th=huber_th, reduction=self.reduction, square_loss=square_loss)

            if square_loss:
                point_plane_loss = self.opt.sf_point_plane_weight * point_plane_loss
                loss_surfels += point_plane_loss
                losses["point_plane_loss"] = point_plane_loss
            else:
                point_plane_loss = np.sqrt(self.opt.sf_point_plane_weight) * point_plane_loss
                loss_surfels.append(point_plane_loss)
                if self.reduction == "mean":
                    losses["point_plane_loss"] = (point_plane_loss**2).mean()
                elif self.reduction == "sum":
                    losses["point_plane_loss"] = (point_plane_loss**2).sum()
        
        # TODO: Try mesh normal consistency loss and mesh laplacian smoothing loss.
        # loss_normal = mesh_normal_consistency(new_src_mesh)
        # loss_laplacian = mesh_laplacian_smoothing(full_new_src_mesh, method="uniform")

        """
        Appearance loss.
        """
        if self.opt.render_loss:
            
            if self.opt.renderer == "grid_sample":
                # diff_render_loss = self.opt.render_loss_weight * torch.abs(target_color - source_color).mean(0)
                diff_render_loss = self.opt.render_loss_weight * torch.abs(target_color - source_color).mean(0)**2
            elif self.opt.renderer == "warp":
                diff_render_loss = self.opt.render_loss_weight * compute_reprojection_loss(warpbackImg, inputs[("prev_color", 0, 0)])
                # diff_render_loss = self.opt.render_loss_weight * torch.abs(warpbackImg - inputs[("prev_color", 0, 0)]).mean(1)**2
            else: # Pulsar
                # diff_render_loss = self.opt.render_loss_weight * SSIM().cuda()(renderImg, inputs[("color", 0, 0)]).mean(1, True)**2 # window=3
                diff_render_loss = self.opt.render_loss_weight * (0.85 * SSIM().cuda()(renderImg, inputs[("color", 0)]).mean(1, True)**2 + \
                                                                  0.15 * torch.abs(renderImg - inputs[("color", 0)]).mean(1, True)**2)

            diff_render_loss = diff_render_loss.mean()
            # if self.reduction == "mean":
            #     diff_render_loss = diff_render_loss.mean()
            # elif self.reduction == "sum":
            #     diff_render_loss = diff_render_loss.sum()
            # else:
            #     assert False
            losses["render_loss"] = diff_render_loss
            render_loss += diff_render_loss

        # """
        # feature loss
        # """
        # if self.opt.feature_loss:
        #     if isinstance(inputs["disp_feature"], list):
        #         new_fmap = torch.cat([F.interpolate(fmap.detach(), 
        #                                             inputs["disp_feature"][-1].size()[2:], 
        #                                             mode="bilinear") \
        #                               for fmap in inputs["disp_feature"]]
        #                               , dim=1)
        #     else:
        #         new_fmap = inputs["disp_feature"].detach()           

        #     new_features = F.grid_sample(new_fmap, 
        #                                  self.new_sf_grid[None, :, None, :]
        #                                 )[0, :, :, 0].permute(1, 0)
        #     feature_loss += (torch.abs(new_features - self.old_features).mean(1)**2).mean()
        #     losses["feature_loss"] = feature_loss

        loss = loss_mesh + loss_surfels + render_loss + feature_loss
        return loss, losses, boundary_edge_type, boundary_face_type

    def forward(self, inputs, src, trg, models, trg_=None):

        src_graph = src.ED_nodes
        src_edge_index = src_graph.edge_index.type(torch.long)

        sf_knn_indices = src.knn_indices[src.isStable]
        sf_knn_w = src.knn_w[src.isStable]
        sf_knn = src_graph.points[sf_knn_indices]
        sf_diff = src.points[src.isStable].unsqueeze(1) - sf_knn
        skew_v = get_skew(sf_diff)
        if hasattr(src, 'seg'):
            self.sf_seg = src.seg[src.isStable]
            self.sf_seg_conf = src.seg_conf[src.isStable]

        # if self.opt.feature_loss:
        #     if isinstance(src.points[src.isStable], list):
        #         old_fmap = torch.cat([F.interpolate(fmap.detach(), 
        #                                             src.prev_disp_feature[-1].size()[2:], 
        #                                             mode="bilinear") \
        #                              for fmap in src.prev_disp_feature]
        #                              , dim=1)
        #     else:
        #         old_fmap = src.prev_disp_feature.detach()   

        #     old_grid_y, old_grid_x, _, _ = pcd2depth(inputs, 
        #                                              src.points[src.isStable], 
        #                                              round_coords=False)

        #     old_grid_x = old_grid_x / self.opt.width * 2. - 1.
        #     old_grid_y = old_grid_y / self.opt.height * 2. - 1.
        #     self.old_features = F.grid_sample(old_fmap, 
        #                                       torch.stack([old_grid_x.detach(), old_grid_y.detach()], dim=1)[None, :, None, :].type(torch.float32)
        #                                      )[0, :, :, 0].permute(1, 0)

        """
        Optimization loop.
        """
        deform_verts = torch.tensor(
            np.repeat(np.array([[1.,0.,0.,0.,0.,0.,0.]]), src_graph.num+1, axis=0),
            dtype=torch.float64, device=torch.device('cuda'), requires_grad=True)
        # radii = torch.tensor(src.radii[src.isStable].cpu().numpy(), dtype=fl32_, 
        #     device=torch.device('cuda'), requires_grad=True)
        # colors = torch.tensor(src.colors[src.isStable].cpu().numpy(), dtype=fl32_, 
        #     device=torch.device('cuda'), requires_grad=True)
        
        # Init optimizer.
        if self.optim == "SGD":
            lr = 0.00005
            optimizer = torch.optim.SGD([deform_verts], lr=lr, momentum=0.9)
        elif self.optim == "Adam":
            lr = 0.001
            optimizer = torch.optim.Adam([deform_verts], lr=lr)
        elif self.optim == "LM":
            u = 1.5
            v = 5.
            minimal_loss = 1e10
            best_deform_verts = copy.deepcopy(deform_verts)
        
        # if self.opt.method == 'semantic-super' and self.opt.mesh_edge:
        #     connected_boundary_edges = torch.tensor(
        #         src_graph.connected_boundary_edges.cpu().numpy(),
        #         dtype=fl64_, device=torch.device('cuda'), requires_grad=True)
        #     edge_optimizer = torch.optim.SGD([connected_boundary_edges], lr=0.01, momentum=0.9)
        
        for i in range(self.Niter):
            if self.optim in ["SGD", "Adam"]:
                optimizer.zero_grad()
            
            # Deform the mesh and surfels.
            new_verts = src_graph.points + deform_verts[:-1,4:]
            new_norms, _ = transformQuatT(src_graph.norms, deform_verts[:-1,0:4])

            new_sf, _ = Trans_points(sf_diff, sf_knn, deform_verts[sf_knn_indices], sf_knn_w, skew_v=skew_v)
            new_sfnorms = torch.sum(new_norms[sf_knn_indices] * sf_knn_w.unsqueeze(-1), dim=1)

            # T_g
            new_verts, _ = transformQuatT(new_verts, deform_verts[-1:, 0:4]) 
            new_verts = new_verts + deform_verts[-1:, 4:]
            new_norms, _ = transformQuatT(new_norms, deform_verts[-1:, 0:4])
            new_norms = F.normalize(new_norms, dim=-1)
            
            new_sf, _ = transformQuatT(new_sf, deform_verts[-1:, 0:4]) 
            new_sf = new_sf + deform_verts[-1:, 4:]
            new_sfnorms, _ = transformQuatT(new_sfnorms, deform_verts[-1:, 0:4])
            new_sfnorms = F.normalize(new_sfnorms, dim=-1)

            self.new_sf = new_sf
            self.new_data = Data(points=new_sf, norms=new_sfnorms, 
                                 colors=src.colors[src.isStable])

            # if self.opt.feature_loss:
            new_sf_y, new_sf_x, _, _ = pcd2depth(inputs, self.new_sf, round_coords=False)
            self.ori_new_sf_grid = torch.stack([new_sf_x, new_sf_y], dim=1)
            self.new_sf_grid = torch.stack([new_sf_x / self.opt.width * 2 - 1, 
                                            new_sf_y / self.opt.height * 2 - 1], dim=1).type(torch.float64)

            loss, losses, boundary_edge_type, boundary_face_type = self.get_losses(i, deform_verts, inputs, trg, src, src_graph, new_verts, src_edge_index, models, init_iter=i==0)

            # Weighted sum of the losses
            if i == self.Niter-1:
                print_text = f"[frame{trg.time}]"
                for key in losses.keys():
                    print_text += f"{key}: {losses[key]}    " #:0.5f
                src.logger.info(print_text)
                
            # Optimization step.
            if self.optim in ["SGD", "Adam"]:
                loss.backward()
                # T_g_grad = deform_verts.grad[-1]
                # ed_grad = deform_verts.grad[:-1]
                # deform_verts.grad[-1] = (T_g_grad - T_g_grad.mean()) / T_g_grad.std() * ed_grad.std() + ed_grad.mean()
                deform_verts.grad[-1] = deform_verts.grad[-1] / src_graph.num
                optimizer.step()
            
            elif self.optim == "LM":
                # loss = torch.cat(loss)
                # jtj = torch.zeros((src_graph.param_num+7, src_graph.param_num+7), layout=torch.sparse_coo, dtype=fl32_).cuda()
                # jtl = torch.zeros((src_graph.param_num+7, 1), layout=torch.sparse_coo, dtype=fl32_).cuda()
                # for _loss_ in loss:
                #     print("$$$")
                #     _loss_.backward(retain_graph=True) # TODO: Too slow.
                #     print("***")
                #     j = deform_verts.grad.reshape(1, -1)
                #     jtj = jtj + torch.sparse.mm(j.T.to_sparse(), j.to_sparse())
                #     jtl = jtl + j.T.to_sparse() * _loss_
                # delta = torch.matmul(
                #             torch.inverse((jtj + u * torch.eye(jtj.size(0), layout=torch.sparse_coo).cuda()).to_dense()),
                #             jtl).reshape(deform_verts.size())
                
                jtj = torch.zeros((src_graph.param_num+7, src_graph.param_num+7), layout=torch.sparse_coo, dtype=fl32_).cuda()
                jtl = torch.zeros((src_graph.param_num+7, 1), dtype=fl32_).cuda()
                for _loss_ in loss:
                    if self.reduction == "mean":
                        _loss_ = torch.abs(_loss_).mean()
                    elif self.reduction == "sum":
                        _loss_ = torch.abs(_loss_).sum()
                    else:
                        assert False
                    _loss_.backward(retain_graph=True)
                    j = deform_verts.grad.reshape(1, -1)
                    jtj = jtj + torch.sparse.mm(j.T.to_sparse(), j.to_sparse())
                    jtl = jtl + j.T * _loss_
                jtj = jtj.to_dense()
                delta = torch.matmul(
                            torch.inverse(jtj + u * torch.eye(jtj.size(0)).cuda()),
                            jtl).reshape(deform_verts.size())
                
                # loss = torch.cat(loss)
                # loss = torch.abs(loss).sum()
                # loss.backward(retain_graph=True)
                # j = deform_verts.grad.reshape(1, -1)
                # jtj = torch.matmul(j.T, j)
                # jtl = j.T * loss
                
                delta = torch.matmul(
                            torch.inverse(jtj + u * torch.eye(jtj.size(0)).cuda()),
                            jtl).reshape(deform_verts.size())
                deform_verts = deform_verts + delta
                
                with torch.no_grad():
                    loss, _, _, _ = self.get_losses(deform_verts, inputs, trg, src, src_graph, new_verts, src_edge_index, nets)
                    if self.reduction == "mean":
                        loss = (torch.cat(loss)**2).mean()
                    elif self.reduction == "sum":
                        loss = (torch.cat(loss)**2).sum()
                    else:
                        assert False

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

        return deform_verts, boundary_edge_type, boundary_face_type