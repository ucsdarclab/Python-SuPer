import cv2
import copy
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch_geometric.data import Data

from super.loss import *

# from RAFT.core.utils.flow_viz import flow_to_image
from optical_flow.feature_matching import LoFTR

from utils.config import *
from utils.utils import *

from bnmorph.layers import grad_computation_tools
# from bnmorph.bnmorph import BNMorph

from depth.monodepth2.layers import SSIM, AdaSSIM, compute_reprojection_loss, get_smooth_loss

class GraphFit(nn.Module):
    def __init__(self, opt):
        super(GraphFit, self).__init__()

        self.opt = opt
        if self.opt.sf_corr:
            self.flow_detector = LoFTR()

        if self.opt.bn_morph or self.opt.sf_bn_morph:
            self.tool = grad_computation_tools(batch_size=self.opt.batch_size, height=self.opt.height,
                                               width=self.opt.width).cuda()

        if self.opt.use_edge_ssim_hints:
            self.ssim = AdaSSIM().cuda()

        self.reduction="sum"
        self.valid_margin = 1
        self.optim = "SGD"
        self.Niter=10

    def get_losses(self, i, deform_verts, inputs, trg, src, src_graph, new_verts, src_edge_index, nets, init_iter=False):
        square_loss = self.optim in ["SGD", "Adam"]

        boundary_edge_type = None
        boundary_face_type = None
        if self.opt.use_edge_ssim_hints:
            if self.opt.mesh_edge:
                boundary_edge_type = src_graph.boundary_edge_type
            if self.opt.mesh_face:
                boundary_face_type = src_graph.boundary_face_type

        if self.opt.render_loss or self.opt.sf_corr or self.opt.mesh_edge or self.opt.mesh_face:
            if self.opt.renderer == "grid_sample":
                with torch.no_grad():
                    renderImg = nets["renderer"](inputs, self.new_data, rad=self.opt.renderer_rad).permute(2,0,1).unsqueeze(0)

                # if i == 0:
                init_y, init_x, _, _ = pcd2depth(inputs, src.points[src.isStable], round_coords=False)
                init_pts = torch.stack([init_x, init_y], dim=1).type(fl32_)

                init_pts[...,0] = init_pts[...,0] / self.opt.width * 2 - 1
                init_pts[...,1] = init_pts[...,1] / self.opt.height * 2 - 1
                source_color = nn.functional.grid_sample(inputs[("color", 0, 0)], init_pts[None, :, None, :])[0, :, :, 0]
                ####

                new_sf_y, new_sf_x, _, _ = pcd2depth(inputs, new_sf, round_coords=False)
                new_sf_grid = torch.stack([new_sf_x, new_sf_y], dim=1).type(fl32_)

                new_sf_grid[...,0] = new_sf_grid[...,0] / self.opt.width * 2 - 1
                new_sf_grid[...,1] = new_sf_grid[...,1] / self.opt.height * 2 - 1
                target_color = nn.functional.grid_sample(inputs[("color", 0, 0)], new_sf_grid[None, :, None, :])[0, :, :, 0]

            elif self.opt.renderer == "warp":
                with torch.no_grad():
                    renderImg = nets["renderer"](inputs, self.new_data, rad=self.opt.renderer_rad).permute(2,0,1).unsqueeze(0)

                base_x, base_y = torch.meshgrid(torch.arange(self.opt.width), torch.arange(self.opt.height), indexing='xy')
                base_grid = torch.stack([base_x, base_y], dim=2).type(fl64_).cuda()
                base_grid = base_grid.reshape(-1, 2)

                # if i == 0:
                init_y, init_x, _, _ = pcd2depth(inputs, src.points[src.isStable], round_coords=False)
                init_pts = torch.stack([init_x, init_y], dim=1)

                knn_dists, knn_ids = find_knn(base_grid, init_pts, k=4)
                knn_weights = F.softmax(torch.exp(-knn_dists), dim=1)
                ####

                sample_sf = (new_sf[knn_ids] * knn_weights[..., None]).sum(1)
                sample_y, sample_x, _, _ = pcd2depth(inputs, sample_sf, round_coords=False)
                sample_grid = torch.stack([sample_x, sample_y], dim=1).reshape(1, self.opt.height, self.opt.width, 2).type(fl32_)
                sample_grid[...,0] = sample_grid[...,0] / self.opt.width * 2 - 1
                sample_grid[...,1] = sample_grid[...,1] / self.opt.height * 2 - 1

                warpbackImg = nn.functional.grid_sample(inputs[("color", 0, 0)], sample_grid)

            elif "renderer" in nets:
                renderImg = nets["renderer"](inputs, self.new_data, rad=self.opt.renderer_rad).permute(2,0,1).unsqueeze(0)
                # render_data = Data(points=torch.cat([self.new_data.points, inputs["del_points"][0]], dim=0), 
                #                    colors=torch.cat([self.new_data.colors, inputs["del_colors"][0]], dim=0))
                # renderImg = nets["renderer"](inputs, render_data, rad=self.opt.renderer_rad).permute(2,0,1).unsqueeze(0)

        if self.opt.sf_corr:
            if "optical_flow" in nets:
                target_images = inputs[("color", 0, 0)]
                if self.opt.sf_corr_match_renderimg:
                    source_images = renderImg
                else:
                    source_images = inputs[("prev_color", 0, 0)]
                if self.opt.sf_corr_use_keyframes:
                    target_images = target_images.repeat(1+src.keyframes[0].size(0), 1, 1, 1)
                    source_images = torch.cat([source_images, src.keyframes[0]], dim=0)
                flow = nets["optical_flow"](source_images, target_images) # x, y
                if isinstance(flow, list):
                    flow = flow[-1]
                
                if self.opt.sf_corr_use_keyframes:
                    keyframe_flows = flow.clone()
                    flow = flow[0:1]

            else:
                if self.opt.sf_corr_match_renderimg:
                    flow = self.flow_detector(renderImg, inputs[("color", 0, 0)]) # x, y
                else:
                    flow = self.flow_detector(inputs[("prev_color", 0, 0)], inputs[("color", 0, 0)])
        ####


        losses = {}
        if square_loss:
            loss_mesh = 0.0
            loss_surfels = 0.0
            render_loss = 0.0
        else:
            loss_mesh = []
            loss_surfels = []
            render_loss = []

        """
        Mesh losses.
        """

        if (self.opt.mesh_edge or self.opt.mesh_face) and (self.opt.method == 'seman-super') and self.opt.use_edge_ssim_hints:
            if self.opt.mesh_edge:
                src_boundary_edge_index = src_edge_index.permute(1, 0)[src_graph.isBoundary]
            if self.opt.mesh_face:
                src_boundary_face_index = src_graph.triangles.permute(1, 0)[src_graph.isBoundaryFace]
            
            if i == self.Niter - 1:
                # Find the init edge end points project location in the image plane.
                init_ed_y, init_ed_x, _, _ = pcd2depth(inputs, src_graph.points, round_coords=False)

                # Find the updated edge end points project location in the image plane.
                current_ed_y, current_ed_x, _, _ = pcd2depth(inputs, new_verts, round_coords=False)

                if self.opt.mesh_edge:
                    moving_boundary_edge = []
                    margin = 10
                    for edge_id1, edge_id2 in src_boundary_edge_index:
                        init_ct_x = 0.5 * (init_ed_x[edge_id1] + init_ed_x[edge_id2])
                        init_ct_y = 0.5 * (init_ed_y[edge_id1] + init_ed_y[edge_id2])
                        current_ct_x = 0.5 * (current_ed_x[edge_id1] + current_ed_x[edge_id2])
                        current_ct_y = 0.5 * (current_ed_y[edge_id1] + current_ed_y[edge_id2])
                        
                        win_h = torch.abs(init_ed_y[edge_id1] - init_ed_y[edge_id2]) + margin
                        win_w = torch.abs(init_ed_x[edge_id1] - init_ed_x[edge_id2]) + margin
                        x1, x2 = int(init_ct_x - win_w / 2), int(init_ct_x + win_w / 2)
                        y1, y2 = int(init_ct_y - win_h / 2), int(init_ct_y + win_h / 2)
                        prev_patch = F.pad(inputs[("prev_color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), 
                                                                        min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
                                        (min(max(0, -x1), win_w), min(max(0, x2-self.opt.width), win_w), 
                                            min(max(0, -y1), win_h), min(max(0, y2-self.opt.height), win_h))
                                        ) # Patch extracted from the previous frame with init projection.
                        cr_patch = F.pad(inputs[("color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), 
                                                                min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
                                        (min(max(0, -x1), win_w), min(max(0, x2-self.opt.width), win_w), 
                                        min(max(0, -y1), win_h), min(max(0, y2-self.opt.height), win_h))
                                        ) # Patch extracted from the current frame with init projection.

                        win_h = torch.abs(current_ed_y[edge_id1] - current_ed_y[edge_id2]) + margin
                        win_w = torch.abs(current_ed_x[edge_id1] - current_ed_x[edge_id2]) + margin
                        x1, x2 = int(current_ct_x - win_w / 2), int(current_ct_x + win_w / 2)
                        y1, y2 = int(current_ct_y - win_h / 2), int(current_ct_y + win_h / 2)
                        moving_patch = F.pad(inputs[("color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), 
                                                                    min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
                                            (min(max(0, -x1), win_w), min(max(0, x2-self.opt.width), win_w), 
                                            min(max(0, -y1), win_h), min(max(0, y2-self.opt.height), win_h))
                                            ) # Patches extracted from the current frame with new projection.
                        moving_boundary_edge.append(torch.any(self.ssim(prev_patch, cr_patch) < \
                                                    self.ssim(F.interpolate(prev_patch[None, ...], moving_patch.size()[1:3], mode='bilinear')[0], moving_patch)))
                    
                    moving_boundary_edge = torch.tensor(moving_boundary_edge).cuda()

                    new_edges_lens = torch.norm(new_verts[src_edge_index[0]][src_graph.isBoundary] - 
                                                new_verts[src_edge_index[1]][src_graph.isBoundary], dim=1)
                    moving_boundary_edge &= (new_edges_lens > 0.2)

                    moving_boundary_edge = moving_boundary_edge.type(fl32_)
                    boundary_edge_type = torch.cat([boundary_edge_type[:, 1:], moving_boundary_edge[:, None]], dim=1)

                if self.opt.mesh_face:
                    moving_boundary_face = []
                    margin = 10
                    for face_id1, face_id2, face_id3 in src_boundary_face_index:
                        init_ct_x = 1./3. * (init_ed_x[face_id1] + init_ed_x[face_id2] + init_ed_x[face_id3])
                        init_ct_y = 1./3. * (init_ed_y[face_id1] + init_ed_y[face_id2] + init_ed_y[face_id3])
                        current_ct_x = 1./3. * (current_ed_x[face_id1] + current_ed_x[face_id2] + current_ed_x[face_id3])
                        current_ct_y = 1./3. * (current_ed_y[face_id1] + current_ed_y[face_id2] + current_ed_y[face_id3])
                        
                        win_h = int(torch.max(torch.tensor([init_ed_y[face_id1], init_ed_y[face_id2], init_ed_y[face_id3]])) - \
                                torch.min(torch.tensor([init_ed_y[face_id1], init_ed_y[face_id2], init_ed_y[face_id3]])) + margin)
                        win_w = int(torch.max(torch.tensor([init_ed_x[face_id1], init_ed_x[face_id2], init_ed_x[face_id3]])) - \
                                torch.min(torch.tensor([init_ed_x[face_id1], init_ed_x[face_id2], init_ed_x[face_id3]])) + margin)
                        x1 = int(init_ct_x - win_w / 2)
                        x2 = x1 + win_w
                        y1 = int(init_ct_y - win_h / 2)
                        y2 = y1 + win_h
                        prev_patch = F.pad(inputs[("prev_color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), 
                                                                        min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
                                        (min(max(0, -x1), win_w), min(max(0, x2-self.opt.width), win_w), 
                                            min(max(0, -y1), win_h), min(max(0, y2-self.opt.height), win_h))
                                        ) # Patch extracted from the previous frame with init projection.
                        cr_patch = F.pad(inputs[("color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), 
                                                                min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
                                        (min(max(0, -x1), win_w), min(max(0, x2-self.opt.width), win_w), 
                                        min(max(0, -y1), win_h), min(max(0, y2-self.opt.height), win_h))
                                        ) # Patch extracted from the current frame with init projection.

                        cr_win_h = int(torch.max(torch.tensor([current_ed_y[face_id1], current_ed_y[face_id2], current_ed_y[face_id3]])) - \
                                torch.min(torch.tensor([current_ed_y[face_id1], current_ed_y[face_id2], current_ed_y[face_id3]])) + margin)
                        cr_win_w = int(torch.max(torch.tensor([current_ed_x[face_id1], current_ed_x[face_id2], current_ed_x[face_id3]])) - \
                                torch.min(torch.tensor([current_ed_x[face_id1], current_ed_x[face_id2], current_ed_x[face_id3]])) + margin)
                        x1 = int(init_ct_x - cr_win_w / 2)
                        x2 = x1 + cr_win_w
                        y1 = int(init_ct_y - cr_win_h / 2)
                        y2 = y1 + cr_win_h
                        cr_x1 = int(current_ct_x - cr_win_w / 2)
                        cr_x2 = cr_x1 + cr_win_w
                        cr_y1 = int(current_ct_y - cr_win_h / 2)
                        cr_y2 = cr_y1 + cr_win_h
                        prev_patch2 = F.pad(inputs[("prev_color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), 
                                                                         min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
                                            (min(max(0, -x1), cr_win_w), min(max(0, x2-self.opt.width), cr_win_w), 
                                            min(max(0, -y1), cr_win_h), min(max(0, y2-self.opt.height), cr_win_h))
                                            )
                        moving_patch = F.pad(inputs[("color", 0, 0)][0, :, min(max(cr_y1,0),self.opt.height): min(max(cr_y2,0),self.opt.height), 
                                                                     min(max(cr_x1,0),self.opt.width): min(max(cr_x2,0),self.opt.width)],
                                             (min(max(0, -cr_x1), cr_win_w), min(max(0, cr_x2-self.opt.width), cr_win_w), 
                                             min(max(0, -cr_y1), cr_win_h), min(max(0, cr_y2-self.opt.height), cr_win_h))
                                            ) # Patches extracted from the current frame with new projection.
                                            
                        moving_boundary_face.append(torch.any(self.ssim(prev_patch, cr_patch) < self.ssim(prev_patch2, moving_patch)))
                    
                    moving_boundary_face = torch.tensor(moving_boundary_face).cuda()
                    # print("####", torch.count_nonzero(moving_boundary_face))

                    new_triangles_areas = torch.cross(new_verts[src_graph.triangles[1]][src_graph.isBoundaryFace] - new_verts[src_graph.triangles[0]][src_graph.isBoundaryFace],
                                    new_verts[src_graph.triangles[2]][src_graph.isBoundaryFace] - new_verts[src_graph.triangles[0]][src_graph.isBoundaryFace],
                                    dim=1)
                    new_triangles_areas = 0.5 * torch.sqrt((new_triangles_areas**2).sum(1) + 1e-13)
                    # print(torch.unique(new_triangles_areas))
                    moving_boundary_face |= (new_triangles_areas < 0.025) # 0.025, 0.01

                    moving_boundary_face = moving_boundary_face.type(fl32_)
                    boundary_face_type = torch.cat([boundary_face_type[:, 1:], moving_boundary_face[:, None]], dim=1)

        # Edge loss.
        if self.opt.mesh_edge:
            new_edges_lens = torch.norm(new_verts[src_edge_index[0]] - new_verts[src_edge_index[1]], dim=1)
            edge_losses = new_edges_lens - src_graph.edges_lens

            if square_loss:
                edge_losses = self.opt.mesh_edge_weight * edge_losses ** 2
            else:
                edge_losses = np.sqrt(self.opt.mesh_edge_weight) * edge_losses
            
            if self.opt.method == 'seman-super' and self.opt.use_edge_ssim_hints:
                connected_edges = torch.ones_like(src_graph.isBoundary)
                connected_edges[src_graph.isBoundary] = src_graph.boundary_edge_type[:, -1] == 1.
                
                edge_losses = edge_losses[connected_edges]

            if square_loss:
                if self.reduction == "mean":
                    edge_losses = edge_losses.mean()
                elif self.reduction == "sum":
                    edge_losses = edge_losses.sum()
                else:
                    assert False
                loss_mesh += edge_losses
                losses["edge_loss"] = edge_losses
            else:
                loss_mesh.append(edge_losses)
                if self.reduction == "mean":
                    losses["edge_loss"] = (edge_losses**2).mean()
                elif self.reduction == "sum":
                    losses["edge_loss"] = (edge_losses**2).sum()
                else:
                    assert False
                

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

            if self.opt.method == 'seman-super' and self.opt.use_edge_ssim_hints:
                connected_faces = torch.ones_like(src_graph.isBoundaryFace)
                connected_faces[src_graph.isBoundaryFace] = src_graph.boundary_face_type[:, -1] == 1.
                
                face_losses = face_losses[connected_faces]

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
            sf_y, sf_x, _, _ = pcd2depth(inputs, self.new_sf, round_coords=False)
            ori_new_sf_grid = torch.stack([sf_x, sf_y], dim=1)
            new_sf_grid = ori_new_sf_grid.clone()
            new_sf_grid[...,0] = new_sf_grid[...,0] / self.opt.width * 2 - 1
            new_sf_grid[...,1] = new_sf_grid[...,1] / self.opt.height * 2 - 1
            new_sf_seman = F.grid_sample(inputs[("seman_conf", 0)], 
                                                        new_sf_grid[None, :, None, :],
                                                        )[0, :, :, 0].argmax(0)
            # new_sf_grid[...,0] = new_sf_grid[...,0] / (self.opt.width-1) * 2 - 1
            # new_sf_grid[...,1] = new_sf_grid[...,1] / (self.opt.height-1) * 2 - 1
            # new_sf_seman = nn.functional.grid_sample(inputs[("seman_conf", 0)], 
            #                                          new_sf_grid[None, :, None, :],
            #                                          align_corners=True
            #                                         )[0, :, :, 0].argmax(0)
                
            sf_morph_val = (~(new_sf_seman==self.sf_seman)) & \
                            (new_sf_grid[...,0] >= -1) & (new_sf_grid[...,0] <= 1) & \
                            (new_sf_grid[...,1] >= -1) & (new_sf_grid[...,1] <= 1)
            sf_inside_morph_val = (new_sf_seman==self.sf_seman) & \
                                  (new_sf_grid[...,0] >= -1) & (new_sf_grid[...,0] <= 1) & \
                                  (new_sf_grid[...,1] >= -1) & (new_sf_grid[...,1] <= 1)

            sorted_sf_seman, sf_sort_indices = torch.sort(self.sf_seman)
            sorted_ori_new_sf_grid = ori_new_sf_grid[sf_sort_indices]
            sorted_sf_morph_val = sf_morph_val[sf_sort_indices]
            sorted_sf_inside_morph_val = sf_inside_morph_val[sf_sort_indices]

            sf_morph_gt = []
            if init_iter:
                with torch.no_grad():
                    kernels = [3, 3, 3]
                    self.edge_pts = []
                    for class_id in range(self.opt.num_classes):
                        seman_grad_bin = self.tool.get_semanticsEdge(
                            inputs[("seman", 0)], foregroundType=[class_id],
                            erode_foreground=True, kernel_size=kernels[class_id])
                        edge_y, edge_x = seman_grad_bin[0,0].nonzero(as_tuple=True)
                        self.edge_pts.append(torch.stack([edge_x, edge_y], dim=1).type(fl64_))

            for class_id in range(self.opt.num_classes):
                _, knn_edge_ids = find_knn(sorted_ori_new_sf_grid[sorted_sf_seman == class_id], 
                                                self.edge_pts[class_id], k=1)
                sf_morph_gt.append(self.edge_pts[class_id][knn_edge_ids])
            sf_morph_gt_seman = torch.cat([class_id * torch.ones(len(_sf_morph_gt_), dtype=long_) \
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
                
            #     morph_gt_seman_win = inputs[("seman", 0)][0, 0, morph_gt_y1:morph_gt_y2, morph_gt_x1:morph_gt_x2]
            #     class_point_numbers = []
            #     class_depths = []
            #     for class_id in range(self.opt.num_classes):
            #         morph_gt_win_class_val = (morph_gt_seman_win == class_id) & morph_gt_depth_win_val
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
            #     class_point_numbers[sf_morph_gt_seman[sf_morph_val_id]] < class_th or \
            #     torch.abs(class_depths[max_class_id] - class_depths[sf_morph_gt_seman[sf_morph_val_id]]) > delta_depth_th:
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
                # sorted_sf_morph_val &= inputs[("seman_valid", 0)][0, 0][sf_morph_gt_scaled[:, 1].type(long_), sf_morph_gt_scaled[:, 0].type(long_)]
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

        assert not (self.opt.sf_point_plane and self.opt.sf_hard_seman_point_plane and self.opt.sf_soft_seman_point_plane), \
            "Need to choose one of them: sf_point_plane, sf_hard_seman_point_plane, sf_soft_seman_point_plane."
        if self.opt.sf_point_plane or self.opt.sf_hard_seman_point_plane or self.opt.sf_soft_seman_point_plane:
            huber_th = -1.

            if self.opt.sf_hard_seman_point_plane or self.opt.sf_soft_seman_point_plane:
                point_plane_loss =  DataLoss.autograd_forward(self.opt, inputs, self.new_data, trg, color_hint=self.opt.use_color_hints,
                        src_seman=self.sf_seman, src_seman_conf=self.sf_seman_conf, soft_seman=self.opt.sf_soft_seman_point_plane,
                        huber_th=huber_th, reduction=self.reduction, square_loss=square_loss)
            else:
                point_plane_loss = DataLoss.autograd_forward(self.opt, inputs, self.new_data, trg, color_hint=self.opt.use_color_hints, 
                                                huber_th=huber_th, reduction=self.reduction, square_loss=square_loss)

            if i >= self.opt.icp_start_iter - 1:
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

        if self.opt.sf_corr:

            if self.opt.sf_corr_use_keyframes:
                trg_loc_x, trg_loc_y = torch.meshgrid(torch.arange(self.opt.width), torch.arange(self.opt.height), indexing='xy')
                trg_loc = torch.stack([trg_loc_x, trg_loc_y], dim=2)[None, ...].type(fl32_).cuda()
                trg_loc = trg_loc.repeat(keyframe_flows.size(0), 1, 1, 1) + keyframe_flows.permute(0, 2, 3, 1)
                target_grid = trg_loc.clone()
                trg_loc[...,0] = trg_loc[...,0] / self.opt.width * 2 - 1
                trg_loc[...,1] = trg_loc[...,1] / self.opt.height * 2 - 1

                warp_images = nn.functional.grid_sample(target_images, trg_loc)
                warp_losses = compute_reprojection_loss(warp_images, source_images)
                
                source_grid = - 2. * self.opt.width * torch.ones((len(src.points), 2), dtype=fl64_).cuda()
                source_grid[src.keyframes[1]] = src.keyframes[2][0]
                init_sf_y, init_sf_x, _, _ = pcd2depth(inputs, src.points, round_coords=False, valid_margin=self.valid_margin)
                init_sf_x[~src.isStable] = - 2. * self.opt.width
                init_sf_y[~src.isStable] = - 2. * self.opt.height
                source_grid = torch.stack([
                                    torch.stack([init_sf_x, init_sf_y], dim=1),
                                    source_grid
                                ], dim=0)[:, None, :, :].type(fl32_)
                source_grid[...,0] = source_grid[...,0] / self.opt.width * 2 - 1
                source_grid[...,1] = source_grid[...,1] / self.opt.height * 2 - 1
                warp_losses = F.grid_sample(warp_losses, source_grid)[:,0,0,:]
                warp_losses = torch.where(
                    (source_grid[...,0,0]>=-1) & (source_grid[...,0,0]<=1) & (source_grid[...,0,1]>=-1) & (source_grid[...,0,1]<=1),
                    warp_losses,
                    torch.tensor(100., dtype=fl32_).cuda()
                    )

                good_ids = warp_losses.argmin(0)
                target_grid = F.grid_sample(target_grid.permute(0, 3, 1, 2), source_grid)[:,:,0,:].permute(0, 2, 1)
                current_sf_flow = torch.where(
                                    good_ids[:, None].repeat(1, 2)==0, 
                                    target_grid[0], 
                                    target_grid[1]
                                )
                val_losses = torch.any(warp_losses < 100., dim=0)
                current_sf_flow_valid = torch.any(warp_losses < 100., dim=0) & (current_sf_flow[:, 1] >= self.valid_margin) & \
                                        (current_sf_flow[:, 1] < self.opt.height-1-self.valid_margin) & \
                                        (current_sf_flow[:, 0] >= self.valid_margin) & \
                                        (current_sf_flow[:, 0] < self.opt.width-1-self.valid_margin)

                current_sf_flow = current_sf_flow[src.isStable.nonzero(as_tuple=True)[0]].permute(1, 0)
                current_sf_flow_valid = current_sf_flow_valid[src.isStable]

                loss_corr = self.opt.sf_corr_weight * \
                    DataLoss.autograd_forward(self.opt, inputs, self.new_data, trg, 
                        correspts=current_sf_flow, correspts_valid=current_sf_flow_valid,
                        loss_type=self.opt.sf_corr_loss_type, reduction=self.reduction)

            else:
                if self.opt.sf_hard_seman_corr or self.opt.sf_soft_seman_corr:
                    loss_corr = self.opt.sf_corr_weight * \
                        DataLoss.autograd_forward(self.opt, inputs, self.new_data, trg, 
                            flow=flow, huber_th=self.opt.sf_corr_huber_th,
                            src_seman=sf_seman, src_seman_conf=sf_seman_conf, soft_seman=self.opt.sf_soft_seman_corr,
                            loss_type=self.opt.sf_corr_loss_type, reduction=self.reduction)
                else:
                    loss_corr = self.opt.sf_corr_weight * \
                        DataLoss.autograd_forward(self.opt, inputs, self.new_data, trg, 
                            flow=flow, loss_type=self.opt.sf_corr_loss_type, reduction=self.reduction)

            loss_surfels += loss_corr
            losses["corr_loss"] = loss_corr
        
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
            else:
                diff_render_loss = self.opt.render_loss_weight * SSIM(window=3).cuda()(renderImg, inputs[("color", 0, 0)]).mean(1, True)**2 #+ \
                                    # self.opt.render_loss_weight * torch.abs(renderImg - inputs[("color", 0, 0)]).mean(1, True)**2

            diff_render_loss = diff_render_loss.mean()
            # if self.reduction == "mean":
            #     diff_render_loss = diff_render_loss.mean()
            # elif self.reduction == "sum":
            #     diff_render_loss = diff_render_loss.sum()
            # else:
            #     assert False
            losses["render_loss"] = diff_render_loss
            render_loss += diff_render_loss

        if self.opt.depth_smooth_loss:
            new_depth_data = self.new_data.clone()
            new_depth_data.colors = self.new_data.points
            renderDepthMap = nets["renderer"](inputs, new_depth_data, rad=self.opt.renderer_rad)[..., 0][None, None, :, :]
            depth_smooth_loss = get_smooth_loss(renderDepthMap, inputs[("color", 0, 0)])
            # depth_smooth_loss = get_smooth_loss(renderDepthMap, inputs[("color", 0, 0)])
            render_loss += self.opt.depth_smooth_loss_weight * depth_smooth_loss

        loss = loss_mesh + loss_surfels + render_loss
        return loss, losses, boundary_edge_type, boundary_face_type

    def forward(self, inputs, src, trg, nets, trg_=None):

        src_graph = src.ED_nodes
        src_edge_index = src_graph.edge_index.type(long_)

        sf_knn_indices = src.knn_indices[src.isStable]
        sf_knn_w = src.knn_w[src.isStable]
        sf_knn = src_graph.points[sf_knn_indices]
        sf_diff = src.points[src.isStable].unsqueeze(1) - sf_knn
        skew_v = get_skew(sf_diff)
        if hasattr(src, 'seman'):
            self.sf_seman = src.seman[src.isStable]
            self.sf_seman_conf = src.seman_conf[src.isStable]

        """
        Optimization loop.
        """
        deform_verts = torch.tensor(
            np.repeat(np.array([[1.,0.,0.,0.,0.,0.,0.]]), src_graph.num+1, axis=0),
            dtype=fl64_, device=torch.device('cuda'), requires_grad=True)
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
        
        # if self.opt.method == 'seman-super' and self.opt.mesh_edge:
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

            loss, losses, boundary_edge_type, boundary_face_type = self.get_losses(i, deform_verts, inputs, trg, src, src_graph, new_verts, src_edge_index, nets, init_iter=i==0)

            # Weighted sum of the losses
            if i == self.Niter-1:
                print_text = f"[frame{trg.time.item()}]"
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