import copy
import torch
import torch.nn as nn
import torchvision.transforms as T

from super.loss import *

# from RAFT.core.utils.flow_viz import flow_to_image
from optical_flow.feature_matching import LoFTR

from utils.config import *
from utils.utils import *

from bnmorph.layers import grad_computation_tools
# from bnmorph.bnmorph import BNMorph

from depth.monodepth2.layers import SSIM, compute_reprojection_loss

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
            self.ssim = SSIM(window=self.opt.edge_ssim_hints_window).cuda()

    def forward(self, inputs, src, trg, nets, Niter=10, trg_=None, reduction="sum"):

        src_graph = src.ED_nodes
        src_edge_index = src_graph.edge_index.type(long_)

        sf_knn_indices = src.knn_indices[src.isStable]
        sf_knn_w = src.knn_w[src.isStable]
        sf_knn = src_graph.points[sf_knn_indices]
        sf_diff = src.points[src.isStable].unsqueeze(1) - sf_knn
        skew_v = get_skew(sf_diff)
        if hasattr(src, 'seman'):
            sf_seman = src.seman[src.isStable]
            sf_seman_conf = src.seman_conf[src.isStable]
        boundary_edge_type = None
        
        losses = {}

        """
        Optimization loop.
        """
        deform_verts = torch.tensor(
            np.repeat(np.array([[1.,0.,0.,0.,0.,0.,0.]]), src_graph.num + 1, axis=0),
            dtype=fl64_, device=torch.device('cuda'), requires_grad=True)
        # Init optimizer.
        # optimizer = torch.optim.SGD([deform_verts], lr=0.00005, momentum=0.9)
        optimizer = torch.optim.Adam([deform_verts], lr=0.00005)
        
        # if self.opt.method == 'seman-super' and self.opt.mesh_edge:
        #     connected_boundary_edges = torch.tensor(
        #         src_graph.connected_boundary_edges.cpu().numpy(),
        #         dtype=fl64_, device=torch.device('cuda'), requires_grad=True)
        #     edge_optimizer = torch.optim.SGD([connected_boundary_edges], lr=0.01, momentum=0.9)
        
            
        for i in range(Niter):
            optimizer.zero_grad()
            
            # Deform the mesh and surfels.
            new_verts = src_graph.points + deform_verts[:-1,4:]
            new_norms, _ = transformQuatT(src_graph.norms, deform_verts[:-1,0:4])
            
            new_sf, _ = Trans_points(sf_diff, sf_knn, deform_verts[sf_knn_indices], sf_knn_w, skew_v=skew_v)
            new_sfnorms = torch.sum(new_norms[sf_knn_indices] * sf_knn_w.unsqueeze(-1), dim=1)

            # T_g
            new_verts = new_verts + deform_verts[-1:,4:]
            new_norms, _ = transformQuatT(new_norms, deform_verts[-1:,0:4])
            new_sf = new_sf + deform_verts[-1:,4:]
            new_sfnorms, _ = transformQuatT(new_sfnorms, deform_verts[-1:,0:4])
            
            new_data = Data(points=new_sf, norms=new_sfnorms, colors=src.colors[src.isStable])

            if self.opt.render_loss or self.opt.sf_corr or self.opt.mesh_edge or self.opt.mesh_face:
                if self.opt.renderer == "grid_sample":
                    with torch.no_grad():
                        renderImg = nets["renderer"](inputs, new_data, rad=self.opt.renderer_rad).permute(2,0,1).unsqueeze(0)

                    if i == 0:
                        init_y, init_x, _, _ = pcd2depth(inputs, src.points[src.isStable], round_coords=False)
                        init_pts = torch.stack([init_x, init_y], dim=1).type(fl32_)

                        init_pts[...,0] = init_pts[...,0] / self.opt.width * 2 - 1
                        init_pts[...,1] = init_pts[...,1] / self.opt.height * 2 - 1
                        source_color = nn.functional.grid_sample(inputs[("color", 0, 0)], init_pts[None, :, None, :])[0, :, :, 0]

                    new_sf_y, new_sf_x, _, _ = pcd2depth(inputs, new_sf, round_coords=False)
                    new_sf_grid = torch.stack([new_sf_x, new_sf_y], dim=1).type(fl32_)

                    new_sf_grid[...,0] = new_sf_grid[...,0] / self.opt.width * 2 - 1
                    new_sf_grid[...,1] = new_sf_grid[...,1] / self.opt.height * 2 - 1
                    target_color = nn.functional.grid_sample(inputs[("color", 0, 0)], new_sf_grid[None, :, None, :])[0, :, :, 0]

                elif self.opt.renderer == "warp":
                    with torch.no_grad():
                        renderImg = nets["renderer"](inputs, new_data, rad=self.opt.renderer_rad).permute(2,0,1).unsqueeze(0)

                    base_x, base_y = torch.meshgrid(torch.arange(self.opt.width), torch.arange(self.opt.height), indexing='xy')
                    base_grid = torch.stack([base_x, base_y], dim=2).type(fl64_).cuda()
                    base_grid = base_grid.reshape(-1, 2)

                    if i == 0:
                        init_y, init_x, _, _ = pcd2depth(inputs, src.points[src.isStable], round_coords=False)
                        init_pts = torch.stack([init_x, init_y], dim=1)

                        knn_dists, knn_ids = find_knn(base_grid, init_pts, k=4)
                        knn_weights = F.softmax(torch.exp(-knn_dists), dim=1)
                    
                    sample_sf = (new_sf[knn_ids] * knn_weights[..., None]).sum(1)
                    sample_y, sample_x, _, _ = pcd2depth(inputs, sample_sf, round_coords=False)
                    sample_grid = torch.stack([sample_x, sample_y], dim=1).reshape(1, self.opt.height, self.opt.width, 2).type(fl32_)
                    sample_grid[...,0] = sample_grid[...,0] / self.opt.width * 2 - 1
                    sample_grid[...,1] = sample_grid[...,1] / self.opt.height * 2 - 1

                    warpbackImg = nn.functional.grid_sample(inputs[("color", 0, 0)], sample_grid)

                    # new_sf_y, new_sf_x, _, _ = pcd2depth(inputs, new_sf, round_coords=False)
                    # new_sf_pts = torch.stack([new_sf_x, new_sf_y], dim=1)

                    # knn_dists, knn_ids = find_knn(base_grid, new_sf_pts, k=4)
                    # knn_weights = F.softmax(torch.exp(-knn_dists), dim=1)
                    
                    # sample_sf = (src.points[src.isStable][knn_ids] * knn_weights[..., None]).sum(1)
                    # sample_y, sample_x, _, _ = pcd2depth(inputs, sample_sf, round_coords=False)
                    # sample_grid = torch.stack([sample_x, sample_y], dim=1).reshape(1, self.opt.height, self.opt.width, 2).type(fl32_)
                    # sample_grid[...,0] = sample_grid[...,0] / self.opt.width * 2 - 1
                    # sample_grid[...,1] = sample_grid[...,1] / self.opt.height * 2 - 1

                    # renderImg = nn.functional.grid_sample(inputs[("prev_color", 0, 0)], sample_grid)

                elif "renderer" in nets:
                    renderImg = nets["renderer"](inputs, new_data, rad=self.opt.renderer_rad).permute(2,0,1).unsqueeze(0)

            if i == 0 and self.opt.sf_corr:
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

                    # import cv2
                    # import torch.nn.functional as F
                    # ren_img = 255 * torch_to_numpy(renderImg[0].permute(1,2,0))
                    # tar_img = inputs[("color", 0, 0)]
                    # x = torch.tensor([1, 2, 3])
                    # grid_y, grid_x = torch.meshgrid(torch.arange(480), torch.arange(640), indexing='ij')
                    # grid = torch.stack([grid_x, grid_y], dim=2)[None, ...].type(fl32_).cuda()
                    # grid[...,0] = (grid[...,0] + flow[:,0]) / 320 - 1
                    # grid[...,1] = (grid[...,1] + flow[:,1]) / 240 - 1
                    # tar_img = F.grid_sample(tar_img, grid)
                    # tar_img = 255 * torch_to_numpy(tar_img[0].permute(1,2,0))
                    # cv2.imwrite(
                    #     "warp.jpg",
                    #     np.concatenate([ren_img, tar_img], axis=0)[...,::-1]
                    # )

                else:
                    if self.opt.sf_corr_match_renderimg:
                        flow = self.flow_detector(renderImg, inputs[("color", 0, 0)]) # x, y
                    else:
                        flow = self.flow_detector(inputs[("prev_color", 0, 0)], inputs[("color", 0, 0)])

            # with torch.no_grad():
            # if self.opt.sf_corr:
            #     current_sf_y, current_sf_x, _, _ = pcd2depth(inputs, new_sf, round_coords=False, valid_margin=1)

            #     grid = torch.stack(
            #         [current_sf_x * 2 / inputs["width"] - 1, 
            #         current_sf_y * 2 / inputs["height"] - 1], dim=1).view(1, -1, 1, 2).type(fl32_)
            #     trg_loc = F.grid_sample(flow, grid)[0,:,:,0]
            #     current_sf_x += trg_loc[0]
            #     current_sf_y += trg_loc[1]

            #     current_sf_flow = torch.stack([current_sf_x, current_sf_y], dim=0)

            #     valid_margin = 1
            #     current_sf_flow_valid = (current_sf_y >= valid_margin) & (current_sf_y < inputs["height"]-1-valid_margin) & \
            #         (current_sf_x >= valid_margin) & (current_sf_x < inputs["width"]-1-valid_margin)

            #     current_sf_y, current_sf_x, _, current_sf_valid = pcd2depth(inputs, new_sf, round_coords=False, valid_margin=1)
            #     # current_sf_grid = -2. * self.opt.width * torch.ones((self.opt.height, self.opt.width, 2)).cuda()
            #     # current_sf_grid[init_sf_y[init_sf_val], init_sf_x[init_sf_val], 0] = current_sf_x[init_sf_val].type(fl32_)
            #     # current_sf_grid[init_sf_y[init_sf_val], init_sf_x[init_sf_val], 1] = current_sf_y[init_sf_val].type(fl32_)
            #     # current_sf_grid[...,0] = current_sf_grid[...,0] / self.opt.width * 2 - 1
            #     # current_sf_grid[...,1] = current_sf_grid[...,1] / self.opt.height * 2 - 1
            #     # warp_back_img = nn.functional.grid_sample(inputs[("color", 0, 0)], current_sf_grid[None, ...].type(fl32_))
            #     # val_warp_back_img = (current_sf_grid[:, :, 0] >= -1) & (current_sf_grid[:, :, 0] <= 1) & (current_sf_grid[:, :, 1] >= -1) & (current_sf_grid[:, :, 1] <= 1)

            #     current_sf_loc = torch.stack([current_sf_x, current_sf_y], dim=1).view(1, -1, 1, 2).type(fl32_)
            #     current_sf_loc[...,0] = current_sf_loc[...,0] / self.opt.width * 2 - 1
            #     current_sf_loc[...,1] = current_sf_loc[...,1] / self.opt.height * 2 - 1
            #     current_sf_flow = nn.functional.grid_sample(flow, current_sf_loc.type(fl32_))[0,:,:,0]
            #     current_sf_flow[0] += current_sf_x
            #     current_sf_flow[1] += current_sf_y
                
            #     valid_margin = 1
            #     current_sf_flow_valid =  current_sf_valid & (current_sf_flow[0] >= valid_margin) & \
            #                              (current_sf_flow[0] < self.opt.width-1-valid_margin) & \
            #                              (current_sf_flow[1] >= valid_margin) & \
            #                              (current_sf_flow[1] < self.opt.height-1-valid_margin)

                # base_x, base_y = torch.meshgrid(torch.arange(self.opt.width), torch.arange(self.opt.height), indexing='xy')
                # base_grid = torch.stack([base_x, base_y], dim=2)[None, ...].type(fl32_).cuda()
                
                # flow_grid = base_grid + flow.permute(0, 2, 3, 1)
                # flow_grid[...,0] = flow_grid[...,0] / self.opt.width * 2 - 1
                # flow_grid[...,1] = flow_grid[...,1] / self.opt.height * 2 - 1
                # warp_flow_img = nn.functional.grid_sample(inputs[("color", 0, 0)], flow_grid.type(fl32_))
                # val_warp_flow_img = (flow_grid[0, :, :, 0] >= -1) & (flow_grid[0, :, :, 0] <= 1) & (flow_grid[0, :, :, 1] >= -1) & (flow_grid[0, :, :, 1] <= 1)
                
                # warp_flow_img_loss = compute_reprojection_loss(warp_flow_img, src.renderImg)

                # # TODO: better way to select which loss to optimize.
                # # sim_compare = compute_reprojection_loss(warp_back_img, inputs[("prev_color", 0, 0)]) < compute_reprojection_loss(warp_flow_img, inputs[("prev_color", 0, 0)])
                # sim_compare = compute_reprojection_loss(warp_back_img, src.renderImg) < compute_reprojection_loss(warp_flow_img, src.renderImg)
                # sim_compare = sim_compare[0, 0]
                # sf_point_plane_ids = index_map[sim_compare & (index_map >= 0) & val_warp_back_img]
                # sf_corr_ids = index_map[(~sim_compare) & (index_map >= 0) & val_warp_flow_img]

                # import cv2
                # outimg = torch.cat([inputs[("prev_color", 0, 0)][0], warp_back_img[0], warp_flow_img[0]], dim=1).permute(1,2,0).cpu().numpy()
                # cv2.imwrite("outimg.jpg", 255 * outimg)
            # else:
                # sf_point_plane_ids = None
                # sf_corr_ids = None

            """
            Mesh losses.
            """
            loss_mesh = 0.0
            
            # # Point-point loss.
            # if model_args['m-point-point'][0]:
            #     radius = 1.0
            #     src2trg_mesh_dists, _ = find_knn(new_verts, trg.points, k=5)
            #     weights = torch.exp(-src2trg_mesh_dists.detach()/(radius**2))
            #     loss_mesh += model_args['m-point-point'][1] * (weights * src2trg_mesh_dists).sum()

            # # Point-plane loss.
            # if model_args['m-point-plane'][0]:
            #     radius = 1.0
            #     src2trg_mesh_dists, src2trg_mesh_idx = find_knn(new_verts, trg.points, k=5)
            #     weights = torch.exp(-src2trg_mesh_dists.detach()/(radius**2))
            #     loss_mesh += model_args['m-point-plane'][1] * (weights * torch_inner_prod(
            #         trg.norms[src2trg_mesh_idx],
            #         new_verts.unsqueeze(1) - trg.points[src2trg_mesh_idx]
            #     )**2).sum()

            if (self.opt.mesh_edge or self.opt.mesh_face) and (self.opt.method == 'seman-super') and self.opt.use_edge_ssim_hints:
                src_boundary_edge_index = src_edge_index.permute(1, 0)[src_graph.isBoundary].reshape(-1)
                
                if i == 0:
                    # static_ssim_map = 1 - 2 * self.ssim(src.renderImg, inputs[("color", 0, 0)]).mean(1, keepdim=True)

                    init_ed_y, init_ed_x, _, _ = pcd2depth(inputs, src_graph.points, round_coords=False)
                    init_ed_y = init_ed_y[src_boundary_edge_index]
                    init_ed_x = init_ed_x[src_boundary_edge_index]

                    prev_patches = []
                    patches = []
                    window_rad = self.opt.edge_ssim_hints_window/2
                    for init_x, init_y in zip(init_ed_x, init_ed_y):
                        y1, x1 = int(init_y-window_rad), int(init_x-window_rad)
                        x2 = x1 + self.opt.edge_ssim_hints_window
                        y2 = y1 + self.opt.edge_ssim_hints_window
                        prev_patches.append(F.pad(inputs[("prev_color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
                                                (min(max(0, -x1), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, x2-self.opt.width), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, -y1), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, y2-self.opt.height),self.opt.edge_ssim_hints_window))
                                            ))
                        if not (prev_patches[-1].size(1) == 6 and prev_patches[-1].size(2)==6):
                            print(inputs[("color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)].size(), x1, x2, y1, y2, 
                            init_y, init_x, window_rad, min(max(0, -x1), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, x2-self.opt.width), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, -y1), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, y2-self.opt.height),self.opt.edge_ssim_hints_window))
                        
                        patches.append(F.pad(inputs[("color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
                                            (min(max(0, -x1), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, x2-self.opt.width), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, -y1), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, y2-self.opt.height),self.opt.edge_ssim_hints_window))
                                            ))

                    prev_patches = torch.stack(prev_patches, dim=0)
                    patches = torch.stack(patches, dim=0)

                current_ed_y, current_ed_x, _, _ = pcd2depth(inputs, new_verts, round_coords=False)
                current_ed_y = current_ed_y[src_boundary_edge_index]
                current_ed_x = current_ed_x[src_boundary_edge_index]
                current_patches = []
                for current_x, current_y in zip(current_ed_x, current_ed_y):
                    # y1, y2, x1, x2 = int(current_y-window_rad), int(current_y+window_rad), int(current_x-window_rad), int(current_x+window_rad)
                    y1, x1 = int(current_y-window_rad), int(current_x-window_rad)
                    x2 = x1 + self.opt.edge_ssim_hints_window
                    y2 = y1 + self.opt.edge_ssim_hints_window
                    current_patches.append(F.pad(inputs[("color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)],
                                                (min(max(0, -x1), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, x2-self.opt.width), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, -y1), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, y2-self.opt.height),self.opt.edge_ssim_hints_window))
                                            ))
                    if not (current_patches[-1].size(1) == 6 and current_patches[-1].size(2)==6):
                        print(inputs[("color", 0, 0)][0, :, min(max(y1,0),self.opt.height): min(max(y2,0),self.opt.height), min(max(x1,0),self.opt.width): min(max(x2,0),self.opt.width)].size(), -x1, x2-self.opt.width, -y1, y2-self.opt.height, 
                        current_y, current_x, window_rad, min(max(0, -x1), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, x2-self.opt.width), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, -y1), self.opt.edge_ssim_hints_window), 
                                                 min(max(0, y2-self.opt.height),self.opt.edge_ssim_hints_window))
                current_patches = torch.stack(current_patches, dim=0)

                static_ssim = 1 - 2 * self.ssim(prev_patches, patches).mean(1)[:, 1, 1].reshape(-1, 2)
                moving_ssim = 1 - 2 * self.ssim(prev_patches, current_patches).mean(1)[:, 1, 1].reshape(-1, 2)

                static_boundary_edge = torch.any(static_ssim > moving_ssim, dim=1)

                # moving_dists = torch.sqrt((current_ed_x - init_ed_x)**2 + (current_ed_y - init_ed_y)**2).reshape(-1, 2)
                # static_boundary_edge &= torch.any(moving_dists > 5, dim=1)
                new_edges_lens = torch.norm(new_verts[src_edge_index[0]][src_graph.isBoundary] - 
                                            new_verts[src_edge_index[1]][src_graph.isBoundary], dim=1)
                # moving_dists = torch.abs(new_edges_lens - src_graph.edges_lens[src_graph.isBoundary])
                static_boundary_edge &= (new_edges_lens > 0.2) # (moving_dists > 2e-2)

                moving_boundary_edge = 1 - static_boundary_edge.type(fl32_)

                boundary_edge_type = src_graph.boundary_edge_type
                boundary_edge_type = torch.cat([boundary_edge_type[:, 1:], moving_boundary_edge[:, None]], dim=1)
                
            # if (self.opt.mesh_edge or self.opt.mesh_face) and (self.opt.method == 'seman-super') and self.opt.use_edge_ssim_hints:
            #     if i == 0:
            #         static_ssim_map = 1 - 2 * self.ssim(src.renderImg, inputs[("color", 0, 0)]).mean(1, keepdim=True)

            #         init_ed_y, init_ed_x, _, _ = pcd2depth(inputs, src_graph.points, round_coords=False)
            #         ori_init_grid = torch.stack([init_ed_x, init_ed_y], dim=1) # Nx2
            #         init_grid = ori_init_grid.clone()
            #         init_grid[...,0] = init_grid[...,0] / self.opt.width * 2 - 1
            #         init_grid[...,1] = init_grid[...,1] / self.opt.height * 2 - 1
            #         static_ssim = nn.functional.grid_sample(static_ssim_map, init_grid[None, None, :, :].type(fl32_))[0, 0, 0]

            #     moving_ssim_map = 1 - 2 * self.ssim(renderImg, inputs[("color", 0, 0)]).mean(1, keepdim=True)

            #     current_ed_y, current_ed_x, _, _ = pcd2depth(inputs, new_verts, round_coords=False)
            #     ori_current_grid = torch.stack([current_ed_x, current_ed_y], dim=1) # Nx2
            #     current_grid = ori_current_grid.clone()
            #     current_grid[...,0] = current_grid[...,0] / self.opt.width * 2 - 1
            #     current_grid[...,1] = current_grid[...,1] / self.opt.height * 2 - 1
            #     moving_ssim = nn.functional.grid_sample(moving_ssim_map, current_grid[None, None, :, :].type(fl32_))[0, 0, 0]
                
            #     static_ed_nodes = (static_ssim > moving_ssim) & (torch.norm(ori_init_grid-ori_current_grid, dim=1) >= 5)
            #     static_boundary_edge = static_ed_nodes[src_edge_index[0]] | static_ed_nodes[src_edge_index[1]]
            #     moving_boundary_edge = 1 - static_boundary_edge.type(fl32_)[src_graph.isBoundary]

            #     boundary_edge_type = src_graph.boundary_edge_type
            #     boundary_edge_type = torch.cat([boundary_edge_type[:, 1:], moving_boundary_edge[:, None]], dim=1)

            # if (self.opt.mesh_edge or self.opt.mesh_face) and (self.opt.method == 'seman-super') and self.opt.use_edge_ssim_hints and i == Niter - 1:
            #     with torch.no_grad():
            #         static_ssim_map = 1 - 2 * self.ssim(src.renderImg, inputs[("color", 0, 0)]).mean(1, keepdim=True)
            #         # static_ssim_map = 1 - 2 * self.ssim(inputs[("prev_color", 0, 0)], inputs[("color", 0, 0)]).mean(1, keepdim=True)
            #         # static_ssim_map = compute_reprojection_loss(inputs[("prev_color", 0, 0)], inputs[("color", 0, 0)])

            #         init_ed_y, init_ed_x, _, _ = pcd2depth(inputs, src_graph.points, round_coords=False)
            #         init_grid = torch.stack([init_ed_x, init_ed_y], dim=1) # Nx2
            #         init_grid[...,0] = init_grid[...,0] / self.opt.width * 2 - 1
            #         init_grid[...,1] = init_grid[...,1] / self.opt.height * 2 - 1
            #         static_ssim = nn.functional.grid_sample(static_ssim_map, init_grid[None, None, :, :].type(fl32_))[0, 0, 0]


            #         moving_ssim_map = 1 - 2 * self.ssim(renderImg, inputs[("color", 0, 0)]).mean(1, keepdim=True)

            #         current_ed_y, current_ed_x, _, _ = pcd2depth(inputs, new_verts, round_coords=False)
            #         current_grid = torch.stack([current_ed_x, current_ed_y], dim=1) # Nx2
            #         current_grid[...,0] = current_grid[...,0] / self.opt.width * 2 - 1
            #         current_grid[...,1] = current_grid[...,1] / self.opt.height * 2 - 1
            #         moving_ssim = nn.functional.grid_sample(moving_ssim_map, current_grid[None, None, :, :].type(fl32_))[0, 0, 0]
                    
            #         # init_sf_y, init_sf_x, _, init_sf_val = pcd2depth(inputs, src.points[src.isStable], round_coords=True)
            #         # current_sf_y, current_sf_x, _, _ = pcd2depth(inputs, new_sf, round_coords=False)
            #         # current_sf_grid = -2. * self.opt.width * torch.ones((self.opt.height, self.opt.width, 2)).cuda()
            #         # current_sf_grid[init_sf_y[init_sf_val], init_sf_x[init_sf_val], 0] = current_sf_x[init_sf_val].type(fl32_)
            #         # current_sf_grid[init_sf_y[init_sf_val], init_sf_x[init_sf_val], 1] = current_sf_y[init_sf_val].type(fl32_)
            #         # current_sf_grid[...,0] = current_sf_grid[...,0] / self.opt.width * 2 - 1
            #         # current_sf_grid[...,1] = current_sf_grid[...,1] / self.opt.height * 2 - 1
            #         # warp_back_img = nn.functional.grid_sample(inputs[("color", 0, 0)], current_sf_grid[None, ...].type(fl32_))
            #         # moving_ssim_map = 1 - 2 * self.ssim(inputs[("prev_color", 0, 0)], warp_back_img).mean(1, keepdim=True)
            #         # # moving_ssim_map = compute_reprojection_loss(inputs[("prev_color", 0, 0)], warp_back_img)
            #         # moving_ssim = nn.functional.grid_sample(moving_ssim_map, init_grid[None, None, :, :].type(fl32_))[0, 0, 0]

            #         # static_ed_nodes = static_ssim < moving_ssim
            #         static_ed_nodes = static_ssim > moving_ssim

            # Edge loss.
            if self.opt.mesh_edge:
                # edge_losses = self.opt.mesh_edge_weight * \
                #     ((torch_distance(
                #         new_verts[src_edge_index[0]], new_verts[src_edge_index[1]])
                #         - src_graph.edges_lens) ** 2)
                new_edges_lens = torch.norm(new_verts[src_edge_index[0]] - new_verts[src_edge_index[1]], dim=1)
                edge_losses = self.opt.mesh_edge_weight * ((new_edges_lens - src_graph.edges_lens) ** 2)

                # if self.opt.method == 'seman-super':
                #     weights = torch.where(src_graph.isBoundary, torch.exp(- 0.1 * new_edges_lens), 1.)
                #     # weights = torch.where(src_graph.isBoundary, 1. / (1. + torch.exp(- 0.1 * new_edges_lens)), 1.)
                    
                #     # weights = torch.where(src_graph.isTool, 10., weights)
                    
                #     edge_losses = weights * edge_losses

                #     # edge_losses = edge_losses[src_graph.seman_edge_val]
                
                if self.opt.method == 'seman-super' and self.opt.use_edge_ssim_hints:
                    connected_edges = torch.ones_like(src_graph.isBoundary)
                    connected_edges[src_graph.isBoundary] = boundary_edge_type.mean(1) >= 0.5
                    # connected_edges[src_graph.isBoundary] = src_graph.boundary_edge_type.mean(1) >= 0.5
                    edge_losses = edge_losses[connected_edges]

                if reduction == "mean":
                    edge_losses = edge_losses.mean()
                elif reduction == "sum":
                    edge_losses = edge_losses.sum()
                else:
                    assert False
                loss_mesh += edge_losses
                losses["edge_loss"] = edge_losses

            if self.opt.mesh_face:
                new_triangles_areas = torch.cross(new_verts[src_graph.triangles[:, 1]] - new_verts[src_graph.triangles[:, 0]],
                                        new_verts[src_graph.triangles[:, 2]] - new_verts[src_graph.triangles[:, 0]],
                                        dim=1)
                new_triangles_areas = 0.5 * torch.sqrt((new_triangles_areas**2).sum(1) + 1e-13)
                
                face_losses = self.opt.mesh_face_weight * \
                    ((new_triangles_areas - src_graph.triangles_areas) ** 2)

                # if self.opt.method == 'seman-super' and self.opt.use_edge_ssim_hints:
                #     if i == 0:
                #         inside_faces = (src_graph.seman[src_graph.triangles[:, 0]] == src_graph.seman[src_graph.triangles[:, 1]]) & \
                #                     (src_graph.seman[src_graph.triangles[:, 0]] == src_graph.seman[src_graph.triangles[:, 2]])
                #         connected_faces = inside_faces | \
                #                             (~ (src_graph.static_ed_nodes[src_graph.triangles[:, 0]] | \
                #                                     src_graph.static_ed_nodes[src_graph.triangles[:, 1]] | \
                #                                     src_graph.static_ed_nodes[src_graph.triangles[:, 2]]))
                #     face_losses = face_losses[connected_faces]

                if reduction == "mean":
                    face_losses = face_losses.mean()
                elif reduction == "sum":
                    face_losses = face_losses.sum()
                else:
                    assert False
                loss_mesh += face_losses
                losses["face_losses"] = face_losses

            # if model_args['m-point-point'][0]:
            #     loss_mesh = model_args['m-point-point'][1] * (fcorr * torch_sq_distance(
            #         new_verts.unsqueeze(1), trg.points.unsqueeze(0))
            #         ).sum() # Point-point loss.

            # if model_args['m-point-plane'][0]:
            #     loss_mesh = model_args['m-point-plane'][1] * (fcorr * torch_inner_prod(
            #         trg.norms.unsqueeze(0), 
            #         new_verts.unsqueeze(1) - trg.points.unsqueeze(0)
            #         )**2).sum() # Point-plane loss.

            # Regularization terms.
            if self.opt.mesh_arap:
                arap_loss = self.opt.mesh_arap_weight * \
                    ARAPLoss.autograd_forward(src_graph, deform_verts[:-1, :], reduction=reduction)
                loss_mesh += arap_loss

                losses["arap_loss"] = arap_loss

            if self.opt.mesh_rot:
                rot_loss = self.opt.mesh_rot_weight * RotLoss.autograd_forward(deform_verts, reduction=reduction)
                loss_mesh += rot_loss

                losses["rot_loss"] = rot_loss

            if self.opt.bn_morph:
                # render_seman = nets["renderer"](
                #     inputs, 
                #     Data(points=new_sf, colors=src.seman_conf[src.isStable]), 
                #     rad=self.opt.renderer_rad) # H x W x #cls
                # render_seman = torch.argmax(render_seman, dim=2)[None, None, ...]

                # render_seman_grad_bin = self.tool.get_semanticsEdge(render_seman) # B x 1 x H x W
                # seman_grad_bin = self.tool.get_semanticsEdge(inputs[("seman", 0)])
                
                with torch.no_grad():
                    # Project ED nodes to the image plane.
                    ED_nodes_y, ED_nodes_x, _, _ = pcd2depth(inputs, new_verts, round_coords=False)
                    ED_nodes_pts = torch.stack([ED_nodes_x, ED_nodes_y], dim=1)
                    morph_th = torch.zeros(src_graph.num, dtype=fl64_).cuda()
                    morph_gt = torch.zeros((src_graph.num, 2), dtype=fl64_).cuda()

                    grid = copy.deepcopy(ED_nodes_pts)
                    grid[...,0] = grid[...,0] / self.opt.width * 2 - 1
                    grid[...,1] = grid[...,1] / self.opt.height * 2 - 1
                    ed_seman = nn.functional.grid_sample(inputs[("seman_conf", 0)], grid[None, :, None, :])[0, :, :, 0]
                    ed_seman = torch.argmax(ed_seman, dim=0)
                    # True: edge points / points need push to edge.
                    # morph_val = src_graph.isBoundary | (~(ed_seman==src_graph.seman))
                    morph_val = ~(ed_seman==src_graph.seman)

                    kernels = [3, 3, 3]
                    for class_id in range(self.opt.num_classes):
                        seman_grad_bin = self.tool.get_semanticsEdge(
                            inputs[("seman", 0)], foregroundType=[class_id],
                            erode_foreground=True, kernel_size=kernels[class_id])
                        edge_y, edge_x = seman_grad_bin[0,0].nonzero(as_tuple=True)
                        edge_pts = torch.stack([edge_x, edge_y], dim=1).type(fl64_)

                        _ED_nodes_ids_ = src_graph.seman == class_id
                        _ED_nodes_pts_ = ED_nodes_pts[_ED_nodes_ids_]
                        _dists_, _sort_idx_ = find_knn(_ED_nodes_pts_, edge_pts, k=2)
                        
                        morph_th[_ED_nodes_ids_] = _dists_[:, 1]
                        morph_gt[_ED_nodes_ids_] = edge_pts[_sort_idx_[:, 1]]

                    valid_margin = 1
                    morph_val &= (morph_gt[:,1] >= valid_margin) & \
                                 (morph_gt[:,1] < inputs["width"]-1-valid_margin) & \
                                 (morph_gt[:,0] >= valid_margin) & \
                                 (morph_gt[:,1] < inputs["height"]-1-valid_margin)

                if torch.any(morph_val):
                    src_graph_seman_conf = src_graph.seman_conf[torch.arange(len(src_graph.seman)), src_graph.seman]

                    ################### Code to visualize the tracking process ####################
                    import cv2
                    ed_v, ed_u, _, _ = pcd2depth(inputs, new_verts, round_coords=False)
                    img = 255 * torch_to_numpy(inputs[("color", 0, 0)][0].permute(1,2,0))[...,::-1]
                    img = img.astype(np.uint8).copy()
                    colors = [(255,0,0),
                            (0,255,0),
                            (0,0,255)]
                    ed_colors = [(150,0,0),
                                (0,150,0),
                                (0,0,150)]
                    for _u_, _v_, _ed_u_, _ed_v_, class_id in \
                    zip(morph_gt[:,0][morph_val], morph_gt[:,1][morph_val], \
                    ed_u[morph_val], ed_v[morph_val], src_graph.seman[morph_val]):
                        img = cv2.rectangle(img, 
                            (int(_u_ - 5), int(_v_ - 5)), (int(_u_ + 5), int(_v_ + 5)),
                            colors[class_id], -1)

                        img = cv2.arrowedLine(img, 
                            (int(_ed_u_+0.3*(_u_-_ed_u_)), int(_ed_v_+0.3*(_v_-_ed_v_))), 
                            (int(_ed_u_+0.6*(_u_-_ed_u_)), int(_ed_v_+0.6*(_v_-_ed_v_))),
                            colors[class_id], 1)

                        img = cv2.circle(img, (int(_ed_u_), int(_ed_v_)), 4, (255, 255, 255), -1)
                        img = cv2.circle(img, (int(_ed_u_), int(_ed_v_)), 3, ed_colors[class_id], -1)

                    for _u_, _v_, class_id, knn_id in zip(ed_u[~morph_val], ed_v[~morph_val], src_graph.seman[~morph_val], src_graph.knn_indices[~morph_val]):
                        img = cv2.circle(img, (int(_u_), int(_v_)), 4, ed_colors[class_id], -1)

                    cv2.imwrite("bnmorph.jpg", img)
                    ################### Code to visualize the tracking process ####################

                    morph_gt_scaled = copy.deepcopy(morph_gt)
                    morph_gt_scaled[:, 0] /= self.opt.width
                    morph_gt_scaled[:, 1] /= self.opt.height
                    
                    bn_morph_loss = torch_sq_distance(
                        torch.stack([ed_u/self.opt.width, ed_v/self.opt.height], dim=1)[morph_val], morph_gt_scaled[morph_val])
                    if reduction == "sum":
                        bn_morph_loss = self.opt.bn_morph_weight * bn_morph_loss.sum()
                    elif reduction == "mean":
                        bn_morph_loss = self.opt.bn_morph_weight * bn_morph_loss.mean()
                    else:
                        assert False

                    loss_mesh += bn_morph_loss
                    losses["bn_morph_loss"] = bn_morph_loss

            if self.opt.sf_bn_morph:
                # Project surfels to the image plane.
                sf_y, sf_x, _, _ = pcd2depth(inputs, new_sf, round_coords=False)
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
                    
                sf_morph_val = (~(new_sf_seman==sf_seman)) & \
                               (new_sf_grid[...,0] >= -1) & (new_sf_grid[...,0] <= 1) & \
                               (new_sf_grid[...,1] >= -1) & (new_sf_grid[...,1] <= 1)

                sorted_sf_seman, sf_sort_indices = torch.sort(sf_seman)
                sorted_ori_new_sf_grid = ori_new_sf_grid[sf_sort_indices]
                sorted_sf_morph_val = sf_morph_val[sf_sort_indices]

                sf_morph_gt = []
                kernels = [3, 3, 3]
                if i == 0:
                    edge_pts = []
                    for class_id in range(self.opt.num_classes):
                        seman_grad_bin = self.tool.get_semanticsEdge(
                            inputs[("seman", 0)], foregroundType=[class_id],
                            erode_foreground=True, kernel_size=kernels[class_id])
                        edge_y, edge_x = seman_grad_bin[0,0].nonzero(as_tuple=True)
                        edge_pts.append(torch.stack([edge_x, edge_y], dim=1).type(fl64_))

                for class_id in range(self.opt.num_classes):
                    _, knn_edge_ids = find_knn(sorted_ori_new_sf_grid[sorted_sf_seman == class_id], 
                                                   edge_pts[class_id], k=1)
                    sf_morph_gt.append(edge_pts[class_id][knn_edge_ids])
                sf_morph_gt = torch.cat(sf_morph_gt, dim=0).mean(1)

                valid_margin = 1
                sorted_sf_morph_val &= (sf_morph_gt[:,0] >= valid_margin) & \
                                (sf_morph_gt[:,0] < inputs["width"]-1-valid_margin) & \
                                (sf_morph_gt[:,1] >= valid_margin) & \
                                (sf_morph_gt[:,1] < inputs["height"]-1-valid_margin)
                # sf_morph_gt[...,0] = sf_morph_gt[...,0] / self.opt.width * 2 - 1       
                # sf_morph_gt[...,1] = sf_morph_gt[...,1] / self.opt.height * 2 - 1 

                # with torch.no_grad():
                #     sf_y, sf_x, _, _ = pcd2depth(inputs, new_sf, round_coords=False)
                #     sf_pts = torch.stack([sf_x, sf_y], dim=1)
                #     sf_morph_gt = torch.zeros((len(new_sf), 2), dtype=fl64_).cuda()

                #     grid = copy.deepcopy(sf_pts)
                #     grid[...,0] = grid[...,0] / self.opt.width * 2 - 1
                #     grid[...,1] = grid[...,1] / self.opt.height * 2 - 1
                #     new_sf_seman = nn.functional.grid_sample(inputs[("seman_conf", 0)], grid[None, :, None, :])[0, :, :, 0]
                #     # print(torch.unique())
                #     new_sf_seman = torch.argmax(new_sf_seman, dim=0)
                #     # True: edge points / points need push to edge.
                #     # print(torch.min(grid), torch.max(grid))
                #     # sf_morph_val = (~(new_sf_seman==sf_seman)) & torch.any(new_sf_seman > 0, dim=0)
                #     sf_morph_val = (~(new_sf_seman==sf_seman)) & (grid[...,0] >- 1) & (grid[...,0] < 1) & (grid[...,1] >- 1) & (grid[...,1] < 1)

                #     kernels = [3, 3, 3]
                #     for class_id in range(self.opt.num_classes):
                #         seman_grad_bin = self.tool.get_semanticsEdge(
                #             inputs[("seman", 0)], foregroundType=[class_id],
                #             erode_foreground=True, kernel_size=kernels[class_id])
                #         edge_y, edge_x = seman_grad_bin[0,0].nonzero(as_tuple=True)
                #         edge_pts = torch.stack([edge_x, edge_y], dim=1).type(fl64_)

                #         _sf_ids_ = sf_seman == class_id
                #         _sf_pts_ = sf_pts[_sf_ids_]
                #         _dists_, _sort_idx_ = find_knn(_sf_pts_, edge_pts, k=1)
                        
                #         sf_morph_gt[_sf_ids_] = edge_pts[_sort_idx_[:, 0]]

                #     valid_margin = 1
                #     sf_morph_val &= (sf_morph_gt[:,0] >= valid_margin) & \
                #                  (sf_morph_gt[:,0] < inputs["width"]-1-valid_margin) & \
                #                  (sf_morph_gt[:,1] >= valid_margin) & \
                #                  (sf_morph_gt[:,1] < inputs["height"]-1-valid_margin)

                if torch.any(sf_morph_val):
                    # sf_morph_gt_scaled = copy.deepcopy(sf_morph_gt)
                    sf_morph_gt_scaled = sf_morph_gt.clone()
                    sf_morph_gt_scaled[:, 0] /= self.opt.width
                    sf_morph_gt_scaled[:, 1] /= self.opt.height

                    # sf_y, sf_x, _, _ = pcd2depth(inputs, new_sf, round_coords=False)
                    # sf_morph_pred_scaled = torch.stack([sf_x, sf_y], dim=1)
                    sf_morph_pred_scaled = sorted_ori_new_sf_grid.clone()
                    sf_morph_pred_scaled[:, 0] /= self.opt.width
                    sf_morph_pred_scaled[:, 1] /= self.opt.height

                    # print(sf_morph_pred_scaled[sorted_sf_morph_val], sf_morph_gt_scaled[sorted_sf_morph_val])

                    sf_bn_morph_loss = torch_sq_distance(
                        sf_morph_pred_scaled[sorted_sf_morph_val], sf_morph_gt_scaled[sorted_sf_morph_val])
                    if reduction == "sum":
                        sf_bn_morph_loss = self.opt.sf_bn_morph_weight * sf_bn_morph_loss.sum()
                    elif reduction == "mean":
                        sf_bn_morph_loss = self.opt.sf_bn_morph_weight * sf_bn_morph_loss.mean()
                    else:
                        assert False

                    loss_mesh += sf_bn_morph_loss
                    losses["sf_bn_morph_loss"] = sf_bn_morph_loss
                
            """
            Surfel losses.
            """
            loss_surfels = 0.0

            assert not (self.opt.sf_point_plane and self.opt.sf_hard_seman_point_plane and self.opt.sf_soft_seman_point_plane), \
                "Need to choose one of them: sf_point_plane, sf_hard_seman_point_plane, sf_soft_seman_point_plane."
            if self.opt.sf_point_plane or self.opt.sf_hard_seman_point_plane or self.opt.sf_soft_seman_point_plane:

                if self.opt.sf_hard_seman_point_plane or self.opt.sf_soft_seman_point_plane:
                    point_plane_loss = self.opt.sf_point_plane_weight * \
                        DataLoss.autograd_forward(self.opt, inputs, new_data, trg, color_hint=self.opt.use_color_hints,
                            src_seman=sf_seman, src_seman_conf=sf_seman_conf, soft_seman=self.opt.sf_soft_seman_point_plane,
                            reduction=reduction)
                else:
                    point_plane_loss = self.opt.sf_point_plane_weight * \
                        DataLoss.autograd_forward(self.opt, inputs, new_data, trg, color_hint=self.opt.use_color_hints, 
                                                  reduction=reduction)

                loss_surfels += point_plane_loss
                losses["point_plane_loss"] = point_plane_loss

            if self.opt.sf_corr:
                # current_sf_y, current_sf_x, _, _ = pcd2depth(inputs, new_sf, round_coords=False, valid_margin=1)

                # current_sf_grid = torch.stack(
                #     [current_sf_x * 2 / inputs["width"] - 1, 
                #     current_sf_y * 2 / inputs["height"] - 1], dim=1).view(1, -1, 1, 2).type(fl32_)
                # trg_loc = F.grid_sample(flow, current_sf_grid)[0,:,:,0]
                # current_sf_x += trg_loc[0]
                # current_sf_y += trg_loc[1]

                # current_sf_flow = torch.stack([current_sf_x, current_sf_y], dim=0)

                # valid_margin = 1
                # current_sf_flow_valid = (current_sf_y >= valid_margin) & (current_sf_y < inputs["height"]-1-valid_margin) & \
                #     (current_sf_x >= valid_margin) & (current_sf_x < inputs["width"]-1-valid_margin)

                # base_x, base_y = torch.meshgrid(torch.arange(self.opt.width), torch.arange(self.opt.height), indexing='xy')
                # base_grid = torch.stack([base_x, base_y], dim=2)[None, ...].type(fl32_).cuda()
                
                # flow_grid = base_grid + flow.permute(0, 2, 3, 1)
                # flow_grid[...,0] = flow_grid[...,0] / self.opt.width * 2 - 1
                # flow_grid[...,1] = flow_grid[...,1] / self.opt.height * 2 - 1
                # warp_flow_img = nn.functional.grid_sample(inputs[("color", 0, 0)], flow_grid.type(fl32_))
                # val_warp_flow_img = (flow_grid[0, :, :, 0] >= -1) & (flow_grid[0, :, :, 0] <= 1) & (flow_grid[0, :, :, 1] >= -1) & (flow_grid[0, :, :, 1] <= 1)

                # warp_flow_img_loss = compute_reprojection_loss(warp_flow_img, src.renderImg)
                # warp_flow_img_loss = nn.functional.grid_sample(warp_flow_img_loss, current_sf_grid)

                # warp_losses = [warp_flow_img_loss[0, 0, :, 0]]
                # warp_correspts = [current_sf_flow]
                # warp_correspts_valid = [current_sf_flow_valid]

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
                    init_sf_y, init_sf_x, _, _ = pcd2depth(inputs, src.points, round_coords=False, valid_margin=1)
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
                    # good_ids = torch.zeros_like(good_ids)
                    target_grid = F.grid_sample(target_grid.permute(0, 3, 1, 2), source_grid)[:,:,0,:].permute(0, 2, 1)
                    current_sf_flow = torch.where(
                                        good_ids[:, None].repeat(1, 2)==0, 
                                        target_grid[0], 
                                        target_grid[1]
                                    )
                    val_losses = torch.any(warp_losses < 100., dim=0)
                    current_sf_flow_valid = torch.any(warp_losses < 100., dim=0) & (current_sf_flow[:, 1] >= valid_margin) & \
                                            (current_sf_flow[:, 1] < inputs["height"]-1-valid_margin) & \
                                            (current_sf_flow[:, 0] >= valid_margin) & \
                                            (current_sf_flow[:, 0] < inputs["width"]-1-valid_margin)

                    current_sf_flow = current_sf_flow[src.isStable.nonzero(as_tuple=True)[0]].permute(1, 0)
                    current_sf_flow_valid = current_sf_flow_valid[src.isStable]

                    loss_corr = self.opt.sf_corr_weight * \
                        DataLoss.autograd_forward(self.opt, inputs, new_data, trg, 
                            correspts=current_sf_flow, correspts_valid=current_sf_flow_valid,
                            loss_type=self.opt.sf_corr_loss_type, reduction=reduction)

                else:
                    if self.opt.sf_hard_seman_corr or self.opt.sf_soft_seman_corr:
                        loss_corr = self.opt.sf_corr_weight * \
                            DataLoss.autograd_forward(self.opt, inputs, new_data, trg, 
                                flow=flow, huber_th=self.opt.sf_corr_huber_th,
                                src_seman=sf_seman, src_seman_conf=sf_seman_conf, soft_seman=self.opt.sf_soft_seman_corr,
                                loss_type=self.opt.sf_corr_loss_type, reduction=reduction)
                    else:
                        loss_corr = self.opt.sf_corr_weight * \
                            DataLoss.autograd_forward(self.opt, inputs, new_data, trg, 
                                flow=flow, loss_type=self.opt.sf_corr_loss_type, reduction=reduction)
                    # huber_th=self.opt.sf_corr_huber_th

                loss_surfels += loss_corr
                losses["corr_loss"] = loss_corr
            
            # TODO: Try mesh normal consistency loss and mesh laplacian smoothing loss.
            # loss_normal = mesh_normal_consistency(new_src_mesh)
            # loss_laplacian = mesh_laplacian_smoothing(full_new_src_mesh, method="uniform")

            """
            Appearance loss.
            """
            render_loss = 0.
            if self.opt.render_loss:
                
                if self.opt.renderer == "grid_sample":
                    # diff_render_loss = self.opt.render_loss_weight * torch.abs(target_color - source_color).mean(0)
                    diff_render_loss = self.opt.render_loss_weight * torch.abs(target_color - source_color).mean(0)**2
                elif self.opt.renderer == "warp":
                    # diff_render_loss = self.opt.render_loss_weight * compute_reprojection_loss(warpbackImg, inputs[("prev_color", 0, 0)])
                    diff_render_loss = self.opt.render_loss_weight * torch.abs(warpbackImg - inputs[("prev_color", 0, 0)]).mean(1)**2
                else:
                    diff_render_loss = self.opt.render_loss_weight * compute_reprojection_loss(renderImg, inputs[("color", 0, 0)])
                    # diff_render_loss = self.opt.render_loss_weight * torch.abs(renderImg - inputs[("color", 0, 0)]).mean(1)**2
                
                if reduction == "mean":
                    diff_render_loss = diff_render_loss.mean()
                elif reduction == "sum":
                    diff_render_loss = diff_render_loss.sum()
                else:
                    assert False
                losses["render_loss"] = diff_render_loss
                render_loss += diff_render_loss

            # Weighted sum of the losses
            loss = loss_mesh + loss_surfels + render_loss
            if i == Niter-1:
                print_text = f"[frame{trg.time.item()}]"
                for key in losses.keys():
                    print_text += f"{key}: {losses[key]}    " #:0.5f
                src.logger.info(print_text)
                
            # Optimization step.
            loss.backward(retain_graph=True)
            # nn.utils.clip_grad_norm_([deform_verts], 1.)
            optimizer.step()
            torch.cuda.empty_cache()

            # ### seman-super: optimize the boundary connection values. ###
            # if self.opt.method == 'seman-super' and self.opt.mesh_edge:
            #     edge_optimizer.zero_grad()
            
            #     boundary_edges = ~src_graph.inside_edges
            #     edge_losses = self.opt.mesh_edge_weight * torch.exp(- 10 * src_graph.edges_lens[boundary_edges]) * \
            #                     ((torch_distance(new_verts[src_edge_index[0][boundary_edges]], 
            #                         new_verts[src_edge_index[1][boundary_edges]])
            #                     - src_graph.edges_lens[boundary_edges]) ** 2)

            #     edge_losses = 1e10 * nn.functional.sigmoid(connected_boundary_edges) * edge_losses

            #     if reduction == "mean":
            #         edge_losses = edge_losses.mean()
            #     elif reduction == "sum":
            #         edge_losses = edge_losses.sum()
            #     else:
            #         assert False

            #     edge_regularization_losses = (nn.functional.sigmoid(connected_boundary_edges) - 
            #                                     torch.ones_like(connected_boundary_edges))**2
            #     if reduction == "mean":
            #         edge_regularization_losses = edge_regularization_losses.mean()
            #     elif reduction == "sum":
            #         edge_regularization_losses = edge_regularization_losses.sum()
            #     else:
            #         assert False
            #     edge_losses += edge_regularization_losses

            #     edge_losses.backward(retain_graph=True)
            #     # print("GRAD: ", connected_boundary_edges.grad)
            #     edge_optimizer.step()
            #     torch.cuda.empty_cache()

        return deform_verts, boundary_edge_type