import datetime
import cv2
import copy
import open3d as o3d

import pandas as pd
from pyntcloud import PyntCloud

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from super.utils import *

from utils.config import *
from utils.utils import *
from utils.labels import id2color
from utils.data_loader import SSIM

from seg.evaluate import general_dice, general_jaccard

class Surfels():

    def __init__(self, opt, models, inputs, data):
        self.opt = opt
        self.models = models
        self.evaluate_tracking = self.opt.tracking_gt_file is not None
        self.mssim = []
        
        if self.opt.method == "seman-super":
            self.power_arg = (1/2, 1/2) # (2/3, 1/3)
        
        if self.opt.phase == 'test':
            self.output_dir = os.path.join(self.opt.sample_dir, f"model{self.opt.mod_id}_exp{self.opt.exp_id}")
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        time_stamp = datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
        custom_log_manager.setup(
            to_file=not self.opt.nologfile, 
            log_prefix=f"{self.opt.method}_exp{self.opt.mod_id}_log",
            time_stamp=time_stamp, 
            logdir=self.output_dir)
        self.logger = custom_log_manager.get_logger('test')
        self.logger.info("Initiate surfels and ED nodes ...: ")
        for arg in vars(self.opt):
            self.logger.info(f"{arg}: {str(getattr(self.opt, arg))}")
        self.logger.info("\n")

        if self.evaluate_tracking:
            tracking_gt_file = os.path.join(self.opt.data_dir, self.opt.tracking_gt_file)
            self.track_pts = np.array(np.load(tracking_gt_file, allow_pickle=True)).tolist()
            for key in self.track_pts.keys():
                for filename in self.track_pts[key]:
                    self.track_pts[key][filename] = numpy_to_torch(self.track_pts[key][filename], dtype=int_)
            
            track_gt = self.track_pts["gt"]
            self.track_num = len(list(track_gt.items())[0][1])
            self.track_id = - torch.ones((self.track_num,), dtype=long_).cuda()
            # -1: Haven't started tracking; 
            # >=0: ID of tracked points; 
            # -2: Lost tracking.
            self.track_rsts = {}

        if self.opt.load_seman_gt:
            self.result_dice = []
            self.result_jaccard = []

        for key, v in data.items():
            if key == 'valid':
                continue
            setattr(self, key, v)
        self.sf_num = len(self.points)
        self.isStable = torch.ones((self.sf_num,), dtype=bool_).cuda()
        self.time_stamp = self.time * torch.ones(self.sf_num).cuda()

        # self.validmap = self.valid.view(HEIGHT,WIDTH)
        # valid_indexs = torch.arange(self.sf_num, dtype=int, device=dev).unsqueeze(1)
        # valid_coords = torch.flip(self.validmap.nonzero(), dims=[-1])
        # # Column direction of self.projdata:
        # # 1) index of projected points in self.points, 
        # # corresponding projection coordinates (x,y) on the image plane 
        # self.projdata = torch.cat([valid_indexs,valid_coords], dim=1).type(tfdtype_)

        self.projdata = torch.flip(
            data.valid.view(inputs["height"], inputs["width"]).nonzero(), 
            dims=[-1]).type(fl32_)

        # init ED nodes
        self.update_ED()
        self.logger.info(f" Number of parameters: {self.ED_nodes.param_num}")
        self.logger.info(f" Average ED graph edge length: {self.ED_nodes.edges_lens.mean()/0.1*5} mm")

        # if hasattr(new_data,'x'):
        #     if model_args['method'] == 'super':
        #         pass
        #     elif model_args['method'] == 'dlsuper':
        #         v, u, _, _ = pcd2depth(self.points, round_coords=False, valid_margin=1)
        #         # sh, sw = new_data.x.size()[2:]
        #         # sh /= HEIGHT
        #         # sw /= WIDTH
        #         # x, _, _ = bilinear_sample([new_data.x[0].permute(1,2,0)], v*sh, u*sw)
        #         x, _, _ = bilinear_sample([new_data.x[0].permute(1,2,0)], v/8., u/8.)
        #         self.x = x[0]

        # if model_args['do_segmentation']:
        #     self.points = self.points.unsqueeze(0).repeat(self.class_num, 1, 1)
        #     self.norms = self.norms.unsqueeze(0).repeat(self.class_num, 1, 1)

        # Init renderer.
        # if render_method == 'proj': self.renderer = Projector().to(dev)
        # elif render_method == 'pulsar': self.renderer = Pulsar().to(dev)

        # Keyframes
        # if self.opt.sf_corr_use_keyframes:
        init_y, init_x, _, _ = pcd2depth(inputs, self.points, round_coords=False, valid_margin=1)
        self.keyframes = (inputs[("color", 0, 0)], 
                            torch.arange(len(init_x))[None, ...].cuda(),
                            torch.stack([init_x, init_y], dim=1)[None, ...].type(fl64_).cuda()
                            )

    # Init/update ED nodes & ED nodes related parameters. TODO
    def update_ED(self, new_nodes=None):
        if new_nodes is None: # Init.
            # Find 8 neighbors of ED nodes.
            if self.opt.hard_seman and False:
                dists, sort_idx = find_knn(self.ED_nodes.points, self.ED_nodes.points, 
                k=self.opt.num_ED_neighbors+1, num_classes=self.opt.num_classes, 
                seman1=self.ED_nodes.seman, seman2=self.ED_nodes.seman)
            else:
                dists, sort_idx = find_knn(self.ED_nodes.points, self.ED_nodes.points, 
                    k=self.opt.num_ED_neighbors+1)
            dists = dists[:, 1:] / self.ED_nodes.radii[:, None]
            sort_idx = sort_idx[:, 1:]
            # dists, sort_idx = find_knn(self.ED_nodes.points, self.ED_nodes.points, 
            #     k=self.opt.num_ED_neighbors+2)
            # dists = dists[:, 1:-1] / dists[:, -1:]
            # sort_idx = sort_idx[:, 1:-1]
            if self.opt.method == "seman-super" and False:
                P = self.ED_nodes.seman_conf[:, None, :]
                Q = self.ED_nodes.seman_conf[sort_idx]
                self.ED_nodes.knn_w = F.softmax(
                                torch.pow(torch.exp(-JSD(P, Q)), self.power_arg[0]) * \
                                torch.pow(torch.exp(-dists), self.power_arg[1])
                                , dim=-1) # Jensen-Shannon divergence
            else:
                self.ED_nodes.knn_w = F.softmax(torch.exp(-dists), dim=-1)
            self.ED_nodes.knn_indices = sort_idx
            
            # Find 4 neighboring ED nodes of surfels.
            # if self.opt.method == "seman-super" and False:
            if self.opt.hard_seman:
                dists, self.knn_indices = find_knn(self.points, self.ED_nodes.points,
                    k=self.opt.num_neighbors, num_classes=self.opt.num_classes,
                    seman1=self.seman, seman2=self.ED_nodes.seman)
            else:
                dists, self.knn_indices = find_knn(self.points, self.ED_nodes.points,
                    k=self.opt.num_neighbors)
            radii = self.ED_nodes.radii[self.knn_indices]
            self.isStable[~torch.any(dists <= radii, dim=1)] = False # If surfels are too far away from its knn ED nodes, this surfels will be disabled.
            if self.opt.method == "seman-super" and not self.opt.hard_seman:
                P = self.ED_nodes.seman_conf[self.knn_indices]
                Q = self.seman_conf[:, None, :]
                # self.knn_w = F.softmax(
                #                 torch.sqrt(
                #                     torch.exp(- KLD(P, Q)) * torch.exp(-KLD(Q, P)) * torch.exp(- dists / radii)
                #                 ), dim=-1)
                self.knn_w = F.softmax(
                                torch.pow(torch.exp(- JSD(P, Q)), self.power_arg[0]) * \
                                torch.pow(torch.exp(- dists / radii), self.power_arg[1])
                                , dim=-1) # Jensen-Shannon divergence
            else:
                self.knn_w = F.softmax(torch.exp(- dists / radii), dim=-1)

    def update(self, deform, time, boundary_edge_type=None, boundary_face_type=None):
        """
        Update surfels and ED nodes with their motions estimated by optimizor.
        """
        # deform, radii, colors = deform
        # self.colors[self.isStable] = colors

        sf_knn = self.ED_nodes.points[self.knn_indices] # All g_i in (10).
        sf_diff = self.points.unsqueeze(1) - sf_knn
        deform_ = deform[self.knn_indices]
        self.points, _ = Trans_points(sf_diff, sf_knn, deform_, self.knn_w)
        self.points += deform[-1:, 4:] # T_g
        
        norms, _ = transformQuatT(
            self.norms.unsqueeze(1).repeat(1,self.opt.num_neighbors,1),
            deform_)
        norms = torch.sum(self.knn_w.unsqueeze(-1) * norms, dim=-2)
        norms, _ = transformQuatT(norms, deform[-1:,0:4]) # T_g
        self.norms = F.normalize(norms, dim=-1)

        # self.time_stamp = time * torch.ones_like(self.time_stamp)
        
        self.ED_nodes.points += deform[:-1,4:]
        self.ED_nodes.points += deform[-1:, 4:] # T_g
        ED_norms, _ = transformQuatT(self.ED_nodes.norms, deform[:-1,0:4])
        ED_norms, _ = transformQuatT(ED_norms, deform[-1:,0:4]) # T_g
        self.ED_nodes.norms = F.normalize(ED_norms, dim=-1)
    
        if boundary_edge_type is not None:
            self.ED_nodes.boundary_edge_type = boundary_edge_type
        if boundary_face_type is not None:
            self.ED_nodes.boundary_face_type = boundary_face_type

        # if self.opt.method == 'seman-super':
        #     new_edges_lens = torch_distance(self.ED_nodes.points[self.ED_nodes.edge_index[0]], self.ED_nodes.points[self.ED_nodes.edge_index[1]])
            
        #     seman_edge_val = torch.ones_like(self.ED_nodes.isBoundary)
        #     seman_edge_val[self.ED_nodes.isBoundary] = new_edges_lens[self.ED_nodes.isBoundary] < torch.quantile(self.ED_nodes.edges_lens[~self.ED_nodes.isBoundary], 0.75)

        #     self.ED_nodes.seman_edge_val = seman_edge_val

    # Fuse the input data into our reference model.
    def fuseInputData(self, inputs, sfdata):
        # Return data[indices]. 'indices' can be either indices or True/False map.
        def get_data(indices, data):
            return data.points[indices], data.norms[indices], data.colors[indices], \
                data.radii[indices], data.confs[indices]

        # Merge data1 & data2, and update self.data[indices].
        def merge_data(data1, indices1, data2, indices2, time, add_new=False):
            p, n, c, r, w = get_data(indices1, data1)
            p_new, n_new, c_new, r_new, w_new = get_data(indices2, data2)

            if len(p) == 0:
                return torch.zeros(len(p), dtype=bool_).cuda()

            # Only merge points that are close enough.
            # print(torch_inner_prod(n, n_new))
            valid = (torch_distance(p, p_new) < self.opt.th_dist) & \
                (torch_inner_prod(n, n_new) > self.opt.th_cosine_ang)
            if self.opt.hard_seman or self.opt.data == "superv1":
                valid &= data1.seman[indices1] == data2.seman[indices2]
            indices = indices1[valid]
            w, w_new = w[valid], w_new[valid]
            w_update = w + w_new
            w /= w_update
            w_new /= w_update

            # Fuse the radius(r), confidence(r), position(p), normal(n) and color(c).
            self.radii[indices] = w * r[valid] + w_new * r_new[valid]
            self.confs[indices] = w_update
            w, w_new = w.unsqueeze(-1), w_new.unsqueeze(-1)
            self.points[indices] = w * p[valid] + w_new * p_new[valid]
            norms = w * n[valid] + w_new * n_new[valid]
            self.norms[indices] = torch.nn.functional.normalize(norms, dim=-1)
            # TODO: Better color update.
            if add_new:
                w_color = w
                w_color_new = w_new * 2
                w_color_sum = w_color + w_color_new
                self.colors[indices] = w_color / w_color_sum * c[valid] + w_color_new / w_color_sum * c_new[valid]
            else:
                self.colors[indices] = w * c[valid] + w_new * c_new[valid]
            self.time_stamp[indices] = time.type(fl32_) # Update time stamps.

            # Merge semantic information.
            if hasattr(self, 'seman'):
                self.seman_conf[indices] = w * data1.seman_conf[indices1][valid] + \
                    w_new * data2.seman_conf[indices2][valid]
                self.seman_conf[indices] = self.seman_conf[indices] / self.seman_conf[indices].sum(1, keepdim=True)
                self.seman[indices] = torch.argmax(self.seman_conf[indices], dim=1).type(long_)

            return valid

        valid = sfdata.valid.clone()

        ## Project surfels onto the image plane. For each pixel, only up to 16 projections with higher confidence.
        # Ignore surfels that have projections outside the image.
        _, _, coords, val_indices = pcd2depth(inputs, self.points)
        val_indices = val_indices & self.isStable
        ids = torch.arange(len(self.points)).cuda()
        # Sort based on confidence.
        _, confs_sort_indices = torch.sort(self.confs, descending=True)
        # Continue sort based on coordinates.
        coords, coords_sort_indices = torch.sort(coords[confs_sort_indices], stable=True)
        sort_indices = confs_sort_indices[coords_sort_indices]
        val_indices = val_indices[sort_indices]
        ids = ids[sort_indices][val_indices]
        coords = coords[val_indices]
        val_indices = val_indices[val_indices]
        # Get the projection maps 1) valid & 2) surfel indices in self.points,
        # map size: map_num x PIXEL_NUM)
        map_num = 16
        val_maps = []
        index_maps = []
        for i in range(map_num):
            if len(coords) == 0: break
            
            if i == 0:
                temp_coords, counts = torch.unique_consecutive(coords, return_counts=True)
                counts_limits = torch.cumsum(counts, dim=0)
                counts = torch.cat([torch.tensor([0]).cuda(), counts_limits[:-1]])
                _counts_ = counts
            else:
                counts += 1
                _counts_ = counts[counts < counts_limits]
                temp_coords = coords[_counts_]

            val_map = torch.zeros((inputs["height"] * inputs["width"]), dtype=bool_).cuda()
            val_map[temp_coords] = True
            val_maps.append(val_map)

            index_map = torch.zeros((inputs["height"] * inputs["width"]), dtype=long_).cuda()
            index_map[temp_coords] = ids[_counts_]
            index_maps.append(index_map)

            val_indices[_counts_] = False
        # Init indices of surfels that will be deleted.
        ids = ids[val_indices]
        del_indices = [ids] if len(ids) > 0 else []

        if not self.opt.disable_merging_new_surfels:
            # Init valid map of new points that will be added.
            add_valid = valid & (~val_maps[0])
            ## Merge new points with existing surfels.
            valid[add_valid] = False
            for val_map, index_map in zip(val_maps, index_maps):
                if not torch.any(valid): break

                val_ = valid & val_map
                index_ = index_map[val_]
                merge_val_ = merge_data(self, index_, sfdata, val_[sfdata.valid], sfdata.time, add_new=True)
                valid[val_] = ~merge_val_
            add_valid |= valid

        ## Merge paired surfels.
        map_num = len(val_maps)
        for i in range(map_num):
            val_map = val_maps[i]

            for j in range(i+1, map_num):
                val_map &= val_maps[j]
                if not torch.any(val_map): continue

                indices1 = index_maps[i][val_map]
                indices2 = index_maps[j][val_map]
                val_merge = merge_data(self, indices1, self, indices2, sfdata.time)
                update_val_map = torch.ones_like(val_map)
                update_val_map[val_map] = ~val_merge
                val_maps[j] &= update_val_map

                # if self.opt.sf_corr_use_keyframes:
                keyframes_pts_ids = self.keyframes[1]
                for tid_keep, tid_del in zip(indices1[val_merge], indices2[val_merge]):
                    if (not tid_keep in keyframes_pts_ids) and (tid_del in keyframes_pts_ids):
                        keyframes_pts_ids[keyframes_pts_ids==tid_del] = tid_keep
                self.keyframes = (self.keyframes[0], keyframes_pts_ids, self.keyframes[2])
                
                indices2 = indices2[val_merge]
                del_indices.append(indices2)
                if hasattr(self, 'track_pts'):
                    indices1 = indices1[val_merge]
                    for k, tid in enumerate(self.track_id):
                        if tid in indices2:
                            self.track_id[k] = indices1[indices2==tid].type(long_)
        
        ## Delete redundant surfels.
        if len(del_indices) > 0:
            del_indices = torch.unique(torch.cat(del_indices))

            if hasattr(self, 'track_pts'):
                for k, tid in enumerate(self.track_id):
                    # if torch.isnan(tid):
                    #     continue
                    if tid in del_indices:
                        self.track_id[k] = torch.tensor(-2)
            
            self.isStable[del_indices] = False

        ## Update the knn weights of existing surfels.
        if self.opt.method == "seman-super":
            # keep_ids = self.seman == self.ED_nodes.seman[self.knn_indices[:,0]]
            # shuffle_ids = ~keep_ids

            # shuffle_dists, shuffle_knn_indices = find_knn(self.points[shuffle_ids], self.ED_nodes.points,
            #     k=self.opt.num_neighbors, num_classes=self.opt.num_classes,
            #     seman1=self.seman[shuffle_ids], seman2=self.ED_nodes.seman)
            # shuffle_radii = self.ED_nodes.radii[shuffle_knn_indices]
            # self.knn_w[shuffle_ids] = F.softmax(torch.exp(- shuffle_dists / shuffle_radii), dim=-1)

            # dists = torch_distance(self.points[keep_ids][:, None, :], \
            #     self.ED_nodes.points[self.knn_indices[keep_ids]])
            # radii = self.ED_nodes.radii[self.knn_indices[keep_ids]]
            # self.knn_w[keep_ids] = F.softmax(torch.exp(- dists / radii), dim=-1)

            dists = torch_distance(self.points.unsqueeze(1), \
                self.ED_nodes.points[self.knn_indices])
            radii = self.ED_nodes.radii[self.knn_indices]

            P = self.ED_nodes.seman_conf[self.knn_indices]
            Q = self.seman_conf[:, None, :]
            # self.knn_w = F.softmax(
            #                 torch.sqrt(
            #                     torch.exp(- KLD(P, Q)) * torch.exp(- KLD(Q, P)) * torch.exp(- dists / radii)
            #                 ), dim=-1)
            self.knn_w = F.softmax(
                            torch.pow(torch.exp(- JSD(P, Q)), self.power_arg[0]) * \
                            torch.pow(torch.exp(- dists / radii), self.power_arg[1])
                            , dim=-1)
            # self.knn_w = F.softmax(torch.exp(- dists / radii), dim=-1)

        else:
            dists = torch_distance(self.points.unsqueeze(1), \
                self.ED_nodes.points[self.knn_indices])
            radii = self.ED_nodes.radii[self.knn_indices]
            self.knn_w = F.softmax(torch.exp(- dists / radii), dim=-1)

        if not self.opt.disable_merging_new_surfels:
            ## Add points that do not have corresponding surfels.
            add_valid = add_valid[sfdata.valid]
            if add_valid.count_nonzero() > 0:
                # Extract new surfels and check if each of them has corresponding ED nodes.
                new_points = sfdata.points[add_valid]
                # Update the knn weights and indicies of new surfels.
                new_isStable = torch.ones(len(new_points), dtype=bool_).cuda()
                # if self.opt.method == "seman-super" and False:
                if self.opt.hard_seman:
                    dists, new_knn_indices = find_knn(new_points, self.ED_nodes.points, 
                        k=self.opt.num_neighbors, num_classes=self.opt.num_classes, 
                        seman1=sfdata.seman[add_valid], seman2=self.ED_nodes.seman)
                else:
                    dists, new_knn_indices = find_knn(new_points, self.ED_nodes.points, 
                        k=self.opt.num_neighbors)
                radii = self.ED_nodes.radii[new_knn_indices]
                new_isStable[~torch.any(dists <= radii, dim=1)] = False # If surfels are too far away from its knn ED nodes, this surfels will be disabled.
                if self.opt.method == "seman-super" and not self.opt.hard_seman:
                    P = self.ED_nodes.seman_conf[new_knn_indices]
                    Q = sfdata.seman_conf[add_valid][:, None, :]
                    # new_knn_w = F.softmax(
                    #                 torch.sqrt(
                    #                     torch.exp(- KLD(P, Q)) * torch.exp(- KLD(Q, P)) * torch.exp(- dists / radii)
                    #                 ), dim=-1)
                    new_knn_w = F.softmax(
                                    torch.pow(torch.exp(- JSD(P, Q)), self.power_arg[0]) * \
                                    torch.pow(torch.exp(- dists / radii), self.power_arg[1])
                                    , dim=-1)
                else:
                    new_knn_w = F.softmax(torch.exp(- dists / radii), dim=-1)

                self.isStable = torch.cat([self.isStable, torch.ones(torch.count_nonzero(new_isStable), dtype=bool_).cuda()])
                self.knn_w = torch.cat([self.knn_w, new_knn_w[new_isStable]], dim=0)
                self.knn_indices = torch.cat([self.knn_indices, new_knn_indices[new_isStable]], dim=0)

                
                new_sf_num = new_isStable.count_nonzero()
                self.points = torch.cat([self.points, new_points[new_isStable]], dim=0)
                for key, v in sfdata.items():
                    if key in ['norms', 'colors']: # 'points', 
                        v = torch.cat([getattr(self, key), v[add_valid][new_isStable]], dim=0)
                    elif key in ['seman', 'seman_conf', 'radii', 'confs', 'dist2edge']:
                        v = torch.cat([getattr(self, key), v[add_valid][new_isStable]])
                    elif key == 'time':
                        v = torch.cat([self.time_stamp, v*torch.ones(new_sf_num).cuda()])
                        key = "time_stamp"
                    else:
                        continue
                    setattr(self, key, v)
                
                # ## Update isED: Only points that are far enough from 
                # ## existing ED nodes can be added as new ED nodes.
                # D = torch.cdist(new_points, self.ED_nodes.points)
                # new_knn_dists, _ = D.topk(k=1, dim=-1, largest=False, sorted=True)
                # isED = new_data.isED[add_valid] & (new_knn_dists[:,-1] > torch.max(sf_knn_dists))
                # self.update_ED(points=new_points[isED], norms=new_norms[isED])
                
                

        v, u, _, _ = pcd2depth(inputs, self.points, round_coords=False)
        self.projdata = torch.stack([u, v], dim=1).type(fl32_)

    # If evaluate on the SuPer dataset, init self.label_index which includes
    # the indicies of the tracked points.
    def init_track_pts(self, sfdata, filename, th=0.2):
        if not filename in self.track_pts["gt"]:
            return
        gt_coords = self.track_pts["gt"][filename]
        
        for k, tid in enumerate(self.track_id):
            # The point hasn't been tracked. & Ground truth exists for this point.
            x, y, v = gt_coords[k]
            gt_id = sfdata.index_map[y,x]
            if tid < 0 and gt_id > 0 and v == 1:
                dists = torch_distance(self.points, sfdata.points[gt_id])
                # inval_id = self.track_id[(self.track_id >= 0) | torch.isnan(self.track_id)]
                inval_id = self.track_id[(self.track_id >= 0) | (self.track_id == -2)]
                if len(inval_id) > 0:
                    dists[inval_id.type(long_)] = 1e13
                    dists[~self.isStable] = 1e13
                if torch.min(dists) < th:
                    self.track_id[k] = torch.argmin(dists)
        # print("Init", self.track_id)
    # def init_track_pts(self, sfdata, filename):
    #     if not filename in self.track_pts["gt"]:
    #         return
    #     gt_coords = self.track_pts["gt"][filename]
        
    #     for k, tid in enumerate(self.track_id):
    #         # The point hasn't been tracked. & Ground truth exists for this point.
    #         x, y, v = gt_coords[k]
    #         gt_id = sfdata.index_map[y,x]
    #         if tid < 0 and gt_id > 0 and v == 1:
    #             u, v = self.projdata[:,0].type(long_), self.projdata[:,1].type(long_)
    #             candidate_tids = torch.cat([
    #                 ((u == x) & (v == y)).nonzero(as_tuple=True)[0],
    #                 ((u == x) & (v == y+1)).nonzero(as_tuple=True)[0],
    #                 ((u == x) & (v == y-1)).nonzero(as_tuple=True)[0],
    #                 ((u == x-1) & (v == y)).nonzero(as_tuple=True)[0],
    #                 ((u == x-1) & (v == y+1)).nonzero(as_tuple=True)[0],
    #                 ((u == x-1) & (v == y-1)).nonzero(as_tuple=True)[0],
    #                 ((u == x+1) & (v == y)).nonzero(as_tuple=True)[0],
    #                 ((u == x+1) & (v == y+1)).nonzero(as_tuple=True)[0],
    #                 ((u == x+1) & (v == y-1)).nonzero(as_tuple=True)[0]
    #             ])
    #             if len(candidate_tids) == 1:
    #                 self.track_id[k] = candidate_tids[0]
    #             elif len(candidate_tids) > 1:
    #                 dists = torch_distance(self.points[candidate_tids], sfdata.points[gt_id])
    #                 self.track_id[k] = candidate_tids[torch.argmin(dists)]

    # If evaluate on the SuPer dataset, update self.label_index.
    def update_track_pts(self, sfdata, filename, th=1e-2):
        self.track_rsts[filename] = torch.zeros((self.track_num, 3)).cuda()
        
        for k, tid in enumerate(self.track_id):
            if tid >= 0:
                self.track_rsts[filename][k, 0:2] = self.projdata[tid.type(long_)] # self.projdata[tid.type(long_), 1:]
                self.track_rsts[filename][k, 2] = 1
        # print("update", self.track_id)

    # Delete unstable surfels & ED nodes.
    def prepareStableIndexNSwapAllModel(self, inputs, sfdata):
        # A surfel is unstable if it 1) hasn't been updated for a long time,
        # and 2) has low confidence.
        # self.isStable = self.isStable & \
        #     ((inputs["time"]-self.time_stamp < self.opt.th_time_steps) |\
        #     (self.confs >= self.opt.th_conf))
        self.isStable = self.isStable & (inputs["time"]-self.time_stamp < self.opt.th_time_steps)
        self.isStable[self.track_id[self.track_id>=0]] = True
        
        self.points = self.points[self.isStable]
        self.norms = self.norms[self.isStable]
        self.colors = self.colors[self.isStable]
        self.confs = self.confs[self.isStable]
        self.radii = self.radii[self.isStable]
        self.time_stamp = self.time_stamp[self.isStable]
        self.knn_indices = self.knn_indices[self.isStable]
        self.knn_w = self.knn_w[self.isStable]
        self.projdata = self.projdata[self.isStable]
        if hasattr(self, "seman"):
            self.seman = self.seman[self.isStable]
            self.seman_conf = self.seman_conf[self.isStable]
            self.dist2edge = self.dist2edge[self.isStable]
        if self.evaluate_tracking:
            id_map = - torch.ones(len(self.isStable), dtype=long_).cuda()
            id_map[self.isStable] = torch.arange(torch.count_nonzero(self.isStable)).cuda()
            track_id_val = self.track_id >= 0
            self.track_id[track_id_val] = id_map[self.track_id[track_id_val]]
        self.isStable = self.isStable[self.isStable]

        if self.evaluate_tracking:
            inval_id = (self.track_id >= 0) & torch.tensor(
                [False if tid == torch.tensor(-2) else ~self.isStable[tid.type(long_)] for tid in self.track_id]).cuda()
            if inval_id.count_nonzero() > 0:
                self.track_id[inval_id] = torch.tensor(-2)
        
        self.surfel_num = self.isStable.count_nonzero()

        self.logger.info(f"surfel number: {self.surfel_num}")

        if self.evaluate_tracking:
            if (self.track_id >= 0).count_nonzero() > 0:
                self.update_track_pts(sfdata, inputs["filename"][0])

            if (self.track_id == -1).count_nonzero() > 0:
                self.init_track_pts(sfdata, inputs["filename"][0])

        # v, u, coords, valid_indexs = pcd2depth(self.points, depth_sort=True, round_coords=False)
        # self.valid = torch.zeros(PIXEL_NUM, dtype=bool_, device=dev)
        # self.valid[coords] = True
        # self.validmap = self.valid.view(HEIGHT,WIDTH)
        # self.projdata = torch.stack([valid_indexs,v,u],dim=1)
        
        # Render the current 3D model to an image.
        self.render_img(inputs)

        ssim_map = 1 - SSIM()(self.renderImg, inputs[("color", 0, 0)]) * 2
        self.mssim.append(ssim_map.mean().item())

        self.viz(inputs, sfdata)

        if self.opt.save_ply or self.opt.save_seman_ply:
            ### Smooth point cloud ###
            disp_points = self.points[self.isStable][self.seman[self.isStable] < 2]
            if self.opt.save_ply:
                disp_colors = self.colors[self.isStable][self.seman[self.isStable] < 2]
            elif self.opt.save_seman_ply:
                disp_colors = torch.flip(numpy_to_torch(id2color), [1])[self.seman[self.isStable]][self.seman[self.isStable] < 2]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(disp_points.cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(disp_colors.cpu().numpy())
            
            pcd = pcd.voxel_down_sample(voxel_size=0.005)
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=50,
                                                    std_ratio=0.1)
            pcd = pcd.select_by_index(ind)

            disp_points = np.asarray(pcd.points).astype(np.float32)
            disp_points = disp_points / 0.1 * 0.005 # baseline: 5mm, unit: m
            if self.opt.save_ply:
                disp_colors = (255 * np.asarray(pcd.colors)).astype(np.uint8)
            elif self.opt.save_seman_ply:
                disp_colors = np.asarray(pcd.colors).astype(np.uint8)
            ######

            pcd_to_save = {'x': disp_points[:,0],'y': disp_points[:,1],'z': disp_points[:,2], 
             'red' : disp_colors[:,0], 'green' : disp_colors[:,1], 'blue' : disp_colors[:,2]}
            cloud = PyntCloud(pd.DataFrame(data=pcd_to_save))
            if self.opt.save_ply:
                ply_output_dir = os.path.join(self.output_dir, f"{os.path.basename(self.opt.sample_dir)}_ply")
            elif self.opt.save_seman_ply:
                ply_output_dir = os.path.join(self.output_dir, f"{os.path.basename(self.opt.sample_dir)}_seman_ply")
            if not os.path.exists(ply_output_dir):
                os.makedirs(ply_output_dir)
            cloud.to_file(os.path.join(ply_output_dir, f"{inputs['filename'][0]}.ply"))

    def render_img(self, inputs):
        # if "del_points" in inputs:
        #     data = Data(points=torch.cat([self.points[self.isStable], inputs["del_points"][0]], dim=0), 
        #                 colors=torch.cat([self.colors[self.isStable], inputs["del_colors"][0]], dim=0))
        # else:
        data = Data(points=self.points[self.isStable], 
                    colors=self.colors[self.isStable]) # radii=self.radii[self.isStable]
        
        with torch.no_grad():
            self.renderImg = self.models["renderer"](inputs, data, 
                                                     rad=self.opt.renderer_rad
                                                    ).permute(2,0,1).unsqueeze(0)
        
        # self.projGraph = self.renderer(self.ED_nodes).permute(2,0,1).unsqueeze(0)

    # Visualize the tracking & reconstruction results.
    def viz(self, inputs, sfdata, bid=0):
        def draw_keypoints_(img_, keypoints, colors="red"):
            keypoints = keypoints[:, 0:2][keypoints[:,2] == 1].unsqueeze(0)
            if len(keypoints) > 0:
                img_ = draw_keypoints(img_, keypoints, colors=colors, radius=3)
            return img_

        filename = inputs["filename"][0]

        render_img = self.renderImg[bid]
        render_img = (255*render_img).type(torch.uint8).cpu()

        img = inputs[("color", 0, 0)][bid]
        img = (255*img).type(torch.uint8).cpu()

        if self.opt.num_classes == 2:
            seman_colors = self.seman_conf[self.isStable]
            seman_colors = torch.cat([seman_colors, torch.zeros_like(seman_colors[:, 0:1])], dim=1)
            y_pred = self.models["renderer"](inputs, 
                                            Data(points=self.points[self.isStable], 
                                                 colors=seman_colors), 
                                            rad=self.opt.renderer_rad)
        elif self.opt.num_classes == 3:
            y_pred = self.models["renderer"](inputs, 
                                            Data(points=self.points[self.isStable], 
                                                 colors=self.seman_conf[self.isStable]), 
                                            rad=self.opt.renderer_rad)
        inval = torch.max(y_pred, dim=2)[0] < 1e-3
        # inval = torch.min(y_pred, dim=2)[0] >= 0.5
        y_pred = torch.argmax(y_pred, dim=2) + 1
        y_pred = torch_to_numpy(y_pred)
        if ("seman_gt", 0) in inputs:
            y_true = inputs[("seman_gt", 0)][0,0] + 1
            y_true[inval] = 0
            
            y_true = torch_to_numpy(y_true)
            self.result_dice += [general_dice(y_true, y_pred)]
            self.result_jaccard += [general_jaccard(y_true, y_pred)]

        if self.opt.save_raw_data:
            raw_data_dir = os.path.join(self.output_dir, f"{os.path.basename(self.opt.sample_dir)}_raw_data", filename)
            if not os.path.exists(raw_data_dir):
                os.makedirs(raw_data_dir)

            raw_img = torch_to_numpy(inputs[("color",0,0)][bid].permute(1,2,0))[...,::-1]
            raw_img = 255 * raw_img
            cv2.imwrite(os.path.join(raw_data_dir, "input.png"), raw_img[:, 32:-32])

            raw_seman = id2color[torch_to_numpy(inputs[("seman",0)][bid, 0])]
            cv2.imwrite(os.path.join(raw_data_dir, "seman_input.png"), raw_seman[:, 32:-32])

            if ("disp",0) in inputs:
                out_disp = torch_to_numpy(inputs[("disp",0)][bid,0])
                out_disp[np.isnan(out_disp)] = 0
                out_disp = 255 * out_disp / np.max(out_disp)
                out_disp = cv2.applyColorMap(out_disp.astype(np.uint8), cv2.COLORMAP_MAGMA)
                cv2.imwrite(os.path.join(raw_data_dir, "depth.png"), out_disp[:, 32:-32])

            out_normal = inputs[("normal",0)].permute(0, 3, 1, 2)
            # out_normal = blur_image(out_normal, kernel=31)
            out_normal = out_normal[bid].permute(1, 2, 0)
            out_normal = torch_to_numpy(out_normal)
            out_normal[np.isnan(out_normal)] = 0
            out_normal = 255 * out_normal
            cv2.imwrite(os.path.join(raw_data_dir, "normal.png"), out_normal[:, 32:-32])

            ### Smooth point cloud ###
            disp_points = self.points[self.isStable][self.seman[self.isStable] < 2]
            disp_colors = self.colors[self.isStable][self.seman[self.isStable] < 2]
            disp_normals = torch.flip(numpy_to_torch(id2color), [1])[self.seman[self.isStable]][self.seman[self.isStable] < 2]

            # disp_semans = self.seman[self.isStable][self.seman[self.isStable] < 2]
            # seman_beef_valid = disp_semans == 0
            # seman_beef_valid[seman_beef_valid] = disp_points[disp_semans==0, 2] < 1.2
            # print(torch.unique(disp_points[disp_semans==0, 2]), torch.unique(disp_points[disp_semans==1, 2]))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(disp_points.cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(disp_colors.cpu().numpy())
            pcd.normals = o3d.utility.Vector3dVector(disp_normals.cpu().numpy())
            
            pcd = pcd.voxel_down_sample(voxel_size=0.005)
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=50,
                                                    std_ratio=0.1)
            pcd = pcd.select_by_index(ind)

            disp_points = numpy_to_torch(np.asarray(pcd.points), dtype=fl64_)
            disp_colors = numpy_to_torch(np.asarray(pcd.colors), dtype=fl64_)
            disp_seman_colors = numpy_to_torch(np.asarray(pcd.normals), dtype=fl64_)
            ######

            theta = - 40 / 180 * np.pi
            Rx = torch.tensor([[1, 0, 0], 
                               [0, np.cos(theta), -np.sin(theta)], 
                               [0, np.sin(theta), np.cos(theta)]], dtype=fl64_).cuda()
            # rot_points = self.points[self.isStable]
            rot_points = disp_points
            rot_points_mean = rot_points.mean(0, keepdim=True)
            rot_points = torch.matmul(rot_points - rot_points_mean, Rx.T) + rot_points_mean
            
            # out_data = Data(points=rot_points, 
            #         colors=self.colors[self.isStable])
            out_data = Data(points=rot_points, 
                    colors=disp_colors)
            out_render_img = self.models["renderer"](inputs, out_data, rad=0.000002)

            out_render_img = torch_to_numpy(255 * out_render_img)[:, :, ::-1]
            cv2.imwrite(os.path.join(raw_data_dir, "raw_render_img.png"), out_render_img[:, 32:-32])
            
            # masked_colors=self.colors[self.isStable]
            # masked_colors[self.seman[self.isStable] == 2] = 1
            # out_data = Data(points=rot_points, 
            #                 colors=masked_colors)
            out_data = Data(points=rot_points, 
                            colors=disp_seman_colors)
            masked_out_render_img = self.models["renderer"](inputs, out_data, rad=0.000002)
            # masked_out_render_img = torch_to_numpy(255 * masked_out_render_img)[:, :, ::-1]
            masked_out_render_img = torch_to_numpy(masked_out_render_img)[:, :, ::-1]
            cv2.imwrite(os.path.join(raw_data_dir, "render_img.png"), masked_out_render_img[:, 32:-32])

            # out_data = Data(points=rot_points, 
            #         colors=self.seman_conf[self.isStable])
            # viz_y_pred = self.models["renderer"](inputs, out_data, rad=0.000002).argmax(2)
            # viz_y_pred = id2color[torch_to_numpy(viz_y_pred)]
            # viz_y_pred[torch_to_numpy(inval)] = 255
            # cv2.imwrite(os.path.join(raw_data_dir, "render_seg.png"), viz_y_pred[:, 32:-32])

            # rot_pcd = inputs[("pcd",0)][bid].type(fl64_)
            # val_rot_pcd = torch.any(torch.isnan(rot_pcd), 2)
            # rot_pcd = rot_pcd[val_rot_pcd]
            # # rot_pcd_mean = rot_pcd.mean(0, keepdim=True)
            # # rot_pcd = torch.matmul(rot_pcd - rot_pcd_mean, Rx.T) + rot_pcd_mean
            # out_data = Data(points=rot_pcd, 
            #         colors=inputs[("color", 0, 0)][bid].permute(1,2,0)[val_rot_pcd])
            out_data = Data(points=sfdata.points, colors=sfdata.colors)
            
            out_new_pcd_render = self.models["renderer"](inputs, out_data, rad=0.000002)
            out_new_pcd_render = torch_to_numpy(255 * out_new_pcd_render)[:, :, ::-1]
            cv2.imwrite(os.path.join(raw_data_dir, "new_pcd_render.png"), out_new_pcd_render[:, 32:-32])

        if inputs["ID"] % self.opt.save_sample_freq == 0:
            # Draw the tracked points.
            colors = {"gt": "blue", "super_cpp": "magenta", "SURF": "lime"}
            if self.evaluate_tracking:
                for key in self.track_pts:
                    if filename in self.track_pts[key]:
                        keypoints = self.track_pts[key][filename]
                        render_img = draw_keypoints_(render_img, keypoints, colors=colors[key])

                if filename in self.track_rsts:
                    keypoints = self.track_rsts[filename]
                    render_img = draw_keypoints_(render_img, keypoints)

            out = torch.cat([render_img, img], dim=1)
            out = torch_to_numpy(out.permute(1,2,0))

            # Visualize mesh onto the image.
            ed_y, ed_x, _, _ = pcd2depth(inputs, self.ED_nodes.points)
            ed_pts = torch.stack([ed_x, ed_y], dim=1).cpu().numpy().astype(int)
            boundary_edge_id = 0
            id2color_strong = [(255, 0, 0), (255, 255, 0), (0, 255, 0)]
            for k, edge_id1 in enumerate(self.ED_nodes.edge_index[0]):
                edge_id2 = self.ED_nodes.edge_index[1][k]
                if self.ED_nodes.seman[edge_id1] == self.ED_nodes.seman[edge_id2]:
                    edge_color = id2color_strong[self.ED_nodes.seman[edge_id1]]#[::-1]
                else:
                    edge_color = (255, 255, 255)
                if hasattr(self.ED_nodes, 'boundary_edge_type'):
                    if self.ED_nodes.isBoundary[k]:
                        if self.ED_nodes.boundary_edge_type[boundary_edge_id].mean() < 0.5:
                            edge_color = (255, 0, 255)
                        boundary_edge_id += 1

                pt1 = ed_pts[edge_id1]
                pt2 = ed_pts[edge_id2]
                
                out = out.astype(np.uint8).copy()
                # out = cv2.line(out, pt1, pt2, (255, 255, 255), 2)
                out = cv2.line(out, pt1, pt2, edge_color, 1)

            # for pt_pos, pt_seman in zip(ed_pts, self.ED_nodes.seman):

            if hasattr(self.ED_nodes, 'boundary_face_type'):
                boundary_id = self.ED_nodes.isBoundaryFace.nonzero(as_tuple=True)[0]
                mesh_out = copy.deepcopy(out)
                for k, face_type in enumerate(self.ED_nodes.boundary_face_type[:, -1]):
                    tri_id1, tri_id2, tri_id3 = self.ED_nodes.triangles[:, boundary_id[k]]
                    vertices = torch.tensor([[ed_x[tri_id1], ed_y[tri_id1]], 
                                            [ed_x[tri_id2], ed_y[tri_id2]], 
                                            [ed_x[tri_id3], ed_y[tri_id3]]]).cpu().numpy().astype(np.int32)
                    if face_type == 1:
                        mesh_out = cv2.drawContours(mesh_out, [vertices], 0, (0, 255, 0), -1)
                    else:
                        # mesh_out = cv2.drawContours(mesh_out, [vertices], 0, (0, 0, 0), -1)
                        mesh_out = cv2.drawContours(mesh_out, [vertices], 0, (255, 0, 255), -1)
                out = 0.6 * out + 0.4 * mesh_out

            # Put ID next to each tracked point.
            if (self.opt.data == "superv2" and filename == "000000") or self.opt.data == "superv1":
            # if filename in self.track_pts["gt"]:
                out = out.astype(np.uint8).copy()
                for k, pts in enumerate(self.track_pts["gt"][filename][:, 0:2]):
                    x = int(pts[0].item())
                    y = int(pts[1].item())

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (x-10, y+20)
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 1
                    out = cv2.putText(out, str(k+1), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)

            if ("disp",0) in inputs:
                disp = torch_to_numpy(inputs[("disp",0)][bid,0])
            else:
                disp = 1 / (torch_to_numpy(inputs[("depth",0)][bid,0])+1e-13)
            disp[np.isnan(disp)] = 0
            disp = 255 * disp / np.max(disp)
            disp = cv2.applyColorMap(disp.astype(np.uint8), cv2.COLORMAP_MAGMA)
            outl = [disp]
            if hasattr(self, 'seman'):
                seman = inputs[("seman", 0)][bid, 0]
                seman = torch_to_numpy(seman)
                seman = id2color[seman]
                # seman[torch_to_numpy(inputs[("seman_valid", 0)][bid, 0]) == 0] = 0

                data = Data(points=self.points[self.isStable], 
                            colors=numpy_to_torch(id2color)[self.seman[self.isStable]]) # radii=self.radii[self.isStable]
                render_seman = self.models["renderer"](inputs, data, rad=self.opt.renderer_rad)
                render_seman = torch_to_numpy(render_seman)
                
                outl += [seman, render_seman]
            else:
                outl += [np.zeros_like(disp)]
            outl = np.concatenate(outl, axis=0)
            h, w, _ = out.shape
            lh, lw, _ = outl.shape
            outl = cv2.resize(outl, (int(lw*h/lh), h))
            out = np.concatenate([outl, out[:,:,::-1]], axis=1)

            cv2.imwrite(os.path.join(self.output_dir, f"{filename}.png"), \
                out)