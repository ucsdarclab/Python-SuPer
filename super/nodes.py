
import os, numpy as np
import copy
import cv2
import torch, torch.nn.functional as F
from torch_geometric.data import Data

from super.utils import *

from utils.utils import draw_keypoints, log_trackpts_err, get_gt, find_knn, pcd2depth, conf2color, plot_pcd
from utils.utils import torch_distance, torch_inner_prod, JSD
from utils.labels import binary_id2color, id2color

from torch.utils.tensorboard import SummaryWriter


def evaluate(gt, est, igonored_ids=[], normalize=False):
    ''' Calculate tracking error '''
    val = (gt[:, 2] == 1) #& (est[:, 2] == 1)
    if len(igonored_ids) > 0:
        val[np.array(igonored_ids) - 1] = False
    dists = np.linalg.norm(gt[:, 0:2] - est[:, 0:2], axis=1)
    dists[~val] = -1

    h = 480
    if normalize: 
        dists /= h

    # # if np.any(est[:, 2][val]==0):
    # if np.max(dists) > 50:
    #     id = np.argmax(dists)
    #     print(id, dists[id], gt[id, 0:2], est[id, 0:2])

    return dists

class Surfels():
    ''' Surfels contains the collection surfels, encoding deformation and point tracking information.
        It contains the following attributes:
        - opt:      Options parsed from argument parser.
        - models:   InitNets object: device, mesh_encoder, renderer, super.
        - output_dir:  Directory to save the output.
        
        - time:       Int. Time stamp of the current frame.
        - time_stamp: Tensor (N,).  Time stamp of each surfel.
        - sf_num:     Int N. Number of surfels.
        - surfel_num: Tensor (). Humber of surfels as 0-dim tensor.

        - track_num:  Int. Number of tracked surfels.
        - track_id: Tensor (20, ). Tracked surfel IDs.
            - -1: Haven't started tracking; 
            - -2: Lost tracking.
            - >=0: ID of tracked points; 
        - tracked_rsts: Dict. Predicted screen coordinate of tracked surfels at the current frame.
        - tracked_pts: Dict of Dict. Ground truth screen coordinate of tracked surfels across 520 frames.
            {'gt':        { '00010': array of shape (20, 3). Homogeneous screen coord. of 20 tracked points at time 000010
                            '00020': array of shape (20, 3). ...}
             'super_cpp': { ... }
             'SURF':      { ... }
            }
        - points:       Tensor (N,3).
        - norms:        Tensor (N,3).
        - colors:       Tensor (N,3).
        - confs:        Tensor (N,).
        - radii:        Tensor (N,).
        - dist2edge:    Tensor (N,).
        - isStable:     Tensor (bool) (N,).
        - index_map:    Tensor (480, 640).
        - knn_indices:  Tensor (N,K=4).
        - knn_w:        Tensor (N,K=4).
        - projdata:     Tensor (N, 2).
        - ED_nodes: torch_geometric.data object. Describes Embedded Deformation Graph (EDG) nodes.
            - keys: ['knn_indices', 'edge_index', 'points', ...]

        - seg:      Tensor (N,). 
        - seg_conf: Tensor (N, 2).
        - renderImg:           Tensor (1,3,H,W). Rendered tensor image.
        - renderImg_conf_heat: Tensor (1,3,H,W). Confidence heatmap as tensor image.

        - logger:         Custom logger manager.
        - summary_writer: Tensorboard summary writer.

        - gt: dict. {
            '000010': array of shape (20, 3). Homogeneous screen coord. of 20 tracked points at time 000010
            '000020': array of shape (20, 3). Homogeneous screen coord. of 20 tracked points at time 000020
            ...
            '000520': array of shape (20, 3). Homogeneous screen coord. of 20 tracked points at time 000520
        }
        - gt_array:   Array (52, 20, 3). Homogeneous screen coord. of 20 tracked points at all times [10, 20, ..., 520]. 
        - gt_intkeys: [10, 20, ..., 520]. List of int time stamps where ground truth info is recorded. 
        - gt_strkeys: ['000010', '000020', ..., '000520']. List of string filename prefixes. 
    '''

    def __init__(self, opt, models, inputs, data):
        self.opt = opt
        self.models = models
        self.evaluate_tracking = self.opt.tracking_gt_file is not None

        ''' For Semantic-Super '''
        if self.opt.method == "semantic-super": self.power_arg = (1/2, 1/2)
        if hasattr(self.opt, 'hard_seg'):
            self.hard_seg = self.opt.hard_seg
        else:
            self.hard_seg = False
        
        ''' For evaluation '''
        if self.opt.phase == 'test':
            self.output_dir = os.path.join(self.opt.output_dir, self.opt.model_name)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # Set up tensorboard logger
            print(f"Tensorboard summary writer dir: {self.output_dir}")
            self.summary_writer = SummaryWriter(log_dir=self.output_dir)

            # Load ground truth tracking information for tracking evaluation 
            if self.evaluate_tracking:
                self.track_pts, self.gt, self.gt_intkeys, \
                                self.gt_strkeys, self.gt_array = get_gt(self.opt)    
                for key in self.track_pts.keys():
                    for filename in self.track_pts[key]:
                        self.track_pts[key][filename] = torch.as_tensor(self.track_pts[key][filename], 
                                                                        dtype=torch.int)
                self.track_num = self.gt_array.shape[1]
                self.track_id = - torch.ones((self.track_num,), dtype=torch.long).cuda()
                self.track_rsts = {}    # Stores predicted tracking results
                self.tracking_eval_errors = {}     # Reprojection error for 20 tracked point of each scene.
                
                if 'super_cpp' in self.track_pts:
                    self.super_cpp_eval_errors = {}
                    for strkey_rsts in sorted(self.track_pts['super_cpp'].keys()):
                        self.super_cpp_eval_errors[strkey_rsts] = evaluate(self.gt[strkey_rsts].numpy(), 
                                                                           self.track_pts['super_cpp'][strkey_rsts].cpu().numpy())

        ''' Init surfels '''
        for key, v in data.items():
            if key == 'valid': continue
            setattr(self, key, v)
        self.sf_num = len(self.points)
        self.isStable = torch.ones((self.sf_num,), dtype=torch.bool).cuda()
        if self.opt.phase == 'test':
            self.time_stamp = self.time * torch.ones(self.sf_num).cuda()

        self.projdata = torch.flip(
            data.valid.view(self.opt.height, self.opt.width).nonzero(), 
            dims=[-1]).type(torch.float32)

        ''' Init ED nodes '''
        self.update_ed()
        self.update_sfed_knn()
        if self.opt.phase == 'test':
            self.summary_writer.add_scalar("graph_info/num_ED_nodes", self.ED_nodes.param_num, self.time)
            self.summary_writer.add_scalar("graph_info/average_ED_graph_edge_length (mm)", self.ED_nodes.edges_lens.mean()/0.1*5, self.time)

    def update_ed(self):
        ''' Init / update ED nodes and their K neighboring ED nodes '''        
        # Find K neighbors of ED nodes.
        if self.hard_seg:
            dists, sort_idx = find_knn(self.ED_nodes.points, self.ED_nodes.points, 
            k=self.opt.num_ED_neighbors+1, num_classes=self.opt.num_classes, 
            seg1=self.ED_nodes.seg, seg2=self.ED_nodes.seg)
        else:
            dists, sort_idx = find_knn(self.ED_nodes.points, self.ED_nodes.points, 
                k=self.opt.num_ED_neighbors+1)
        dists = dists[:, 1:] / self.ED_nodes.radii[:, None]
        sort_idx = sort_idx[:, 1:]
    
        self.ED_nodes.knn_w = F.softmax(torch.exp(-dists), dim=-1)  # (J,K)
        self.ED_nodes.knn_indices = sort_idx                        # (J,K)            
    
    def update_sfed_knn(self):
        ''' Update K neighboring ED nodes of each surfel. '''
        if self.hard_seg:
            dists, self.knn_indices = find_knn(self.points, self.ED_nodes.points,
                k=self.opt.num_neighbors, num_classes=self.opt.num_classes,
                seg1=self.seg, seg2=self.ED_nodes.seg)
        else:
            dists, self.knn_indices = find_knn(self.points, self.ED_nodes.points,
                k=self.opt.num_neighbors)
        radii = self.ED_nodes.radii[self.knn_indices]
        
        # Disable surfels too far away from its knn ED nodes
        self.isStable[~torch.any(dists <= radii, dim=1)] = False
        if self.opt.method == "semantic-super" and not self.hard_seg:
            P = self.ED_nodes.seg_conf[self.knn_indices]
            Q = self.seg_conf[:, None, :]
            self.knn_w = F.softmax(
                            torch.pow(torch.exp(- JSD(P, Q)), self.power_arg[0]) * \
                            torch.pow(torch.exp(- dists / radii), self.power_arg[1])
                            , dim=-1) # Jensen-Shannon divergence
        else:
            self.knn_w = F.softmax(torch.exp(- dists / radii), dim=-1)

    def update(self, deform):
        """ Update surfels and ED nodes with their motions estimated by optimizor.
        Inputs:
            - deform: (J,7) [q;b] deformation parameters
        """
        if deform is None: return

        sf_knn = self.ED_nodes.points[self.knn_indices] # All g_i in (10).
        sf_diff = self.points.unsqueeze(1) - sf_knn
        deform_ = deform[self.knn_indices]
        self.points, _ = Trans_points(sf_diff, sf_knn, deform_, self.knn_w)
        if not self.opt.use_derived_gradient:
            self.points += deform[-1:, 4:] # T_g
        
        norms, _ = transformQuatT(
            self.norms.unsqueeze(1).repeat(1,self.opt.num_neighbors,1),
            deform_)
        norms = torch.sum(self.knn_w.unsqueeze(-1) * norms, dim=-2)
        if not self.opt.use_derived_gradient:
            norms, _ = transformQuatT(norms, deform[-1:,0:4]) # T_g
        self.norms = F.normalize(norms, dim=-1)
        
        if self.opt.use_derived_gradient:
            self.ED_nodes.points += deform[:,4:]
            ED_norms, _ = transformQuatT(self.ED_nodes.norms, deform[:,0:4])
        else:
            self.ED_nodes.points += deform[:-1,4:]
            self.ED_nodes.points += deform[-1:, 4:] # T_g
            ED_norms, _ = transformQuatT(self.ED_nodes.norms, deform[:-1,0:4])
            ED_norms, _ = transformQuatT(ED_norms, deform[-1:,0:4]) # T_g
        self.ED_nodes.norms = F.normalize(ED_norms, dim=-1)

    def init_track_pts(self, sfdata, filename, th=0.2):
        ''' Initialize tracked points
        Inputs:
            - sfdata: torch_geometric.data object returned by `data_loader.depth_preprocessing()`. 
                - index_map: Tensor (H=480, W=640).
        '''
        if not filename in self.gt: return
        
        self.track_rsts[filename] = torch.zeros((self.track_num, 3)).cuda()
        # If evaluate on the SuPer dataset, init self.label_index which includes the indicies of the tracked points.
        gt_coords = torch.as_tensor(self.gt[filename], dtype=torch.int)  # (20,3)
        
        for k, tid in enumerate(self.track_id):
            x, y, v = gt_coords[k]
            gt_id = sfdata.index_map[y,x]
            if tid < 0 and gt_id > 0 and v == 1:
                dists = torch_distance(self.points, sfdata.points[gt_id])
                inval_id = self.track_id[(self.track_id >= 0) | (self.track_id == -2)]
                if len(inval_id) > 0:
                    dists[inval_id.type(torch.long)] = 1e13
                    dists[~self.isStable] = 1e13
                if torch.min(dists) < th:
                    self.track_id[k] = torch.argmin(dists)
            self.track_rsts[filename][k, 0:2] = self.projdata[tid.type(torch.long)] # self.projdata[tid.type(long_), 1:]
            self.track_rsts[filename][k, 2] = 1    
        
    def update_track_pts(self, sfdata, filename, th=1e-2):
        ''' If evaluate on the SuPer dataset, update self.label_index. '''
        if filename not in set(self.gt_strkeys):
            return 
        
        if filename not in self.track_rsts.keys():
            self.init_track_pts(sfdata, filename, th)
            return

        for k, tid in enumerate(self.track_id):
            if tid < 0: continue 
            self.track_rsts[filename][k, 0:2] = self.projdata[tid.type(torch.long)] # self.projdata[tid.type(long_), 1:]
            self.track_rsts[filename][k, 2] = 1

        return


    ''' For surfel fusion '''

    def fuseInputData(self, inputs, sfdata):
        ''' Fuse the input data into our reference model.
        Inputs:
            - inputs: a dict from dataloader containing the following keys:
                ['filename', 'ID', 'time', ('color', 0), 'K', 'inv_K', 'divterm', 
                'stereo_T', ('color_aug', 0), ('seg_conf', 0), ('seg', 0), ('disp', 0), ('depth', 0)]
            - sfdata: torch_geometric.data object. Returned by data_loader.depth_preprocessing(). 
                Has the following attributes:
                - colors:       Tensor (N, 3)
                - time:         Int.
                - valid:        Bool Tensor (640*480, )
                - seg_conf:     Tensor (N, 2)
                - dist2edge:    Tensor (N, )
                - ED_nodes:     torch_geometric.data object. [num, seg_conf, edge_lens, 
                    radii, norms, triangles, seg, inside_edges, points, 
                    static_ed_nodes, param_num, edge_index, triangles_areas]
                - radii:        Tensor (N, ) 
                - index_map:    Int Tensor (640*480, )
                - norms:        Tensor (N, 3)
                - seg:          Int Tensor (N, )
                - points:       Tensor (N, 3)
                - confs:        Tensor (N, )
        Note: `sfdata` is derived from `inputs`.
        '''
        def get_data(indices, data):
            ''' Return data[indices]. 'indices' can be either indices or True/False map. '''
            
            return data.points[indices], data.norms[indices], data.colors[indices], \
                data.radii[indices], data.confs[indices]


        def merge_data(data1, indices1, data2, indices2, time=None, add_new=False):
            
            # Merge data1 & data2, and update self.data[indices].
            p, n, c, r, w = get_data(indices1, data1)
            p_new, n_new, c_new, r_new, w_new = get_data(indices2, data2)

            if len(p) == 0:
                return torch.zeros(len(p), dtype=torch.bool).cuda()

            # Only merge points that are close enough.
            valid = (torch_distance(p, p_new) < self.opt.th_dist) & \
                (torch_inner_prod(n, n_new) > self.opt.th_cosine_ang)
            
            if (self.hard_seg or self.opt.data == "superv1") and \
            hasattr(data1, 'seg') and hasattr(data2, 'seg'):
                valid &= data1.seg[indices1] == data2.seg[indices2]

            indices = indices1[valid] 
            w, w_new = w[valid], w_new[valid]
            w_update = w + w_new
            w /= w_update
            w_new /= w_update

            # Calculate per-point velocity for Kalman filter motion model.

            points_update =  w.unsqueeze(-1) * p[valid] + w_new.unsqueeze(-1) * p_new[valid]
            norms_update = w.unsqueeze(-1) * n[valid] + w_new.unsqueeze(-1) * n_new[valid]

            # Fuse the radius(r), confidence(w), position(p), normal(n) and color(c).
            self.radii[indices] = w * r[valid] + w_new * r_new[valid]
            self.confs[indices] = w_update
            w, w_new = w.unsqueeze(-1), w_new.unsqueeze(-1)
            self.points[indices] = points_update
            self.norms[indices] = torch.nn.functional.normalize(
                                    norms_update, dim=-1)

            if add_new:
                w_color = w
                w_color_new = w_new * 3
                w_color_sum = w_color + w_color_new
                self.colors[indices] = w_color / w_color_sum * c[valid] + w_color_new / w_color_sum * c_new[valid]
            else:
                self.colors[indices] = w * c[valid] + w_new * c_new[valid]
            
            if time is not None:
                self.time_stamp[indices] = time # Update time stamps.

            # Merge semantic information.
            if hasattr(self, 'seg'):
                self.seg_conf[indices] = w * data1.seg_conf[indices1][valid] + \
                    w_new * data2.seg_conf[indices2][valid]
                self.seg_conf[indices] = self.seg_conf[indices] / self.seg_conf[indices].sum(1, keepdim=True)
                self.seg[indices] = torch.argmax(self.seg_conf[indices], dim=1).type(torch.long)

            return valid

        self.time = sfdata.time
        valid = sfdata.valid.clone()

        ''' Project surfels onto the image plane. 
            - For each pixel, only up to 16 projections with higher confidence.
            - Ignore surfels that have projections outside the image. '''
        _, _, coords, val_indices = pcd2depth(inputs, self.points)  # (N,), (N,). Int pixel coordinates and valid indices.
        val_indices = val_indices & self.isStable               # Boolean tensor (N,) 
        ids = torch.arange(len(self.points)).cuda()             # Point ids.
        
        _, confs_sort_indices = \
            torch.sort(self.confs, descending=True)             # Sort based on confidence.
        coords, coords_sort_indices = \
            torch.sort(coords[confs_sort_indices], stable=True) # Continue sort based on coordinates.
        sort_indices = confs_sort_indices[coords_sort_indices]  # Sorted point indices.

        val_indices = val_indices[sort_indices] 
        ids = ids[sort_indices][val_indices]    
        coords = coords[val_indices]
        val_indices = val_indices[val_indices]
        
        val_maps, index_maps = [], []   # Get the projection maps 1) valid & 2) surfel indices in self.points,
        for i in range(map_num:=16):    # Map size: map_num x PIXEL_NUM
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

            val_map = torch.zeros((self.opt.height * self.opt.width), dtype=torch.bool).cuda()
            val_map[temp_coords] = True 
            val_maps.append(val_map)    # Boolean tensor. (self.opt.height * self.opt.width, )

            index_map = torch.zeros((self.opt.height * self.opt.width), dtype=torch.long).cuda()
            index_map[temp_coords] = ids[_counts_]
            index_maps.append(index_map)

            val_indices[_counts_] = False
        
        ids = ids[val_indices]  # Init indices of surfels that will be deleted.
        del_indices = [ids] if len(ids) > 0 else []

        if len(val_maps) == 0: 
            print(f'No valid index maps to add for fusion at frame {self.time}')
            self.logger.warning(f'No valid index maps to add for fusion at frame {self.time}')

        ''' Merge paired new surfels '''
        add_valid = None
        if not self.opt.disable_merging_new_surfels and len(val_maps) > 0:         
            add_valid = valid & (~val_maps[0])  # Init valid map of new points that will be added.
            valid[add_valid] = False            # Merge new points with existing surfels.
            for val_map, index_map in zip(val_maps, index_maps):
                if not torch.any(valid): break
                val_ = valid & val_map
                index_ = index_map[val_]
                merge_val_ = merge_data(self, index_, sfdata, val_[sfdata.valid], 
                                        time=sfdata.time if self.opt.phase=="test" else None, 
                                        add_new=True)
                valid[val_] = ~merge_val_
            add_valid |= valid      # (H*W, ) 

        ''' Merge paired existing surfels '''
        if not self.opt.disable_merging_exist_surfels and len(val_maps)>0:
            for i in range(map_num := len(val_maps)):
                val_map = val_maps[i]

                for j in range(i+1, map_num):
                    val_map &= val_maps[j]     # Masks for valid points in both map i and map j.
                    if not torch.any(val_map): continue

                    indices1 = index_maps[i][val_map]
                    indices2 = index_maps[j][val_map]
                    val_merge = merge_data(self, indices1, self, indices2, 
                                            time=sfdata.time if self.opt.phase=="test" else None)
                    update_val_map = torch.ones_like(val_map)
                    update_val_map[val_map] = ~val_merge
                    val_maps[j] &= update_val_map

                    indices2 = indices2[val_merge]
                    del_indices.append(indices2)
                    if hasattr(self, 'track_pts'):
                        indices1 = indices1[val_merge]
                        for k, tid in enumerate(self.track_id):
                            if tid in indices2:
                                self.track_id[k] = indices1[indices2==tid].type(torch.long)
        
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

        if add_valid is None:
            print(f'add_valid is None at frame {self.time}')
            self.logger.warning(f'add_valid is None at frame {self.time}')

        ## Update the knn weights of existing surfels.
        if self.opt.method == "semantic-super":
            dists = torch_distance(self.points.unsqueeze(1), \
                self.ED_nodes.points[self.knn_indices])
            radii = self.ED_nodes.radii[self.knn_indices]

            P = self.ED_nodes.seg_conf[self.knn_indices]
            Q = self.seg_conf[:, None, :]
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

        if not self.opt.disable_adding_new_surfels and add_valid is not None:
            ## Add points that do not have corresponding surfels.
            add_valid = add_valid[sfdata.valid]
            if add_valid.count_nonzero() > 0:
                # Extract new surfels and check if each of them has corresponding ED nodes.
                new_points = sfdata.points[add_valid]
                # Update the knn weights and indicies of new surfels.
                new_isStable = torch.ones(len(new_points), dtype=torch.bool).cuda()
                if self.hard_seg:
                    dists, new_knn_indices = find_knn(new_points, self.ED_nodes.points, 
                        k=self.opt.num_neighbors, num_classes=self.opt.num_classes, 
                        seg1=sfdata.seg[add_valid], seg2=self.ED_nodes.seg)
                else:
                    dists, new_knn_indices = find_knn(new_points, self.ED_nodes.points, 
                        k=self.opt.num_neighbors)
                radii = self.ED_nodes.radii[new_knn_indices]
                new_isStable[~torch.any(dists <= radii, dim=1)] = False # If surfels are too far away from its knn ED nodes, this surfels will be disabled.
                if self.opt.method == "semantic-super" and not self.hard_seg:
                    P = self.ED_nodes.seg_conf[new_knn_indices]
                    Q = sfdata.seg_conf[add_valid][:, None, :]
                    new_knn_w = F.softmax(
                                    torch.pow(torch.exp(- JSD(P, Q)), self.power_arg[0]) * \
                                    torch.pow(torch.exp(- dists / radii), self.power_arg[1])
                                    , dim=-1)
                else:
                    new_knn_w = F.softmax(torch.exp(- dists / radii), dim=-1)

                self.isStable = torch.cat([self.isStable, torch.ones(torch.count_nonzero(new_isStable), dtype=torch.bool).cuda()])
                self.knn_w = torch.cat([self.knn_w, new_knn_w[new_isStable]], dim=0)
                self.knn_indices = torch.cat([self.knn_indices, new_knn_indices[new_isStable]], dim=0)

                
                new_sf_num = new_isStable.count_nonzero()
                # FIXME: Make this compatible with self.velocities
                self.points = torch.cat([self.points, new_points[new_isStable]], dim=0)
                for key, v in sfdata.items():
                    if key in ['norms', 'colors', 'velocities']: 
                        v = torch.cat([getattr(self, key), v[add_valid][new_isStable]], dim=0)
                    elif key in ['seg', 'seg_conf', 'radii', 'confs', 'dist2edge']:
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
                # self.update_ed(points=new_points[isED], norms=new_norms[isED])
                
        v, u, _, _ = pcd2depth(inputs, self.points, round_coords=False)
        self.projdata = torch.stack([u, v], dim=1).type(torch.float32)

    def prepareStableIndexNSwapAllModel(self, inputs, sfdata):
        ''' Delete unstable surfels & ED nodes.
        Inputs:
            - inputs: a dict corresponding to a dataloader item containing the following keys:
                ['filename', 'ID', 'time', ('color', 0), 'K', 'inv_K', 'divterm', 
                'stereo_T', ('color_aug', 0), ('seg_conf', 0), ('seg', 0), ('disp', 0), ('depth', 0)]
            - sfdata: torch_geometric.data object returned by `data_loader.depth_preprocessing()`. 
            Contains the following attributes:
                [points, norms, colors, radii, confs(of valid verts), validverts, index_map, time]
                
        '''
        if not self.opt.disable_removing_unstable_surfels:
            # A surfel is unstable if it 1) hasn't been updated for a long time,
            # and 2) has low confidence.
            self.isStable = self.isStable & (inputs["time"]-self.time_stamp < self.opt.th_time_steps) # self.confs >= self.opt.th_conf
            if self.evaluate_tracking:
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
            if hasattr(self, 'velocities'):
                self.velocities = self.velocities[self.isStable]
            if hasattr(self, "seg"):
                self.seg = self.seg[self.isStable]
                self.seg_conf = self.seg_conf[self.isStable]
                self.dist2edge = self.dist2edge[self.isStable]
            if self.evaluate_tracking:
                id_map = - torch.ones(len(self.isStable), dtype=torch.long).cuda()
                id_map[self.isStable] = torch.arange(torch.count_nonzero(self.isStable)).cuda()
                track_id_val = self.track_id >= 0
                self.track_id[track_id_val] = id_map[self.track_id[track_id_val]]
            self.isStable = self.isStable[self.isStable]

        if self.evaluate_tracking:
            inval_id = (self.track_id >= 0) & torch.tensor(
                [False if tid == torch.tensor(-2) else ~self.isStable[tid.type(torch.long)] for tid in self.track_id]).cuda()
            if inval_id.count_nonzero() > 0:
                self.track_id[inval_id] = torch.tensor(-2)
        
        self.surfel_num = self.isStable.count_nonzero()

        if self.opt.phase == 'test' and self.time % self.opt.save_sample_freq == 0:
            self.summary_writer.add_scalar(f"graph_info/Number of surfels", self.surfel_num, self.time)

        if self.evaluate_tracking:
            if (self.track_id >= 0).count_nonzero() > 0:
                self.update_track_pts(sfdata, inputs["filename"][0])

            if (self.track_id == -1).count_nonzero() > 0:
                self.init_track_pts(sfdata, inputs["filename"][0])
        
        # Visualize tracked point cloud.
        if self.opt.phase == 'test' and self.time % self.opt.save_sample_freq == 0:
            if 'v1_520_pairs' in self.opt.data_dir:
                view_params = (45, 125)
            elif 'trial_3' in self.opt.data_dir or 'trial_4' in self.opt.data_dir:
                view_params = (50, 55)
            elif 'trial_8' in self.opt.data_dir or 'trial_9' in self.opt.data_dir:
                view_params = (50, 35)
            elif '090923_t2' in self.opt.data_dir:
                view_params = (10, -10)
            elif '082323_traj_4points_t2' in self.opt.data_dir:
                view_params = (0, 0)
            elif '082323_traj_4points_t4' in self.opt.data_dir:
                view_params = (0, 0)
            else:
                view_params = None

            pcd_image = plot_pcd(self.points.cpu().numpy(), 
                                 self.colors.cpu().numpy(),
                                 view_params = view_params)
            self.summary_writer.add_image('visualization/pcd', 
                                          torch.from_numpy(pcd_image).permute(2, 0, 1), 
                                          self.time)

        self.render_img(inputs) # Render the current 3D model to an image.
        if self.time % self.opt.save_sample_freq == 0:
            self.viz(inputs, sfdata)
    
    ''' For Prob/Bayesian SuPer '''
    def render_(self, inputs):
        ''' Helper for render_img. See render_img below '''
        data = Data(points=self.points[self.isStable], 
                    colors=self.colors[self.isStable])
        confidence_heat = conf2color(self.confs)[self.isStable]
        
        self.renderImg = self.models.renderer(
                inputs, data, colors = data.colors, rad=self.opt.renderer_rad
        ).permute(2,0,1).unsqueeze(0)
        
        self.renderImg_conf_heat = self.models.renderer(
            inputs, data, colors = confidence_heat, rad=self.opt.renderer_rad
        ).permute(2,0,1).unsqueeze(0)

    def render_img(self, inputs):  
        ''' Render two images and store in a `Surfels` object.
            img1: Colorful reconstruction of the original scene
            img2: Confidence heat map (using plt `magma` color map)
        '''
        with torch.no_grad(): 
            self.render_(inputs)

    def viz(self, inputs, sfdata, bid=0): 
        ''' 
        Visualize tracking, reconstruction results
        '''
        
        def get_valid_keypoints(keypoints):
            return keypoints[:, 0:2][keypoints[:,2] == 1]

        filename = inputs["filename"][bid]

        self.summary_writer.add_image('visualization/raw', inputs[("color", 0)][bid], self.time)

        # disparity (inverse depth)
        if ("disp",0) in inputs:
            disp = inputs[("disp",0)][bid,0]
        else:
            disp = 1 / (inputs[("depth",0)][bid,0] + 1e-13)
        disp[torch.isnan(disp)] = 0
        if self.opt.data == 'superv1':
            if self.opt.depth_model == 'raft_stereo':
                disp = 255 * torch.clip(0.02 * disp, 0., 1.)
            else:
                disp = 255 * torch.clip(0.15 * disp, 0., 1.)
        elif self.opt.data == 'superv2':
            disp = 255 * torch.clip(0.9 * disp, 0., 1.)
        disp = cv2.applyColorMap(disp.cpu().numpy().astype(np.uint8), cv2.COLORMAP_MAGMA)
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        disp = torch.as_tensor(disp).permute(2, 0, 1)
        self.summary_writer.add_image('visualization/disparity', disp, self.time)
        
        render_img = self.renderImg[bid]
        render_img = (255*render_img).type(torch.uint8)
        
        # Draw tracked points on rendered img.
        colors = {"gt": "blue", "super_cpp": "magenta", "SURF": "lime"}
        if self.evaluate_tracking:
            render_img_keypoints = copy.deepcopy(render_img)

            for key in self.track_pts:
                if filename in self.track_pts[key]:
                    keypoints = self.track_pts[key][filename]
                    render_img_keypoints = draw_keypoints(render_img_keypoints, 
                                                          get_valid_keypoints(keypoints)[None, :, :], 
                                                          colors=colors[key]
                                                         )

            if filename in self.track_rsts:
                keypoints = self.track_rsts[filename]
                render_img_keypoints = draw_keypoints(render_img_keypoints, 
                                                      get_valid_keypoints(keypoints)[None, :, :],
                                                      colors="red"
                                                     )
        self.summary_writer.add_image('visualization/render', render_img_keypoints, self.time)

        # Overlay deformed mesh onto RGB image.
        ed_y, ed_x, _, _ = pcd2depth(inputs, self.ED_nodes.points)
        ed_pts = torch.stack([ed_x, ed_y], dim=1).cpu().numpy().astype(int)
        render_img = render_img.permute(1, 2, 0).cpu().numpy()
        for k, edge_id1 in enumerate(self.ED_nodes.edge_index[0]):
            edge_id2 = self.ED_nodes.edge_index[1][k]
            edge_color = (255, 255, 255)

            pt1 = ed_pts[edge_id1]
            pt2 = ed_pts[edge_id2]
            render_img = cv2.line(render_img, pt1, pt2, edge_color, 1)
        self.summary_writer.add_image('visualization/mesh', torch.as_tensor(render_img).permute(2, 0, 1), self.time)

        # uncertainty heatmap
        if self.renderImg_conf_heat is not None:
            render_img_conf_heat = self.renderImg_conf_heat
            render_img_conf_heat = (255*render_img_conf_heat).type(torch.uint8)[0]
            self.summary_writer.add_image('visualization/uncertainty', render_img_conf_heat, self.time)

        # segmentation map
        if hasattr(self, 'seg'):
            seg = inputs[("seg", 0)][bid, 0]
            if self.opt.data == 'superv1':
                seg = binary_id2color[seg].permute(2, 0, 1)
                data = Data(points=sfdata.points, 
                            colors=binary_id2color[sfdata.seg].to(sfdata.points.device)
                           )
            else:
                seg = id2color[seg].permute(2, 0, 1)
                data = Data(points=sfdata.points, 
                            colors=id2color[sfdata.seg].to(sfdata.points.device)
                           )
            
            self.summary_writer.add_image('visualization/seg_pred', seg, self.time)

            render_seg = self.models.renderer(inputs, 
                                              data, 
                                              rad=self.opt.renderer_rad).permute(2, 0, 1)
            self.summary_writer.add_image('visualization/seg_render', render_seg, self.time)

            if hasattr(self, "seg_conf"):
                seg_pred_conf = self.models.renderer(inputs, 
                                                     Data(points=self.points[self.isStable], 
                                                          colors=self.seg_conf[self.isStable].max(1)[0][:, None].repeat(1, 3)), 
                                                     rad=self.opt.renderer_rad)
                seg_pred_conf = (seg_pred_conf * 255).type(torch.uint8).permute(2, 0, 1)
                self.summary_writer.add_image('visualization/seg_pred_conf', seg_pred_conf, self.time)

    def evaluate(self):
        ''' Expect the following result format (can be encapsulated in a list wrapper, due to the original
        impementation's attempt of support multiple results for multiple approaches):
        dict containing {
            '000010': (20, 3) tensor of homogeneous screen coord. of 20 tracked points at time 000011.
            '000020': (20, 3) tensor of homogeneous screen coord. of 20 tracked points at time 000012.
            ...
            '000510': (20, 3) tensor of homogeneous screen coord. of 20 tracked points at time 000520.
        }
        
        Inputs:
            - inputs: a dict from dataloader containing the following keys:
                ['filename', 'ID', 'time', ('color', 0), 'K', 'inv_K', 'divterm', 
                'stereo_T', ('color_aug', 0), ('seg_conf', 0), ('seg', 0), ('disp', 0), ('depth', 0)]
        '''
        if len(self.track_rsts.keys()) == 0: return
        
        gt, gt_intkeys, gt_strkeys, gt_array = self.gt, self.gt_intkeys, self.gt_strkeys, self.gt_array

        # Build result array from self.track_rsts
        for strkey_rsts in self.track_rsts.keys() & set(self.gt_strkeys):
            if strkey_rsts not in gt_strkeys or strkey_rsts in self.tracking_eval_errors: 
                continue
            self.tracking_eval_errors[strkey_rsts] = evaluate(gt[strkey_rsts].numpy(), 
                                                              self.track_rsts[strkey_rsts].cpu().numpy())

        # Add SuPer C++ results as reference.
        if 'super_cpp' in self.track_pts:
            super_cpp_err_array = []
            for k in sorted(self.tracking_eval_errors.keys()):
                if k in self.super_cpp_eval_errors:
                    super_cpp_err_array.append(self.super_cpp_eval_errors[k])
            if len(super_cpp_err_array) > 0:
                super_cpp_err_array = np.stack(super_cpp_err_array, axis=0)
                self.summary_writer.add_scalar('reprojerr/supercpp_mean', np.mean(super_cpp_err_array), self.time)
                self.summary_writer.add_scalar('reprojerr/supercpp_std', np.std(super_cpp_err_array), self.time)

        log_trackpts_err(
            err_array = np.stack([self.tracking_eval_errors[k]
                for k in sorted(self.tracking_eval_errors.keys())], axis=0), # (N, 20), 
            res_array = np.stack([self.track_rsts[k].cpu().numpy()
                for k in sorted(self.track_rsts.keys())], axis=0),   # (N, 20, 3) 
            edge_ids = self.opt.edge_ids if hasattr(self.opt, 'edge_ids') else [],
            logdir = self.output_dir,
            gt_array = self.gt_array, # (N, 20, 3)
            summary_writer=self.summary_writer, 
            time=self.time
        )    

        return