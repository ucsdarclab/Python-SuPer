import cv2

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from super.utils import *

from utils.config import *
from utils.utils import *


class Surfels():

    def __init__(self, model_args, data):
        self.model_args = model_args

        if model_args['evaluate_tracking']:
            tracking_gt_file = os.path.join(model_args['data_dir'], model_args['tracking_gt_file'])
            self.track_pts = np.array(np.load(tracking_gt_file, allow_pickle=True)).tolist()
            for key in self.track_pts.keys():
                for filename in self.track_pts[key]:
                    self.track_pts[key][filename] = numpy_to_torch(self.track_pts[key][filename], dtype=int_)
            
            track_gt = self.track_pts["gt"]
            self.track_num = len(list(track_gt.items())[0][1])
            self.track_id = - torch.ones((self.track_num,), device=dev)
            # -1: Haven't started tracking; 
            # >=0: ID of tracked points; 
            # nan: Lost tracking.
            self.track_rsts = {}

        for key, v in data.items():
            if key == 'valid':
                continue
            setattr(self, key, v)
        self.sf_num = len(self.points)
        self.isStable = torch.ones((self.sf_num,), dtype=bool_, device=dev)
        self.time_stamp = self.time * torch.ones(self.sf_num, device=dev)

        # if 'seman-super' in model_args['method']:
        #     self.seg_conf = new_data.seg_conf
        #     self.class_num = new_data.seg_conf.size(-1)

        # self.validmap = self.valid.view(HEIGHT,WIDTH)
        # valid_indexs = torch.arange(self.sf_num, dtype=int, device=dev).unsqueeze(1)
        # valid_coords = torch.flip(self.validmap.nonzero(), dims=[-1])
        # # Column direction of self.projdata:
        # # 1) index of projected points in self.points, 
        # # corresponding projection coordinates (x,y) on the image plane 
        # self.projdata = torch.cat([valid_indexs,valid_coords], dim=1).type(tfdtype_)

        self.projdata = torch.flip(
            data.valid.view(model_args['CamParams'].HEIGHT, model_args['CamParams'].WIDTH).nonzero(), 
            dims=[-1]).type(fl32_)
        

        # init ED nodes
        self.update_ED()

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
    
    # Init/update ED nodes & ED nodes related parameters. TODO
    def update_ED(self, new_nodes=None):
        if new_nodes is None: # Init.
            # Find 8 neighbors of ED nodes.
            dists, sort_idx = find_knn(self.ED_nodes.points, self.ED_nodes.points, 
                k=self.model_args['CamParams'].ED_n_neighbors+1)
            dists = dists[:, 1:]
            sort_idx = sort_idx[:, 1:]
            self.ED_nodes.knn_w = F.softmax(torch.exp(- dists), dim=-1)
            # if 'seman-super' in self.model_args['method']:
            #     seg_cls = torch.argmax(self.ED_nodes.seg_conf, dim=-1)
            #     self.ED_nodes.ED_knn_w[
            #         ~(seg_cls.unsqueeze(1).repeat(1, ED_n_neighbors)==seg_cls[sort_idx])
            #         ] = 0.
            self.ED_nodes.knn_indices = sort_idx
            
            # Find 4 neighboring ED nodes of surfels.
            dists, self.knn_indices = find_knn(self.points, self.ED_nodes.points,
                k=self.model_args['CamParams'].n_neighbors)
            radii = (self.ED_nodes.radii[0]**2)[self.knn_indices]
            self.knn_w = F.softmax(torch.exp(- dists / radii), dim=-1)
            # if 'seman-super' in self.model_args['method']:
            #     sf_seg_cls = torch.argmax(self.seg_conf, dim=-1)
            #     weights[
            #         ~(sf_seg_cls.unsqueeze(1).repeat(1, n_neighbors)==seg_cls[sort_idx])
            #         ] = 0.

            # if self.model_args['do_segmentation']:
            #     self.ED_nodes.points = self.ED_nodes.points.unsqueeze(0).repeat(self.class_num, 1, 1)
            #     self.ED_nodes.norms = self.ED_nodes.norms.unsqueeze(0).repeat(self.class_num, 1, 1)


    def update(self, deform, time):
        """
        Update surfels and ED nodes with their motions estimated by optimizor.
        """
        sf_knn = self.ED_nodes.points[self.knn_indices] # All g_i in (10).
        sf_diff = self.points.unsqueeze(1) - sf_knn
        deform_ = deform[self.knn_indices]
        self.points, _ = Trans_points(sf_diff, sf_knn, deform_, self.knn_w)

        norms, _ = transformQuatT(
            self.norms.unsqueeze(1).repeat(1,self.model_args['CamParams'].n_neighbors,1),
            deform_[...,0:4])
        norms = torch.sum(self.knn_w.unsqueeze(-1) * norms, dim=-2)
        self.norms = torch.nn.functional.normalize(norms, dim=-1)

        self.time_stamp = time * torch.ones_like(self.time_stamp)
        
        self.ED_nodes.points += deform[:,4:]
        self.ED_nodes.norms, _ = transformQuatT(self.ED_nodes.norms, deform[...,0:4])
    

    # Fuse the input data into our reference model.
    def fuseInputData(self, sfdata):
        # Return data[indices]. 'indices' can be either indices or True/False map.
        def get_data(indices, data):
            return data.points[indices], data.norms[indices], data.colors[indices], \
                data.radii[indices], data.confs[indices]

        # Merge data1 & data2, and update self.data[indices].
        def merge_data(data1, indices1, data2, indices2, time):
            p, n, c, r, w = get_data(indices1, data1)
            p_new, n_new, c_new, r_new, w_new = get_data(indices2, data2)

            # Only merge points that are close enough.
            valid = (torch_distance(p, p_new) < self.model_args['CamParams'].THRESHOLD_DISTANCE) & \
                (torch_inner_prod(n, n_new) > self.model_args['CamParams'].THRESHOLD_COSINE_ANGLE)
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
            self.colors[indices] = w * c[valid] + w_new * c_new[valid]
            self.time_stamp[indices] = time.type(fl32_) # Update time stamps.

            return valid

        valid = sfdata.valid.clone()
        # time = new_data.time

        ## Project surfels onto the image plane. For each pixel, only up to 16 projections with higher confidence.
        # Ignore surfels that have projections outside the image.
        _, _, _, indices_ = pcd2depth(self.model_args['CamParams'], self.points)
        indices_ = indices_ & self.isStable
        indices_ = indices_.nonzero(as_tuple=True)[0]
        points_ = self.points[indices_]
        conf_ = self.confs[indices_]
        # Get the projection maps 1) valid & 2) surfel indices in self.points,
        # map size: map_num x PIXEL_NUM)
        map_num = 16
        val_maps = []
        index_maps = []
        for i in range(map_num):
            if len(points_) == 0: break

            _, _, temp_coords, temp_indices = \
                pcd2depth(self.model_args['CamParams'], points_, conf_sort=True, conf=conf_)
            
            val_map = torch.zeros(
                (self.model_args['CamParams'].HEIGHT * self.model_args['CamParams'].WIDTH), 
                dtype=bool_, device=dev)
            val_map[temp_coords] = True
            val_maps.append(val_map)

            index_map = torch.zeros(
                (self.model_args['CamParams'].HEIGHT * self.model_args['CamParams'].WIDTH),
                dtype=long_, device=dev)
            index_map[temp_coords] = indices_[temp_indices]
            index_maps.append(index_map)

            points_ = torch_delete(points_, temp_indices)
            conf_ = torch_delete(conf_, temp_indices)
            indices_ = torch_delete(indices_, temp_indices)
        # Init indices of surfels that will be deleted.
        del_indices = [indices_] if len(indices_) > 0 else []

        # Init valid map of new points that will be added.
        add_valid = valid & (~val_maps[0])
        ## Merge new points with existing surfels.
        valid[add_valid] = False
        for val_map, index_map in zip(val_maps, index_maps):
            if not torch.any(valid): break

            val_ = valid & val_map
            index_ = index_map[val_]
            merge_val_ = merge_data(self, index_, sfdata, val_[sfdata.valid], sfdata.time)
            valid[val_] = ~merge_val_
        add_valid |= valid

        ## Merge paired surfels.
        map_num = len(val_maps)
        for i in range(map_num):
            val_ = val_maps[i]

            for j in range(i+1, map_num):
                val_ &= val_maps[j]
                if not torch.any(val_): continue

                indices1 = index_maps[i][val_]
                indices2 = index_maps[j][val_]
                merge_val_ = merge_data(self, indices1, self, indices2, sfdata.time)
                val_maps[j][val_] = ~merge_val_
                
                indices2 = indices2[merge_val_]
                del_indices.append(indices2)
                if hasattr(self, 'track_pts'):
                    indices1 = indices1[merge_val_]
                    for k, index_ in enumerate(indices2):
                        m = (self.track_id==index_).nonzero(as_tuple=True)[0]
                        if len(m) > 0:
                            self.track_id[m] = indices1[k].type(fl32_)
        
        ## Delete redundant surfels.
        if len(del_indices) > 0:
            del_indices = torch.unique(torch.cat(del_indices))

            if hasattr(self, 'track_pts'):
                for k, tid in enumerate(self.track_id):
                    if torch.isnan(tid):
                        continue
                    if tid in del_indices:
                        self.track_id[k] = torch.tensor(np.nan)
            
            self.isStable[del_indices] = False
            # self.surfel_num = len(self.points)
        
        ## Add points that do not have corresponding surfels.
        add_valid = add_valid[sfdata.valid]
        # new_point_num = torch.count_nonzero(add_valid)
        # if new_point_num > 0:
        if add_valid.count_nonzero() > 0:
            ## Update the knn weights of existing surfels.
            dists = torch_distance(self.points.unsqueeze(1), \
                self.ED_nodes.points[self.knn_indices])
            radii = (self.ED_nodes.radii[0]**2)[self.knn_indices]
            self.knn_w = F.softmax(torch.exp(- dists / radii), dim=-1)
            new_sf_num = add_valid.count_nonzero()
            for key, v in sfdata.items():
                if key in ['points', 'norms', 'colors']:
                    v = torch.cat([getattr(self, key), v[add_valid]], dim=0)
                elif key in ['radii', 'confs']:
                    v = torch.cat([getattr(self, key), v[add_valid]])
                elif key == 'time':
                    v = torch.cat([self.time_stamp, v*torch.ones(new_sf_num, device=dev)])
                    key = "time_stamp"
                else:
                    continue
                setattr(self, key, v)

            self.isStable = torch.cat([self.isStable, torch.ones((new_sf_num,), dtype=bool_, device=dev)])
            
            # ## Update isED: Only points that are far enough from 
            # ## existing ED nodes can be added as new ED nodes.
            # D = torch.cdist(new_points, self.ED_nodes.points)
            # new_knn_dists, _ = D.topk(k=1, dim=-1, largest=False, sorted=True)
            # isED = new_data.isED[add_valid] & (new_knn_dists[:,-1] > torch.max(sf_knn_dists))
            # self.update_ED(points=new_points[isED], norms=new_norms[isED])
            
            ## Update the knn weights and indicies of new surfels.
            dists, new_knn_indices = find_knn(self.points[-new_sf_num:], self.ED_nodes.points, 
                k=self.model_args['CamParams'].n_neighbors)
            radii = (self.ED_nodes.radii[0]**2)[new_knn_indices]
            new_knn_w = F.softmax(torch.exp(- dists / radii), dim=-1)

            self.knn_w = torch.cat([self.knn_w, new_knn_w], dim=0)
            self.knn_indices = torch.cat([self.knn_indices, new_knn_indices], dim=0)

        v, u, _, valproj = pcd2depth(self.model_args['CamParams'], self.points, round_coords=False)
        self.projdata = torch.stack([u[valproj], v[valproj]], dim=1).type(fl32_)


    # If evaluate on the SuPer dataset, init self.label_index which includes
    # the indicies of the tracked points.
    def init_track_pts(self, sfdata, filename, th=0.1):
        if not filename in self.track_pts["gt"]:
            return
        gt_coords = self.track_pts["gt"][filename]
        
        for k, tid in enumerate(self.track_id):
            # The point hasn't been tracked. & Ground truth exists for this point.
            x, y, v = gt_coords[k]
            gt_id = sfdata.index_map[y,x]
            if tid < 0 and gt_id > 0 and v == 1:
                dists = torch_distance(self.points, sfdata.points[gt_id])
                inval_id = self.track_id[(self.track_id >= 0) | torch.isnan(self.track_id)]
                if len(inval_id) > 0:
                    dists[inval_id.type(long_)] = 1e13

                if torch.min(dists) < th:
                    self.track_id[k] = torch.argmin(dists)

    # If evaluate on the SuPer dataset, update self.label_index.
    def update_track_pts(self, sfdata, filename, th=1e-2):
        self.track_rsts[filename] = torch.zeros((self.track_num, 3), device=dev)
        
        for k, tid in enumerate(self.track_id):
            if tid >= 0:
                self.track_rsts[filename][k, 0:2] = self.projdata[tid.type(long_)] # self.projdata[tid.type(long_), 1:]
                self.track_rsts[filename][k, 2] = 1

    # Delete unstable surfels & ED nodes.
    def prepareStableIndexNSwapAllModel(self, model, data, sfdata):
        # A surfel is unstable if it 1) hasn't been updated for a long time,
        # and 2) has low confidence.
        self.isStable = self.isStable & \
            ((data["time"]-self.time_stamp < self.model_args['CamParams'].STABLE_TH) |\
             (self.confs >= self.model_args['CamParams'].CONF_TH))

        inval_id = (self.track_id >= 0) & torch.tensor(
            [False if torch.isnan(tid) else ~self.isStable[tid.type(long_)] for tid in self.track_id],
            device=dev)
        if inval_id.count_nonzero() > 0:
            self.track_id[inval_id] = torch.tensor(np.nan)
        
        self.surfel_num = self.isStable.count_nonzero()

        if hasattr(self, 'track_pts'):
            if (self.track_id >= 0).count_nonzero() > 0:
                self.update_track_pts(sfdata, data["filename"][0])

            if (self.track_id < 0).count_nonzero() > 0:
                self.init_track_pts(sfdata, data["filename"][0])

        # v, u, coords, valid_indexs = pcd2depth(self.points, depth_sort=True, round_coords=False)
        # self.valid = torch.zeros(PIXEL_NUM, dtype=bool_, device=dev)
        # self.valid[coords] = True
        # self.validmap = self.valid.view(HEIGHT,WIDTH)
        # self.projdata = torch.stack([valid_indexs,v,u],dim=1)
        
        # Render the current 3D model to an image.
        self.render_img(model["renderer"])

        if sfdata.ID[0]%10 == 0:
            self.viz(data)

    def render_img(self, renderer):
        data = Data(points=self.points[self.isStable], colors=self.colors[self.isStable])
        self.renderImg = renderer(self.model_args['CamParams'], data).permute(2,0,1).unsqueeze(0)
        
        # self.projGraph = self.renderer(self.ED_nodes).permute(2,0,1).unsqueeze(0)

    # Visualize the tracking & reconstruction results.
    def viz(self, data, bid=0):
        def draw_keypoints_(img_, keypoints, colors="red"):
            keypoints = keypoints[:, 0:2][keypoints[:,2] == 1].unsqueeze(0)
            if len(keypoints) > 0:
                img_ = draw_keypoints(img_, keypoints, colors=colors, radius=3)
            return img_
        
        colors = {"gt": "blue", "super_cpp": "magenta", "SURF": "lime"}

        filename = data["filename"][0]

        render_img = de_normalize(self.renderImg)[bid]
        render_img = (255*render_img).type(torch.uint8).cpu()

        img = de_normalize(data[("color", 0, 0)])[bid]
        img = (255*img).type(torch.uint8).cpu()

        # render_img[self.projGraph > 0] = 1.
        if self.model_args['evaluate_tracking']:
            for key in self.track_pts:
                if filename in self.track_pts[key]:
                    keypoints = self.track_pts[key][filename]
                    render_img = draw_keypoints_(render_img, keypoints, colors=colors[key])

            if filename in self.track_rsts:
                keypoints = self.track_rsts[filename]
                render_img = draw_keypoints_(render_img, keypoints)

        out = torch.cat([render_img, img], dim=1)
        out = torch_to_numpy(out.permute(1,2,0))

        cv2.imwrite(os.path.join(self.model_args['sample_dir'], filename+'.png'), \
            out[:,:,::-1])