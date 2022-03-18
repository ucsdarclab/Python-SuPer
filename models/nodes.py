import open3d as o3d
import torch
from torch_geometric.data import Data
import cv2

from utils.config import *
from utils.utils import *
from utils.renderer import *

class Surfels():

    def __init__(self, new_data):

        self.valid = new_data.valid

        # Init surfels.
        self.points = new_data.points[self.valid]
        self.norms = new_data.norms[self.valid]
        self.colors = new_data.colors[self.valid]
        if qual_color_eval:
            self.eval_colors = init_qual_color(self.points)
        self.rad = new_data.rad[self.valid]
        self.conf = new_data.conf[self.valid]
        self.surfel_num = len(self.points)
        self.time_stamp = new_data.time * torch.ones(self.surfel_num, device=dev)

        self.evaluate_super = False

        self.validmap = self.valid.view(HEIGHT,WIDTH)
        valid_indexs = torch.arange(self.surfel_num, dtype=int, device=dev).unsqueeze(1)
        valid_coords = self.validmap.nonzero()
        # Column direction of self.projdata:
        # 1) index of projected points in self.points, 
        # corresponding projection coordinates (y,x) on the image plane 
        self.projdata = torch.cat([valid_indexs,valid_coords], dim=1).type(tfdtype_)

        # init ED nodes
        isED = new_data.isED[self.valid]
        self.update_ED(points=self.points[isED], norms=self.norms[isED], init=True)

        # self.rgb = rgb
        self.ID = new_data.ID
        
        # TODO
        # if open3d_visualize:
        #     self.pcd = []
        #     self.ED_pcd = []
        #     self.get_o3d_pcd(init=True)
        # self.get_o3d_pcd(init=True)

        # Init renderer.
        if render_method == 'proj': self.renderer = Projector()
        elif render_method == 'pulsar': self.renderer = Pulsar()
    
    # Init/update ED nodes & ED nodes related parameters. TODO
    def update_ED(self, points=None, norms=None, init=False):
        
        ## Grid Sample
        if init:
            if ED_sample_method == 'uniform':
                points, ED_indices = find_ED_nodes(points)
                norms = norms[ED_indices]

            num = len(points)
            self.ED_nodes = Data(points=points, norms=norms, \
                colors=init_qual_color(points, margin=50.), \
                num=num, param_num=num*7)

            # Find 8 neighbors of ED nodes.
            _, self.ED_knn_indices, _ = update_KNN_weights(points=points, \
                targets=points, n_neighbors_=ED_n_neighbors)

            self.sf_knn_weights, self.sf_knn_indices, self.sf_knn_div_indices = \
                update_KNN_weights(points=self.points, targets=points, n_neighbors_=n_neighbors)

        elif points is not None:
            if ED_sample_method == 'grid':
                self.ED_nodes.points = torch.cat([self.ED_nodes.points, points], dim=0)
                self.ED_nodes.norms = torch.cat([self.ED_nodes.norms, norms], dim=0)
                self.ED_nodes.colors = torch.cat([self.ED_nodes.colors, \
                    255 * torch.ones((len(points), 3), device=dev)], dim=0)

            elif ED_sample_method == 'uniform':
                self.ED_nodes.points, new_indices = find_ED_nodes(points, EDs=self.ED_nodes.points)
                points = self.ED_nodes.points[self.ED_nodes.num:]
                if len(new_indices) > 0:
                    self.ED_nodes.norms = torch.cat([self.ED_nodes.norms, norms[new_indices]], dim=0)
                    self.ED_nodes.colors = torch.cat([self.ED_nodes.colors, \
                        255 * torch.ones((len(new_indices), 3), device=dev)], dim=0)

            self.ED_nodes.num = len(self.ED_nodes.points)
            self.ED_nodes.param_num = self.ED_nodes.num*7

            # Add 8 neighbors of new ED nodes.
            _, new_ED_knn_indices, _ = update_KNN_weights(points=points, \
                targets=self.ED_nodes.points, n_neighbors_=ED_n_neighbors)
            self.ED_knn_indices = torch.cat([self.ED_knn_indices, new_ED_knn_indices], dim=0)
            
        self.ED_knn_indices = self.ED_knn_indices.contiguous()

    # If evaluate on the SuPer dataset, init self.label_index which includes
    # the indicies of the tracked points.
    def init_track_pts(self, eva_ids, labelPts):
        self.evaluate_super = True
        self.eva_ids = eva_ids
        self.labelPts = labelPts

        self.label_index = torch.zeros(20, dtype=long_, device=dev)
        for k, pt in enumerate(labelPts['gt'][0]):
            Y, X = round(pt[0]), round(pt[1])
            self.label_index[k] = ((self.projdata[:,1].long()==Y) & \
                (self.projdata[:,2].long()==X)).nonzero(as_tuple=True)[0]

        self.track_rsts = []

    # If evaluate on the SuPer dataset, update self.label_index.
    def update_track_pts(self, ID, labelPts, rgb):
        
        # Save tracking results.
        evaluate_id = np.argwhere(self.eva_ids==ID)
        if len(evaluate_id) > 0:
            evaluate_id = np.squeeze(evaluate_id)

            v, u, _, _ = pcd2depth(self.points[self.label_index], round_coords=False)
            self.track_rsts.append(torch_to_numpy(torch.stack((v, u), dim=-1)))

            if vis_super_track_rst:
                filename = os.path.join(F_super_track,str(evaluate_id+1)+".jpg")
                vis_track_rst(labelPts, self.track_rsts[-1], evaluate_id, rgb, filename)

    # Init/update open3d data for display. TODO
    def get_o3d_pcd(self, init=False):

        # pcd for surfels
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(self.colors.cpu().numpy()/255.)
        
        # # pcd for ED nodes
        # ed_pcd = o3d.geometry.PointCloud()
        # ed_pcd.points = o3d.utility.Vector3dVector(self.ED_points)
        # display_colors = np.zeros_like(self.ED_colors)
        # display_colors[:,0] = 1.0
        # ed_pcd.colors = o3d.utility.Vector3dVector(display_colors)

        self.pcd = pcd
        # self.ED_pcd = ed_pcd

    # Update surfels and ED nodes with their motions estimated by optimizor.
    def update(self, deform):
        sf_knn = self.ED_nodes.points[self.sf_knn_indices] # All g_i in (10).
        sf_diff = self.points.unsqueeze(1) - sf_knn
        deform_ = deform[self.sf_knn_indices]
        self.points, _ = Trans_points(sf_diff, sf_knn, deform_, self.sf_knn_weights)

        norms, _ = transformQuatT(self.norms.unsqueeze(1).repeat(1,n_neighbors,1), \
            deform_[...,0:4])
        norms = torch.sum(self.sf_knn_weights.unsqueeze(-1) * norms, dim=-2)
        self.norms = torch.nn.functional.normalize(norms, dim=-1)
        
        self.ED_nodes.points += deform[:,4:]
        self.ED_nodes.norms, _ = transformQuatT(self.ED_nodes.norms, deform[...,0:4])

    # Delete unstable surfels & ED nodes.
    def prepareStableIndexNSwapAllModel(self, time, ID, init=False):

        # A surfel is unstable if it 1) hasn't been updated for a long time,
        # and 2) has low confidence.
        isStable = (time-self.time_stamp < STABLE_TH) | (self.conf >= CONF_TH)

        self.points = self.points[isStable]
        self.norms = self.norms[isStable]
        self.colors = self.colors[isStable]
        if qual_color_eval:
            self.eval_colors = self.eval_colors[isStable]
        self.conf = self.conf[isStable]
        self.rad = self.rad[isStable]
        self.time_stamp = self.time_stamp[isStable]
        self.surfel_num = len(self.points)

        self.sf_knn_indices = self.sf_knn_indices[isStable]
        self.sf_knn_div_indices = self.sf_knn_div_indices[isStable]
        self.sf_knn_weights = self.sf_knn_weights[isStable]

        if not init:
            if self.evaluate_super:
                # isStable = torch_to_numpy(isStable)
                # indices = isStable.astype(int)
                indices = isStable.type(long_)
                indices[isStable] = torch.arange(self.surfel_num, device=dev)
                self.label_index = indices[self.label_index]

        v, u, coords, valid_indexs = pcd2depth(self.points, depth_sort=True, round_coords=False)
        self.valid = torch.zeros(PIXEL_NUM, dtype=bool_, device=dev)
        self.valid[coords] = True
        self.validmap = self.valid.view(HEIGHT,WIDTH)
        self.projdata = torch.stack([valid_indexs,v,u],dim=1)

        # self.get_o3d_pcd() # TODO
        
        # Render the current 3D model to an image.
        self.renderImg = self.renderer.forward(self)
        self.validRender = torch.mean(self.renderImg.view(-1,3), axis=1) > 10 # TODO: Better valid map.
        self.validRender &= self.valid
        if save_render_img:
            out = torch_to_numpy(self.renderImg)
            if vis_ED_nodes:
                if render_method == 'proj':
                    ED_img = torch_to_numpy(self.renderer.forward(self.ED_nodes))
                elif render_method == 'pulsar':
                    ED_img = torch_to_numpy(self.renderer.forward(self.ED_nodes, rad=0.04))
                valid_ED_img = ED_img[:,:,0] > 10
                out[cv2.dilate(valid_ED_img.astype('uint8'), \
                    np.ones((3, 3), 'uint8'), iterations=1) > 0] = 255
                out[valid_ED_img] = ED_img[valid_ED_img]
            cv2.imwrite(os.path.join(F_render_img, "{:06d}.png".format(ID)), \
                out[:,:,::-1])

        # Save the cont. color image for qualitative evaluation.
        if qual_color_eval:
            cv2.imwrite(os.path.join(F_qual_color_img, "{:06d}.png".format(ID)), \
                torch_to_numpy(self.renderer.forward(self, qual_color=True))[:,:,::-1])

    # Fuse the input data into our reference model.
    # def fuseInputData(self, new_data, ID):
    def fuseInputData(self, new_data):

        # Return data[indices]. 'indices' can be either indices or True/False map.
        def get_data(indices, data):
            return data.points[indices], data.norms[indices], data.colors[indices], \
                data.rad[indices], data.conf[indices]

        # Merge data1 & data2, and update self.data[indices].
        def merge_data(data1, indices1, data2, indices2, time):
            p, n, c, r, w = get_data(indices1, data1)
            p_new, n_new, c_new, r_new, w_new = get_data(indices2, data2)

            # Only merge points that are close enough.
            valid = (torch_distance(p, p_new) < THRESHOLD_DISTANCE) & \
                (torch_inner_prod(n, n_new) > THRESHOLD_COSINE_ANGLE)
            indices = indices1[valid]
            w, w_new = w[valid], w_new[valid]
            w_update = w + w_new
            w /= w_update
            w_new /= w_update

            # Fuse the radius(r), confidence(r), position(p), normal(n) and color(c).
            self.rad[indices] = w * r[valid] + w_new * r_new[valid]
            self.conf[indices] = w_update
            w, w_new = w.unsqueeze(-1), w_new.unsqueeze(-1)
            self.points[indices] = w * p[valid] + w_new * p_new[valid]
            norms = w * n[valid] + w_new * n_new[valid]
            self.norms[indices] = torch.nn.functional.normalize(norms, dim=-1)
            # TODO: Better color update.
            self.colors[indices] = w * c[valid] + w_new * c_new[valid]
            if qual_color_eval and data2 is None:
                update_indices = torch.max(self.eval_colors[indices], dim=1)[0] == 0
                if torch.any(update_indices):
                    self.eval_colors[indices[update_indices]] = \
                        self.eval_colors[indices2[valid][update_indices]]
            self.time_stamp[indices] = time # Update time stamps.

            return valid

        valid = new_data.valid
        time = new_data.time

        ## Project surfels onto the image plane. For each pixel, only up to 16 projections with higher confidence.
        # Ignore surfels that have projections outside the image.
        _, _, _, indices_ = pcd2depth(self.points)
        points_ = self.points[indices_]
        conf_ = self.conf[indices_]
        # Get the projection maps 1) valid & 2) surfel indices in self.points,
        # map size: map_num x PIXEL_NUM)
        map_num = 16
        val_maps = []
        index_maps = []
        for i in range(map_num):
            if len(points_) == 0: break

            _, _, temp_coords, temp_indices = pcd2depth(points_, conf_sort=True, conf=conf_)
            
            val_map = torch.zeros((PIXEL_NUM), dtype=bool_, device=dev)
            val_map[temp_coords] = True
            val_maps.append(val_map)

            index_map = torch.zeros((PIXEL_NUM), dtype=long_, device=dev)
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
            merge_val_ = merge_data(self, index_, new_data, val_, time)
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
                merge_val_ = merge_data(self, indices1, self, indices2, time)
                val_maps[j][val_] = ~merge_val_
                
                indices2 = indices2[merge_val_]
                del_indices.append(indices2)
                if self.evaluate_super:
                    indices1 = indices1[merge_val_]
                    for k, index_ in enumerate(indices2):
                        m = (self.label_index==index_).nonzero(as_tuple=True)[0]
                        if len(m) > 0: self.label_index[m] = indices1[k]
        
        ## Delete redundant surfels.
        if len(del_indices) > 0:
            del_indices = torch.unique(torch.cat(del_indices))

            if self.evaluate_super:
                new_valid = torch.ones(self.surfel_num, dtype=bool_, device=dev)
                new_valid[del_indices] = False

                new_indices = torch.zeros(self.surfel_num, dtype=long_, device=dev)
                new_indices[new_valid] = torch.arange(self.surfel_num - len(del_indices), \
                    device=dev)

                self.label_index = new_indices[self.label_index]
            
            self.points = torch_delete(self.points, del_indices)
            self.norms = torch_delete(self.norms, del_indices)
            self.colors = torch_delete(self.colors, del_indices)
            if qual_color_eval:
                self.eval_colors = torch_delete(self.eval_colors, del_indices)
            self.rad = torch_delete(self.rad, del_indices)
            self.conf = torch_delete(self.conf, del_indices)
            self.time_stamp = torch_delete(self.time_stamp, del_indices)

            self.surfel_num = len(self.points)

            self.sf_knn_indices = torch_delete(self.sf_knn_indices, del_indices)
            self.sf_knn_div_indices = torch_delete(self.sf_knn_div_indices, del_indices)
            self.sf_knn_weights = torch_delete(self.sf_knn_weights, del_indices)
        
        ## Add points that do not have corresponding surfels.
        new_point_num = torch.count_nonzero(add_valid)
        if new_point_num > 0:
            ## Update the knn weights of existing surfels.
            sf_knn_dists = torch_distance(self.points.unsqueeze(1), \
                self.ED_nodes.points[self.sf_knn_indices])
            self.sf_knn_weights = update_KNN_weights(dists=torch.cat([sf_knn_dists, \
                torch_distance(self.points, self.ED_nodes.points[self.sf_knn_div_indices], keepdim=True)], \
                dim=1))

            new_points = new_data.points[add_valid]
            self.points = torch.cat([self.points, new_points], dim=0)
            new_norms = new_data.norms[add_valid]
            self.norms = torch.cat([self.norms, new_norms], dim=0)
            
            self.colors = torch.cat([self.colors, new_data.colors[add_valid]], dim=0)
            if qual_color_eval:
                new_eval_colors = float('nan')*torch.ones((new_point_num,3), device=dev)
                self.eval_colors = torch.cat([self.eval_colors, new_eval_colors], dim=0)
            self.rad = torch.cat([self.rad, new_data.rad[add_valid]])
            self.conf = torch.cat([self.conf, new_data.conf[add_valid]])
            self.time_stamp = torch.cat([self.time_stamp, \
                time*torch.ones(new_point_num, device=dev)])
            self.surfel_num += new_point_num
            
            ## Update isED: Only points that are far enough from 
            ## existing ED nodes can be added as new ED nodes.
            D = torch.cdist(new_points, self.ED_nodes.points)
            new_knn_dists, _ = D.topk(k=1, dim=-1, largest=False, sorted=True)
            isED = new_data.isED[add_valid] & (new_knn_dists[:,-1] > torch.max(sf_knn_dists))
            self.update_ED(points=new_points[isED], norms=new_norms[isED])
            # Add the knn weights of new surfels.
            new_sf_knn_weights, new_sf_knn_indices, new_sf_knn_div_indices = \
                update_KNN_weights(points=new_points, targets=self.ED_nodes.points, \
                n_neighbors_=n_neighbors)
            self.sf_knn_weights = torch.cat([self.sf_knn_weights, new_sf_knn_weights], dim=0)
            self.sf_knn_indices = torch.cat([self.sf_knn_indices, new_sf_knn_indices], dim=0)
            self.sf_knn_div_indices = torch.cat([self.sf_knn_div_indices, new_sf_knn_div_indices])