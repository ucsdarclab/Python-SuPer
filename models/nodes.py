import open3d as o3d
# from sklearn.neighbors import NearestNeighbors
import torch
# import copy
import cv2

from utils.config import *
from utils.utils import *
from utils.renderer import *

class Surfels():

    def __init__(self, points, norms, colors, rad, conf, isED, valid, rgb, time, ID):

        # Init surfels.
        self.points = points
        self.norms = norms
        self.colors = colors
        if qual_color_eval:
            self.eval_colors = init_qual_color(self.points)
        self.rad = rad
        self.conf = conf
        self.surfel_num = len(points)
        self.time_stamp = time * torch.ones(self.surfel_num, device=dev)

        self.evaluate_super = False

        self.valid = valid
        self.validmap = valid.view(HEIGHT,WIDTH)
        valid_indexs = torch.arange(self.surfel_num, dtype=int, device=dev).unsqueeze(1)
        valid_coords = self.validmap.nonzero()
        # Column direction of self.projdata:
        # 1) index of projected points in self.points, 
        # corresponding projection coordinates (y,x) on the image plane 
        self.projdata = torch.cat([valid_indexs,valid_coords], dim=1).type(tfdtype_)

        # init ED nodes
        self.get_ED(isED, init=True)

        self.param_num = self.ED_num * 7

        # self.rgb = rgb
        self.ID = ID
        
        # TODO
        # if open3d_visualize:
        #     self.pcd = []
        #     self.ED_pcd = []
        #     self.get_o3d_pcd(init=True)
        # self.get_o3d_pcd(init=True)

        # Init renderer. Options: Projector(), Pulsar()
        self.renderer = Pulsar()
    
    # Init/update ED nodes & ED nodes related parameters. TODO
    def get_ED(self, isED, init=False, points=None):
        
        if init:
            self.ED_points = self.points[isED]
            # self.ED_colors = self.colors[isED]
            self.ED_norms = self.norms[isED]
            # self.ED_rad = self.rad[isED]
            # self.ED_conf = self.conf[isED]

            self.update_KNN()

        else:
            self.ED_points = torch.cat([self.ED_points,points[isED]], dim=0)

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

    # TODO
    def update_KNN(self):

            # finding the 8 neighbour for ED nodes
            self.ednode_knn_indexs, _ = self.findKNN(self.ED_points, n_neighbors=ED_n_neighbors, isSurfel=False)

            # finding the 4NN ED nodes for every surfel in new index
            self.surfel_knn_indexs, self.surfel_knn_weights, _ = self.findKNN(self.points)

            self.ED_num = len(self.ED_points)

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

        sf_knn = self.ED_points[self.surfel_knn_indexs] # All g_i in (10).
        sf_diff = self.points.unsqueeze(1) - sf_knn
        deform_ = deform[self.surfel_knn_indexs]
        self.points, _ = Trans_points(sf_diff, sf_knn, deform_, self.surfel_knn_weights)

        norms, _ = transformQuatT(self.norms.unsqueeze(1).repeat(1,n_neighbors,1), \
            deform_[...,0:4])
        norms = torch.sum(self.surfel_knn_weights.unsqueeze(-1) * norms, dim=-2)
        self.norms = torch.nn.functional.normalize(norms, dim=-1)
        
        self.ED_points += deform[:,4:]
        # self.ED_norms, _ = transformQuatT(self.ED_norms, deform[...,0:4]) # TODO

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
        self.param_num = self.ED_num * 7

        self.surfel_knn_indexs = self.surfel_knn_indexs[isStable]
        self.surfel_knn_weights = self.surfel_knn_weights[isStable]

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
            cv2.imwrite(os.path.join(F_render_img, "{:06d}.png".format(ID)), \
                torch_to_numpy(self.renderImg)[:,:,::-1])

        # Save the cont. color image for qualitative evaluation.
        if qual_color_eval:
            cv2.imwrite(os.path.join(F_qual_color_img, "{:06d}.png".format(ID)), \
                torch_to_numpy(self.renderer.forward(self, qual_color=True))[:,:,::-1])

    # Fuse the input data into our reference model.
    def fuseInputData(self, new_data, ID):

        # Return data[indices]. 'indices' can be either indices or True/False map.
        def get_data(indices, data):
            if data is None:
                return self.points[indices], self.norms[indices], self.colors[indices], \
                self.rad[indices], self.conf[indices]
            else:
                points, norms, colors, rad, conf = data
                return points[indices], norms[indices], colors[indices], \
                    rad[indices], conf[indices]

        # Merge data1 & data2, and update self.data[indices].
        def merge_data(indices1, indices2, time, data1=None, data2=None):

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

        points, norms, valid, colors, rad, conf, isED, time = new_data

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
            merge_val_ = merge_data(index_, val_, time, \
                data2=[points, norms, colors, rad, conf])
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
                merge_val_ = merge_data(indices1, indices2, time)
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
        
        ## Add points that do not have corresponding surfels.
        new_point_num = torch.count_nonzero(add_valid)
        if new_point_num > 0:
            new_points = points[add_valid]
            self.points = torch.cat([self.points, points[add_valid]], dim=0)
            self.norms = torch.cat([self.norms, norms[add_valid]], dim=0)
            self.colors = torch.cat([self.colors, colors[add_valid]], dim=0)
            if qual_color_eval:
                new_eval_colors = float('nan')*torch.ones((new_point_num,3), device=dev)
                self.eval_colors = torch.cat([self.eval_colors, new_eval_colors], dim=0)
            self.rad = torch.cat([self.rad, rad[add_valid]])
            self.conf = torch.cat([self.conf, conf[add_valid]])
            self.time_stamp = torch.cat([self.time_stamp, \
                time*torch.ones(new_point_num, device=dev)])
            self.surfel_num += new_point_num
            
            # Update isED: Only points that are far enough from 
            # existing ED nodes can be added as new ED nodes. 
            # TODO: Better ways?
            _, _, new_knn_dists = self.findKNN(new_points)
            isED = isED[add_valid] & (new_knn_dists[:,0] > UPPER_ED_DISTANCE)
            self.get_ED(isED, points=new_points)

        # TODO: Better ways?
        ## Delete ED nodes that are too close to each other.
        SFknn_idxs, _, _ = self.findKNN(self.points)
        _, node_count = torch.unique( torch.cat([SFknn_idxs.flatten(), \
            torch.arange(len(self.ED_points), device=dev)]), \
            sorted=True, return_counts=True)
        bad_idxs = node_count < CLS_SIZE_TH
        # Find the 8 neighbour for ED nodes.
        ednode_knn_indexs, ednode_knn_dists = self.findKNN(self.ED_points, \
            n_neighbors=ED_n_neighbors, isSurfel=False)
        m = ((ednode_knn_dists[:,0] < LOWER_ED_DISTANCE) & (~bad_idxs)).nonzero()
        n = ednode_knn_indexs[m,0]
        bad_idxs = torch.cat([bad_idxs.nonzero(), torch.where(node_count[m]<node_count[n],m,n)])
        # del node_count, ednode_knn_indexs, ednode_knn_dists, m, n
        self.ED_points = torch_delete(self.ED_points, bad_idxs)
        self.update_KNN()

    # TODO: Speed up KNN
    def findKNN(self, surfels, n_neighbors=n_neighbors, isSurfel=True):

        D = torch.cdist(surfels, self.ED_points)
        dists, sort_idx = D.topk(k=n_neighbors+1, dim=-1, largest=False, sorted=True)

        if isSurfel:
            knn_indexs = sort_idx[:,0:n_neighbors]
            knn_dists = dists[:,0:n_neighbors]

            # TODO: Need better weight calculation method.
            knn_weights = 1. - knn_dists / (dists[:,-1].view(-1,1)+1e-8)
            knn_weights *= knn_weights
            
            # Normalize the weights.
            # knn_weights *= self.ED_conf[knn_indexs]
            knn_weights = torch.nn.functional.softmax(knn_weights, dim=1)

            return knn_indexs, knn_weights, knn_dists
        
        else:
            return sort_idx[:, 1:].contiguous(), dists[:, 1:]