import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import torch
import copy
import cv2

from utils.config import *
from utils.utils import *

class Surfels:

    def __init__(self, points, norms, colors, rad, conf, time_stamp, isED, valid, \
            evaluate_super=False, eva_ids=None, compare_rsts=None):

        # init surfels
        self.points = points
        self.norms = norms
        self.colors = colors
        self.rad = rad
        self.conf = conf
        self.time_stamp = time_stamp
        self.surfel_num = len(points)

        self.valid = valid
        self.validmap = valid.view(HEIGHT,WIDTH)
        valid_indexs = torch.arange(self.surfel_num, dtype=int, device=cuda0).view(-1,1)
        valid_coords = self.validmap.nonzero()
        # Column direction of self.projdata:
        # 1) index of projected points in self.points, 
        # corresponding projection coordinates (y,x) on the image plane 
        self.projdata = torch.concat([valid_indexs,valid_coords], dim=1)

        if qual_color_eval:
            self.eval_colors = self.init_eval_colors(self.points)

        # init ED nodes
        self.get_ED(isED, init=True)

        self.evaluate_super = evaluate_super
        if evaluate_super:
            self.label_index = None
            self.eva_ids = eva_ids
            self.compare_rsts = compare_rsts

        if open3d_visualize:
            self.pcd = []
            self.ED_pcd = []
            self.get_o3d_pcd(init=True)
    
    # init/update ED nodes & ED nodes related parameters
    def get_ED(self, isED, init=False, points=None):
        
        if init:

            self.ED_points = self.points[isED]
            # self.ED_colors = self.colors[isED]
            # self.ED_norms = self.norms[isED]
            # self.ED_rad = self.rad[isED]
            # self.ED_conf = self.conf[isED]

            if qual_color_eval:
                self.ED_eval_colors = self.eval_colors[isED]

            self.update_KNN()

        else:
            self.ED_points = torch.concat([self.ED_points,points[isED]], dim=0)

    def update_KNN(self):

            # finding the 8 neighbour for ED nodes
            self.ednode_knn_indexs, _ = self.findKNN(self.ED_points, n_neighbors=ED_n_neighbors, isSurfel=False)

            # finding the 4NN ED nodes for every surfel in new index
            self.surfel_knn_indexs, self.surfel_knn_weights, _ = self.findKNN(self.points)

            self.ED_num = len(self.ED_points)

    def init_eval_colors(self, points, init=True):
        
        if init:
            max_dist, _ = torch.max(points, dim = 0, keepdim=True)
            min_dist, _ = torch.min(points, dim = 0, keepdim=True)
 
            # return 255. * (max_dist - self.points) / (max_dist - min_dist)
            margin = 20.
            return (255. - margin * 2) * (max_dist - self.points) / (max_dist - min_dist) + margin

        else:
            knn_indexs, _, _ = self.findKNN(points)
            anchor_points = self.ED_points[knn_indexs]
            anchor_colors = self.ED_eval_colors[knn_indexs]

            # anchor_points & anchor_colors: N x n_neighbors x 3
            # points: N x 3
            N = len(points)
            max_dist, max_idx = torch.max(anchor_points, dim = 1)
            max_color = anchor_colors[torch.arange(N, device=cuda0).unsqueeze(1).expand(N,3), \
                max_idx, torch.tensor([[0,1,2]], device=cuda0).expand(N,3)]
            min_dist, min_idx = torch.min(anchor_points, dim = 1)
            min_color = anchor_colors[torch.arange(N, device=cuda0).unsqueeze(1).expand(N,3), \
                min_idx, torch.tensor([[0,1,2]], device=cuda0).expand(N,3)]

            out = (points - min_dist) / (max_dist - min_dist) * (max_color - min_color) + min_color
            min_out, _ = torch.min(out, dim=1)
            max_out, _ = torch.max(out, dim=1)
            out[min_out < 0.] = 0.
            out[max_out > 255.] = 0.
            return out

    # init/update open3d data for display
    def get_o3d_pcd(self, init=False):

        # pcd for surfels
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors/255.)
        
        # pcd for ED nodes
        ed_pcd = o3d.geometry.PointCloud()
        ed_pcd.points = o3d.utility.Vector3dVector(self.ED_points)
        display_colors = np.zeros_like(self.ED_colors)
        display_colors[:,0] = 1.0
        ed_pcd.colors = o3d.utility.Vector3dVector(display_colors)

        self.pcd = pcd
        self.ED_pcd = ed_pcd

    # Visualize the ICP cost on each pixel.
    def vis_opt_error(self, points, norms, valid):

        v = self.projdata[:,1].detach().cpu().numpy()
        u = self.projdata[:,2].detach().cpu().numpy()

        proj_points = np.zeros((HEIGHT,WIDTH,3))
        proj_points[v,u] = self.points[self.projdata[:,0]].detach().cpu().numpy()

        proj_norms = np.zeros((HEIGHT,WIDTH,3))
        proj_norms[v,u] = self.norms[self.projdata[:,0]].detach().cpu().numpy()

        points = points.view(HEIGHT,WIDTH,3).detach().cpu().numpy()
        norms = norms.view(HEIGHT,WIDTH,3).detach().cpu().numpy()
        data_errors = inner_prod(norms, proj_points-points)
        data_errors *= data_errors
        valid_ = valid.view(HEIGHT,WIDTH).detach().cpu().numpy()
        data_errors[~(self.validmap.detach().cpu().numpy() & valid_)] = 0.0

        return data_errors

    # Save the ICP cost map before & after optimization.
    def save_opt_error_maps(self, error_bf, error_af, filename):
        bf_max = np.max(error_bf)
        af_max = np.max(error_af)
        valid_bf = error_bf > 0
        bf_mean = np.mean(error_bf[valid_bf])
        valid_af = error_af > 0
        af_mean = np.mean(error_af[valid_af])
        
        scale = 255./np.max([bf_max,af_max])
        error_bf *= scale
        error_af *= scale
        error_bf = cv2.applyColorMap(error_bf.astype(np.uint8), cv2.COLORMAP_JET)
        error_af = cv2.applyColorMap(error_af.astype(np.uint8), cv2.COLORMAP_JET)
        error_bf = put_text(error_bf, 'Max: '+str(round(bf_max,3))+', mean: '+str(round(bf_mean,3)) )
        error_af = put_text(error_af, 'Max: '+str(round(af_max,3))+', mean:'+str(round(af_mean,3)) )

        out_img = np.concatenate([error_bf, error_af], axis=1)
        cv2.imwrite(filename, out_img)

    def projSurfel(self, colors, show_ednodes=False):

        proj_index = torch.round(self.projdata[:,0]).long()
        v = torch.round(self.projdata[:,1]).long()
        u = torch.round(self.projdata[:,2]).long()

        renderImg = torch.zeros((HEIGHT,WIDTH,3), dtype=torch.uint8, device=cuda0)
        renderImg[v,u] = colors[proj_index].byte()

        if show_ednodes:
            # ED_indexs_in_surfels = self.isED[proj_index].nonzero()
            # renderImg[v[ED_indexs_in_surfels], u[ED_indexs_in_surfels]] = \
            #     torch.tensor([[255,255,255]], dtype=torch.uint8, device=cuda0)
            v, u, _, valid = pcd2depth(self.ED_points, vis_only=True)
            # print(renderImg.size(), torch.min(v), torch.max(v), torch.min(u), torch.max(u))
            renderImg[v[valid], u[valid]] = torch.tensor([[255,255,255]], dtype=torch.uint8, device=cuda0)

        return renderImg.cpu().numpy()

    # Project the surfels onto current image plane
    def surfelProj(self, maxProjNum=16):

        # def get_best_projection(_coords, _index, cont=True):
        def get_best_projection(coords, index):

            projMap = -torch.ones(PIXEL_NUM, dtype=int, device=cuda0)
            
            temp_index = torch.arange(len(coords), device=cuda0)
            projMap[coords] = temp_index
            temp_index = projMap[projMap >= 0]

            projMap[coords[temp_index]] = index[temp_index]
            coords = torch_delete(coords, temp_index)
            index = torch_delete(index, temp_index)

            return projMap.view(HEIGHT,WIDTH), coords, index

        if print_time:
            start = timeit.default_timer()

        _, _, coords, index = pcd2depth(self.points)

        # get all neighbor points
        # index_temp = []
        projMaps = []
        for i in range(maxProjNum):
            
            if len(coords) == 0: break
            
            projMap, coords, index = get_best_projection(coords, index)
            projMaps.append(projMap)

        return projMaps, i

    # ommit the motion estimated by optimizor
    def update(self, deform, lm):

        self.points, self.norms = self.trans_points_norms(deform)

        # update ed node positions & norms
        # self.ED_points = self.points[self.ED_indexs_in_surfels]
        # self.ED_norms = self.norms[self.ED_indexs_in_surfels]
        self.ED_points += deform[:,4:]

    # Save the tracking results onto the current frame
    def save_tracking_rst(self, track_rst, evaluate_id, rgb, filename):
                
        gt_ = self.compare_rsts['gt'][evaluate_id]
        super_cpp_ = self.compare_rsts['super_cpp'][evaluate_id]
        SURF_ = self.compare_rsts['SURF'][evaluate_id]
                
        test_img = rgb[:,:,::-1].astype(np.uint8) # RGB2BGR
        for k, coord in enumerate(track_rst):

            offset = 6
            gt_y = int(gt_[k][0])
            gt_x = int(gt_[k][1])
            if gt_y > 0 and gt_x > 0:
                test_img[gt_y-offset:gt_y+offset, \
                        gt_x-offset:gt_x+offset,:] = \
                        np.array([[0, 255, 0]])

                offset = 4
                super_cpp_y = int(super_cpp_[k][0])
                super_cpp_x = int(super_cpp_[k][1])
                if super_cpp_y > 0 and super_cpp_x > 0:
                    test_img = cv2.line(test_img, (gt_x,gt_y), (super_cpp_x,super_cpp_y), (0,255,0), 1)
                    test_img[super_cpp_y-offset:super_cpp_y+offset, \
                            super_cpp_x-offset:super_cpp_x+offset,:] = \
                            np.array([[0, 0, 255]])

                offset = 3
                SURF_y = int(SURF_[k][0])
                SURF_x = int(SURF_[k][1])
                if SURF_y > 0 and SURF_x > 0:
                    test_img[SURF_y-offset:SURF_y+offset, \
                            SURF_x-offset:SURF_x+offset,:] = \
                            np.array([[255, 0, 0]])

                offset = 2
                y = int(coord[0])
                x = int(coord[1])
                test_img = cv2.line(test_img, (gt_x,gt_y), (x,y), (0,255,0), 1)
                test_img[y-offset:y+offset, \
                        x-offset:x+offset,:] = \
                        np.array([[255, 0, 255]])

            cv2.imwrite(filename,test_img)

    # Delete unstable surfels & ED nodes
    def prepareStableIndexNSwapAllModel(self, time, ID, rgb=None):

        if print_time:
            start = timeit.default_timer()

        # A surfel is unstable if it 1) hasn't been updated for a long time and 2) has low confidence.
        isStable = (time-self.time_stamp < FUSE_INIT_TIME) | (self.conf >= THRESHOLD_CONFIDENCE)

        pre_surfel_num = copy.deepcopy(self.surfel_num)

        self.points = self.points[isStable]
        self.norms = self.norms[isStable]
        self.colors = self.colors[isStable]
        self.conf = self.conf[isStable]
        self.rad = self.rad[isStable]
        self.time_stamp = self.time_stamp[isStable]
        self.surfel_num = len(self.points)

        if qual_color_eval:
            self.eval_colors = self.eval_colors[isStable]

        self.surfel_knn_indexs = self.surfel_knn_indexs[isStable]
        self.surfel_knn_weights = self.surfel_knn_weights[isStable]

        v, u, coords, valid_indexs = pcd2depth(self.points, vis_only=True, round_coords=False)
        self.valid = torch.zeros(PIXEL_NUM, dtype=bool, device=cuda0)
        self.valid[coords] = True
        self.validmap = self.valid.view(HEIGHT,WIDTH)
        self.projdata = torch.stack([valid_indexs,v[valid_indexs],u[valid_indexs]],dim=1)
        
        # Save rendered image.
        if qual_color_eval:
            qual_color_img = self.projSurfel(self.eval_colors, show_ednodes=True)
            cv2.imwrite(os.path.join(qual_color_folder, "{:06d}.png".format(ID)), qual_color_img[:,:,::-1])
            del qual_color_img
        self.renderImg = self.projSurfel(self.colors)
        cv2.imwrite(os.path.join(render_folder, "{:06d}.png".format(ID)), self.renderImg[:,:,::-1])

        # If evaluate on the 20 labeled points
        if self.evaluate_super:

            # Initialize 20 point IDs
            if self.label_index is None:
                self.label_index = np.zeros(20)

                for k, pt in enumerate(self.compare_rsts['gt'][0]):
                    Y, X = round(pt[0]), round(pt[1])
                    self.label_index[k] = ((self.projdata[:,1].long()==Y)&(self.projdata[:,2].long()==X)).nonzero()[0,0].detach().cpu().numpy()
                self.label_index = self.label_index.astype(int)

                self.track_rsts = []
            else:
                new_indexs = np.ones(pre_surfel_num) * np.nan
                new_indexs[isStable.detach().cpu().numpy()] = np.arange(self.surfel_num)
                self.label_index = new_indexs[self.label_index].astype(int)

            # save tracking results
            evaluate_id = np.argwhere(self.eva_ids==ID)
            if len(evaluate_id) > 0:
                evaluate_id = np.squeeze(evaluate_id)

                v, u, _, _ = pcd2depth(self.points[self.label_index])
                self.track_rsts.append(torch.stack((v, u), dim=-1).detach().cpu().numpy())

                if save_20pts_tracking_result:
                    filename = os.path.join(tracking_rst_folder,str(evaluate_id+1)+".png")
                    self.save_tracking_rst(self.track_rsts[-1], evaluate_id, rgb, filename)

        if print_time:
            stop = timeit.default_timer()
            print('Surfel fusing time: {}s'.format(stop - start))

    def mergeWithPoint(self, indexs_a, indexs_b, data_a=None, data_b=None):

        def weighted_sum(w, w_new, w_update, old_data, new_data):
            return (w * old_data + w_new * new_data) / w_update

        p, n, c, r, w = self.get_data_by_idx(indexs_a, data=data_a)
        p_new, n_new, c_new, r_new, w_new = self.get_data_by_idx(indexs_b, data=data_b)

        #fusing the radius(r), confidence(r), position(p), normal(n) and color(c)
        w_update = w + w_new
        self.rad[indexs_a] = weighted_sum(w, w_new, w_update, r, r_new)
        self.conf[indexs_a] = w_update
        #
        w = w.unsqueeze(-1)
        w_new = w_new.unsqueeze(-1)
        w_update = w_update.unsqueeze(-1)
        self.points[indexs_a] = weighted_sum(w, w_new, w_update, p, p_new)
        norms_new = weighted_sum(w, w_new, w_update, n, n_new)
        self.norms[indexs_a] = torch.nn.functional.normalize(norms_new, dim=-1)
        self.colors[indexs_a] = weighted_sum(w, w_new, w_update, c, c_new)

    def get_data_by_idx(self, indexs, data=None):
        if data is None:
            points = self.points[indexs]
            norms = self.norms[indexs]
            colors = self.colors[indexs]
            rad = self.rad[indexs]
            w = self.conf[indexs]
        else:
            points, norms, colors, rads, confs = data
            points = points[indexs]
            norms = norms[indexs]
            colors = colors[indexs]
            rad = rads[indexs]
            w = confs[indexs]

        return points, norms, colors, rad, w

    # Fuse the input data into our reference model
    def fuseInputData(self, points, norms, colors, rad, conf, time_stamp, valid, isED, time, ID, rgb):
        
        def get_surfel_map(IDmap):
            out_points = copy.deepcopy(zero_img)
            out_norms = copy.deepcopy(zero_img)
            
            ValidIDMap = IDmap >= 0
            ValidIDs = IDmap[ValidIDMap]
            out_points[ValidIDMap] = self.points[ValidIDs]
            out_norms[ValidIDMap] = self.norms[ValidIDs]
            return out_points, out_norms, ValidIDMap

        _,_, proj_coords, proj_valid_index = pcd2depth(self.points, vis_only=True)
        # Find correspondence
        pair_valid = valid[proj_coords]
        match_surfels = proj_valid_index[pair_valid] # Index for paired surfels
        match_point_coords = proj_coords[pair_valid] # Index for paired new comers

        ## merge surfels with new comer
        # TODO For each new commer that has radius > 1.5 * corresponding surfel radius,
        # only update the surfel's confidence. (?)
        # $$ ELSE:
        self.mergeWithPoint(match_surfels, match_point_coords, data_b=[points, norms, colors, rad, conf])
        self.time_stamp[match_surfels] = time
        
        # add new comers that do not have corresponding surfels
        projCoords = torch.concat([match_point_coords,(~valid).nonzero().squeeze(1)])
        new_points = torch_delete(points, projCoords)
        if len(new_points) > 0:
            self.points = torch.concat([self.points, new_points], dim=0)
            self.norms = torch.concat([self.norms, torch_delete(norms, projCoords)], dim=0)
            self.colors = torch.concat([self.colors, torch_delete(colors, projCoords)], dim=0)
            self.rad = torch.concat([self.rad, torch_delete(rad, projCoords)])
            self.conf = torch.concat([self.conf, torch_delete(conf, projCoords)])
            self.time_stamp = torch.concat([self.time_stamp, torch_delete(time_stamp, projCoords)])
            self.surfel_num = len(self.points)

            if qual_color_eval:
                if colorize_new_surfels:
                    new_eval_colors = self.init_eval_colors(new_points, init=False)
                else:
                    new_eval_colors = torch.zeros_like(new_points)
                self.eval_colors = torch.concat([self.eval_colors, new_eval_colors], dim=0)
            
            # update isED
            isED = torch_delete(isED, projCoords)
            _, _, new_knn_dists = self.findKNN(new_points)
            isED = isED & (new_knn_dists[:,0] > UPPER_ED_DISTANCE)
            self.get_ED(isED, points=new_points)
            if qual_color_eval:
                self.ED_eval_colors = torch.concat([self.ED_eval_colors,new_eval_colors[isED]], dim=0)

        # reproject surfels onto the current image plane
        projMaps, projMapNum = self.surfelProj()
        # merge neighbor surfels
        deleteIndex = []

        for i in range(projMapNum):

            pmap1 = projMaps[i]
            if torch.max(pmap1) < 0: continue

            for j in range(i+1,projMapNum):

                pmap2 = projMaps[j]
                if torch.max(pmap2) < 0: continue

                surfelMap1, normMap1, ValidIDMap1 = get_surfel_map(pmap1)
                surfelMap2, normMap2, ValidIDMap2 = get_surfel_map(pmap2)
                
                dis = torch_distance(surfelMap1, surfelMap2)
                ang = torch_inner_prod(normMap1, normMap2)

                ValidIDMap = ValidIDMap1 & ValidIDMap2 & (dis < THRESHOLD_DISTANCE) & (ang > THRESHOLD_COSINE_ANGLE)
                
                if torch.sum(ValidIDMap) == 0: continue
                ValidIDs1 = pmap1[ValidIDMap]
                ValidIDs2 = pmap2[ValidIDMap]

                self.mergeWithPoint(ValidIDs1, ValidIDs2)
                if qual_color_eval:
                    ref_colors = torch.maximum(self.eval_colors[ValidIDs1], self.eval_colors[ValidIDs2])
                    self.eval_colors[ValidIDs1] = torch.where(self.eval_colors[ValidIDs1]>0, \
                        self.eval_colors[ValidIDs1], ref_colors)

                self.time_stamp[ValidIDs1] = torch.maximum(self.time_stamp[ValidIDs1], self.time_stamp[ValidIDs2])

                # if self.label_index is not None:
                if self.evaluate_super:
                    for update_id, label_pt_id in enumerate(self.label_index):
                        index_ = (ValidIDs2==label_pt_id).nonzero()
                        if len(index_) > 0:
                            self.label_index[update_id] = ValidIDs1[index_[0,0]].detach().cpu().numpy()
                    self.label_index = self.label_index.astype(int)

                deleteIndex.append(ValidIDs2)

                projMaps[j][ValidIDMap] = -1
        
        if len(deleteIndex) > 0:
            deleteIndex = torch.concat(deleteIndex)

            surfel_num = len(self.points)
            new_valid = torch.ones(surfel_num, dtype=bool, device=cuda0)
            new_indexmap = -new_valid.long()
            new_valid[deleteIndex] = False
            new_indexmap[new_valid] = torch.arange(surfel_num-len(deleteIndex), device=cuda0)
            if self.evaluate_super:
                self.label_index = new_indexmap[self.label_index].detach().cpu().numpy()
            
            self.points = torch_delete(self.points, deleteIndex)
            self.norms = torch_delete(self.norms, deleteIndex)
            self.colors = torch_delete(self.colors, deleteIndex)
            self.eval_colors = torch_delete(self.eval_colors, deleteIndex)
            self.rad = torch_delete(self.rad, deleteIndex)
            self.conf = torch_delete(self.conf, deleteIndex)
            self.time_stamp = torch_delete(self.time_stamp, deleteIndex)

        # Delete ED nodes that are too close to each other
        surfel_knn_indexs, _, _ = self.findKNN(self.points)
        _, node_count = torch.unique(surfel_knn_indexs, sorted=True, return_counts=True)
        # finding the 8 neighbour for ED nodes
        ednode_knn_indexs, ednode_knn_dists = self.findKNN(self.ED_points, n_neighbors=ED_n_neighbors, isSurfel=False)
        m = (ednode_knn_dists[:,0] < LOWER_ED_DISTANCE).nonzero()
        n = ednode_knn_indexs[m,0]
        bad_idxs = torch.where(node_count[m]<node_count[n],m,n)
        del ednode_knn_indexs, ednode_knn_dists, m, n
        self.ED_points = torch_delete(self.ED_points, bad_idxs)
        if qual_color_eval:
            self.ED_eval_colors = torch_delete(self.ED_eval_colors, bad_idxs)
        self.update_KNN()

        self.surfel_num = len(self.points)

        # # update ED nodes (did in prepareStableIndexNSwapAllModel())
        # self.get_ED()

    #### TODO: Speed up KNN
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

    def trans_points_norms(self, quat_, n_neighbors=4):

        quat_ = quat_[self.surfel_knn_indexs]
        qw = quat_[...,0:1]
        qv = quat_[...,1:4]
        t = quat_[...,4:7]

        ednodes = self.ED_points[self.surfel_knn_indexs]
        trans_surfels = self.transformQuatT(self.points.unsqueeze(1)-ednodes, qw, qv, t=t)
        trans_surfels += ednodes
        surfel_knn_weights = self.surfel_knn_weights.unsqueeze(-1)
        trans_surfels = torch.sum(surfel_knn_weights * trans_surfels, dim=-2)

        trans_norms = self.transformQuatT(torch.tile(self.norms.unsqueeze(1),(1,n_neighbors,1)), qw, qv)
        trans_norms = torch.sum(surfel_knn_weights * trans_norms, dim=-2)
        trans_norms = torch.nn.functional.normalize(trans_norms, dim=-1)
        
        return trans_surfels, trans_norms

    # q: quarternion; t: translation
    def transformQuatT(self, v, qw, qv, t=None):

        cross_prod = torch.cross(qv, v, dim=-1)
        out = v + 2.0 * qw * cross_prod + \
            2.0 * torch.cross(qv, cross_prod, dim=-1)

        if t is None: return out
        else: return out + t