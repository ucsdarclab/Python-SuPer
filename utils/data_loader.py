import os
import numpy as np
import random
from skimage import io
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torchvision.transforms as T

from depth.monodepth2.layers import disp_to_depth, BackprojectDepth, Project3D, SSIM
from seg.inference import generate_mask
from utils.utils import torch_dilate, pcd2depth, find_knn, find_edge_region, get_grid_coords

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class GeneralDataset(Dataset):
    def __init__(self,
                 opt
                 ):
        super(GeneralDataset, self).__init__()
        self.opt = opt
        self.start_id = opt.start_id
        self.end_id = opt.end_id
        self.height = opt.height
        self.width = opt.width
        self.img_ext = opt.img_ext
        self.depth_ext = opt.depth_ext
        self.min_depth = opt.min_depth
        self.max_depth = opt.max_depth
        self.seg_ext = opt.seg_ext

        self.load_depth = opt.load_depth
        self.load_seg = opt.load_seg
        self.phase = opt.phase

        self.loader = pil_loader
        self.to_tensor = T.ToTensor()
        # for data augmentation
        if self.phase == 'train':
            try:
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                T.ColorJitter.get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
            except TypeError:
                self.brightness = 0.2
                self.contrast = 0.2
                self.saturation = 0.2
                self.hue = 0.1

        self.filenames = self.get_files()

        self.interp = Image.ANTIALIAS
        self.resize = T.Resize((self.height, self.width), interpolation=self.interp)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im = k
                inputs[(n, im)] = self.resize(inputs[(n, im)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im = k
                inputs[(n, im)] = self.to_tensor(f)
                inputs[(n + "_aug", im)] = color_aug(f)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        inputs = {}

        if self.opt.disable_side_shuffle:
            side = "l"
        else:
            side = "r" if self.phase=='train' and random.random() > 0.5 else "l"
        
        if self.opt.disable_color_aug:
            do_color_aug = False
        else:
            do_color_aug = self.phase=='train' and random.random() > 0.5
        
        if self.opt.disable_horizontal_flip:
            do_flip = False
        else:
            do_flip = self.phase=='train' and random.random() > 0.5
        
        if self.opt.disable_vertical_flip:
            do_vertical_flip = False
        else:
            do_vertical_flip = self.phase=='train' and random.random() > 0.5

        files = self.filenames[idx]
        inputs["filename"] = self.get_filename(files)
        inputs["ID"] = int(inputs["filename"])
        inputs["time"] = float(inputs["filename"])

        file_id = {"l": 0, "r": 1}[side]
        inputs[("color", 0)] = self.get_color(files, file_id, do_flip, do_vertical_flip)

        K = self.get_K()
        K[0, :] *= self.width
        K[1, :] *= self.height
        inv_K = np.linalg.pinv(K)
        inputs["K"] = torch.from_numpy(K)
        inputs["inv_K"] = torch.from_numpy(inv_K)
        inputs["divterm"] = 1./(2.*0.6*0.6)

        rescale_fac = 1
        stereo_T = np.eye(4, dtype=np.float32)
        baseline_sign = -1 if do_flip else 1
        side_sign = -1 if side == "l" else 1
        stereo_T[0, 3] = side_sign * baseline_sign * 0.1 * rescale_fac
        inputs["stereo_T"] = torch.from_numpy(stereo_T)

        if do_color_aug:
            color_aug = T.Compose(
                [T.ToTensor(),
                T.ColorJitter(brightness=self.brightness,
                    contrast=self.contrast, saturation=self.saturation, hue=self.hue),
                ]
            )
        else:
            color_aug = T.ToTensor()

        self.preprocess(inputs, color_aug)
        if self.load_seg:
            inputs[("seg_conf",0)], inputs[("seg",0)] = self.get_seg(inputs, files, side, do_flip, do_vertical_flip)

        if self.load_depth:
            inputs[("disp",0)], inputs[("depth",0)] = self.get_depth(files, side, do_flip, do_vertical_flip)

        # backproject_depth = BackprojectDepth(1, self.height, self.width)
        # cam_points = backproject_depth(inputs[("depth",0)], inputs["inv_K"][None, ...])
        # pcd = cam_points.reshape(4, self.height, self.width)
        # pcd = pcd.permute(1, 2, 0)[..., 0:3]
        # inputs["pcd"] = pcd

        # # Warp the right image to the left and use SSIM to estimate depth confidence.
        # if self.opt.use_ssim_conf or self.opt.use_seg_conf:
        #     project_3d = Project3D(1, self.height, self.width)
        #     pix_coords = project_3d(cam_points, inputs["K"][None, ...], 
        #         inputs["stereo_T"][None, ...])

        #     if self.opt.use_ssim_conf:
        #         target_img = inputs[("color",0)][None, ...]
        #         warp_img = F.grid_sample(target_img, pix_coords)
        #         ssim = SSIM()
        #         ssim_map = 1. - 2 * ssim(warp_img, target_img)[0].mean(0)
        #         inputs[("disp_conf",0)] = ssim_map[None, ...]

        #     if self.opt.use_seg_conf:
        #         warp_seg = F.grid_sample(inputs[("seg_conf",0)][None, ...], pix_coords.type(torch.float64))
        #         inputs[("warp_seg_conf", "s")] = warp_seg[0]

        # if self.opt.data == "superv1":
        #     inval = torch.zeros_like(inputs[("seg",0)]).bool()

        #     for del_class_id in self.opt.del_seg_classes:
        #         inval |= inputs[("seg",0)] == del_class_id

        #     if self.opt.del_seg_kernel > 0:
        #         inval = torch_dilate(inval[None, ...].float(), kernel=self.opt.del_seg_kernel)[0]
            
        #     # Filter invalid depth
        #     inval |= inputs[("depth", 0)] <= 0

        #     # Filter large depth
        #     inval |= inputs[("depth", 0)] > 1.5
            
        #     inputs[("disp", 0)][inval] = np.nan
        #     inputs[("depth", 0)][inval] = np.nan
        #     inputs["pcd"][inval[0]] = np.nan

        # elif self.opt.data == "superv2":

        #     inval = torch.zeros((1, self.opt.height, self.opt.width), dtype=bool_).cuda()
            
        #     if self.opt.load_depth:
        #         inval |= inputs[("depth", 0)] == 0

        #         start_id = int(0.1 * self.opt.width)
        #         inval[:, :, 0:start_id] = True
            
        #     elif "depth" in self.models:
        #         start_id = int(self.opt.depth_width_range[0] * self.opt.width)
        #         inval[:, :, 0 : start_id] = True

        #         end_id = int(self.opt.depth_width_range[1] * self.opt.width)
        #         inval[:, :, end_id:] = True

        #     # TODO: Use input parameters to control which region to track.
        #     # Filter out invalid values.
        #     for del_class_id in self.opt.del_seg_classes:

        #         inval |= inputs[("seg", 0)] == del_class_id

        #     if ("disp", 0) in inputs:
        #         inputs[("disp", 0)][inval] = np.nan
        #     inputs[("depth", 0)][inval] = np.nan
        #     inputs[("pcd", 0)][inval[0]] = np.nan

        return inputs

    def get_files(self):
        raise NotImplementedError

    def get_filename(self, files):
        raise NotImplementedError

    def get_K(self):
        raise NotImplementedError

    def get_color(self, files, file_id, do_flip, do_vertical_flip):
        raise NotImplementedError

    def get_seg(self, inputs, files, side, do_vertical_flip):
        raise NotImplementedError

    def get_depth(self, files, side, do_vertical_flip):
        raise NotImplementedError

class SuPerDataset(GeneralDataset):
    def __init__(self, opt):
        super(SuPerDataset, self).__init__(opt)

    def get_files(self):
        rgb_dir = "rgb"
        depth_dir = "depth"
        seg_dir = self.opt.data_seg_dir

        frames = []
        if self.phase == 'train':
            filelist = [file for file in os.listdir(os.path.join(self.opt.data_dir, rgb_dir)) \
                        if file.endswith(f"left{self.img_ext}")]
        else:
            if self.end_id is None:
                self.end_id = 0
                for filename in os.listdir(os.path.join(self.opt.data_dir, rgb_dir)):
                    if filename.endswith(self.img_ext):
                        id_temp = int(filename.split('-')[0])
                        if id_temp > self.end_id:
                            self.end_id = id_temp
                self.end_id += 1
            filelist = ["{:06d}-left{}".format(id,self.img_ext) for id in range(self.start_id, self.end_id)]
        
        for file in filelist:
            filenames = [os.path.join(self.opt.data_dir, rgb_dir, file),
                        os.path.join(self.opt.data_dir, rgb_dir, file.replace('left', 'right'))]

            filenames += [os.path.join(self.opt.data_dir, depth_dir, file.split('-')[0] + self.depth_ext)]

            filenames += [os.path.join(self.opt.data_dir, seg_dir, file.replace(self.img_ext, self.seg_ext)),
                            os.path.join(self.opt.data_dir, seg_dir, file.replace('left', 'right').replace(self.img_ext, self.seg_ext))]
                
            frames.append(filenames)
        
        return frames

    def get_filename(self, files):
        return os.path.basename(files[0]).split('-')[0]

    def get_K(self):
        if self.opt.data == 'superv1':
            # return np.array([[883.0, 0, 445.06, 0],
            #                 [0, 883.0, 190.24, 0],
            #                 [0, 0, 1, 0],
            #                 [0, 0, 0, 1]], dtype=np.float32)
            return np.array([[1.3796875, 0, 0.69540625, 0],
                            [0, 1.8395833333333333, 0.39633333333333337, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
        elif self.opt.data == 'superv2':
            # return np.array([[768.98551924, 0, 292.8861567, 0],
            #                 [0, 768.98551924, 291.61479526, 0],
            #                 [0, 0, 1, 0],
            #                 [0, 0, 0, 1]], dtype=np.float32)
            return np.array([[1.20153987381, 0, 0.45763461984, 0],
                            [0, 1.60205316508, 0.60753082345, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)

    def get_color(self, files, file_id, do_flip, do_vertical_flip):
        color = self.loader(os.path.join(self.opt.data_dir, files[file_id]))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        if do_vertical_flip:
            color = color.transpose(pil.FLIP_TOP_BOTTOM)

        return color

    def get_seg(self, inputs, files, side, do_flip, do_vertical_flip):
        file_id = {"l": 3, "r":4}[side]
        semantic_path = files[file_id]
        
        if os.path.exists(semantic_path):
            if semantic_path.endswith('.npy'):
                semantic_conf = np.load(semantic_path).astype(np.float32)
                semantic_label = np.argmax(semantic_conf, axis=0, keepdims=True)
                semantic_label = torch.as_tensor(semantic_label, dtype=torch.long)

                if do_flip:
                    semantic_label = T.functional.hflip(semantic_label)

                if do_vertical_flip:
                    semantic_label = T.functional.vflip(semantic_label)
                    
                return semantic_conf, semantic_label

            elif semantic_path.endswith('.png'):
                semantic_label = T.ToTensor()(Image.open(semantic_path)) * 255.
                semantic_label = semantic_label.long()

                semantic_conf = F.one_hot(semantic_label[0]).permute(2, 0, 1).double()

                return semantic_conf, semantic_label
        else:
            return None

    def get_depth(self, files, side, do_flip, do_vertical_flip):
        depth_path = files[2]
        
        if os.path.exists(depth_path):
            if self.depth_ext == '.png':
                disp = io.imread(depth_path).astype(np.float32)
            elif self.depth_ext == '.npy':
                disp = np.load(depth_path).astype(np.float32)
            disp = torch.as_tensor(disp, dtype=torch.float32)[None, ...]

            if do_flip:
                disp = T.functional.hflip(disp)

            if do_vertical_flip:
                disp = T.functional.vflip(disp)

            disp, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
                
            return disp, depth
        
        else:
            return None

def pred_depth(opt, models, inputs):
    if opt.post_process:
        # Post-processed results require each image to have two forward passes
        input_color = torch.concat((inputs[("color", 0)], 
                                    torch.flip(inputs[("color", 0)], 
                                    [3])), 0)
    else:
        input_color = inputs[("color", 0)]
    
    features = models.encoder(input_color)
    outputs = models.depth({"features":features}, None, None)

    disp = outputs[("disp", 0)]
        
    # if opt.feature_loss or opt.optical_flow_model == "raft_github":
    #     if opt.feature_loss_option == "depth" or opt.raft_option == "depth":
    #         inputs["disp_feature"] = outputs["disp_feature"][0] # TODO: average when post process?
        
    #     if opt.feature_loss_option == "seg" or opt.raft_option == "seg":
    #         inputs["seg_feature"] = outputs["seg_feature"][0] # TODO: average when post process?

    if opt.post_process:
        disp, _ = disp_to_depth(disp, opt.min_depth, opt.max_depth)
        N = disp.size(0) // 2
        disp = batch_post_process_disparity(disp[:N], torch.flip(disp[N:],[-1]))
        if opt.depth_filter_kernel_size > 0:
            disp = blur_image(disp, kernel=opt.depth_filter_kernel_size)
        inputs[("disp", 0)] = disp
        depth = 1 / disp
    else:
        if opt.depth_filter_kernel_size > 0:
            disp = blur_image(disp, kernel=opt.depth_filter_kernel_size)
        disp, depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)
        inputs[("disp", 0)] = disp
    inputs[("depth", 0)] = depth

    return inputs

def pred_seg(opt, models, inputs):
    # if opt.share_depth_seg_model:
    #     s_features = models.encoder(inputs[("color", "s")])
    #     outputs[("seg", "s")] = models.depth({"features":s_features}, None, None)["seg"]
    #     if self.opt.post_process:
    #         outputs["seg"] = outputs["seg"][:self.opt.batch_size]
    # else:
    #     features = models.seg_encoder(inputs[("color", 0)])
    #     s_features = models.seg_encoder(inputs[("color", "s")])
    #     outputs.update(models.seg({"features":features}, None, None))
    #     outputs[("seg", "s")] = models.seg({"features":s_features}, None, None)["seg"]

    # inputs[("seg", 0)] = F.interpolate(outputs["seg"], (self.opt.height, self.opt.width), mode='bilinear')
    # inputs[("seg", "s")] = F.interpolate(outputs[("seg", "s")], (self.opt.height, self.opt.width), mode='bilinear')

    seg_left = generate_mask(models.seg, inputs[("color", 0)])
    seg_conf_smoother = nn.AvgPool2d(11, stride=1, padding=5, count_include_pad=False)
    seg_left = seg_conf_smoother(seg_left)
    inputs[("seg", 0)] = torch.argmax(seg_left, dim=1, keepdim=True).long()

    # seg_right = generate_mask(models.seg, inputs[("color", "s")])
    # seg_right = seg_conf_smoother(seg_right)
    # inputs[("seg", "s")] = torch.argmax(seg_right, dim=0, keepdim=True).long()

    if opt.hard_seg:
        inputs[("seg_conf", 0)] = F.one_hot(inputs[("seg", 0)]).permute(0, 3, 1, 2).type(torch.float64)
        # inputs[("seg_conf", "s")] = F.one_hot(inputs[("seg", "s")]).permute(0, 3, 1, 2).type(torch.float64)
    else:
        inputs[("seg_conf", 0)] = seg_left.type(torch.float64)
        # inputs[("seg_conf", "s")] = seg_right.type(torch.float64)

    return inputs

def depth_preprocessing(opt, models, inputs):
    """
    Convert depth map to 3D point cloud (x, y, z, norms) in the camera frame.
    """
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    backproject_depth = BackprojectDepth(1, opt.height, opt.width).to(device)
    cam_points = backproject_depth(inputs[("depth",0)], inputs["inv_K"])
    pcd = cam_points.reshape(cam_points.size(0), 4, opt.height, opt.width)
    pcd = pcd.permute(0, 2, 3, 1)[..., 0:3]
    inputs["pcd"] = pcd

    # Warp the right image to the left and use SSIM to estimate depth confidence.
    if opt.use_ssim_conf or opt.use_seg_conf:
        project_3d = Project3D(1, opt.height, opt.width)
        pix_coords = project_3d(cam_points, inputs["K"], 
            inputs["stereo_T"])

        if opt.use_ssim_conf:
            target_img = inputs[("color",0)]
            warp_img = F.grid_sample(target_img, pix_coords)
            ssim = SSIM()
            ssim_map = 1. - 2 * ssim(warp_img, target_img).mean(1, keepdim=True)
            inputs[("disp_conf",0)] = ssim_map

        if opt.use_seg_conf:
            warp_seg = F.grid_sample(inputs[("seg_conf",0)], pix_coords.float())
            inputs[("warp_seg_conf", "s")] = warp_seg[0]

    if opt.data == "superv1":
        inval = torch.zeros_like(inputs[("seg",0)]).bool()

        for del_class_id in opt.del_seg_classes:
            inval |= inputs[("seg",0)] == del_class_id

        if opt.del_seg_kernel > 0:
            inval = torch_dilate(inval.float(), kernel=opt.del_seg_kernel)
        
        # Filter invalid depth
        inval |= inputs[("depth", 0)] <= 0

        # Filter large depth
        inval |= inputs[("depth", 0)] > 1.5
        inputs[("disp", 0)][inval] = np.nan
        inputs[("depth", 0)][inval] = np.nan
        inputs["pcd"][inval[:, 0]] = np.nan

    elif opt.data == "superv2":

        inval = torch.zeros(inputs[("disp", 0)].size(), dtype=torch.bool).cuda()
        
        if opt.load_depth:
            inval |= inputs[("depth", 0)] == 0

            start_id = int(0.1 * opt.width)
            inval[:, :, 0:start_id] = True
        
        else:
            start_id = int(opt.depth_width_range[0] * opt.width)
            inval[:, :, 0 : start_id] = True

            end_id = int(opt.depth_width_range[1] * opt.width)
            inval[:, :, end_id:] = True

        # TODO: Use input parameters to control which region to track.
        # Filter out invalid values.
        for del_class_id in opt.del_seg_classes:

            inval |= inputs[("seg", 0)] == del_class_id

        if ("disp", 0) in inputs:
            inputs[("disp", 0)][inval] = np.nan
        inputs[("depth", 0)][inval] = np.nan
        inputs["pcd"][inval[0]] = np.nan


    with torch.no_grad():
        if "pcd" in inputs:
            points = inputs["pcd"]
            if opt.normal_model == 'naive':
                norms, valid = getN(points)
            elif opt.normal_model == '8neighbors':
                norms, valid = getN(points, colors=inputs[("color", 0)])
            valid &= ~torch.any(torch.isnan(points), dim=3)
            Z = - inputs[("depth", 0)]
        else:
            Z = inputs[("depth", scale)]
            Z[Z==0] = np.nan # Convert invalid depth values (0) to nan.
            Z = blur_image(Z)
            Z *= inputs["depth_scale"]

            # Estimate (x,y,z) and norms from the depth map.
            points, norms, valid = pcd2norm(opt, inputs, Z)
        inputs[("normal", 0)] = norms
        points = points[0].type(torch.float64)
        norms = norms[0].type(torch.float64)
        if ("seg", 0) in inputs:
            seg = inputs[("seg", 0)][0, 0]
        if ("seg_conf", 0) in inputs:
            seg_conf = inputs[("seg_conf", 0)][0].permute(1, 2, 0)
        valid = valid[0]

        # # For visualizing the normals.
        # normal_map = torch.zeros_like(norms)
        # normal_map[valid] = norms[valid]

        # # Init surfel graph, get the edges and faces of the graph.
        # valid_verts, edge_index, face_index = init_graph(valid)
        valid_verts = valid
        points = points[valid_verts]
        norms = norms[valid_verts]

        index_map = - torch.ones((opt.height, opt.width), dtype=torch.long)
        index_map[valid_verts] = torch.arange(valid_verts.count_nonzero())
        
        # Calculate the radii.
        radii = Z[0,0][valid_verts] / (np.sqrt(2) * inputs["K"][0,0,0] * \
            torch.clamp(torch.abs(norms[...,2]), 0.26, 1.0))
        # radii = 0.002 * torch.ones(torch.count_nonzero(valid_verts), dtype=fl64_).cuda()

        # Calculate the confidence.
        if opt.use_ssim_conf:
            confs = torch.sigmoid(inputs[("disp_conf", 0)][0, 0])
        else:
            U, V = get_grid_coords(opt.height, opt.width)
            scale_u, scale_v = U / opt.width, V / opt.height
            dc2 = (2.*scale_u-1.)**2 + (2.*scale_v-1.)**2
            confs = torch.exp(-dc2 * inputs["divterm"]).cuda()
        
        if opt.use_seg_conf:
            P = inputs[("warp_seg_conf", "s")][0].float()
            Q = inputs[("seg_conf", 0)][0].float()
            seg_confs = torch.exp(- 0.1 * (0.5 * (P * (P / (Q + 1e-13) + 1e-13).log()).sum(0) + \
                            0.5 * (Q * (Q / (P + 1e-13) + 1e-13).log()).sum(0))
                            )
            confs = 0.5 * (confs + seg_confs)

        rgb = inputs[("color", 0)]
        colors = rgb[0].permute(1,2,0)[valid_verts]
    data = Data(points=points,
                norms=norms,
                colors=colors,
                radii=radii, 
                confs=confs[valid_verts],
                valid=valid_verts.view(-1),
                index_map=index_map,
                time=int(inputs["filename"][0]))
    # Deleted parameters: edge_index, face_index, rgb, ID=inputs["ID"]
    
    if ("seg", 0) in inputs:
        data.seg = seg[valid_verts]
    if ("seg_conf", 0) in inputs:
        data.seg_conf = seg_conf[valid_verts]

        # Calculate the (normalized) distance of project points to the semantic region edges.
        with torch.no_grad():
            kernels = [3, 3, 3]
            edge_pts = []
            for class_id in range(opt.num_classes):
                seg_grad_bin = find_edge_region(inputs[("seg", 0)], 
                                                num_classes=opt.num_classes,
                                                class_list=[class_id],
                                                kernel=kernels[class_id])
                edge_y, edge_x = seg_grad_bin[0,0].nonzero(as_tuple=True)
                edge_pts.append(torch.stack([edge_x/opt.width, edge_y/opt.height], dim=1).type(torch.float64))

            sf_y, sf_x, _, _ = pcd2depth(inputs, data.points, round_coords=False)
            sf_coords = torch.stack([sf_x/opt.width, sf_y/opt.height], dim=1)
            dist2edge = torch.zeros_like(data.radii)
            for class_id in range(opt.num_classes):
                val_points = data.seg == class_id
                knn_dist2edge, knn_edge_ids = find_knn(sf_coords[val_points], 
                                                       edge_pts[class_id], k=1)
                dist2edge[val_points] = knn_dist2edge[:, 0]
            data.dist2edge = dist2edge
    
    return data, inputs

"""
getN() and pcd2norm() together are used to estimate the normal for
each point in the input point cloud.
Method: Estimate normals from central differences (Ref[1]-Section3, 
        link: https://en.wikipedia.org/wiki/Finite_difference), 
        i.e., f(x+h/2)-f(x-h/2), of the vertext map.
"""
def getN(points, colors=None):
    if colors is None:
        b, h, w, _ = points.size()
        points = torch.nn.functional.pad(points, (0,0,1,1,1,1), value=float('nan'))
        hL = points[:, 1:-1, :-2, :].reshape(-1,3)
        hR = points[:, 1:-1, 2:, :].reshape(-1,3)
        hD = points[:, :-2, 1:-1, :].reshape(-1,3)
        hU = points[:, 2:, 1:-1, :].reshape(-1,3)

        N = torch.cross(hR-hL, hD-hU)
        N = torch.nn.functional.normalize(N, dim=-1).reshape(b,h,w,3)
        
        return N, ~torch.any(torch.isnan(N), -1)
    
    else:
        b, h, w, _ = points.size()

        colors = colors.permute(0, 2, 3, 1)
        colors = torch.nn.functional.pad(colors, (0,0,1,1,1,1), value=float('nan'))
        col_cen = colors[:, 1:-1, 1:-1, :].reshape(-1,3)
        col_hL = torch.exp(-torch.mean(torch.abs(colors[:, 1:-1, :-2, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hLU = torch.exp(-torch.mean(torch.abs(colors[:, :-2, :-2, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hU = torch.exp(-torch.mean(torch.abs(colors[:, :-2, 1:-1, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hRU = torch.exp(-torch.mean(torch.abs(colors[:, :-2, 2:, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hR = torch.exp(-torch.mean(torch.abs(colors[:, 1:-1, 2:, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hRD = torch.exp(-torch.mean(torch.abs(colors[:, 2:, 2:, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hD = torch.exp(-torch.mean(torch.abs(colors[:, 2:, 1:-1, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        col_hDL = torch.exp(-torch.mean(torch.abs(colors[:, 2:, :-2, :].reshape(-1,3) - col_cen), dim=1, keepdim=True))
        
        points = torch.nn.functional.pad(points, (0,0,1,1,1,1), value=float('nan'))
        cen = points[:, 1:-1, 1:-1, :].reshape(-1,3)
        hL = (points[:, 1:-1, :-2, :].reshape(-1,3) - cen) * col_hL
        hLU = (points[:, :-2, :-2, :].reshape(-1,3) - cen) * col_hLU
        hU = (points[:, :-2, 1:-1, :].reshape(-1,3) - cen) * col_hU
        hRU = (points[:, :-2, 2:, :].reshape(-1,3) - cen) * col_hRU
        hR = (points[:, 1:-1, 2:, :].reshape(-1,3) - cen) * col_hR
        hRD = (points[:, 2:, 2:, :].reshape(-1,3) - cen) * col_hRD
        hD = (points[:, 2:, 1:-1, :].reshape(-1,3) - cen) * col_hD
        hDL = (points[:, 2:, :-2, :].reshape(-1,3) - cen) * col_hDL

        N = torch.stack([
            torch.cross(hL, hLU + hU + hRU + hR + hRD + hD + hDL),
            torch.cross(hLU, hU + hRU + hR + hRD + hD + hDL),
            torch.cross(hU, hRU + hR + hRD + hD + hDL),
            torch.cross(hRU, hR + hRD + hD + hDL),
            torch.cross(hR, hRD + hD + hDL),
            torch.cross(hRD, hD + hDL),
            torch.cross(hD, hDL)],
            dim=2).sum(2)
        N = torch.nn.functional.normalize(N, dim=-1).reshape(b,h,w,3)
        
        return N, ~torch.any(torch.isnan(N), -1)

def pcd2norm(opt, inputs, Z):
    ZF = blur_image(Z)
    
    X, Y, Z = depth2pcd(inputs, Z)
    points = torch.stack([X,Y,Z], dim=-1)
    if opt.normal_model == 'naive':
        norms, valid = getN(points)
    elif opt.normal_model == '8neighbors':
        norms, valid = getN(points, colors=inputs[("color", 0, 0)])

    # Ref[1]-"A copy of the depth map (and hence associated vertices and normals) are also 
    # denoised using a bilateral filter (for camera pose estimation later)."
    XF, YF, ZF = depth2pcd(inputs, ZF)
    pointsF = torch.stack([XF,YF,ZF], dim=-1)
    if opt.normal_model == 'naive':
        normsF, validF = getN(pointsF)
    elif opt.normal_model == '8neighbors':
        normsF, validF = getN(pointsF, colors=inputs[("color", 0, 0)])

    # Update the valid map.
    norm_angs = torch_inner_prod(norms, normsF)
    pt_dists = torch_distance(points, pointsF)
    valid = valid & validF & (norm_angs >= opt.th_cosine_ang) & \
        (pt_dists <= opt.th_dist)

    return points, norms, valid

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    # _, h, w = l_disp.shape
    # m_disp = 0.5 * (l_disp + r_disp)
    # l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    # l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    # r_mask = l_mask[:, :, ::-1]
    # return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp
    b, h, w = l_disp.size(0), l_disp.size(-2), l_disp.size(-1)
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = get_grid_coords(h, w, batch_size=b)
    l = l.to(l_disp.device)
    l_mask = 1.0 - torch.clip(20 * (l/w - 0.05), min=0, max=1)
    r_mask = torch.flip(l_mask, [-1])
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp