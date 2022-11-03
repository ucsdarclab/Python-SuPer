import os
import numpy as np
import random
from skimage import io
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from seg.seg_models import *
from seg.inference import *

from depth.monodepth2.layers import disp_to_depth, BackprojectDepth, Project3D, SSIM

from utils.config import *
from utils.utils import *


class SuPerDataset(Dataset):

    def __init__(self, opt, transform, phase='test', aug=None, flip_aug=True,
    models=None):
        self.opt = opt
        self.root = opt.data_dir
        self.phase = phase
        self.load_depth = not "depth" in models
        if self.opt.load_seman:
            assert not "seman" in models, "Segmentation model is not required"
        self.models = models

        self.seman_conf_smoother = nn.AvgPool2d(11, stride=1, padding=5, count_include_pad=False)

        # if self.opt.data == 'superv1':
        #     image_folder_ = "stereo_rgb"
        # elif self.opt.data == 'superv2':
        image_folder_ = ""
        
        # if self.load_depth:
        #     depth_folder_ = "depth"
        
        # seg_mask_folder_ = "tool_mask"
        if self.opt.load_seman:
            seman_folder_ = "seman"

        self.transform = transform
        self.aug = aug
        self.flip_aug = flip_aug if phase == 'train' else False

        files = os.listdir(os.path.join(self.root, image_folder_))
        frame_num = 0
        for file in files:
            if not file.endswith('.png'):
                continue

            id = int(file.split('-')[0])
            if id > frame_num:
                frame_num = id
        
        self.frames = []
        # split_id = int(0.25 * len(files))
        for id in range(frame_num):
            if self.opt.data == "superv1" and id < 4:
            # if self.opt.data == "superv1" and id < 160:
                continue
            file = "{:06d}-left.png".format(id)
            
            # # TODO: Better train-test design for super.
            # if not file.endswith('png') or \
            #     (phase == 'train' and int(file.split('-')[0]) >= split_id) or \
            #     (phase == 'eval' and int(file.split('-')[0]) < split_id) or \
            #     (phase == 'test' and int(file.split('-')[0]) < split_id):
            #     continue

            imagel_file = os.path.join(image_folder_, file)
            imager_file = os.path.join(image_folder_, file.replace('left', 'right'))
            files = [imagel_file, imager_file]
            
            if self.load_depth:
                # depth_file = os.path.join(depth_folder_, 
                #     file.replace('left', 'depth'))
                depth_file = file.split('-')[0] + self.opt.depth_type
                files += [depth_file]

            # if 'seman-super' in self.opt.method:
            if self.opt.load_seman:
                if opt.seg_model == 'deeplabv3':
                    semantic_folder = os.path.join(seman_folder_, 'DeepLabV3+')
                elif opt.seg_model == 'unet':
                    semantic_folder = os.path.join(seman_folder_, 'UNet')
                elif opt.seg_model == 'unet++':
                    semantic_folder = os.path.join(seman_folder_, 'UNet++')
                elif opt.seg_model == 'manet':
                    semantic_folder = os.path.join(seman_folder_, 'MANet')
                tool_maskl_file = os.path.join(semantic_folder, file[:-4] + self.opt.seman_type)
                files = files + [tool_maskl_file]

                # tool_maskr_file = os.path.join(seman_folder_, file.replace('left', 'right'))
                # files = files + [tool_maskl_file, tool_maskr_file]

            # if ~np.any([not os.path.exists(os.path.join(self.root, file_)) for file_ in files]):
            self.frames.append(files)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        """Returns a single training item from the dataset as a dictionary.
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:
            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("seman", <frame_id>)                   for semantic segmentation map,
            ("depth", <frame_id>)                   for depth map,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics,
            "filename"                              for filename.
        """
        inputs = {}

        if torch.is_tensor(idx):
            idx = idx.tolist()

        files = self.frames[idx]
        inputs["filename"] = os.path.basename(files[0]).split('-')[0]
        inputs["ID"] = torch.tensor(int(inputs["filename"]))
        inputs["time"] = inputs["ID"].type(fl32_)
        
        imagel = io.imread(os.path.join(self.root, files[0])) # RGB, [0,255]
        imager = io.imread(os.path.join(self.root, files[1]))

        # Load camera parameters.
        if self.opt.data == 'superv1':
            CamParams = OldSuPerParams
        elif self.opt.data == 'superv2':
            CamParams = SuPerParams
        inputs["depth_scale"] = CamParams.depth_scale
        inputs["height"] = CamParams.HEIGHT
        inputs["width"] = CamParams.WIDTH
        inputs["divterm"] = CamParams.DIVTERM
        inputs["fx"] = CamParams.fx
        inputs["fy"] = CamParams.fy
        inputs["cx"] = CamParams.cx
        inputs["cy"] = CamParams.cy

        K = np.array([[inputs["fx"], 0, inputs["cx"], 0],
                      [0, inputs["fy"], inputs["cy"], 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        inv_K = np.linalg.pinv(K)
        inputs[("K", 0)] = torch.from_numpy(K)
        inputs[("inv_K", 0)] = torch.from_numpy(inv_K)

        # TODO: update dataloader to KITTI format.
        do_flip = False
        side = "l"
        rescale_fac = 1
        stereo_T = np.eye(4, dtype=np.float32)
        baseline_sign = -1 if do_flip else 1
        side_sign = -1 if side == "l" else 1
        stereo_T[0, 3] = side_sign * baseline_sign * 0.1 * rescale_fac
        inputs["stereo_T"] = torch.from_numpy(stereo_T)

        if self.aug is not None:
            inputs[("color_aug", 0, 0)] = self.aug(imagel).cuda()
            inputs[("color_aug", "s", 0)] = self.aug(imager).cuda()
        inputs[("color", 0, 0)] = self.transform(imagel).cuda()
        inputs[("color", "s", 0)] = self.transform(imager).cuda()

        # Noisy segmentation mask based on HSV colorspace.
        # if 'seman-super' in self.opt.method:
        if self.opt.load_seman:
            if self.load_depth:
                left_id, right_id = 3, 4
            else:
                left_id, right_id = 2, 3

            if self.opt.seman_type == ".npy":
                seman_left = np.load(os.path.join(self.root, files[left_id]))
                inputs[("seman_conf", 0)] = torch.as_tensor(seman_left).type(fl64_)
                inputs[("seman", 0)] = torch.argmax(inputs[("seman_conf", 0)], dim=0, keepdim=True)

            # seman_left = io.imread(os.path.join(self.root, files[left_id]))
            # inputs[("seman", 0)] = torch.as_tensor(seman_left)[None, ...].type(long_)
            # inputs[("seman_conf", 0)] = F.one_hot(inputs[("seman", 0)][0]).permute(2, 0, 1).type(fl64_)

            # seman_right = io.imread(os.path.join(self.root, files[right_id]))
            # inputs[("seman", "s")] = torch.as_tensor(seman_right)[None, ...].type(long_)
            # inputs[("seman_conf", "s")] = F.one_hot(inputs[("seman", "s")][0]).permute(2, 0, 1).type(fl64_)
        
        elif "seman" in self.models:
            with torch.no_grad():
                seman_left = generate_mask(self.models["seman"], inputs[("color", 0, 0)])
                seman_right = generate_mask(self.models["seman"], inputs[("color", "s", 0)])

                seman_left = self.seman_conf_smoother(seman_left)
                seman_right = self.seman_conf_smoother(seman_right)
                
                inputs[("seman", 0)] = torch.argmax(seman_left, dim=0, keepdim=True).type(long_)
                inputs[("seman", "s")] = torch.argmax(seman_right, dim=0, keepdim=True).type(long_)

                # inputs[("seman_valid", 0)] = erode_dilate_seg(inputs[("seman", 0)])
                # inputs[("seman_valid", "s")] = erode_dilate_seg(inputs[("seman", "s")])

                if self.opt.hard_seman:
                    inputs[("seman_conf", 0)] = F.one_hot(inputs[("seman", 0)][0]).permute(2, 0, 1).type(fl64_)
                    inputs[("seman_conf", "s")] = F.one_hot(inputs[("seman", "s")][0]).permute(2, 0, 1).type(fl64_)
                else:
                    inputs[("seman_conf", 0)] = seman_left.type(fl64_)
                    inputs[("seman_conf", "s")] = seman_right.type(fl64_)
        
        if self.opt.load_seman_gt:
            seman_left_gt_path = os.path.join(self.root, "seman_gt", f"{inputs['filename']}-left.png")
            if os.path.exists(seman_left_gt_path):
                seman_left_gt = cv2.imread(seman_left_gt_path, 0)
                inputs[("seman_gt", 0)] = torch.as_tensor(seman_left_gt)[None, ...].type(long_)
        
        if self.load_depth:
            # depth = io.imread(os.path.join(self.root, files[2])).astype(np.float32)
            try:
                depth = np.load(os.path.join(self.opt.depth_dir, files[2]))[0][None, None, :, :] # TODO
            except:
                depth = np.load(os.path.join(self.opt.depth_dir, files[2]))[None, None, :, :]
            # depth = cv2.resize(depth, dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
            depth = numpy_to_torch(depth, dtype=fl32_)

            inputs[("depth", 0)] = depth[0]
        else:
            with torch.no_grad():
                input_color = inputs[("color", 0, 0)]

                if self.opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.stack((input_color, torch.flip(input_color, [2])), 0)
                    # if 'seman_gt' in inputs:
                    #     inputs['seman_gt'] = torch.cat((inputs['seman_gt'], torch.flip(inputs['seman_gt'], [3])), 0)
                else:
                    input_color = input_color[None, ...]

                if "encoder" in self.models:
                    features = self.models["encoder"](input_color)
                    disp = self.models["depth"](features)[("disp", 0)]

                    if self.opt.post_process:
                        assert not self.phase == 'train', "Post prossing is not used for training."

                        disp, _ = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                        N = disp.shape[0] // 2
                        disp = batch_post_process_disparity(disp[:N], torch.flip(disp[N:],[-1]))
                        if self.opt.depth_filter_kernel_size > 0:
                            disp = blur_image(disp, kernel=self.opt.depth_filter_kernel_size)
                        inputs[("disp", 0)] = disp[0]
                        depth = 1 / disp
                    elif self.phase == 'test':
                        if self.opt.depth_filter_kernel_size > 0:
                            disp = blur_image(disp, kernel=self.opt.depth_filter_kernel_size)
                        disp, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                        inputs[("disp", 0)] = disp[0]
                    else:
                        assert False, "TODO: Write the code."
                    inputs[("depth", 0)] = depth[0]
                else:
                    assert False, "TODO: Write the code."

        backproject_depth = BackprojectDepth(1, inputs["height"], inputs["width"]).cuda()
        cam_points = backproject_depth(depth, inputs[("inv_K", 0)][None, ...].cuda())
        pcd = cam_points.reshape(4, inputs["height"], inputs["width"])
        pcd = pcd.permute(1, 2, 0)[..., 0:3]
        inputs[("pcd", 0)] = pcd

        ########################################################################
        # Warp the right image to the left and use SSIM to estimate depth confidence.
        if self.opt.use_ssim_conf or self.opt.use_seman_conf:
            project_3d = Project3D(1, inputs["height"], inputs["width"]).cuda()
            pix_coords = project_3d(cam_points, inputs[("K", 0)][None, ...].cuda(), 
                inputs["stereo_T"][None, ...].cuda())

            if self.opt.use_ssim_conf:
                target_img = inputs[("color", "s", 0)][None, ...]
                warp_img = F.grid_sample(target_img, pix_coords)
                ssim = SSIM().cuda()
                ssim_map = 1. - 2 * ssim(warp_img, target_img)[0].mean(0)
                inputs[("disp_conf", 0)] = ssim_map[None, ...]

            if self.opt.use_seman_conf:
                warp_seman = F.grid_sample(inputs[("seman_conf", "s")][None, ...].cuda(), pix_coords.type(fl64_))
                inputs[("warp_seman_conf", "s")] = warp_seman[0]

        # For DEBUG superv1: load the depth map to extract valid region.
        if self.opt.data == "superv1":
            superv1_depth_root = '/media/bear/77f1cfad-f74f-4e12-9861-557d86da4f681/research_proj/datasets/3d_data/super_dataset/super_exp_520/depth'
            depth_gt = io.imread(os.path.join(superv1_depth_root, 
                os.path.basename(files[0]).replace('left', 'depth'))).astype(np.float32)
            depth_gt = depth_gt
            val = (depth_gt > 0).astype(np.uint8)

            kernel = np.ones((31, 31), np.uint8)
            val = cv2.erode(val, kernel, iterations=1)
            val = cv2.dilate(val, kernel, iterations=1)
            kernel = np.ones((11, 11), np.uint8)
            val = cv2.dilate(val, kernel, iterations=1)
            inval = val == 0.
            cv2.imwrite("inval.jpg", inval * 255)

            inval = numpy_to_torch(inval, dtype=bool_)[None, ...]

            inputs[("full_depth", 0)] = copy.deepcopy(inputs[("depth", 0)])
            inval_full = torch.ones_like(inval)
            for class_id in range(self.opt.num_classes):
                tool_valid_map = inputs[("seman",0)] == class_id
                tool_valid_map = 1. - nn.MaxPool2d(7, stride=1, padding=3).cuda()(1. - tool_valid_map.type(fl32_))
                inval_full[tool_valid_map == 1] = False
            inputs[("full_depth", 0)][inval_full] = np.nan

            inval[inputs[("seman",0)] == 1] = True
            # inval = inval_full
            inputs[("disp", 0)][inval] = np.nan
            inputs[("depth", 0)][inval] = np.nan
            inputs[("pcd", 0)][inval[0]] = np.nan

        elif self.opt.data == "superv2":

            # # Warp the right image to the left and use SSIM to filter bad depth.
            # project_3d = Project3D(1, inputs["height"], inputs["width"]).cuda()
            # pix_coords = project_3d(cam_points, inputs[("K", 0)][None, ...].cuda(), 
            #     inputs["stereo_T"][None, ...].cuda())
            # target_img = inputs[("color", "s", 0)][None, ...]
            # warp_img = F.grid_sample(target_img, pix_coords)
            # ssim_map = compute_reprojection_loss(warp_img, target_img)
            # inval = ssim_map[0] >= 1

            inval = torch.zeros((1, self.opt.height, self.opt.width), dtype=bool_).cuda()
            
            # if self.opt.use_ssim_conf:
            #     inval[inputs[("disp_conf", 0)] <= 0.0] = True
            
            if self.load_depth:
                inval |= inputs[("depth", 0)] == 0

                start_id = int(0.1 * self.opt.width)
                inval[:, :, 0:start_id] = True

                # inval |= ssim_map[None, ...] < 0.2
            
            elif "depth" in self.models:
                start_id = int(self.opt.depth_width_range[0] * self.opt.width)
                inval[:, :, 0 : start_id] = True

                end_id = int(self.opt.depth_width_range[1] * self.opt.width)
                inval[:, :, end_id:] = True

            # TODO: Use input parameters to control which region to track.
            # Filter out invalid values.
            for del_class_id in self.opt.del_seman_classes:
                # del_boolmap = ((~ inval) & (inputs[("seman", 0)] == del_class_id))[0]
                # if "del_points" in inputs:
                #     inputs["del_points"] = torch.cat([inputs["del_points"], inputs[("pcd", 0)][del_boolmap]], dim=0)
                #     inputs["del_colors"] = torch.cat([inputs["del_colors"], inputs[("color", 0, 0)].permute(1, 2, 0)[del_boolmap]], dim=0)
                # else:
                #     inputs["del_points"] = inputs[("pcd", 0)][del_boolmap]
                #     inputs["del_colors"] = inputs[("color", 0, 0)].permute(1, 2, 0)[del_boolmap]

                inval |= inputs[("seman", 0)] == del_class_id

            # if ("seman_valid", 0) in inputs:
            #     inval |= ~inputs[("seman_valid", 0)]
            
            if ("disp", 0) in inputs:
                inputs[("disp", 0)][inval] = np.nan
            inputs[("depth", 0)][inval] = np.nan
            inputs[("pcd", 0)][inval[0]] = np.nan

            # if self.opt.bn_morph:
            #     scaledDisp = inputs[("disp", 0)][None, ...]
            #     real_scale_disp = scaledDisp * (torch.abs(inputs[("K", 0)][0, 0][None, ...] * inputs["stereo_T"][0, 3][None, ...]).view(self.opt.batch_size, 1, 1,1).expand_as(scaledDisp).cuda())
            #     occlu_mask = self.models["OccluMask"](real_scale_disp, inputs['stereo_T'][0, 3][None, ...])[0]
            #     inval = occlu_mask == 1

            #     inputs[("disp", 0)][inval] = np.nan
            #     inputs[("depth", 0)][inval] = np.nan
            #     inputs[("pcd", 0)][inval[0]] = np.nan
            #     # if ("seman", 0) in inputs:
            #     #     inputs[("seman", 0)][inval] = -1

            #     # '''
            #     # Warp right semantic map to the left view using depth, if a semantic 
            #     # value in the warped map does not matches the semantic value in the 
            #     # real left semantic map, either the semantic or the depth estimation
            #     # is wrong and this point will not be used for tracking.
            #     # '''
            #     # # if opt.bn_morph:
            #     # #     # valid &= (inputs[("seman", 0)] == inputs[("warp_seman", "s")])[0, 0, :, :]
        
        ########################################################################

        return inputs

class HamlynDataset(Dataset):

    def __init__(self, args, transform, phase='train', aug=None, flip_aug=True):
        self.root = args['data_dir']

        if phase == 'train': subroot = 'train'
        elif phase == 'eval': subroot = 'test'
        elif phase == 'test': subroot = 'test'

        imagel_folder_ = "image_0"
        imager_folder_ = "image_1"
        segl_folder_ = "seg_0"
        segr_folder_ = "seg_1"

        self.args = args
        self.transform = transform
        self.aug = aug
        self.flip_aug = flip_aug if phase == 'train' else False
        
        self.frames = []
        for file in os.listdir(os.path.join(self.root, subroot,imagel_folder_)):
            if not file.endswith('png'):
                continue

            imagel_file = os.path.join(subroot, imagel_folder_, file)
            imager_file = os.path.join(subroot, imager_folder_, file)
            
            segl_file = os.path.join(subroot, segl_folder_, file)
            segr_file = os.path.join(subroot, segr_folder_, file)

            files = [imagel_file, imager_file, segl_file, segr_file]
            self.frames.append(files)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        files = self.frames[idx]
        imagel = io.imread(os.path.join(self.root, files[0])) # RGB, [0,255]
        imager = io.imread(os.path.join(self.root, files[1]))
        segl = io.imread(os.path.join(self.root, files[2]))
        segr = io.imread(os.path.join(self.root, files[3]))
        filename = os.path.basename(files[0]).split('.')[0]

        # TODO: Add data augmentation.
        if self.aug is not None:
            imagel_color_aug = self.aug(imagel).to(dev)
            imager_color_aug = self.aug(imager).to(dev)
        imagel = self.transform(imagel).to(dev)
        imager = self.transform(imager).to(dev)
        segl = numpy_to_torch(segl).type(long_)
        segr = numpy_to_torch(segr).type(long_)

        if self.args['height'] is not None and self.args['width'] is not None:
            target_size = (self.args['height'], self.args['width'])
            resize = T.Compose([T.Resize(target_size)])
            # nst_resize = T.Compose([T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST)])
            if self.aug is None:
                imagel = resize(imagel)
                imager = resize(imager)
            else:
                imagel_color_aug = resize(imagel_color_aug)
                imager_color_aug = resize(imager_color_aug)
            # segl = nst_resize(segl.unsqueeze(0)).squeeze(0)
            # segr = nst_resize(segr.unsqueeze(0)).squeeze(0)

        if self.flip_aug:
            flip = random.random() > 0.5
            if flip:
                imager_temp = torch.flip(imagel, [-1])
                imagel = torch.flip(imager, [-1])
                imager = imager_temp

                if self.aug is not None:
                    imager_color_aug_temp = torch.flip(imagel_color_aug, [-1])
                    imagel_color_aug = torch.flip(imager_color_aug, [-1])
                    imager_color_aug = imager_color_aug_temp

                segr_temp = torch.flip(segl, [-1])
                segl = torch.flip(segr, [-1])
                segr = segr_temp
        
        if self.aug is None:
            return imagel, imager, segl, segr, filename
        else:
            return imagel, imagel_color_aug, imager, imager_color_aug, \
                segl, segr, filename

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