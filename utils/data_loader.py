import os
import numpy as np
import random
from skimage import io

import torch
from torch.utils.data import Dataset

from seg.seg_models import *

from utils.config import *
from utils.utils import *


class SuPerDataset(Dataset):

    def __init__(self, model_args, transform, phase='test', aug=None, flip_aug=True):
        self.root = model_args['data_dir']

        image_folder_ = "stereo_rgb"
        depth_folder_ = "depth"
        seg_mask_folder_ = "tool_mask" # seg_mask_folder_ = "full_mask"

        self.model_args = model_args
        self.transform = transform
        self.aug = aug
        self.flip_aug = flip_aug if phase == 'train' else False

        files = os.listdir(os.path.join(self.root, image_folder_))
        frame_num = 0
        for file in files:
            id = int(file.split('-')[0])
            if id > frame_num:
                frame_num = id
        
        self.frames = []
        split_id = int(0.25 * len(files))
        for id in range(frame_num):
            file = "{:06d}-left.png".format(id)
            
            # # TODO: Better train-test design for super.
            # if not file.endswith('png') or \
            #     (phase == 'train' and int(file.split('-')[0]) >= split_id) or \
            #     (phase == 'eval' and int(file.split('-')[0]) < split_id) or \
            #     (phase == 'test' and int(file.split('-')[0]) < split_id):
            #     continue

            imagel_file = os.path.join(image_folder_, file)
            imager_file = os.path.join(image_folder_, file.replace('left', 'right'))
            
            depth_file = os.path.join(depth_folder_, 
                file.replace('left', 'depth'))
            
            files = [imagel_file, imager_file, depth_file]

            if 'seman-super' in model_args['method']:
                tool_maskl_file = os.path.join(seg_mask_folder_, file.replace('left', 'viewerleft'))
                tool_maskr_file = os.path.join(seg_mask_folder_, file.replace('left', 'viewerright'))
                files = files + [tool_maskl_file, tool_maskr_file]
            if ~np.any([not os.path.exists(os.path.join(self.root, file_)) for file_ in files]):
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
        inputs["ID"] = torch.tensor(int(inputs["filename"]), device=dev)
        inputs["time"] = inputs["ID"].type(fl32_)
        
        imagel = io.imread(os.path.join(self.root, files[0])) # RGB, [0,255]
        imager = io.imread(os.path.join(self.root, files[1]))
        depth = io.imread(os.path.join(self.root, files[2])).astype(np.float32)

        if self.aug is not None:
            inputs[("color_aug", 0, 0)] = self.aug(imagel).to(dev)
            inputs[("color_aug", "s", 0)] = self.aug(imager).to(dev)
        inputs[("color", 0, 0)] = self.transform(imagel).to(dev)
        inputs[("color", "s", 0)] = self.transform(imager).to(dev)
        inputs[("depth", 0)] = numpy_to_torch(depth, device=dev, dtype=fl32_).unsqueeze(0)

        # Noisy segmentation mask based on HSV colorspace.
        if 'seman-super' in self.model_args['method']:
            tooll_mask = io.imread(os.path.join(self.root, files[3]))
            inputs[("seman", 0)] = HSVSeg(inputs[("color", 0, 0)], numpy_to_torch(tooll_mask))
            toolr_mask = io.imread(os.path.join(self.root, files[4].replace('viewerleft', 'viewerright')))
            inputs[("seman", "s")] = HSVSeg(inputs[("color", "s", 0)], numpy_to_torch(toolr_mask))

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