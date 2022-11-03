'''
Semantic Segmentation Network Inference
Stand-alone utility to generate segmentation mask for 1 surgical image as input

Note: If you get error with DeepLabV3+, please see bottom of inference.py
'''

# torch imports
from tkinter import Frame
import torch

import segmentation_models_pytorch as smp
import cv2
import utils
import matplotlib.pyplot as plt

# general imports
import argparse
import os

# utility imports
import numpy as np
from torchvision.transforms.transforms import ToTensor
from PIL import Image
import json
from tqdm import tqdm

from utils.config import *
from utils.utils import *

# model imports
import torchvision.transforms.functional as TF
import torch.nn.functional as F



def disentangleKey(key):
    '''
        Disentangles the key for class and labels obtained from the
        JSON file
        Returns a python dictionary of the form:
            {Class Id: RGB Color Code as numpy array}
    '''
    dKey = {}
    for i in range(len(key)):
        class_id = int(key[i]['id'])
        c = key[i]['color']
        c = c.split(',')
        c0 = int(c[0][1:])
        c1 = int(c[1])
        c2 = int(c[2][:-1])
        color_array = np.asarray([c0, c1, c2])
        dKey[class_id] = color_array

    return dKey


def reverseOneHot(batch, key):
    '''
        Generates the segmented image from the output of a segmentation network.
        Takes a batch of numpy oneHot encoded tensors and returns a batch of
        numpy images in RGB (not BGR).
    '''

    seg = []

    # Iterate over all images in a batch
    for i in range(len(batch)):
        vec = batch[i]
        idxs = vec

        segSingle = np.zeros([idxs.shape[0], idxs.shape[1], 3])

        # Iterate over all the key-value pairs in the class Key dict
        for k in range(len(key)):
            rgb = key[k]
            mask = idxs == k
            segSingle[mask] = rgb

        segMask = np.expand_dims(segSingle, axis=0)

        seg.append(segMask)

    seg = np.concatenate(seg)

    return seg


def normalize(batch, mean, std):
    '''
        Normalizes a batch of images, provided the per-channel mean and
        standard deviation.
    '''

    mean.unsqueeze_(1).unsqueeze_(1)
    std.unsqueeze_(1).unsqueeze_(1)
    for i in range(len(batch)):
        img = batch[i, :, :, :]
        img = img.sub(mean).div(std).unsqueeze(0)

        if 'concat' in locals():
            concat = torch.cat((concat, img), 0)
        else:
            concat = img

    return concat


def load_seg_model(classes_path, model_path):
    # GPU Check
    use_gpu = torch.cuda.is_available()
    curr_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(curr_device)
    device = torch.device('cuda' if use_gpu else 'cpu')

    print("CUDA AVAILABLE:", use_gpu, flush=True)
    print("DEVICE NAME:", device_name, flush=True)

    classes = json.load(open(classes_path))["classes"]
    key = disentangleKey(classes)
    num_classes = len(key)

    seg_model = smp.DeepLabV3Plus(
        encoder_name="resnet18",
        in_channels=3,
        classes=num_classes
    )

    seg_model.to(device)
    checkpoint = torch.load(model_path)
    seg_model.load_state_dict(checkpoint['state_dict'])

    seg_model.eval()

    return seg_model, key


def generate_mask(seg_model, img):
    img = img[None, ...].repeat(2, 1, 1, 1) # TODO: Solve the batch_size=1 bug.

    # Add option to reflect confidence level not mask
    seg = seg_model(img)
    seg = F.softmax(seg, dim=1)

    return seg[0]


'''
If you get an error with DeepLabV3+ such as:
RuntimeError: Sizes of tensors must match except in dimension 2. Got 270 and 268 (The offending index is 0)

Please add these lines to "/home/username/miniconda3/envs/torch/lib/python3.8/site-packages/segmentation_models_pytorch/deeplabv3/decoder.py"
prior to line 102:

max_dim_2 = max(high_res_features.size(2), aspp_features.size(2))
aspp_features = F.pad(aspp_features, (0, 0, 0, max_dim_2-aspp_features.size(2), 0, 0), "constant", 0)

Overall, the forward pass for DeepLabV3PlusDecoder() should look like:

def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])
        max_dim_2 = max(high_res_features.size(2), aspp_features.size(2))
        aspp_features = F.pad(aspp_features, (0, 0, 0, max_dim_2-aspp_features.size(2), 0, 0), "constant", 0)
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features
'''
