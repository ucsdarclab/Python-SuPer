import os
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import *
from utils.config import *

def normalize_coordinate(inputs):
    div = HEIGHT
    inputs[...,0] = inputs[...,0]/div
    inputs[...,1] = inputs[...,1]/div
    return inputs

def evaluate(gt, est):

    diff = distance(gt,est)
    
    mean_ = []
    std_ = []
    valid = (gt[:,:,0] > 0) & (gt[:,:,1] > 0)
    for i in range(20):
        diff_temp = diff[:,i][valid[:,i]]
        mean_.append(np.mean(diff_temp))
        std_.append(np.std(diff_temp))

    return np.array(mean_), np.array(std_)

# Read the ground truth & SURF tracking and C++ super tracking results
data_dir="/home/shan/Projects/datasets/3d_data/super_dataset/grasp1_2_psmnet/grasp1_2/exp"
with open(os.path.join(data_dir, 'labelPts.npy'), 'rb') as f:
    labelPts = np.load(f,allow_pickle=True).tolist()

# Load the python super tracking results
with open(os.path.join(output_folder,'exp1_track_rst.npy'), 'rb') as f:
    pysuper_rst = np.load(f)

frame_num = pysuper_rst.shape[0]
gt = labelPts['gt']#[1:frame_num+1]
super_cpp_rst = labelPts['super_cpp']#[1:frame_num+1]
SURF_rst = labelPts['SURF']#[1:frame_num+1]

# normalize the coordinates by image size
gt = normalize_coordinate(gt)
super_cpp_rst = normalize_coordinate(super_cpp_rst)
SURF_rst = normalize_coordinate(SURF_rst)
pysuper_rst = normalize_coordinate(pysuper_rst)

super_cpp_mean, super_cpp_std = evaluate(gt, super_cpp_rst)
SURF_mean, SURF_std = evaluate(gt, SURF_rst)
pysuper_mean, pysuper_std = evaluate(gt, pysuper_rst)

plt.figure(figsize=(10,3))

ind = np.arange(20)
width = 0.4

plt.bar(ind, SURF_mean, width, color='b', edgecolor='k', label='Mean error of SURF')
plt.bar(ind+0.2, super_cpp_mean, width, color='r', edgecolor='k', label='Mean error of C++ SuPer')
plt.bar(ind+0.4, pysuper_mean, width, color='g', edgecolor='k', label='Mean error of Python SuPer')

plt.errorbar(ind, SURF_mean, SURF_std, linestyle='None', color='y', capsize=3.5, label='STD of SURF')
plt.errorbar(ind+0.2, super_cpp_mean, super_cpp_std, linestyle='None', color='m', capsize=3.5, label='STD of C++ SuPer')
plt.errorbar(ind+0.4, pysuper_mean, pysuper_std, linestyle='None', color='darkorange', capsize=3.5, label='STD of Python SuPer')

plt.legend(loc='upper right')
plt.xticks(ind, (ind+1).astype(int))
plt.xlabel('Tracked point ID')
plt.ylabel('Error in percentage of image size')
plt.ylim([0.0,0.5])
plt.grid(True)
plt.show()
