import os
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import *
from utils.config import *

# Normalize the coordinates by image size
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
data_dir="/media/bear/77f1cfad-f74f-4e12-9861-557d86da4f68/research_proj/datasets/3d_data/super_dataset/grasp1_2/exp"
with open(os.path.join(data_dir, 'labelPts.npy'), 'rb') as f:
    labelPts = np.load(f,allow_pickle=True).tolist()
gt = labelPts['gt']
gt = normalize_coordinate(gt)
disp_compare_param = {'SURF':['b','k','y'], 'super_cpp':['r','k','m']}

# Load the python super tracking results
with open(os.path.join(output_folder,'exp1_data_arap_rot.npy'), 'rb') as f:
    pysuper_rst = np.load(f).astype(float)
pysuper_data = {'exp1_data_arap_rot.npy':pysuper_rst}
disp_pysuper_param = {'exp1_data_arap_rot.npy':['g','k','darkorange']}

# super_cpp_rst = labelPts['super_cpp']
# SURF_rst = labelPts['SURF']

# super_cpp_rst = normalize_coordinate(super_cpp_rst)
# SURF_rst = normalize_coordinate(SURF_rst)
# pysuper_rst = normalize_coordinate(pysuper_rst)

# super_cpp_mean, super_cpp_std = evaluate(gt, super_cpp_rst)
# SURF_mean, SURF_std = evaluate(gt, SURF_rst)
# pysuper_mean, pysuper_std = evaluate(gt, pysuper_rst)

plt.figure(figsize=(10,3))

ind = np.arange(20)
bar_width = 0.4
offset = 0.0

for key in disp_compare_param.keys():
    data_ = labelPts[key]
    data_ = normalize_coordinate(data_)
    mean_, std_ = evaluate(gt, data_)

    color_, edgecolor_, barcolor_ = disp_compare_param[key]
    
    plt.bar(ind+offset, mean_, bar_width, color=color_, edgecolor=edgecolor_, \
        label='Mean error of '+key)
    plt.errorbar(ind+offset, mean_, std_, linestyle='None', color=barcolor_, \
        capsize=3.5, label='STD of '+key)

    offset += 0.2

for key in disp_pysuper_param.keys():
    try:
        data_ = pysuper_data[key]
        data_ = normalize_coordinate(data_)
        mean_, std_ = evaluate(gt, data_)

        color_, edgecolor_, barcolor_ = disp_pysuper_param[key]
        
        plt.bar(ind+offset, mean_, bar_width, color=color_, edgecolor=edgecolor_, \
            label='Mean error of '+key)
        plt.errorbar(ind+offset, mean_, std_, linestyle='None', color=barcolor_, \
            capsize=3.5, label='STD of '+key)

        offset += 0.2
    except:
        pass

plt.legend(loc='upper right')
plt.xticks(ind, (ind+1).astype(int))
plt.xlabel('Tracked point ID')
plt.ylabel('Error in percentage of image size')
# plt.ylim([0.0,0.5])
plt.grid(True)
plt.show()

# ID = 20
# img = np.zeros((HEIGHT,WIDTH,3))
# gt_ys = (gt[ID,:,0]*HEIGHT).astype(int)
# gt_xs = (gt[ID,:,1]*HEIGHT).astype(int)
# data_ys = (data_[ID,:,0]*HEIGHT).astype(int)
# data_xs = (data_[ID,:,1]*HEIGHT).astype(int)
# for gt_y, gt_x, data_y, data_x in zip(gt_ys, gt_xs, data_ys, data_xs):

#     offset = 6
#     img[gt_y-offset:gt_y+offset, \
#         gt_x-offset:gt_x+offset,:] = \
#         np.array([[0, 255, 0]])

#     offset = 2
#     img[data_y-offset:data_y+offset, \
#         data_x-offset:data_x+offset,:] = \
#         np.array([[255, 0, 255]])
# cv2.imwrite("test.jpg",img)
