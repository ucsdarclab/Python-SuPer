import os
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import *
from utils.config import *

def evaluate(gt, est, normalize=False):

    diff = []
    diff_valid = []
    for key in est.keys():
        diff.append(distance(gt[int(key)],est[key]))
        diff_valid.append(gt[int(key)][:,0] >= 0)

    # print(diff_valid)
    diff = np.stack(diff)
    diff_valid = np.stack(diff_valid)
    
    mean_ = []
    std_ = []
    for i in range(diff.shape[1]):
        diff_temp = diff[:,i][diff_valid[:,i]]
        # Normalize the coordinates by image size.
        if normalize: diff_temp /= HEIGHT
        mean_.append(np.mean(diff_temp))
        std_.append(np.std(diff_temp))

    return np.array(mean_), np.array(std_)

def main():
    new_evaluate_folder = os.path.join(result_folder, evaluate_folder)

    data_dir = "/media/bear/77f1cfad-f74f-4e12-9861-557d86da4f68/research_proj/datasets/3d_data/super_dataset/new_super"
    gt_file = "uncover1l_pts.npy"
    gt_file = os.path.join(data_dir, gt_file)
    gt = np.array(np.load(gt_file, allow_pickle=True)).tolist()['gt']

    colors = ['tab:red', 'tab:green', 'tab:orange', \
        'tab:blue', 'tab:pink', 'tab:gray', 'tab:cyan']
    files = {
        "SuPer(data,arap,rot)": "exp1.npy", \
        "semantic-SuPer(data,arap,rot)": "exp2.npy", \
        }

    plt.figure(figsize=(10,3))
    ind = np.arange(gt.shape[1])
    bar_width = 0.4
    offset = 0.0

    for key, color_ in zip(files.keys(), colors):
        # Load the corresponding tracking results.
        rst = np.load(os.path.join(new_evaluate_folder, files[key]), allow_pickle=True).tolist()

        mean_, std_ = evaluate(gt, rst)
        
        plt.bar(ind+offset, mean_, bar_width, color=color_, label=key)
        plt.errorbar(ind+offset, mean_, std_, \
            linestyle='None', color='k', elinewidth=.8, capsize=3)
        offset += 0.2

    plt.legend(loc='upper right')
    plt.xticks(ind, (ind+1).astype(int))
    plt.xlabel('Tracked point ID')
    plt.ylabel('Error in percentage of image size')
    plt.grid(True)
    plt.savefig("evaluate.png", bbox_inches='tight')

if __name__ == '__main__':
    main()