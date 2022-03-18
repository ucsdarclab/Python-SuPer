import os
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import *
from utils.config import *

def evaluate(gt, est, normalize=True, start=18, end=30):
# def evaluate(gt, est, normalize=True, start=0, end=52):

    diff = distance(gt,est)
    
    mean_ = []
    std_ = []
    valid = (gt[:,:,0] > 0) & (gt[:,:,1] > 0)
    for i in range(20):
        diff_temp = diff[:,i][valid[:,i]]
        # Normalize the coordinates by image size.
        if normalize: diff_temp /= HEIGHT
        mean_.append(np.mean(diff_temp[start-1:end]))
        std_.append(np.std(diff_temp[start-1:end]))

    return np.array(mean_), np.array(std_)

def main():

    # Load ground truth.
    with open(os.path.join(evaluate_folder, "gt.npy"), 'rb') as f:
            gt = np.load(f).astype(float)

    colors = ['tab:red', 'tab:green', 'tab:orange', \
        'tab:blue', 'tab:pink', 'tab:gray', 'tab:cyan']
    files = {"C++SuPer":"cpp_super.npy", \
        # "pySuPer(data,arap,rot)":"super-pulsar_data_ARAP_rot.npy", \
        "pySuPer (data,arap,rot)":"default.npy", \
        # "pySuPer+grid+smoothZ+SNE(data,arap,rot)":"default-SNE.npy", \
        # "pySuPer+uniform+smoothZ(data,arap,rot)":"default-uniform.npy", \
        # "pySuPer+grid+smoothZ+proj(data,arap,rot)":"default-proj.npy", \
        # "pySuPer+grid+proj(data,arap,rot)":"default-proj-raw.npy", \
        "pySuPer (data,arap,rot,corr)":"super_data_ARAP_rot.npy", \
        }
    # "SURF":"surf.npy"
    # Default setting: super; grid, pulsar, filterZ; losses: data, ARAP, rot
    ## Compare with default:
    # a) default-SNE: worse
    # b) default-uniform: no diff
    # c) default-proj: no diff
    # d) default-proj-raw: worse, raw--- no smooth
    # e) FeatLoss: Use smaller lambda (0.1) for depth / point loss; depth loss is 
    #    slightly worse than data loss; point loss is slightly better than data
    #    loss; no diff when data+depth / data+point.
    # f) CorrLoss: opencv matcher, large weight --> worse, small weight --> no diff

    plt.figure(figsize=(10,3))
    ind = np.arange(20)
    bar_width = 0.4
    offset = 0.0

    for key, color_ in zip(files.keys(), colors):
        # Load the corresponding tracking results.
        with open(os.path.join(evaluate_folder, files[key]), 'rb') as f:
            rst = np.load(f).astype(float)

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