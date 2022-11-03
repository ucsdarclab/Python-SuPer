import os
from skimage import io
import numpy as np
import cv2

dir='/media/bear/77f1cfad-f74f-4e12-9861-557d86da4f681/research_proj/datasets/3d_data/super_dataset/062422/trial_9/seman'

for file in os.listdir(dir):
    seman = io.imread(os.path.join(dir, file))
    new_seman = np.zeros(seman.shape[0:2])
    new_seman[seman[:,:,0] == 128] = 1
    new_seman[seman[:,:,1] == 128] = 2
    
    cv2.imwrite(os.path.join(dir, file), new_seman)