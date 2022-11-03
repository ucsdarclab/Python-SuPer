import os
import cv2
import numpy as np

dir1 = "./results/super/trial_8/model0_exp0"
dir2 = "./results/super/trial_8/model3_exp3"
out_dir = f"./results/{os.path.basename(dir1)}_VS_{os.path.basename(dir2)}"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for filename in os.listdir(dir1):
    if filename.endswith('.png') and os.path.exists(os.path.join(dir2, filename)):
        img1 = cv2.imread(os.path.join(dir1, filename))
        img2 = cv2.imread(os.path.join(dir2, filename))

        out_img = np.concatenate([img1, img2], axis=1)
        cv2.imwrite(os.path.join(out_dir, filename), out_img)