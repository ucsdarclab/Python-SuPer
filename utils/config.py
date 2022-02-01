import numpy as np
import timeit
import torch
import os

cuda0 = torch.device('cuda:0')

SQRT2 = np.sqrt(2)
DIVTERM = 1./(2.*0.6*0.6)

print_time = False # If true, print time spent by each important function.
debug_mode = False

visualize = False # If true, visualize surfels as the program runsç.
open3d_visualize = False # If true, visualize using open3d. Otherwise, visualize using opencv.

qual_color_eval = True
colorize_new_surfels = True

save_20pts_tracking_result = True # If true, save images with tracking ground truth & results of the 20 labeled points in the SuPer dataset.
save_img_feature_matching = False # If true, save images with matched feature points.
save_opt_rst = False # If true, save data/images of the errors before & after LM Algorithm.

use_mask = False #  If true, read the segmentation mask to exclude instrument depth info from the depth map
estimate_cam_pose = False

dataset_class = 'RGBD_mask'

arap_lambda = np.sqrt(10.)
rot_lambda = 10. # sqrt(100.)
corr_lambda = np.sqrt(10.)

# parameters of super data set
fx = 768.98551924
fy = 768.98551924
cx = 292.8861567
cy = 291.61479526
HEIGHT = 480
WIDTH = 640
PIXEL_NUM = HEIGHT * WIDTH
zero_img = torch.zeros((HEIGHT,WIDTH,3), dtype=float, device=cuda0)

# K = np.array([[fx,  0, cx],
#                 [ 0, fy, cy],
#                 [ 0,  0,  1]])# Camera intrinsic matrix

depth_scale = 1/100. # depth unit: cm -> m
# Hmmm, but I feel the unit of the input depth maps is mm.

# cross_prod_skew_mat = torch.sparse_coo_tensor([[2,1,2,0,1,0],[1,2,3,5,6,7]], \
#     [-1,1,1,-1,-1,1], (3, 9), device=cuda0)

# U = range(0, WIDTH)
# V = range(0, HEIGHT)
# U, V = np.meshgrid(U, V)
# U = U.astype(float)
# V = V.astype(float)
U = torch.arange(WIDTH, dtype=float, device=cuda0)
V = torch.arange(HEIGHT, dtype=float, device=cuda0)
V, U = torch.meshgrid(V, U)

THRESHOLD_COSINE_ANGLE = 0.4 # C++ code: accept angle < 67 degree
THRESHOLD_DISTANCE = 0.05 # C++ code value: 0.2
THRESHOLD_CONFIDENCE = 10.0
FUSE_INIT_TIME = 30.0

UPPER_ED_DISTANCE = 0.30 # Threshold (30mm) for deciding if a new surfel should be an ED node.
LOWER_ED_DISTANCE = 0.05 # Threshold (5mm) for deciding if a ED node should be deleted.

n_neighbors = 4
ED_n_neighbors = 8

output_folder = "./outputs"
tracking_rst_folder = os.path.join(output_folder,"track_rst")
match_folder = os.path.join(output_folder,"matches")
render_folder = os.path.join(output_folder,"render_images")
qual_color_folder = os.path.join(output_folder,"qual_color_images")
error_folder = os.path.join(output_folder,"error_images")
o3d_display_dir = os.path.join(output_folder,"o3d_display")

#####################################################################################################################
# PWC-Net MODEL INFO
#####################################################################################################################

model_module_to_load = "full_model" # A: "only_flow_net", B: "full_model"
model_name           = "model_A"    # your model's name, e.g., "model_A", "chairs_things"
model_iteration      = 0            # iteration number of the model you want to load

saved_model = os.path.join("./experiments", "models", model_name, f"{model_name}_{model_iteration}.pt")

# freeze_optical_flow_net = False
