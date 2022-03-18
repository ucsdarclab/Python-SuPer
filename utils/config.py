import numpy as np
import timeit
import torch
import os
import torch

dev = torch.device('cuda:0')

torch_version = torch.__version__

tfdtype_ = torch.float64
dtype_ = torch.float32

fl64_ = torch.float64
fl32_ = torch.float32
int_ = torch.int
long_ = torch.int64
bool_ = torch.bool

#############################

# Camera Parameter

#############################
# Parameters of images in super data set
fx = 768.98551924
fy = 768.98551924
cx = 292.8861567
cy = 291.61479526
# K = np.array([[fx,  0, cx],
#                 [ 0, fy, cy],
#                 [ 0,  0,  1]])# Camera intrinsic matrix

#############################

# Configuarations.

#############################
n_neighbors = 4
ED_n_neighbors = 8

ED_sample_method = 'grid' # ED node sampling method.
if ED_sample_method == 'uniform': ED_rad = 0.2

render_method = 'pulsar' # Options: pulsar, proj
corr_method = 'kornia' # Options: opencv, kornia (LoFTR)

estimate_cam_pose = False

#############################

# Display setting

#############################
P_time = True # If true, print time spent by each important function.

output_folder = "./outputs" # Parent folder.
evaluate_folder = "./evaluates" # Folder of results for evaluation.
folders = [] # Init the list of child folders.

# If ture, save an image of the reconstructed 3D scene, where the colors 
# represent the normal directions.
vis_depth_preprocessing = True
if vis_depth_preprocessing: 
    F_depth_prepro = os.path.join(output_folder,"depth_prepro")
    folders.append(F_depth_prepro)

# If true, save the rendered image after each iter.
save_render_img = True
vis_ED_nodes = True
if save_render_img:
    F_render_img = os.path.join(output_folder,"render_imgs")
    folders.append(F_render_img)

# If true, save the image of points with continuous colors for qualitative evaluation.
qual_color_eval = False
if qual_color_eval:
    F_qual_color_img = os.path.join(output_folder,"qual_color_imgs")
    folders.append(F_qual_color_img)

# If true, save images with tracking ground truth & results of the 20 labeled points 
# in the SuPer dataset.
vis_super_track_rst = True
if vis_super_track_rst:
    F_super_track = os.path.join(output_folder,"super_track")
    folders.append(F_super_track)

# If true, save images with matched feature points.
vis_img_matching = True
if vis_img_matching:
    F_img_matching = os.path.join(output_folder,"img_matching")
    folders.append(F_img_matching)

debug_mode = False # TODO
visualize = False # TODO If true, visualize surfels as the program runsç.
open3d_visualize = False # TODO If true, visualize using open3d. Otherwise, visualize using opencv.



save_opt_rst = False # If true, save data/images of the errors before & after LM Algorithm.

use_mask = False #  If true, read the segmentation mask to exclude instrument depth info from the depth map





# tracking_rst_folder = os.path.join(output_folder,"track_rst")
# match_folder = os.path.join(output_folder,"matches")
# qual_color_folder = os.path.join(output_folder,"qual_color_images")
# error_folder = os.path.join(output_folder,"error_images")
# o3d_display_dir = os.path.join(output_folder,"o3d_display")




HEIGHT = 480
WIDTH = 640
PIXEL_NUM = HEIGHT * WIDTH
zero_img = torch.zeros((HEIGHT,WIDTH,3), dtype=dtype_, device=dev)

U = torch.arange(WIDTH, dtype=dtype_, device=dev)
V = torch.arange(HEIGHT, dtype=dtype_, device=dev)
V, U = torch.meshgrid(V, U)

################
# inputStream.py
################

dataset_class = 'RGBD_mask'

depth_scale = 1/100. # depth unit: cm -> m
# But I feel the unit of the input depth maps is mm.

SQRT2 = np.sqrt(2)
DIVTERM = 1./(2.*0.6*0.6)

THRESHOLD_COSINE_ANGLE = 0.4 # C++ code: accept angle < 30 degree (0.87), 67 degree (0.4)
THRESHOLD_DISTANCE = 0.2 # C++ code value: 0.2

CONF_TH = 10.0 # TODO: Select better value.
STABLE_TH = 30.0

UPPER_ED_DISTANCE = 0.30 # Threshold (30mm) for deciding if a new surfel should be an ED node.
LOWER_ED_DISTANCE = 0.05 # Threshold (5mm) for deciding if a ED node should be deleted.
CLS_SIZE_TH = 300






# #####################################################################################################################
# # PWC-Net MODEL INFO
# #####################################################################################################################

# model_module_to_load = "full_model" # A: "only_flow_net", B: "full_model"
# model_name           = "model_A"    # your model's name, e.g., "model_A", "chairs_things"
# model_iteration      = "pt"            # iteration number of the model you want to load

# saved_model = os.path.join("./experiments", "models", model_name, f"{model_name}_{model_iteration}.pt")

# # freeze_optical_flow_net = False