import os
import numpy as np
import random
import argparse

import torch
import torch.backends.cudnn as cudnn

from utils.shared_functions import init_dataset, InitNets

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu",
                        type=int,
                        default=0)

    #### Shared parameters ####

    parser.add_argument('--phase', 
                        default='test', 
                        help='[train] is for our future work that allows super to be end-to-end trainable.',
                        choices=['train', 'test'])
    parser.add_argument('--mod_id', 
                        type=int,
                        required=True)
    parser.add_argument('--exp_id', 
                        type=int, 
                        required=True)
    parser.add_argument("--seed",
                        type=int,
                        default=0)
    parser.add_argument('--save_sample_freq', 
                        type=int, 
                        default=10)
    parser.add_argument('--sample_dir', 
                        dest='sample_dir', 
                        default='sample')
    # parser.add_argument('--save_raw_data',
    #                     action='store_true')
    parser.add_argument('--save_ply',
                        action='store_true')

    # Data options.
    parser.add_argument('--data', 
                        default='super')
    parser.add_argument('--start_id', 
                        type=int, 
                        default=0)
    parser.add_argument('--end_id', 
                        type=int)
    parser.add_argument('--data_dir', 
                        required=True)
    parser.add_argument('--data_seg_dir', 
                        help='path to segmentation data under data_dir',
                        default='seg/DeepLabV3+')
    parser.add_argument('--tracking_gt_file', 
                        dest='tracking_gt_file')
    parser.add_argument('--height', 
                        type=int, 
                        default=480)
    parser.add_argument('--width', 
                        type=int, 
                        default=640)
    parser.add_argument('--img_ext', 
                        default='.png')
    parser.add_argument("--load_depth",
                        help="if set, load the depth map",
                        action="store_true")
    parser.add_argument('--depth_ext', 
                        default='.npy')
    parser.add_argument("--load_seg",
                        help="if set, load the segmentation map",
                        action="store_true")
    parser.add_argument('--seg_ext', 
                        default='.npy')
    parser.add_argument("--del_seg_classes", 
                        nargs="+", 
                        type=int, 
                        help="classes ID that will be ignored for tracking", 
                        default=[])
    parser.add_argument('--del_seg_kernel', 
                        type=int,
                        default=5)
    parser.add_argument("--depth_width_range",
                        nargs="+", 
                        type=float, 
                        default=[0.02, 0.98])
    # Augmentation
    parser.add_argument("--disable_side_shuffle", 
                        action="store_true")
    parser.add_argument("--disable_color_aug", 
                        action="store_true")    
    parser.add_argument("--disable_horizontal_flip", 
                        action="store_true")
    parser.add_argument("--disable_vertical_flip", 
                        action="store_true")

    # Surfels and ED graph
    parser.add_argument("--downsample_params", 
                        nargs="+", 
                        type=float, 
                        help="only for encoding ED nodes", 
                        default=[0.1, 50, 0.1])
    parser.add_argument("--ball_piv_radii", 
                        nargs="+", 
                        type=float, 
                        help="only for encoding ED nodes", 
                        default=[0.08])
    parser.add_argument('--depth_filter_kernel_size', 
                        type=int,
                        default=-1)
    parser.add_argument('--num_ED_neighbors', 
                        type=int, 
                        default=8)
    parser.add_argument('--num_neighbors', 
                        type=int, 
                        default=4)

    # Model options.
    parser.add_argument('--method', 
                        default='super',
                        help='Use naive SuPer or Semantic-SuPer.', 
                        choices=['super', 'semantic-super'])
    # Renderer
    parser.add_argument('--renderer',
                        default='pulsar',
                        help='Method to render image from the tracked surfels.', 
                        choices=['pulsar'])
    parser.add_argument('--renderer_rad', 
                        type=float, 
                        help='required for pulsar',
                        default=0.0005)
    # Encoder
    parser.add_argument("--num_layers",
                        type=int,
                        help="number of resnet layers",
                        default=0,
                        choices=[0, 18, 34, 50, 101, 152])
    parser.add_argument("--seg_num_layers",
                        type=int,
                        help="number of resnet layers for segmentation model")
    parser.add_argument('--pretrained_encoder_checkpoint_dir', 
                        dest='pretrained_encoder_checkpoint_dir', 
                        help='Path to pretrained encoder checkpoints.')
    # Depth model
    parser.add_argument('--depth_model')
    parser.add_argument('--pretrained_depth_checkpoint_dir', 
                        help='Path to pretrained depth model checkpoints.')
    # Monodepth2
    parser.add_argument("--min_depth",
                        type=float,
                        default=0.1)
    parser.add_argument("--max_depth",
                        type=float,
                        default=100.0)
    parser.add_argument("--post_process",
                        help="if set will perform the flipping post processing "
                            "from the original monodepth paper",
                        action="store_true")
    parser.add_argument("--weights_init",
                        type=str,
                        help="pretrained or scratch",
                        default="pretrained",
                        choices=["pretrained", "scratch"])
    # Segmentation model
    parser.add_argument('--seg_model')
    parser.add_argument('--pretrained_seg_checkpoint_dir',
                        help='Path to pretrained segmentation model checkpoints.')
    # # Depth+Seg
    # parser.add_argument("--share_depth_seg_model",
    #                     help="If set, the depth and segmentation model share encoder and decoder.",
    #                     action="store_true")
    # # Optical flow model
    # parser.add_argument('--optical_flow_model')
    # parser.add_argument('--pretrained_optical_flow_checkpoint_dir',
    #                     help='Path to pretrained optical flow model checkpoints.')
    # parser.add_argument('--optical_flow_features')
    # parser.add_argument("--raft_option",
    #                     help="the model type of RAFT",
    #                     default="small", 
    #                     choices=["small", "large", "depth", "seg"])

    # LOGGING options
    parser.add_argument("--log_frequency",
                        type=int,
                        help="number of batches between each tensorboard log",
                        default=30)
    parser.add_argument('--nologfile', 
                        action='store_true')

    parser.add_argument('--save_seg_ply',
                        action='store_true')
    ########

    #### SuPer ####
    parser.add_argument("--num_classes",
                        type=int,
                        help="number of semantic classes",
                        default=2)
    parser.add_argument("--normal_model",
                        help="method to estimate the normal from depth",
                        default="8neighbors")
    parser.add_argument('--mesh_step_size', 
                        type=int,
                        default=30)
    parser.add_argument('--th_dist',
                        type=float,
                        default=0.1)
    parser.add_argument('--th_cosine_ang', 
                        type=float,
                        default=0.4)
    parser.add_argument("--use_derived_gradient",
                        help="if set, use the dericed gradient to update parameter instead of pytorch autograd",
                        action="store_true")
    parser.add_argument('--th_conf', 
                        type=float,
                        help='Confidence threshold for stable surfels. If negative, confidence is not used.', 
                        default=10)
    parser.add_argument('--th_time_steps', 
                        type=int,
                        help='a surfel is unstable if it has not been updated for th_num_time_steps and has low confidence', 
                        default=30)
    parser.add_argument("--disable_merging_new_surfels",
                        help="if set, after initialization, new surfels will not be added",
                        action="store_true")

    # Losses
    parser.add_argument('--sf_point_plane', 
                        help='point-plane ICP loss',
                        action='store_true')
    parser.add_argument('--sf_point_plane_weight', 
                        type=float, 
                        default=1.)
    
    # parser.add_argument('--feature_loss', 
    #                     action='store_true')
    # parser.add_argument("--feature_loss_option",
    #                     help="which feature to be used",
    #                     default="depth", 
    #                     choices=["depth", "seg"])
    # parser.add_argument('--feature_loss_weight', 
    #                     type=float, 
    #                     default=1)

    parser.add_argument('--mesh_arap', 
                        action='store_true', 
                        help='as rigid as possible loss on ED graph')
    parser.add_argument('--mesh_arap_weight', 
                        type=float, 
                        default=10)

    parser.add_argument('--mesh_rot', 
                        action='store_true', 
                        help='quaternion normalization term (Rot loss)')
    parser.add_argument('--mesh_rot_weight', 
                        type=float, 
                        default=10.)

    # # Correlation loss
    # parser.add_argument('--sf_corr', 
    #                     action='store_true')
    # parser.add_argument('--sf_corr_end_iter', 
    #                     type=int,
    #                     default=10)
    # parser.add_argument('--sf_hard_seg_corr', 
    #                     action='store_true')
    # parser.add_argument('--sf_soft_seg_corr', 
    #                     action='store_true')
    # parser.add_argument('--sf_corr_use_keyframes', 
    #                     action='store_true')
    # parser.add_argument('--sf_corr_match_renderimg', 
    #                     action='store_true')
    # parser.add_argument('--sf_corr_weight', 
    #                     type=float, 
    #                     dest='sf_corr_weight', 
    #                     default=1)
    # parser.add_argument('--sf_corr_huber_th', 
    #                     type=float, 
    #                     default=-1) # 1e-2
    # parser.add_argument('--sf_corr_loss_type', 
    #                     default='point-plane')
    ########

    #### Semantic-SuPer ####
    # confidence options
    parser.add_argument("--use_ssim_conf",
                        help="if set, warp right image to the left view and use the ssim between the real and warp image as confidence",
                        action="store_true")
    parser.add_argument("--use_seg_conf",
                        help="if set, warp right semantic map to the left view and compare the real and warp semantic map as confidence",
                        action="store_true")
    parser.add_argument("--hard_seg",
                        action="store_true",
                        help="if set, only search for same semantic class when finding knn surfels for ED nodes")
    
    # Losses
    # Semantic-aware point-to-plane ICP
    parser.add_argument('--sf_hard_seg_point_plane', 
                        help='if set, only match surfels from the same semantic class',
                        action='store_true')
    parser.add_argument('--sf_soft_seg_point_plane', 
                        help='if set, weight ICP loss with Jensen–Shannon divergence between the softmax segmentation output',
                        action='store_true')
    # Rendering loss
    parser.add_argument('--render_loss', 
                        help='rendering loss',
                        action='store_true')
    parser.add_argument('--render_loss_weight', 
                        type=float, 
                        default=1)
    # Morph loss
    parser.add_argument("--sf_bn_morph",
                        help='semantic-aware morphing loss',
                        action="store_true")
    parser.add_argument("--sf_bn_morph_weight",
                        type=float, 
                        default=10)
    # Face loss
    parser.add_argument('--mesh_face', 
                        action='store_true')
    parser.add_argument('--mesh_face_weight', 
                        type=float, 
                        dest='mesh_face_weight', 
                        default=10)
    # parser.add_argument('--mesh_edge', 
    #                     help='edge loss',
    #                     action='store_true')
    # parser.add_argument('--mesh_edge_weight', 
    #                     type=float, 
    #                     default=10)
    # parser.add_argument('--use_edge_ssim_hints', 
    #                     action='store_true')
    # parser.add_argument('--edge_ssim_hints_window', 
    #                     type=int,
    #                     default=6)
    ########
    
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu) # Set GPU.

    """
    Fixed random seed.
    """
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    ########################################################

    # Track and update the 3D scene (surfels) frame-by-frame.

    ########################################################
    if args.method in ['super', 'semantic-super'] or args.phase == 'test':

        # Init data loader.
        testloader = init_dataset(args)

        models = InitNets(args)

        for inputs in testloader:
            models.super(models, inputs)

        # mssim_mean = np.mean(models.super.sf.mssim)
        # mssim_std = np.std(models.super.sf.mssim)
        # models.super.sf.logger.info(f"ssim mean: {mssim_mean}, ssim std: {mssim_std}")
        
        if args.tracking_gt_file is not None:
            with open(os.path.join(args.sample_dir, f"model{args.mod_id}_exp{args.exp_id}", f"tracking_rst.npy"), 'wb') as f:
                np.save(f, models.super.sf.track_rsts)

    # elif args.phase == 'train':
    #     # TODO
    
    # else:
    #     assert False, "Invalid configuration."
    
if __name__ == '__main__':
    main()