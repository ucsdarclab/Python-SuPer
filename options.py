from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

class SuPerOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="options")

        ''' METHOD options '''
        self.parser.add_argument('--method', 
                                 default='super')
        self.parser.add_argument('--phase', 
                                 default='test')
        self.parser.add_argument('--start_id', 
                                 type=int, 
                                 default=4)
        self.parser.add_argument('--end_id', 
                                 type=int, 
                                 default=521)
        self.parser.add_argument("--seed", 
                                 type=int, 
                                 default=0)
        self.parser.add_argument("--use_derived_gradient", 
                                 action="store_true",
                                 help="SuPer: if set, use the derived gradient to update parameter instead of pytorch autograd",)
        self.parser.add_argument('--save_sample_freq', 
                                 type=int,
                                 default=10)
        self.parser.add_argument('--output_dir', 
                                 default='results')
        self.parser.add_argument('--model_name',
                                 required=True)

        self.parser.add_argument("--optimizer",
                                 default="SGD")
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 default=5e-5)
        self.parser.add_argument('--num_optimize_iterations', 
                                 type=int, 
                                 default=10,
                                 help="SuPer: number of iterations to optimize deformation")
        self.parser.add_argument('--num_ED_neighbors', 
                                 type=int, 
                                 default=4)
        self.parser.add_argument('--num_neighbors', 
                                 type=int, 
                                 default=4)
        self.parser.add_argument('--th_dist', 
                                 type=float, 
                                 default=0.1)
        self.parser.add_argument('--th_cosine_ang', 
                                 type=float, 
                                 default=0.4)
        # self.parser.add_argument('--th_conf', 
        #                          type=float, 
        #                          default=10,
        #                          help='SuPer: Confidence threshold for stable surfels. If negative, confidence is not used.')
        self.parser.add_argument('--th_time_steps', 
                                 type=int, 
                                 default=30,
                                 help='SuPer: a surfel is unstable if it has not been updated for th_num_time_steps and has low confidence')
        self.parser.add_argument("--disable_removing_unstable_surfels", 
                                 action="store_true",
                                 help="SuPer: if set, after updating surfels, surfels considered as unstable (according to SuPer) will not be added")
        self.parser.add_argument("--disable_merging_new_surfels", 
                                 action="store_true",
                                 help="SuPer: if set, after initialization, new surfels will not be merged with existing surfels")
        self.parser.add_argument("--disable_merging_exist_surfels", 
                                 action="store_true",
                                 help="SuPer: if set, after initialization, existing surfels will not be fused")
        self.parser.add_argument("--disable_adding_new_surfels", 
                                 action="store_true",
                                 help="SuPer: if set, after initialization, new surfels will not be added")

        self.parser.add_argument("--normal_model", 
                                 default="8neighbors", 
                                 help="method to estimate the normal from depth")
        self.parser.add_argument("--deform_udpate_method", 
                                 default='super_edg', 
                                 help='Method to update surfel and ED node deformation. [edg] corresponds to SuPer method which updates deformation via EDG.')
        self.parser.add_argument("--downsample_params", 
                                  nargs="+", 
                                  type=float, 
                                  default=[0.1, 50, 0.1],
                                  help="only for encoding ED nodes")
        self.parser.add_argument("--ball_piv_radii", 
                                 nargs="+", 
                                 type=float, 
                                 default=[0.08],
                                 help="only for encoding ED nodes")
        self.parser.add_argument('--mesh_step_size', 
                                 type=int, 
                                 default=30)

        self.parser.add_argument("--load_depth", 
                                 action="store_true",
                                 help="if set, load the depth map")
        self.parser.add_argument('--depth_ext', 
                                 default='.npy')
        self.parser.add_argument("--min_depth", 
                                 type=float, 
                                 default=0.1, 
                                 help='Monodepth2: minimum depth')
        self.parser.add_argument("--max_depth", 
                                 type=float, 
                                 default=80.0,
                                 help='Monodepth2: maximum depth')
        self.parser.add_argument('--depth_model')
        # Parameters for Monodepth2.
        self.parser.add_argument('--num_layers', 
                                 type=int, 
                                 default=0, 
                                 choices=[0, 18, 34, 50, 101, 152],
                                 help="Encoder: number of resnet encoder layers")
        # Parameters for RAFTStereo.
        self.parser.add_argument('--valid_iters',
                                 type=int,
                                 default=32)
        self.parser.add_argument('--hidden_dims',
                                 nargs="+", 
                                 type=int,
                                 default = [128, 128, 128])
        self.parser.add_argument('--corr_levels',
                                 type=int,
                                 default=4)
        self.parser.add_argument('--corr_radius',
                                 type=int,
                                 default=4)
        self.parser.add_argument('--shared_backbone',
                                 action="store_true")
        self.parser.add_argument('--n_downsample',
                                 type=int,
                                 default=2)
        self.parser.add_argument('--context_norm',
                                 default='batch')
        self.parser.add_argument('--slow_fast_gru',
                                 action="store_true")
        self.parser.add_argument('--n_gru_layers',
                                 type=int,
                                 default=3)
        self.parser.add_argument('--corr_implementation',
                                 default='reg')
        self.parser.add_argument('--mixed_precision',
                                 action="store_true")
        # Checkpoints.
        self.parser.add_argument('--pretrained_depth_checkpoint_dir', 
                                 help='Path to pretrained depth model checkpoints.')
        self.parser.add_argument('--pretrained_encoder_checkpoint_dir', 
                                 dest='pretrained_encoder_checkpoint_dir', 
                                 help='Path to pretrained encoder checkpoints.')
        self.parser.add_argument("--post_process", 
                                 action="store_true",
                                 help="Monodepth2: if set will perform the flipping post processing from the original monodepth paper")
        self.parser.add_argument("--depth_width_range", 
                                 nargs="+", 
                                 type=float, 
                                 default=[0.02, 0.98])
        self.parser.add_argument('--depth_filter_kernel_size', 
                                 type=int, 
                                 default=-1)
        self.parser.add_argument("--weights_init", 
                                 type=str, 
                                 default="pretrained", 
                                 choices=["pretrained", "scratch"],
                                 help="pretrained or scratch")

        self.parser.add_argument('--optical_flow_model')

        self.parser.add_argument('--renderer', 
                                 default='pulsar', 
                                 choices=['pulsar'],
                                 help='Renderer: Method to render image from the tracked surfels.')
        self.parser.add_argument('--renderer_rad', 
                                 type=float, 
                                 default=0.0002,
                                 help='Renderer rad parameter: required for pulsar')

        ''' DATA options '''
        self.parser.add_argument('--data_dir', 
                                 default=os.path.join(file_dir, 'v1_520_pairs'),
                                 help='path to data directory')
        self.parser.add_argument('--rgb_dir', 
                                 default='rgb')
        self.parser.add_argument('--depth_dir', 
                                 default='depth')
        self.parser.add_argument('--seg_dir', 
                                 default='seg/DeepLabV3+')
        self.parser.add_argument('--data', 
                                 default='superv1')
        self.parser.add_argument('--height', 
                                 type=int, 
                                 default=480)
        self.parser.add_argument('--width', 
                                 type=int, 
                                 default=640)
        self.parser.add_argument('--img_ext', 
                                 default='.png')

        self.parser.add_argument("--load_valid_mask", 
                                 action="store_true",
                                 help="if set, load the valid mask where invalid regions will be excluded for surfel initialization")
        self.parser.add_argument('--valid_mask_dir',  
                                 default='seg/tissue')
        self.parser.add_argument('--dilate_invalid_kernel', 
                                 type=int, 
                                 default=5)

        ''' ABLATION options '''
        self.parser.add_argument('--sf_point_plane', 
                                 action='store_true', 
                                 help='Losses: point-plane ICP loss')
        self.parser.add_argument('--sf_point_plane_weight', 
                                 type=float, 
                                 default=1.)

        self.parser.add_argument('--mesh_arap', 
                                 action='store_true', 
                                 help='Losses: as-rigid-as-possible (ARAP) loss on ED graph')
        self.parser.add_argument('--mesh_arap_weight', 
                                 type=float, 
                                 default=10.)
        self.parser.add_argument('--mesh_face', 
                                 action='store_true',
                                 help='Losses: Face loss, an alternative loss for ARAP loss')
        self.parser.add_argument('--mesh_face_weight', 
                                 type=float, 
                                 default=1.)

        self.parser.add_argument('--mesh_rot', 
                                 action='store_true', 
                                 help='Losses: quaternion normalization term (Rot loss)')
        self.parser.add_argument('--mesh_rot_weight', 
                                 type=float, 
                                 default=1.)
        
        self.parser.add_argument('--sf_corr', 
                                 action='store_true', 
                                 help='Losses: Surfel correspondence loss')
        self.parser.add_argument('--sf_corr_match_renderimg', 
                                 action='store_true')
        self.parser.add_argument('--sf_corr_weight', 
                                 type=float, 
                                 default=0.001)
        self.parser.add_argument('--sf_corr_loss_type', 
                                 default='point-point')

        ''' SYSTEM options '''
        self.parser.add_argument("--gpu", 
                                 type=int, 
                                 default=0)

        ''' EVALUATION options '''
        self.parser.add_argument('--tracking_gt_file')

        # Backup parameters.
        # parser.add_argument('--files', nargs="+", default=['tracking_rst.npy'],
        #                     help='Accepcts a list of arguments. Can specify multiple tracking result files to evaluate (tracking_rst.npy)')       
        # parser.add_argument('--files_legends', nargs="+", default=['DefSLAM'],
        #                     help='Accepcts a list of arguments. For each result file, annotate its method as a legend')    
        # parser.add_argument('--files_to_plot', type=int, nargs="+", default=[1],
        #                     help='Accepcts a list of arguments. For each result file, put `1` to plot, `0` to not plot')
        # parser.add_argument('--files_to_plot_time_error', type=int, nargs="+", default=[0],
        #                     help='Accepcts a list of arguments.')  
        # # Point tracking evalutation
        # parser.add_argument('--num_points', type=int, default=20, help='number of tracked points')
        # parser.add_argument('--igonored_ids', type=int, nargs="+", default=[])  
        #  
        # parser.add_argument('--start_timestamp', type=int, default=1)
        # parser.add_argument('--end_timestamp', type=int, default=519) 
        # parser.add_argument('--traj_ids', type=int, nargs="+",  default=[0,1,2,3]) 
        # parser.add_argument('--traj_start_timestamp', type=int, default=60)
        # parser.add_argument('--traj_end_timestamp', type=int, default=120) 
        # parser.add_argument('--output_dir', default="./results/eval_output")   
        # parser.add_argument('--output_figure_name', default="evaluate.png")      
        # parser.add_argument('--save_seg_ply', action='store_true')

    def parse(self, args=None):
        self.options = self.parser.parse_args(args)
        return self.options

class SemanticSuPerOptions(SuPerOptions):
    def __init__(self):
        super(SemanticSuPerOptions, self).__init__()

        ''' METHOD options '''
        self.parser.set_defaults(method='semantic-super')

        self.parser.add_argument("--load_seg", 
                                 action="store_true",
                                 help="if set, load the segmentation map")
        self.parser.add_argument('--seg_ext', 
                                 default='.npy')
        self.parser.add_argument('--seg_model')
        self.parser.add_argument('--pretrained_seg_checkpoint_dir',
                                 help='Segmentation: Path to pretrained segmentation model checkpoints.')
        self.parser.add_argument("--seg_num_layers", 
                                 type=int,
                                 help="number of resnet layers for segmentation model")
        self.parser.add_argument("--hard_seg", 
                                 action="store_true",
                                 help="SemanticSuper Confidence: if set, only search for same semantic class when finding knn surfels for ED nodes")
        self.parser.add_argument("--del_seg_classes", 
                                 nargs="+", 
                                 type=int, 
                                 default=[],
                                 help="classes ID that will be ignored for tracking")

        self.parser.add_argument("--disable_ssim_conf", 
                                 action="store_true",
                                 help="if set, warp right image to the left view and use the ssim between the real and warp image as surfel init confidence")

        ''' DATA options '''
        self.parser.set_defaults(data='superv2')
        self.parser.set_defaults(start_id=0)
        self.parser.set_defaults(end_id=151)
        self.parser.set_defaults(seg_dir='seg')
        self.parser.add_argument("--num_classes", 
                                 type=int, 
                                 default=3,
                                 help="number of semantic classes")
        self.parser.add_argument('--edge_ids', 
                                 type=int, 
                                 nargs="+", 
                                 default=[]) 

        ''' ABLATION options '''
        self.parser.add_argument('--sf_hard_seg_point_plane', 
                                 action='store_true',
                                 help='Semantic-SuPer Losses: Semantic-aware point-to-plane ICP loss. If set, only match surfels from the same semantic class')
        self.parser.add_argument('--sf_soft_seg_point_plane', 
                                 action='store_true',
                                 help='Semantic-SuPer Losses: Semantic-aware point-to-plane ICP loss. If set, weight ICP loss with Jensen-Shannon divergence between the softmax segmentation output')

        self.parser.add_argument("--sf_bn_morph", 
                                 action="store_true",
                                 help='Semantic-SuPer Losses: Semantic-aware morphing loss')
        self.parser.add_argument("--sf_bn_morph_weight", 
                                 type=float,
                                 default=0.1)

        self.parser.add_argument('--render_loss', 
                                 action='store_true', 
                                 help='Losses: Rendering loss')
        self.parser.add_argument('--render_loss_weight', 
                                 type=float, 
                                 default=1e-4)