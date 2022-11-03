import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import shutil
import numpy as np
import random
import argparse
from reprint import output

import torch
import torch.backends.cudnn as cudnn

from models.submodules import *

from utils.utils import *


def main():

    parser = argparse.ArgumentParser()

    # DATA options
    parser.add_argument('--data', 
                        dest='data', 
                        default='super')
    parser.add_argument('--data_dir', 
                        dest='data_dir', 
                        required=True)
    parser.add_argument('--height', 
                        type=int, 
                        default=480)
    parser.add_argument('--width', 
                        type=int, 
                        default=640)
    parser.add_argument('--sample_dir', 
                        dest='sample_dir', 
                        default='sample')
    parser.add_argument('--save_raw_data',
                        action='store_true', 
                        default=False)
    parser.add_argument('--save_ply',
                        action='store_true', 
                        default=False)
    parser.add_argument('--save_seman_ply',
                        action='store_true', 
                        default=False)

    # MODEL options
    parser.add_argument('--mod_id', 
                        type=int,
                        required=True)
    parser.add_argument('--exp_id', 
                        type=int, 
                        required=True)
    parser.add_argument('--save_sample_freq', 
                        type=int, 
                        default=10)
    parser.add_argument('--method', 
                        dest='method', 
                        default='super')
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
    parser.add_argument("--num_layers",
                        type=int,
                        help="number of resnet layers",
                        default=0,
                        choices=[0, 18, 34, 50, 101, 152])
    parser.add_argument('--pretrained_encoder_checkpoint_dir', 
                        dest='pretrained_encoder_checkpoint_dir', 
                        help='Path to pretrained encoder checkpoints.')
    parser.add_argument('--depth_model', 
                        dest='depth_model')
    parser.add_argument('--pretrained_depth_checkpoint_dir', 
                        dest='pretrained_depth_checkpoint_dir', 
                        help='Path to pretrained depth model checkpoints.')
    parser.add_argument("--depth_dir",
                        help="if load_depth, input the path to depth map files")
    parser.add_argument("--depth_type",
                        help="if load_depth, type of depth map file",
                        default=".npy")
    parser.add_argument("--normal_model",
                        default="8neighbors")
    parser.add_argument('--seg_model', 
                        dest='seg_model')
    parser.add_argument('--pretrained_seg_checkpoint_dir',
                        dest='pretrained_seg_checkpoint_dir',
                        help='Path to pretrained segmentation model checkpoints.')
    parser.add_argument("--load_seman",
                        help="if set, load the predicted segmentation map",
                        action="store_true")
    parser.add_argument("--load_seman_gt",
                        help="if set, load the segmentation ground truth",
                        action="store_true")
    parser.add_argument("--seman_type",
                        help="if load_seman, type of semantic map file",
                        default=".npy")
    parser.add_argument("--hard_seman",
                        action="store_true",
                        default=False)
    parser.add_argument("--del_seman_classes", 
                        nargs="+", 
                        type=int, 
                        help="classes ID that will be ignored for tracking", 
                        default=[])
    parser.add_argument("--scales", 
                        nargs="+", 
                        type=int, 
                        help="scales used in the loss", 
                        default=[0])
    parser.add_argument('--renderer')
    parser.add_argument('--renderer_rad', 
                        type=float, 
                        default=0.02)
    parser.add_argument('--optical_flow_model', 
                        dest='optical_flow_model')
    parser.add_argument('--mesh_step_size', 
                        type=int,
                        default=30)
    parser.add_argument('--depth_filter_kernel_size', 
                        type=int,
                        default=-1)
    parser.add_argument('--num_ED_neighbors', 
                        type=int, 
                        default=8)
    parser.add_argument('--num_neighbors', 
                        type=int, 
                        default=4)
    parser.add_argument('--th_cosine_ang', 
                        type=float,
                        default=0.6) # 0.4
    parser.add_argument('--th_dist',
                        type=float,
                        default=0.02) # 0.2
    parser.add_argument('--th_conf', 
                        type=float,
                        help='Confidence threshold for stable surfels. If negative, confidence is not used.', 
                        default=10)
    parser.add_argument('--th_time_steps', 
                        type=int,
                        help='a surfel is unstable if it has not been updated for th_num_time_steps and has low confidence', 
                        default=30)

    # TRAINING options
    parser.add_argument("--num_classes",
                        type=int,
                        help="number of semantic classes",
                        default=3)
    parser.add_argument("--min_depth",
                        type=float,
                        help="minimum depth",
                        default=0.1)
    parser.add_argument("--max_depth",
                        type=float,
                        help="maximum depth",
                        default=100.0)
    parser.add_argument("--depth_width_range",
                        nargs="+", 
                        type=float, 
                        default=[0.02, 0.98])
    parser.add_argument("--seed",
                        type=int,
                        help="seed",
                        default=0)

    # ABLATION options
    parser.add_argument("--weights_init",
                        type=str,
                        help="pretrained or scratch",
                        default="pretrained",
                        choices=["pretrained", "scratch"])
    parser.add_argument("--disable_merging_new_surfels",
                        help="if set, after initialization, new surfels will not be added",
                        action="store_true")


    # CONFIDENCE options
    parser.add_argument("--use_ssim_conf",
                        help="if set, warp right image to the left view and use the ssim between the real and warp image as confidence",
                        action="store_true")
    parser.add_argument("--use_seman_conf",
                        help="if set, warp right semantic map to the left view and compare the real and warp semantic map as confidence",
                        action="store_true")

    # EVALUATION options
    parser.add_argument("--post_process",
                        help="if set will perform the flipping post processing "
                            "from the original monodepth paper",
                        action="store_true")


    """
    Cost functions for LM optim.
    """
    parser.add_argument('--m_point_plane', action='store_true', dest='m_point_plane')
    parser.add_argument('--m_point_point', action='store_true', dest='m_point_point')
    parser.add_argument('--m_pp_lambda', type=float, dest='m_pp_lambda', default=1.)
    parser.set_defaults(m_point_plane=False)
    parser.set_defaults(m_point_point=False)
    parser.add_argument('--mesh_edge', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--use_edge_ssim_hints', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--edge_ssim_hints_window', 
                        type=int,
                        default=6)
    parser.add_argument('--mesh_edge_weight', 
                        type=float, 
                        dest='mesh_edge_weight', 
                        default=10)
    parser.add_argument('--mesh_face', 
                        action='store_true', 
                        dest='mesh_face',
                        default=False)
    parser.add_argument('--mesh_face_weight', 
                        type=float, 
                        dest='mesh_face_weight', 
                        default=10)
    parser.add_argument('--mesh_arap', 
                        action='store_true', 
                        dest='mesh_arap', 
                        default=False)
    parser.add_argument('--mesh_arap_weight', 
                        type=float, 
                        default=10)
    parser.add_argument('--mesh_rot', 
                        action='store_true', 
                        dest='mesh_rot', 
                        default=False)
    parser.add_argument('--mesh_rot_weight', 
                        type=float, 
                        default=10.)
    parser.add_argument("--bn_morph",
                        help="if set, use border morphing loss",
                        action="store_true")
    parser.add_argument("--bn_morph_weight",
                        type=float, 
                        default=1e3)
    parser.add_argument("--sf_bn_morph",
                        help="if set, use border morphing loss",
                        action="store_true")
    parser.add_argument("--sf_bn_morph_weight",
                        type=float, 
                        default=10)
    parser.add_argument('--sf_point_plane', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--sf_hard_seman_point_plane', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--sf_soft_seman_point_plane', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--icp_start_iter', 
                        type=int, 
                        default=0)
    parser.add_argument('--use_color_hints', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--sf_point_plane_weight', 
                        type=float, 
                        dest='sf_point_plane_weight', 
                        default=1.)
    parser.add_argument('--sf_corr', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--sf_hard_seman_corr', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--sf_soft_seman_corr', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--sf_corr_use_keyframes', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--sf_corr_match_renderimg', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--sf_corr_weight', 
                        type=float, 
                        dest='sf_corr_weight', 
                        default=1)
    parser.add_argument('--sf_corr_huber_th', 
                        type=float, 
                        default=-1) # 1e-2
    parser.add_argument('--sf_corr_loss_type', 
                        default='point-plane')
    parser.add_argument('--merge_sf_pointplane_corr', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--render_loss', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--render_loss_weight', 
                        type=float, 
                        default=1)
    parser.add_argument('--depth_smooth_loss', 
                        action='store_true', 
                        default=False)
    parser.add_argument('--depth_smooth_loss_weight', 
                        type=float, 
                        default=1)

    # End2end Options.
    parser.add_argument('--phase', 
                        dest='phase', 
                        default='test', 
                        choices=['train', 'test'])

    ## Loss functions for end-to-end.
    parser.add_argument('--e2e_sf_point_plane', action='store_true', dest='e2e_sf_point_plane')
    parser.add_argument('--e2e_sf_pp_lambda', type=float, dest='e2e_sf_pp_lambda', default=1.)
    parser.set_defaults(e2e_sf_point_plane=False)
    parser.add_argument('--e2e_photo', action='store_true', dest='e2e_photo')
    # If e2e_dy_photo = True, render image with colors extracted from the previous frame.
    parser.add_argument('--e2e_dy_photo', action='store_true', dest='e2e_dy_photo')
    parser.add_argument('--e2e_photo_lambda', type=float, dest='e2e_photo_lambda', default=1.)
    parser.set_defaults(e2e_photo=False)
    parser.set_defaults(e2e_dy_photo=False)
    parser.add_argument('--e2e_feat', action='store_true', dest='e2e_feat')
    parser.add_argument('--e2e_dy_feat', action='store_true', dest='e2e_dy_feat')
    parser.add_argument('--e2e_feat_lambda', type=float, dest='e2e_feat_lambda', default=1.)
    parser.set_defaults(e2e_feat=False)
    parser.set_defaults(e2e_dy_feat=False)

    # Parameters for training.
    parser.add_argument('--stage', action='store', dest='stage', default='train_graph_enc', help='')
    parser.add_argument('--batch_size', action='store', type=int, dest='batch_size', default=1, help='')
    parser.add_argument('--epoch_num', action='store', type=int, dest='epoch_num', default=2, help='')
    parser.add_argument('--seq_len', action='store', type=int, dest='seq_len', default=2, help='')
    parser.add_argument('--lr', action='store', type=float, dest='lr', default=0.001, help='')
    parser.add_argument('--lr_decay_rate', action='store', type=int, dest='lr_decay_rate', default=5, help='')
    
    
    parser.add_argument('--checkpoint_dir', action='store', dest='checkpoint_dir', default='checkpoints', help='Path to checkpoints.')
    parser.add_argument('--save_checkpoint_freq', action='store', type=int, dest='save_checkpoint_freq', default=100, help='')
    parser.add_argument('--save_tr_sample_freq', action='store', type=int, dest='save_tr_sample_freq', default=20, 
    help='Frequency to save sample results during training.')

    # # Parameters for RAFT (optical flow).
    # parser.add_argument('--optical_flow_method', dest='optical_flow_method')
    # parser.add_argument('--small', action='store_true', help='use small model')
    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    # parser.add_argument('--optical_flow_pretrain_dir', dest='optical_flow_pretrain_dir', 
    #     default='./RAFT/models/raft-small.pth')
    # parser.add_argument('--no_optical_flow_pretrain', action='store_false', dest='optical_flow_pretrain')
    # parser.set_defaults(optical_flow_pretrain=True)

    # Parameters for PSMNet.
    parser.add_argument('--depth_est_method', dest='depth_est_method')
    parser.add_argument('--depth_est_pretrain_dir', dest='depth_est_pretrain_dir', 
        default='./psm/pretrained_model_KITTI2015.tar')
    parser.add_argument('--no_depth_est_pretrain', action='store_false', dest='depth_est_pretrain')
    parser.set_defaults(depth_est_pretrain=True)
    
    # Parameters for testing.
    parser.add_argument('--tracking_gt_file', dest='tracking_gt_file')
    # parser.add_argument('--gt_idx_file', dest='gt_idx_file')

    # LOGGING options
    parser.add_argument("--log_frequency",
                        type=int,
                        help="number of batches between each tensorboard log",
                        default=30)
    parser.add_argument('--nologfile', 
                        action='store_true', 
                        dest='nologfile', 
                        default=False)

    # Choose modules for graph encoding.
    parser.add_argument('--graph_enc_method', dest='graph_enc_method', default='grid')
    args = parser.parse_args()

    """
    Fixed random seed.
    """
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    models = init_nets(args) # Init models /,& networks.

    # prepare_folders(args)

    ########################################################

    # Track and update the 3D scene (surfels) frame-by-frame.

    ########################################################
    if args.method in ['super', 'seman-super'] or args.phase == 'test':

        # Init data loader.
        testloader = init_dataset(args, models=models)

        # init_surfels = True # True: Need to init the surfels with a depth map.
        iter_time_list = [] # List of time spent in each iteration.

        prev_color = None
        for inputs in testloader:
            if prev_color is not None:
                inputs[("prev_color", 0, 0)] = prev_color

            models["super"](models, inputs)
            prev_color = inputs[("color", 0, 0)]

        mssim_mean = np.mean(models["super"].sf.mssim)
        mssim_std = np.std(models["super"].sf.mssim)
        models["super"].sf.logger.info(f"ssim mean: {mssim_mean}, ssim std: {mssim_std}")
        if args.load_seman_gt:
            if len(models['super'].sf.result_dice) > 0:
                models["super"].sf.logger.info(f"mdice of {len(models['super'].sf.result_dice)} images: {np.mean(models['super'].sf.result_dice)}")
            if len(models['super'].sf.result_jaccard) > 0:
                models["super"].sf.logger.info(f"mjaccard (miou) of {len(models['super'].sf.result_jaccard)} images: {np.mean(models['super'].sf.result_jaccard)}")

        # write_args(os.path.join(args.sample_dir, f"model{args.mod_id}_exp{args.exp_id}", "config.txt"), args)
        
        if args.tracking_gt_file is not None:
            with open(os.path.join(args.sample_dir, f"model{args.mod_id}_exp{args.exp_id}", f"tracking_rst.npy"), 'wb') as f:
                np.save(f, models["super"].sf.track_rsts)

                
    # # End-to-end training.
    # elif model_args['is_training']:

    #     def draw_curve(losses):
    #         """
    #         Plot and save loss curve.
    #         """
    #         fig = plt.figure()
    #         ax = fig.add_subplot(1, 1, 1)
            
    #         train_loss = losses['train']
    #         train_loss = torch_to_numpy(torch.stack(train_loss, dim=0))
    #         ax.plot(train_loss, color='tab:blue') #torch.range(len(losses['train'])), 
    #         # ax.set_ylim([0,300])
    #         fig.savefig('loss_curve.jpg')

    #     stage = args.stage
    #     save_checkpoint_freq = args.save_checkpoint_freq
    #     save_tr_sample_freq = args.save_tr_sample_freq

    #     batch_size = args.batch_size
    #     epoch_num = args.epoch_num
    #     lr = args.lr
    #     lr_decay_rate = args.lr_decay_rate
    #     seq_len = args.seq_len

    #     # Init data loader.
    #     # TODO batch size > 1.; Augmentation.
    #     trainset = SuPerDataset(model_args, nets=nets, transform=None, img_transform=img_transform)

    #     gamma = 0.9
    #     losses = {'train':[], 'val':[]}
    #     if stage == 'train_graph_enc':
    #         l1_loss = nn.L1Loss()

    #         optimizer = optim.Adam(get_deform_graph.parameters(), lr=lr)
    #         lr_scheduler = ExponentialLR(optimizer, gamma=0.9)

    #         # Train iteration.
    #         trainset.train_shuffle(batch_size, 1)
    #         iter_num = len(trainset)
    #         iter_id = 0
    #         for epoch in range(epoch_num):
    #             trainloader = DataLoader(trainset, batch_size=batch_size,
    #                                             shuffle=True, num_workers=2)

    #             for new_data in trainloader:
    #                 new_data = new_data.to(dev)

    #                 # zero the parameter gradients
    #                 optimizer.zero_grad()

    #                 # forward + backward + optimize
    #                 # new_ED_nodes, _ = get_deform_graph(new_data)
    #                 # loss = get_deform_graph.loss(epoch, new_data)
    #                 new_ED_nodes = get_deform_graph(new_data)
    #                 loss = get_deform_graph.loss(new_data, new_ED_nodes)
    #                 loss.backward()
    #                 optimizer.step()

    #                 losses['train'].append(loss)
    #                 draw_curve(losses)

    #                 # # Save checkpoints.
    #                 # if (iter_id+1)%save_checkpoint_freq == 0:
    #                 #     save_checkpoint(checkpoint_dir, model_name, str(iter_id), get_deform_graph)
                    
    #                 # Save sample results during training.
    #                 if (iter_id+1)%save_tr_sample_freq == 0:
    #                     image = new_data.rgb.detach()
    #                     cv2.imwrite(
    #                         os.path.join(F_render_img, "{:02d}_{:02d}.png".format(epoch, iter_id)), 
    #                         proj_mesh(torch_to_numpy(new_data.rgb)[...,::-1], new_ED_nodes)
    #                         )

    #                 lr = optimizer.param_groups[0]['lr']
    #                 print("[Epoch {:d}/{:d}, iter {:d}/{:d}, lr {:.05f}] loss: {:.03f}"\
    #                 .format(epoch, epoch_num, np.mod(iter_id, iter_num), iter_num, lr, loss))
                    
    #                 iter_id += 1
    #                 seq_id = 0

    #             if (epoch+1)%lr_decay_rate == 0:
    #                 lr_scheduler.step()

    #     elif stage == 'full':
    #         # Init optimizer.
    #         optimizer = optim.Adam(nets["optical_flow"].parameters(), lr=lr)
    #         lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
            
    #         # For calculate loss.
    #         l1_loss = nn.L1Loss()

    #         # Train iteration.
    #         for epoch in range(epoch_num):
    #             trainset.train_shuffle(batch_size, seq_len)
    #             trainloader = iter(DataLoader(trainset, batch_size=batch_size,
    #                                             shuffle=False, num_workers=2))

    #             iter_num = len(trainset) // seq_len
    #             iter_id = 0
    #             for i in range(iter_num):
    #                 # zero the parameter gradients
    #                 optimizer.zero_grad()
    #                 # Reset loss.
    #                 graph_enc_loss = 0.0
    #                 super_loss = 0.0
    #                 render_img = []

    #                 for seq_id in range(seq_len):
    #                     with torch.no_grad():
    #                         new_data = next(trainloader).to(dev) 
    #                         new_data.x = nets["optical_flow"].fnet((2 * new_data.rgb - 1.0).contiguous())

    #                     new_ED_nodes = get_deform_graph(new_data)
    #                     # # Graph reconstruction loss (?)
    #                     # graph_enc_loss += get_deform_graph.loss(new_data, new_ED_nodes)

    #                     if seq_id == 0:
    #                         # Init surfels.
    #                         sfModel = mod.init_surfels(new_data, new_ED_nodes)

    #                     else:
    #                         w = gamma**(seq_len-1-seq_id)
    #                         sfModel = mod.fusion(sfModel, new_data, new_ED_nodes, nets=nets)

    #                         super_loss += w * mod.get_loss(sfModel, new_data)
    #                         sfModel.rgb = new_data.rgb # TODO
    #                         if end2end_args['e2e_dy_feat'][0]:
    #                             sfModel.x = new_data.x # TODO

    #                         # render_img.append(
    #                         #     proj_mesh(
    #                         #         torch_to_numpy(255 * de_normalize(sfModel.renderImg).permute(1,2,0))[...,::-1], 
    #                         #         sfModel.ED_nodes)
    #                         # )
    #                         with torch.no_grad():
    #                             render_img_temp = de_normalize(sfModel.renderImg)
    #                             render_img_temp[sfModel.projGraph > 0] = 1.
    #                             render_img.append(transforms.Resize((int(HEIGHT/3), int(WIDTH/3)))(render_img_temp))

    #                 losses['train'].append(super_loss.detach())
    #                 draw_curve(losses)

    #                 # TODO: Optimization
    #                 (graph_enc_loss + super_loss).backward()
    #                 optimizer.step()

    #                 # Save checkpoints.
    #                 if (iter_id+1)%save_checkpoint_freq == 0:
    #                     save_checkpoint(checkpoint_dir, "raft", str(iter_id), nets["optical_flow"])

    #                 print("[Epoch {:d}/{:d}, iter {:d}/{:d}] graph_enc_loss: {:.02f}, super_loss: {:.03f}"\
    #                 .format(epoch, epoch_num, iter_id, iter_num, graph_enc_loss, super_loss))
    #                 iter_id += 1
                    
    #                 # render_img = np.concatenate(render_img, axis=1)
    #                 # cv2.imwrite(
    #                 #     os.path.join(F_render_img, "{:02d}.png".format(iter_id)), render_img
    #                 #     )
    #                 save_image(make_grid(torch.cat(render_img, dim=0)),
    #                 os.path.join(F_render_img, "{:02d}.png".format(iter_id))
    #                 )

    #     RAFT_PATH = os.path.join(checkpoint_dir, f"raft_exp{mod_id}_end.pt")
    #     torch.save(nets["optical_flow"].state_dict(), RAFT_PATH)
    #     write_args(config_PATH, args)
    #     print('Finished Training')


if __name__ == '__main__':
    main()