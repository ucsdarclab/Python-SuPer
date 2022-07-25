import os
import shutil
import numpy as np
np.random.seed(0)
import random
random.seed(0)
import argparse
from reprint import output

import torch
torch.manual_seed(0)

from models.submodules import *

from utils.utils import *


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', default='super')
    parser.add_argument('--data_dir', dest='data_dir', required=True)
    parser.add_argument('--height', type=int, dest='height')
    parser.add_argument('--width', type=int, dest='width')

    parser.add_argument('--mod_id', type=int, dest='mod_id')
    parser.add_argument('--exp_id', type=int, dest='exp_id')
    parser.add_argument('--method', dest='method', default='super')
    parser.add_argument('--sample_dir', dest='sample_dir', default='sample')
    # parser.add_argument('--evaluate_tracking', action='store_false', dest='evaluate_tracking')
    # parser.set_defaults(evaluate_tracking=False)

    """
    Cost functions for LM optim.
    """
    # For meshes.
    parser.add_argument('--m_point_plane', action='store_true', dest='m_point_plane')
    parser.add_argument('--m_point_point', action='store_true', dest='m_point_point')
    parser.add_argument('--m_pp_lambda', type=float, dest='m_pp_lambda', default=1.)
    parser.set_defaults(m_point_plane=False)
    parser.set_defaults(m_point_point=False)
    parser.add_argument('--m_edge', action='store_true', dest='m_edge')
    parser.add_argument('--m_edge_lambda', type=float, dest='m_edge_lambda', default=10.)
    parser.set_defaults(m_edge=False)
    parser.add_argument('--m_arap', action='store_true', dest='m_arap')
    parser.add_argument('--m_arap_lambda', type=float, dest='m_arap_lambda', default=10.)
    parser.set_defaults(m_arap=False)
    parser.add_argument('--m_rot', action='store_true', dest='m_rot')
    parser.add_argument('--m_rot_lambda', type=float, dest='m_rot_lambda', default=10.)
    parser.set_defaults(m_rot=False)
    # For surfels.
    parser.add_argument('--sf_point_plane', action='store_true', dest='sf_point_plane')
    parser.add_argument('--sf_pp_lambda', type=float, dest='sf_pp_lambda', default=1.)
    parser.set_defaults(sf_point_plane=False)
    parser.add_argument('--sf_corr', action='store_true', dest='sf_corr')
    parser.add_argument('--sf_corr_lambda', type=float, dest='sf_corr_lambda', default=10.)
    parser.set_defaults(sf_corr=False)

    """
    Parameters for end-to-end.
    """
    parser.add_argument('--phase', action='store', dest='phase', default='test', help='train/test')

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

    # Parameters for RAFT (optical flow).
    parser.add_argument('--optical_flow_method', dest='optical_flow_method')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--optical_flow_pretrain_dir', dest='optical_flow_pretrain_dir', 
        default='./RAFT/models/raft-small.pth')
    parser.add_argument('--no_optical_flow_pretrain', action='store_false', dest='optical_flow_pretrain')
    parser.set_defaults(optical_flow_pretrain=True)

    # Parameters for PSMNet.
    parser.add_argument('--depth_est_method', dest='depth_est_method')
    parser.add_argument('--depth_est_pretrain_dir', dest='depth_est_pretrain_dir', 
        default='./psm/pretrained_model_KITTI2015.tar')
    parser.add_argument('--no_depth_est_pretrain', action='store_false', dest='depth_est_pretrain')
    parser.set_defaults(depth_est_pretrain=True)
    
    # Parameters for testing.
    parser.add_argument('--tracking_gt_file', dest='tracking_gt_file')
    # parser.add_argument('--gt_idx_file', dest='gt_idx_file')

    # Choose modules for graph encoding.
    parser.add_argument('--graph_enc_method', dest='graph_enc_method', default='grid')

    args = parser.parse_args()
    if args.phase == 'test':
        args.sample_dir = os.path.join(args.sample_dir, f"model{args.mod_id}_exp{args.exp_id}")
    model_args = init_params(args)

    model = init_nets(model_args) # Init models /,& networks.

    prepare_folders(args)

    ########################################################

    # Track and update the 3D scene (surfels) frame-by-frame.

    ########################################################
    if args.method in ['super', 'seman-super'] or args.phase == 'test':

        # Init data loader.
        testloader = init_dataset(model_args)

        # init_surfels = True # True: Need to init the surfels with a depth map.
        iter_time_list = [] # List of time spent in each iteration.

        # with output(initial_len=15, interval=0) as output_lines:
        for data in testloader:
            print(data["ID"].item())

            model["super"](model, data)

        write_args(os.path.join(args.sample_dir, "config.txt"), args)
        
        if model_args['evaluate_tracking']:
            with open(os.path.join(args.sample_dir, f"tracking_rst.npy"), 'wb') as f:
                np.save(f, model["super"].sf.track_rsts)

                
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