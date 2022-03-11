import numpy as np
import random
import argparse
import copy

import torch.nn as nn
import torch.optim as optim

from reprint import output

import models
from utils.utils import *

from utils.config import *
from utils.inputStream import *
from utils.render import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', action='store', dest='data_dir', required=True, help='Provide input depth directory')
    parser.add_argument('--method', action='store', dest='method', default='super', help='') # super: original version of super
    parser.add_argument('--phase', action='store', dest='phase', default='test', help='train, test')
    parser.add_argument('--img_format', action='store', dest='img_format', help='Provide image (rgb/depth/seg_mask) format')
    parser.add_argument('--evaluate_super', action='store', type=float, dest='evaluate_super', default=0.0, help='')
    
    # Cost Functions
    parser.add_argument('--data_cost', action='store', type=float, dest='data_cost', default=0.0, help='')
    parser.add_argument('--depth_cost', action='store', type=float, dest='depth_cost', default=0.0, help='')
    parser.add_argument('--coord_cost', action='store', type=float, dest='coord_cost', default=0.0, help='')
    parser.add_argument('--corr_cost', action='store', type=float, dest='corr_cost', default=0.0, help='')
    parser.add_argument('--arap_cost', action='store', type=float, dest='arap_cost', default=0.0, help='')
    parser.add_argument('--rot_cost', action='store', type=float, dest='rot_cost', default=0.0, help='')

    # Use square root of the lambda.
    parser.add_argument('--data_lambda', action='store', type=float, dest='data_lambda', default=1.0, help='')
    parser.add_argument('--depth_lambda', action='store', type=float, dest='depth_lambda', default=1.0, help='')
    parser.add_argument('--deep_depth_lambda', action='store', type=float, dest='deep_depth_lambda', default=1.0, help='')
    parser.add_argument('--coord_lambda', action='store', type=float, dest='coord_lambda', default=1.0, help='')
    parser.add_argument('--corr_lambda', action='store', type=float, dest='corr_lambda', default=np.sqrt(10.), help='')
    parser.add_argument('--arap_lambda', action='store', type=float, dest='arap_lambda', default=np.sqrt(10.), help='')
    parser.add_argument('--rot_lambda', action='store', type=float, dest='rot_lambda', default=10.0, help='')

    args = parser.parse_args()

    data_dir = args.data_dir
    img_format = args.img_format
    if not img_format:
        img_format = "png"
    evaluate_super = bool(args.evaluate_super) # If evaluate the tracking performance on the 20 labeled points in the SuPer dataset.

    data_cost = bool(args.data_cost)
    depth_cost = bool(args.depth_cost)
    corr_cost = bool(args.corr_cost)
    arap_cost = bool(args.arap_cost)
    rot_cost = bool(args.rot_cost)
    if args.method == 'super':
        deep_depth_cost = False
        coord_cost = False
    else:
        deep_depth_cost = bool(args.deep_depth_cost)
        coord_cost = bool(args.coord_cost)

    if args.phase == "test":
        out_filename = name_file(args.method, data_cost, depth_cost, \
            arap_cost, rot_cost, corr_cost, coord_cost, \
            args.data_lambda, args.depth_lambda, args.arap_lambda, \
            args.rot_lambda, args.corr_lambda, args.coord_lambda)

    # Prepare empty folder for saving rendering/tracking/etc. results
    reset_folder(output_folder)
    if len(folders) > 0:
        for folder in folders:
            os.makedirs(folder)
    if not os.path.exists(evaluate_folder):
        os.makedirs(evaluate_folder)
    # if evaluate_super and save_20pts_tracking_result:
    #     os.makedirs(tracking_rst_folder)
    # if corr_cost and save_img_feature_matching:
    #     os.makedirs(match_folder)
    # if save_opt_rst:
    #     os.makedirs(error_folder)
    # if qual_color_eval:
    #     os.makedirs(qual_color_folder)
    # os.makedirs(render_folder)
    # os.makedirs(os.path.join(output_folder, "debug")) ##### TODO: For debug, delete

    if evaluate_super:

        # Coordinates of the 20 points.
        with open(os.path.join(data_dir,'labelPts.npy'), 'rb') as f:
            labelPts = np.load(f,allow_pickle=True).tolist()

        # IDs of frames with labels.
        with open(os.path.join(data_dir,'eva_id.npy'), 'rb') as f:
            eva_ids = np.load(f,allow_pickle=True).tolist()
        eva_ids = np.array(eva_ids)

    ########################################################

    # Track and update the 3D scene (surfels) frame-by-frame.

    ########################################################
    if args.method == 'super' or\
        (args.method == 'dlsuper' and args.phase == 'test'):

        # TODO
        if visualize:
            win_subject = 'Project Surfels'

        time = torch.tensor(0.0, device=dev) # Timestamp; TODO: If reading imgs from rosbag, use the true timestamps.
        init_surfels = True # True: Need to init the surfels with a depth map.
        iter_time_list = [] # List of time spent in each iteration.

        # Find the max frame ID.
        imglist = os.listdir(data_dir)
        imglist.sort()
        while True:
            imgname = imglist.pop(-1)
            if imgname.endswith(img_format):
                frame_num = int(imgname.split('-')[0])
                break
        print("Total frame number: {} \n".format(frame_num))

        # Init the tracking & reconstruction method.
        mod = models.SuPer(args.method, data_cost, depth_cost, \
            deep_depth_cost, coord_cost, arap_cost, rot_cost, corr_cost, \
            args.data_lambda, args.depth_lambda, args.deep_depth_lambda, \
            args.coord_lambda, args.corr_lambda, args.arap_lambda, \
            args.rot_lambda, evaluate_super)

        if P_time:
            print("Running time (s):")
            P_table_line = "----------------------------------------------"
            print(P_table_line)
            print("        Steps       |  Ave.  |  Min.  |  Max. ")
            print(P_table_line)

            # Init step running time: times-ave-min-max
            depth_prepro_time = runtime_moving_average(init=True)
            lm_time = runtime_moving_average(init=True)
            update_time = runtime_moving_average(init=True)
            fusion_time = runtime_moving_average(init=True)
            del_unstable_time = runtime_moving_average(init=True)

        with output(initial_len=15, interval=0) as output_lines:
            
            for depth_ID in range(frame_num):

                # Read depth map, color image, & instance segmentation mask.
                rgb, depth = read_imgs(data_dir, depth_ID, img_format, use_mask=use_mask)
                if rgb is None or depth is None:
                    time += 1.
                    P_cond = "Faild to read images, move to the next iteration."
                
                elif np.max(depth) <= 0:
                    time += 1.
                    P_cond = "Invalid depth map, move to the next iteration."

                else:
                    if P_time: start = timeit.default_timer()
                    
                    # Preprocessing.
                    points, norms, isED, rad, conf, valid = depthProcessing(rgb, depth, depth_ID=depth_ID)
                    if P_time: depth_prepro_time = runtime_moving_average(inputs=depth_prepro_time, \
                        new_time=timeit.default_timer() - start)
                    
                    rgb_flatten = torch.as_tensor(rgb, dtype=dtype_, device=dev).view(-1,3)
                    
                    if init_surfels:

                        allModel = mod.init_surfels(points, norms, rgb, rgb_flatten, \
                            rad, conf, isED, valid, time, depth_ID)

                        if evaluate_super:
                            allModel.init_track_pts(eva_ids, labelPts)

                        init_surfels = False

                        # TODO
                        if visualize:
                            if open3d_visualize:
                                vis.add_geometry(allModel.pcd)

                        P_cond = "Init surfels and ED nodes"

                    else:

                        if save_opt_rst:
                            filename = os.path.join(error_folder,str(depth_ID)+".png")
                            bf_data_errors = allModel.vis_opt_error(points, norms, valid)
                        
                        allModel, times_ = mod.fusion(allModel, points, norms, valid, \
                            rgb_flatten, rad, conf, isED, rgb, time, depth_ID)

                        if P_time: 
                            lm_time = runtime_moving_average(inputs=lm_time, new_time=times_[0])
                            update_time = runtime_moving_average(inputs=update_time, new_time=times_[1])
                            fusion_time = runtime_moving_average(inputs=fusion_time, new_time=times_[2])
                            del_unstable_time = runtime_moving_average(inputs=del_unstable_time, new_time=times_[3])

                        if evaluate_super:
                            allModel.update_track_pts(depth_ID, labelPts, rgb)

                        # if save_opt_rst:
                        #     af_data_errors = allModel.vis_opt_error(points, norms, valid)
                        #     allModel.save_opt_error_maps(bf_data_errors, af_data_errors, filename)

                        if visualize:

                            if open3d_visualize:
                                allModel.get_o3d_pcd()

                            visualize_surfel(allModel, vis, os.path.join(o3d_display_dir,str(depth_ID)+'.png'))

                        if P_time:
                            iter_time_ = timeit.default_timer() - start
                            P_cond = "Iter time: {0:.4f}s".format(iter_time_)

                time += 1.0

                if P_time:
                    if init_surfels or lm_time[0] == 0:
                        output_lines[0] = "Depth preprocessing |   -    |   -    |   -    "
                        output_lines[1] = " Tracking (optim.)  |   -    |   -    |   -    "
                        output_lines[2] = "  Update 3D Model   |   -    |   -    |   -    "
                        output_lines[3] = "  Depth map fusion  |   -    |   -    |   -    "
                        output_lines[4] = "  Delete unstable   |   -    |   -    |   -    "
                    else:
                        output_lines[0] = "Depth preprocessing | {0:.4f} | {1:.4f} | {2:.4f}".format(depth_prepro_time[1], depth_prepro_time[2], depth_prepro_time[3])
                        output_lines[1] = " Tracking (optim.)  | {0:.4f} | {1:.4f} | {2:.4f}".format(lm_time[1], lm_time[2], lm_time[3])
                        output_lines[2] = "  Update 3D Model   | {0:.4f} | {1:.4f} | {2:.4f}".format(update_time[1], update_time[2], update_time[3])
                        output_lines[3] = "  Depth map fusion  | {0:.4f} | {1:.4f} | {2:.4f}".format(fusion_time[1], fusion_time[2], fusion_time[3])
                        output_lines[4] = "  Delete unstable   | {0:.4f} | {1:.4f} | {2:.4f}".format(del_unstable_time[1], del_unstable_time[2], del_unstable_time[3])
                    output_lines[5] = P_table_line
                    output_lines[6] = "Iter: {}".format(depth_ID)
                    output_lines[7] = P_cond
                    if not init_surfels:
                        output_lines[8] = "Current ED num: {}, current surfel num: {}".format(allModel.ED_num, allModel.surfel_num)

            # TODO
            # print("Finish tracking, close the window.")
            # if visualize and open3d_visualize:
            #         vis.destroy_window()

            os.rename(output_folder, output_folder+"_"+out_filename)
            
            if evaluate_super:
                track_rsts = np.stack(allModel.track_rsts,axis=0)
                with open(os.path.join(evaluate_folder, out_filename + ".npy"), 'wb') as f:
                    np.save(f, track_rsts)

    elif args.method == 'dlsuper' and args.phase == 'train':

        # Init models.
        mod = models.DeepSuPer(args.method, args.phase, data_cost, depth_cost, \
            deep_depth_cost, coord_cost, arap_cost, rot_cost, corr_cost, \
            args.data_lambda, args.depth_lambda, args.deep_depth_lambda, \
            args.coord_lambda, args.corr_lambda, args.arap_lambda, \
            args.rot_lambda, evaluate_super).cuda()
        net = mod.lm.matcher.flow_net

        criterion = nn.L1Loss()
        optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

        # Find the max frame ID.
        imglist = os.listdir(data_dir)
        imglist.sort()
        while True:
            imgname = imglist.pop(-1)
            if imgname.endswith(img_format):
                frame_num = int(imgname.split('-')[0])
                break
        print("frame number: ", frame_num)

        running_loss = []
        idxs = np.arange(frame_num)[5:]
        seed = 23

        epoch_num = 50
        seq_len = 5
        gamma = 0.9
        losses = []
        for epoch in range(epoch_num):  # loop over the dataset multiple times

            # random.seed(seed)
            # random.shuffle(idxs)

            iter_num = frame_num // seq_len

            for i in range(iter_num):

                ids = idxs[i*seq_len:(i+1)*seq_len]
                if len(ids) < seq_len: continue

                # zero the parameter gradients
                optimizer.zero_grad()
                loss = 0.0

                for k, id_ in enumerate(ids):

                    # Read images.
                    if use_mask:
                        rgb, depth, mask = read_imgs(data_dir, id_, img_format, use_mask=use_mask)
                    else:
                        rgb, depth = read_imgs(data_dir, id_, img_format, use_mask=use_mask)

                    # Preprocessing.
                    if use_mask:
                        points, norms, isED, rad, conf, valid = depthProcessing(rgb, depth, maks=mask)
                    else:
                        points, norms, isED, rad, conf, valid = depthProcessing(rgb, depth)
                    rgb_flatten = torch.as_tensor(rgb, dtype=tfdtype_, device=dev).view(-1,3)

                    if k == 0:
                        allModel, init_surfels = mod.init_surfels(points, norms, rgb, \
                            rgb_flatten, rad, conf, isED, valid)
                    else:
                        # forward + backward + optimize
                        # outputs = mod.optim(allModel, points, norms, rgb, rgb_flatten, \
                        #     rad, conf, isED, valid, id_, prev_rgb, i)
                        # rgb = torch.as_tensor(rgb, dtype=dtype_, device=dev)
                        # outputs_valid = outputs > 1.
                        # loss += (gamma**(seq_len-1-k)) * criterion(outputs[outputs_valid], rgb[outputs_valid])
                        outputs, target = mod.optim(allModel, points, norms, rgb, rgb_flatten, \
                            rad, conf, isED, valid, id_, prev_rgb, i)
                        loss += (gamma**(seq_len-1-k)) * torch.mean(outputs)
                    prev_rgb = rgb
                
                loss.backward()
                optimizer.step()

                # print statistics
                losses.append(loss.item())
                running_loss.append(loss.item())
                if i == 0 or i % 10 == 9:    # print every 10 mini-batches
                    print('[%d/%5d] loss: %.3f' %
                        (epoch + 1, i + 1, np.mean(running_loss)))
                    running_loss = []

            if epoch % 1 == 0:
                # end_model_iteration = epoch * iter_num
                # updated_model = os.path.join("./experiments", "models", model_name, \
                #     f"{model_name}_{end_model_iteration}.pt")
                updated_model = get_PWCNet_model(str(epoch*iter_num))
                torch.save(net.state_dict(), updated_model)

                np.save("loss_"+ out_filename + ".npy", np.array(losses))

        # end_model_iteration = epoch_num * iter_num
        # updated_model = os.path.join("./experiments", "models", model_name, \
        #     f"{model_name}_{end_model_iteration}.pt")
        updated_model = get_PWCNet_model(str(epoch_num*iter_num))
        torch.save(net.state_dict(), updated_model)
        print('Finished Training')


if __name__ == '__main__':
    main()