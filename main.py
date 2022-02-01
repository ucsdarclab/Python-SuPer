import numpy as np
import random
import argparse
import copy

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
    parser.add_argument('--img_format', action='store', dest='img_format', help='Provide image (depth/seg mask/etc) format')
    parser.add_argument('--evaluate_super', action='store', type=float, dest='evaluate_super', default=0.0, help='')
    parser.add_argument('--data_cost', action='store', type=float, dest='data_cost', default=1.0, help='')
    parser.add_argument('--depth_cost', action='store', type=float, dest='depth_cost', default=1.0, help='')
    parser.add_argument('--corr_cost', action='store', type=float, dest='corr_cost', default=1.0, help='')
    parser.add_argument('--arap_cost', action='store', type=float, dest='arap_cost', default=1.0, help='')
    parser.add_argument('--rot_cost', action='store', type=float, dest='rot_cost', default=1.0, help='')

    parser.add_argument('--track_filename', action='store', dest='track_filename', default='exp1_data_arap_rot.npy', help='')

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

    # Prepare empty folder for saving rendering/tracking/etc. results
    reset_folder(output_folder)
    if evaluate_super and save_20pts_tracking_result:
        os.makedirs(tracking_rst_folder)
    if corr_cost and save_img_feature_matching:
        os.makedirs(match_folder)
    if save_opt_rst:
        os.makedirs(error_folder)
    if qual_color_eval:
        os.makedirs(qual_color_folder)
    os.makedirs(render_folder)
    os.makedirs(os.path.join(output_folder, "debug")) ##### TODO: For debug, delete

    if evaluate_super:

        with open(os.path.join(data_dir,'labelPts.npy'), 'rb') as f:
            labelPts = np.load(f,allow_pickle=True).tolist()

        with open(os.path.join(data_dir,'eva_id.npy'), 'rb') as f:
            eva_ids = np.load(f,allow_pickle=True).tolist()
        eva_ids = np.array(eva_ids)

    # Tracking frame-by-frame.
    if args.method == 'super' or\
        (args.method == 'dlsuper' and args.phase == 'test'):

        if visualize:
            if open3d_visualize:
                vis = o3d.visualization.Visualizer()
                vis.create_window()
            else:
                vis = 'Project Surfels'

        time = 0.0 # Current timestamp
        init_surfels = True # If true, init the surfels.
        iter_time_list = [] # List of time spent in each iteration

        # Find the max frame ID.
        if evaluate_super:
            frame_num = np.max(eva_ids)+1
        else:
            imglist = os.listdir(data_dir)
            imglist.sort()
            while True:
                imgname = imglist.pop(-1)
                # if imgname[-3:]==img_format:
                if imgname.endswith(img_format):
                    frame_num = int(imgname.split('-')[0])
                    break
        print("frame number: ", frame_num)

        # Init models.
        mod = models.SuPer(args.method, data_cost, depth_cost, arap_cost, rot_cost, corr_cost, evaluate_super)
        # if args.method == 'super':
        #     mod = models.SuPer(args.method, data_cost, depth_cost, arap_cost, rot_cost, corr_cost, evaluate_super)
        # elif args.method == 'dlsuper':
        #     mod = models.DLSuPer(args.method, data_cost, depth_cost, arap_cost, rot_cost, corr_cost, evaluate_super)

        for depth_id in range(frame_num):
            start = timeit.default_timer()

            # Read depth map, color image, & instance segmentation mask.
            print("Data ID: ", depth_id)
            try:
                if use_mask:
                    rgb, depth, mask = read_imgs(data_dir, depth_id, img_format, use_mask=use_mask)
                else:
                    rgb, depth = read_imgs(data_dir, depth_id, img_format, use_mask=use_mask)
                print("Finish reading images.")
            except:
                time += 1.
                print("Faild to read images, move to the next iteration.")
                continue
            
            # Preprocessing.
            if use_mask:
                points, norms, isED, rad, conf, time_stamp, valid = depthProcessing(rgb, depth, time, maks=mask)
            else:
                points, norms, isED, rad, conf, time_stamp, valid = depthProcessing(rgb, depth, time)
            rgb_flatten = torch.as_tensor(rgb, dtype=float, device=cuda0).view(-1,3)
            
            if init_surfels:

                if evaluate_super:
                    allModel, init_surfels = mod.init_surfels(points, norms, rgb, rgb_flatten, \
                        rad, conf, time_stamp, isED, valid, time, depth_id, eva_ids=eva_ids, compare_rsts=labelPts)
                else:
                    allModel, init_surfels = mod.init_surfels(points, norms, rgb, rgb_flatten, \
                        rad, conf, time_stamp, isED, valid, time, depth_id)

                if not init_surfels and visualize:
                    if open3d_visualize:
                        vis.add_geometry(allModel.pcd)

            else:
                if save_opt_rst:
                    filename = os.path.join(error_folder,str(depth_id)+".png")
                    bf_data_errors = allModel.vis_opt_error(points, norms, valid)

                allModel = mod.fusion(allModel, points, norms, rgb, rgb_flatten, \
                    rad, conf, time_stamp, isED, valid, time, depth_id)

                if save_opt_rst:
                    af_data_errors = allModel.vis_opt_error(points, norms, valid)
                    allModel.save_opt_error_maps(bf_data_errors, af_data_errors, filename)

                if visualize:

                    if open3d_visualize:
                        allModel.get_o3d_pcd()

                    visualize_surfel(allModel, vis, os.path.join(o3d_display_dir,str(depth_id)+'.png'))

            time += 1.0

            stop = timeit.default_timer()
            iter_time_ = stop - start
            print('Interation time: {}s'.format(iter_time_))
            if not init_surfels:
                iter_time_list.append(iter_time_)

            # if depth_id == 100:
            #     break

        print("Finish tracking (average iteration time: {}s), close the window.".format(np.mean(iter_time_list)))
        if visualize and open3d_visualize:
                vis.destroy_window()

        if evaluate_super:
            track_rsts = allModel.track_rsts
            track_rst = np.stack(track_rsts,axis=0)
            with open(os.path.join(output_folder, args.track_filename), 'wb') as f:
                np.save(f, track_rst)


if __name__ == '__main__':
    main()