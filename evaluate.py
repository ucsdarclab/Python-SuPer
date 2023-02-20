import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

from utils.utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def evaluate(gt, est, igonored_ids, normalize=False):
    val = (gt[:, 2] == 1) #& (est[:, 2] == 1)
    if len(igonored_ids) > 0:
        val[np.array(igonored_ids) - 1] = False
    dists = np.linalg.norm(gt[:, 0:2] - est[:, 0:2], axis=1)
    dists[~val] = -1

    h = 480
    if normalize: 
        dists /= h

    # # if np.any(est[:, 2][val]==0):
    # if np.max(dists) > 50:
    #     id = np.argmax(dists)
    #     print(id, dists[id], gt[id, 0:2], est[id, 0:2])

    return dists

# def evaluate(gt, est, normalize=False):

#     diff = []
#     diff_valid = []
#     for key in est.keys():
#         diff.append(distance(gt[int(key)],est[key]))
#         diff_valid.append(gt[int(key)][:,0] >= 0)

#     # print(diff_valid)
#     diff = np.stack(diff)
#     diff_valid = np.stack(diff_valid)
    
#     mean_ = []
#     std_ = []
#     for i in range(diff.shape[1]):
#         diff_temp = diff[:,i][diff_valid[:,i]]
#         # Normalize the coordinates by image size.
#         if normalize: diff_temp /= HEIGHT
#         mean_.append(np.mean(diff_temp))
#         std_.append(np.std(diff_temp))

#     return np.array(mean_), np.array(std_)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', 
                        default='superv1')
    parser.add_argument('--gt_dir', 
                        default="/media/bear/77f1cfad-f74f-4e12-9861-557d86da4f681/research_proj/datasets/3d_data/super_dataset/super_exp_520/left_pts.npy")
    parser.add_argument('--num_points',
                        type=int,
                        default=20)
    parser.add_argument('--igonored_ids',
                        type=int,
                        nargs="+",  
                        default=[])  
    parser.add_argument('--edge_ids',
                        type=int,
                        nargs="+",  
                        default=[])  
    parser.add_argument('--start_timestamp',
                        type=int,
                        default=1)
    parser.add_argument('--end_timestamp',
                        type=int,
                        default=519) 
    parser.add_argument('--traj_ids',
                        type=int,
                        nargs="+",  
                        default=[]) 
    parser.add_argument('--traj_start_timestamp',
                        type=int,
                        default=60)
    parser.add_argument('--traj_end_timestamp',
                        type=int,
                        default=120) 
    parser.add_argument('--result_folder', 
                        required=True)
    parser.add_argument('--files',
                        nargs="+",  
                        required=True)   
    parser.add_argument('--files_legends',
                        nargs="+",  
                        required=True)    
    parser.add_argument('--files_to_plot',
                        type=int,
                        nargs="+",  
                        required=True)  
    parser.add_argument('--files_to_plot_time_error',
                        type=int,
                        nargs="+")  
    parser.add_argument('--output_dir',   
                        default="")   
    parser.add_argument('--output_figure_name',   
                        default="evaluate.png")      
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    log_file_path = os.path.join(args.output_dir, f"{args.output_figure_name[:-4]}_log.txt")
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    custom_log_manager.setup(
        to_file=True, 
        log_prefix=f"{args.output_figure_name[:-4]}_log",
        logdir=args.output_dir)
    logger = custom_log_manager.get_logger('test')
    logger.info("Initiate surfels and ED nodes ...: ")
    for arg in vars(args):
        logger.info(f"{arg}: {str(getattr(args, arg))}")
    logger.info("\n")

    gt_file = np.array(np.load(args.gt_dir, allow_pickle=True)).tolist() 
    # keys: 'gt', 'super_cpp', 'SURF'
    results = [np.load(os.path.join(args.result_folder, file), allow_pickle=True).tolist() for file in args.files]
    gt = gt_file['gt']
    colors = ['tab:pink', 'tab:orange', 'tab:blue', 'tab:green', 'tab:red', \
        'tab:gray', 'tab:cyan']
    if np.any(np.array(args.files_to_plot) == 1):
        plt.figure(figsize=(12,3))
    bar_width = 0.25
    offset = 0.0

    if 'super_cpp' in gt_file and False:
        tracking_errors = {'super_cpp': []}
        super_cpp = gt_file['super_cpp']
        for key in super_cpp.keys():
            super_cpp[key] = torch.as_tensor(super_cpp[key])
        results = [super_cpp] + results
        args.files_legends = ['super_cpp'] + args.files_legends
        args.files_to_plot = [1] + args.files_to_plot
    else:
        tracking_errors = {}
        results_array = {}
    for legend in args.files_legends:
        tracking_errors[legend] = []
        results_array[legend] = []

    gt_array = []
    keys = [int(key) for key in gt.keys()]
    keys = np.sort(keys)
    for intkey in keys:
        key = f"{intkey:06d}"
        gt_array.append(gt[key])
        for result_key, result in zip(args.files_legends, results):
            if key in result and intkey >= args.start_timestamp and intkey <= args.end_timestamp:
                # print(result_key, key)
                results_array[result_key].append(result[key].cpu().numpy())
                tracking_errors[result_key].append(evaluate(gt[key], result[key].cpu().numpy(), args.igonored_ids))

            # TODO: Count track length.
    gt_array = np.stack(gt_array, axis=0)
    for key in results_array.keys():
        results_array[key] = np.stack(results_array[key], axis=0)

    val_evaluate = torch.ones(args.num_points, dtype=torch.bool)
    val_evaluate[np.array(args.igonored_ids)-1] = False
    ind = np.arange(torch.count_nonzero(val_evaluate))
    offset = 0
    color_id = 0
    time_error = {}
    for k, key in enumerate(tracking_errors):
        tracking_error = tracking_errors[key]
        tracking_error = np.stack(tracking_error, axis=0)

        val = tracking_error >= 0

        time_error[key] = np.array([np.mean(_time_error_[_val_]) \
                                    for _time_error_, _val_ in zip(tracking_error, val)])

        _mean_ = []
        _std_ = []
        _min_ = []
        _max_ = []
        for i in range(args.num_points):
            if val_evaluate[i]:
                _mean_.append(np.mean(tracking_error[:, i][val[:, i]]))
                _std_.append(np.std(tracking_error[:, i][val[:, i]]))
                _min_.append(np.min(tracking_error[:, i][val[:, i]]))
                _max_.append(np.max(tracking_error[:, i][val[:, i]]))

        if args.files_to_plot[k] == 1:
            print(key, ind, _std_, len(_std_), len(ind))
            plt.bar(ind+offset, _mean_, bar_width, color=colors[color_id], label=key)
            plt.errorbar(ind+offset, _mean_, _std_, \
                linestyle='None', color='k', elinewidth=.8, capsize=3)
            # plt.errorbar(ind+offset, _mean_, torch.stack([torch.tensor(_min_), torch.tensor(_max_)], dim=0), \
            #     linestyle='None', color='k', elinewidth=.8, capsize=3)
            offset += 0.15
            color_id += 1
        
        _edge_mean_ = []
        _edge_std_ = []
        for i in range(args.num_points):
            if val_evaluate[i] and i in args.edge_ids:
                _edge_mean_.append(np.mean(tracking_error[:, i][val[:, i]]))
                _edge_std_.append(np.std(tracking_error[:, i][val[:, i]]))

        for i in range(len(_mean_)):
            if np.isnan(_mean_[i]): print(i+1)
        logger.info(f"Mean error of {key:24}: {np.mean(_mean_):0.1f}, std: {np.mean(_std_):0.1f}.      Mean error of edge points: {np.mean(_edge_mean_):0.1f}, std: {np.mean(_edge_std_):0.1f}")

    if np.any(np.array(args.files_to_plot) == 1):
        plt.legend(loc='upper right')
        val_ids = torch_to_numpy(val_evaluate.nonzero(as_tuple=True)[0]) + 1
        plt.xticks(ind, val_ids.astype(int))
        for k, ticklabel in enumerate(plt.gca().get_xticklabels()):
            if val_ids[k] in args.edge_ids:
                tickcolor = 'r'
            else:
                tickcolor = 'b'
            ticklabel.set_color(tickcolor)
        plt.xlabel('Tracked point ID')
        # plt.ylabel('Error in percentage of image size')
        plt.ylabel('Error (unit: pixel)')
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, args.output_figure_name), bbox_inches='tight')

    if args.files_to_plot_time_error is not None:
        output_data = {}
        plt.figure(figsize=(6.2,2.6))
        comp_colors = [['tab:green', 'lime']]
        x = torch.arange(keys[-1]-1)
        window_size, offset = 5, 2
        # for i in range(int(len(args.files_to_plot_time_error)/2)):
        #     id1, id2 = args.files_to_plot_time_error[int(2*i)], args.files_to_plot_time_error[int(2*i+1)]
            
        #     y1 = time_error[args.files_legends[id1]]
        #     # p1 = np.poly1d(np.polyfit(x, y1, 30))
        #     # plt.plot(x, p1(x), color=comp_colors[i][0], label=args.files_legends[id1])
        #     # plt.plot(x, y1, color=comp_colors[i][0], label=args.files_legends[id1])
        #     numbers_series_y1 = pd.Series(y1).rolling(window_size).mean().tolist()
        #     y1 = numbers_series_y1[window_size - 1:]
        #     plt.plot(x[2:-2], y1, color=comp_colors[i][0], label=args.files_legends[id1])
            
        #     y2 = time_error[args.files_legends[id2]]
        #     # p2 = np.poly1d(np.polyfit(x, y2, 30))
        #     # plt.plot(x, p2(x), color=comp_colors[i][1], label=args.files_legends[id2])
        #     # plt.plot(x, y2, color=comp_colors[i][1], label=args.files_legends[id2])
        #     numbers_series_y2 = pd.Series(y2).rolling(window_size).mean().tolist()
        #     y2 = numbers_series_y2[window_size - 1:]
        #     plt.plot(x[2:-2], y2, color=comp_colors[i][1], label=args.files_legends[id2])
        for k, id in enumerate(args.files_to_plot_time_error):
            y1 = time_error[args.files_legends[id]]
            plt.plot(x, y1, '.', color=colors[k], markersize=2)
            output_data[args.files_legends[id]] = [colors[k], x, y1]
            numbers_series_y1 = pd.Series(y1).rolling(window_size).mean().tolist()
            y1 = numbers_series_y1[window_size - 1:]
            plt.plot(x[2:-2], y1, color=colors[k], linewidth=0.8, label=args.files_legends[id])
            output_data[args.files_legends[id]] += [x[2:-2], y1]
            
        plt.legend(loc='upper left')
        plt.ylabel('Reprojection Error (unit: pixel)')
        plt.xlabel('Video Frame ID')
        plt.savefig(os.path.join(args.output_dir, args.output_figure_name.replace('.','_timeerror.')), bbox_inches='tight')
        plt.rcParams['pdf.fonttype'] = 42
        plt.savefig(os.path.join(args.output_dir, args.output_figure_name.replace('.png','_timeerror.pdf')), bbox_inches='tight', format='pdf')
    
        np.save(os.path.join(args.output_dir, args.output_figure_name.replace('.png','.npy')), output_data)

    if len(args.traj_ids) > 0:
        window_size = 5
        for k, traj_id in enumerate(args.traj_ids):
            plt.figure(figsize=(3,3))
            # plot the ground truth
            x = gt_array[1: -1][args.traj_start_timestamp: args.end_timestamp, traj_id-1, 0]
            y = gt_array[1: -1][args.traj_start_timestamp: args.end_timestamp, traj_id-1, 1]
            valid = gt_array[1: -1][args.traj_start_timestamp: args.end_timestamp, traj_id-1, 2] == 1
            x = x[valid]
            y = y[valid]
            numbers_series_x = pd.Series(x).rolling(window_size).mean().tolist()
            numbers_series_y = pd.Series(y).rolling(window_size).mean().tolist()
            x = numbers_series_x[window_size - 1:]
            y = numbers_series_y[window_size - 1:]
            if k == len(args.traj_ids) - 1:
                plt.plot(x, y, color=colors[0], label="Ground Truth") 
            else:
                plt.plot(x, y, color=colors[0])

            # plot the tracking results
            color_id = 1
            for j, key in enumerate(args.files_legends):
                if args.files_to_plot[j] == 1:
                    x = results_array[key][args.traj_start_timestamp: args.end_timestamp, traj_id-1, 0]
                    y = results_array[key][args.traj_start_timestamp: args.end_timestamp, traj_id-1, 1]
                    valid = results_array[key][args.traj_start_timestamp: args.end_timestamp, traj_id-1, 2] == 1
                    
                    x = x[valid]
                    y = y[valid]
                    numbers_series_x = pd.Series(x).rolling(window_size).mean().tolist()
                    numbers_series_y = pd.Series(y).rolling(window_size).mean().tolist()
                    x = numbers_series_x[window_size - 1:]
                    y = numbers_series_y[window_size - 1:]
                    
                    if k == len(args.traj_ids) - 1:
                        plt.plot(x, y, color=colors[color_id], label=key) 
                    else:
                        plt.plot(x, y, color=colors[color_id])
                    color_id += 1

                    # p1 = np.poly1d(np.polyfit(x[valid], y[valid], 100))
                    # plt.plot(x[valid], p1(x[valid]), color=colors[k], label=key)

            if k == len(args.traj_ids) - 1:
                plt.legend(bbox_to_anchor = (1.00, 1.00))
            plt.axis('off')
            # plt.gca().invert_yaxis()
            plt.savefig(os.path.join(args.output_dir, args.output_figure_name.replace('.',f"_traj_pt{k}.")), bbox_inches='tight')

if __name__ == '__main__':
    main()