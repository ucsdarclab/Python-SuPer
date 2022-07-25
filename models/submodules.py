from torch.utils.data import DataLoader
import torchvision.transforms as T

from graph.graph_encoder import *

from super.super import *

from renderer.renderer import *

from utils.data_loader import *


def init_params(args):
    model_args = {
        'data': args.data,
        'data_dir': args.data_dir,
        'height': args.height,
        'width': args.width,

        'method': args.method,
        'phase': args.phase,
        'evaluate_tracking': args.tracking_gt_file is not None,
        'tracking_gt_file': args.tracking_gt_file,
        'sample_dir': args.sample_dir,

        # LM-optim cost function options.
        'm-point-plane': [args.m_point_plane, args.m_pp_lambda],
        'm-point-point': [args.m_point_point, args.m_pp_lambda],
        'm-edge': [args.m_edge, args.m_edge_lambda],
        'm-arap': [args.m_arap, args.m_arap_lambda],
        'm-rot': [args.m_rot, args.m_rot_lambda],
        'sf-point-plane': [args.sf_point_plane, args.sf_pp_lambda],
        'sf-corr': [args.sf_corr, args.sf_corr_lambda],

        # Parameters for end-to-end.
        'e2e_sf_point_plane': [args.e2e_sf_point_plane, args.e2e_sf_pp_lambda],
        'e2e_photo': [args.e2e_photo, args.e2e_photo_lambda],
        'e2e_dy_photo': [args.e2e_dy_photo],
        'e2e_feat': [args.e2e_feat, args.e2e_feat_lambda],
        'e2e_dy_feat': [args.e2e_dy_feat],
    }

    if args.data == 'super':
        model_args['CamParams'] = OldSuPerParams

    return model_args

def init_nets(model_args: dict,):
    model = {}
    if model_args['method'] == 'super':
        model["super"] = SuPer(model_args)
        model["mesh_encoder"] = DirectDeformGraph(model_args).to(dev)
        model["renderer"] =  Pulsar().to(dev) # Projector().to(dev)
        
    return model

    # nets = {}
    # if args.graph_enc_method == 'grid':
    #     # Option 1: Directly init deform graph by griding.
    #     get_deform_graph = DirectDeformGraph().to(dev)
    # # elif args.graph_enc_method == 'resnet50':
    # #     # Option 2: Graph UNet
    # #     in_channels = 3
    # #     hidden_channels = 64
    # #     out_channels = 3
    # #     depth=4
    # #     pool_ratios = 0.25
    # #     get_deform_graph = GraphUNet(in_channels, hidden_channels, out_channels, 
    # #     depth, pool_ratios=pool_ratios).to(dev).double()
    # #     # Option 3: CNN
    # #     get_deform_graph = CNNGraph().to(dev)

    # if args.optical_flow_method == 'raft':
    #     optical_flow_net = RAFT(args).to(dev)
    #     if model_args['is_training'] and args.optical_flow_pretrain:
    #         optical_flow_net.load_state_dict(
    #             torch.load(args.optical_flow_pretrain_dir), strict=False)
    #     elif not model_args['is_training']:
    #         optical_flow_net.load_state_dict(
    #             torch.load(f"{args.optical_flow_method}_exp{args.mod_id}_end.pt"), strict=False)
    #     nets["optical_flow"] = optical_flow_net
    
    # if args.depth_est_method == 'psm':
    #     depth_est_net = PSMNetStereo()
    #     depth_est_net.model.to(dev)
    #     if model_args['is_training'] and args.depth_est_pretrain:
    #         depth_est_net.model.load_state_dict(
    #             torch.load(args.depth_est_pretrain_dir)['state_dict'])
    #     elif not model_args['is_training']:
    #         depth_est_net.model.load_state_dict(
    #             torch.load(f"{args.depth_est_method}_exp{args.mod_id}_end.pt"))
    #     nets["depth"] = depth_est_net

def prepare_folders(args: dict,):
    """
    * Prepare folders to save results.
    * Read point tracking ground truth if evaluation is required.
    """
    reset_folder(args.sample_dir)
    if args.phase == 'train':
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        if args.mod_id is None: 
            mod_id = 1
            while True:
                config_PATH = os.path.join(args.checkpoint_dir, f"exp{mod_id}_config.txt")
                if os.path.exists(config_PATH):
                    mod_id += 1
                else:
                    break
        else:
            mod_id = args.mod_id
            config_PATH = os.path.join(args.checkpoint_dir, f"exp{mod_id}_config.txt")

    # else:
    #     if args.tracking_gt_file is not None:
    #         tracking_folder = os.path.join(args.sample_dir, "tracking")
    #         reset_folder(tracking_folder)

            
            # except:
            #     print("ERROR: The ground truth of tracked points is required for evaluation.")
            #     sys.exit(1)

        #     try:
        #         gt_idx_file = os.path.join(model_args['data_dir'], args.gt_idx_file)
        #         eva_ids = np.array(np.load(gt_idx_file, allow_pickle=True)).tolist()
        #     except:
        #         eva_ids = np.arange(labelPts['gt'].shape[0])

        

    # if len(folders) > 0:
    #     for folder in folders:
    #         if not os.path.exists(folder):
    #             os.makedirs(folder)

def init_dataset(model_args: dict,):
    if model_args['phase'] == 'train':
        pass
    else:
        testset = SuPerDataset(model_args, normalize)
        testloader = DataLoader(testset, batch_size=1, shuffle=False)
        return testloader