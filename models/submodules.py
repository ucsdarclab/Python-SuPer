from torch.utils.data import DataLoader
import torchvision.transforms as T

from torchvision.models.optical_flow import raft_large, raft_small
from optical_flow.RAFT.raft import RAFT

from collections import OrderedDict

from graph.graph_encoder import *

from super.super import *

from renderer.renderer import *
from seg.inference import *
import json

from utils.data_loader import *

# from depth.psm.stackhourglass import PSMNet
import depth.monodepth2 as mono2
# import depth.monodepth2.resnet_encoder as mono2_resnet_encoder
# import depth.monodepth2.depth_decoder as mono2_depth_decoder

import segmentation_models_pytorch as smp

# From "the edge of depth"
# TODO move it
class OccluMask(nn.Module):
    def __init__(self, maxDisp = 21):
        super(OccluMask, self).__init__()
        self.maxDisp = maxDisp
        self.pad = self.maxDisp
        self.init_kernel()
        self.boostfac = 400
    def init_kernel(self):
        convweights = torch.zeros(self.maxDisp, 1, 3, self.maxDisp + 2)
        for i in range(0, self.maxDisp):
            convweights[i, 0, :, 0:2] = 1/6
            convweights[i, 0, :, i+2:i+3] = -1/3
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=self.maxDisp, kernel_size=(3,self.maxDisp + 2), stride=1, padding=self.pad, bias=False)
        self.conv.bias = nn.Parameter(torch.arange(self.maxDisp).type(torch.FloatTensor) + 1, requires_grad=False)
        self.conv.weight = nn.Parameter(convweights, requires_grad=False)


        self.detectWidth = 19  # 3 by 7 size kernel
        self.detectHeight = 3
        convWeightsLeft = torch.zeros(1, 1, self.detectHeight, self.detectWidth)
        convWeightsRight = torch.zeros(1, 1, self.detectHeight, self.detectWidth)
        convWeightsLeft[0, 0, :, :int((self.detectWidth + 1) / 2)] = 1
        convWeightsRight[0, 0, :, int((self.detectWidth - 1) / 2):] = 1
        self.convLeft = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                        kernel_size=(self.detectHeight, self.detectWidth), stride=1,
                                        padding=[1, int((self.detectWidth - 1) / 2)], bias=False)
        self.convRight = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                         kernel_size=(self.detectHeight, self.detectWidth), stride=1,
                                         padding=[1, int((self.detectWidth - 1) / 2)], bias=False)
        self.convLeft.weight = nn.Parameter(convWeightsLeft, requires_grad=False)
        self.convRight.weight = nn.Parameter(convWeightsRight, requires_grad=False)
    def forward(self, dispmap, bsline):
        with torch.no_grad():
            maskl = self.computeMask(dispmap, direction='l')
            maskr = self.computeMask(dispmap, direction='r')
            lind = bsline < 0
            rind = bsline > 0
            mask = torch.zeros_like(dispmap)
            mask[lind,:, :, :] = maskl[lind,:, :, :]
            mask[rind, :, :, :] = maskr[rind, :, :, :]
            return mask

    def computeMask(self, dispmap, direction):
        with torch.no_grad():
            width = dispmap.shape[3]
            if direction == 'l':
                output = self.conv(dispmap)
                output = torch.clamp(output, max=0)
                output = torch.min(output, dim=1, keepdim=True)[0]
                output = output[:, :, self.pad - 1:-(self.pad - 1):, -width:]
                output = torch.tanh(-output)
                mask = (output > 0.05).float()
            elif direction == 'r':
                dispmap_opp = torch.flip(dispmap, dims=[3])
                output_opp = self.conv(dispmap_opp)
                output_opp = torch.clamp(output_opp, max=0)
                output_opp = torch.min(output_opp, dim=1, keepdim=True)[0]
                output_opp = output_opp[:, :, self.pad - 1:-(self.pad - 1):, -width:]
                output_opp = torch.tanh(-output_opp)
                mask = (output_opp > 0.05).float()
                mask = torch.flip(mask, dims=[3])
            return mask

def init_nets(opt):
    models = {}
    
    if opt.method in ["super", "seman-super"]:
        models["super"] = SuPer(opt)
        models["mesh_encoder"] = DirectDeformGraph(opt)
    

    # Renderer.
    if opt.renderer is not None:
        if opt.renderer == 'matrix_transform':
            models["renderer"] =  Projector(method="direct")
        elif opt.renderer == 'opencv':
            models["renderer"] =  Projector(method="opencv")
        elif opt.renderer == "pulsar":
            models["renderer"] =  Pulsar()
        elif opt.renderer in ['grid_sample', 'warp']:
            models["renderer"] =  Pulsar()
        else:
            assert False, "This renderer is not available."


    # Depth model.   
    if opt.num_layers > 0:
        assert not opt.load_depth, "Choose either one: use depth estimation model or load depth."
        models["encoder"] = mono2.resnet_encoder.ResnetEncoder(
                                opt.num_layers, 
                                opt.weights_init == "pretrained"
                            )
        # self.parameters_to_train += list(self.models["encoder"].parameters())
        
        if opt.pretrained_encoder_checkpoint_dir is not None:
            models["encoder"] = load_checkpoints(models["encoder"], 
                                    opt.pretrained_encoder_checkpoint_dir, 
                                    model_key="encoder"
                                )

    if opt.depth_model is not None:
        assert not opt.load_depth, "Choose either one: use depth estimation model or load depth."
        if opt.depth_model == 'monodepth2_stereo':
            models["depth"] = mono2.depth_decoder.DepthDecoder(
                opt, models["encoder"].num_ch_enc, opt.scales)  # TODO: Remove semantic output.

        # elif opt.depth_model == 'psm':
        #     models["depth"] = PSMNet(opt)

        if opt.pretrained_depth_checkpoint_dir is not None:
            models["depth"] = load_checkpoints(models["depth"], 
                opt.pretrained_depth_checkpoint_dir, model_key="depth")


    # Segmentation model.
    if opt.seg_model is not None:
        assert not opt.load_seman, "Choose either one: use segmentation model or load segmentation mask."
        
        if opt.seg_num_layers is not None:
            seg_num_layers = opt.seg_num_layers
        else:
            seg_num_layers = opt.num_layers
        
        if not opt.share_depth_seg_model and opt.seg_model in ["adabins", "monodepth2_stereo"]:
            models["seg_encoder"] = mono2.seg_resnet_encoder.ResnetEncoder(
                                        seg_num_layers, 
                                        opt.weights_init == "pretrained"
                                    )

            if opt.pretrained_seg_checkpoint_dir is not None:
                models["seg_encoder"] = load_checkpoints(models["seg_encoder"], 
                                            opt.pretrained_seg_checkpoint_dir, 
                                            model_key="encoder", 
                                            state_pairs=("encoder", "seg_encoder")
                                        )

    if opt.seg_model is not None and not opt.load_seman:
        if opt.seg_model == 'deeplabv3':
            models["seman"] = smp.DeepLabV3Plus(
                encoder_name=f"resnet{seg_num_layers}",
                in_channels=3,
                classes=opt.num_classes
            )

        elif opt.seg_model == 'unet':
            models["seman"] = smp.Unet(
                encoder_name=f"resnet{seg_num_layers}",
                in_channels=3,
                classes=opt.num_classes
            )

        elif opt.seg_model == 'unet++':
            models["seman"] = smp.UnetPlusPlus(
                encoder_name=f"resnet{seg_num_layers}",
                in_channels=3,
                classes=opt.num_classes
            )
        
        elif opt.seg_model == 'manet':
            models["seman"] = smp.MAnet(
                encoder_name=f"resnet{seg_num_layers}",
                in_channels=3,
                classes=opt.num_classes
            )

        # elif opt.seg_model == "monodepth2_stereo":
        #     if not opt.share_depth_seg_model:
        #         models["seman"] = mono2.seg_decoder.DepthDecoder(
        #             opt, models["encoder"].num_ch_enc, opt.scales) # TODO: Remove depth output.

        #         models["seman"] = load_checkpoints(models["seman"], 
        #                 opt.pretrained_seg_checkpoint_dir, 
        #                 model_key="depth",
        #                 state_pairs=("decoder", "sep_decoder"))

        if opt.pretrained_seg_checkpoint_dir is not None:
            if not opt.seg_model in ["adabins", "monodepth2_stereo"]:
                models["seman"] = load_checkpoints(models["seman"], opt.pretrained_seg_checkpoint_dir,
                                                    model_key='state_dict')
            # models["seman"].eval()

    if opt.optical_flow_model == 'raft_small':
        models["optical_flow"] = raft_small(pretrained=True, progress=True)
    elif opt.optical_flow_model == 'raft_large':
        models["optical_flow"] = raft_large(pretrained=True, progress=True)
    elif opt.optical_flow_model == "raft_github":
        models["optical_flow"] = RAFT(opt)

        if opt.pretrained_optical_flow_checkpoint_dir is not None:
            models["optical_flow"] = load_checkpoints(models["optical_flow"], 
                                                    opt.pretrained_optical_flow_checkpoint_dir,
                                                    state_pairs=("module.", ""))
                                                    # strict=True
    # if opt.bn_morph:
    #     models["OccluMask"] = OccluMask().cuda()

    for key in models.keys():
        models[key].cuda()
    
    return models

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

def load_checkpoints(model, model_path, model_key=None, state_pairs=None, strict=False):
    if os.path.exists(model_path):
        state = torch.load(model_path)
        if isinstance(state, dict):
            # assert model_key is not None, "A model_key is needed to extract model parameters."
            if not model_key in state:
                model_key = "state_dict"
            
            if state_pairs is None:
                if model_key is None:
                    model.load_state_dict(state, strict=strict)
                else:
                    model.load_state_dict(state[model_key], strict=strict)
                print(f"Restored model with key {model_key}")
            else:
                old_state, new_state = state_pairs
                new_state_dict = OrderedDict()
                if model_key is None:
                    for key, value in state.items():
                        new_key = key.replace(old_state, new_state)
                        # print(key, '--->', new_key)
                        new_state_dict[new_key] = value
                else:
                    for key, value in state[model_key].items():
                        # new_key = key
                        # for state_pair in state_pairs:
                        #     old_state, new_state = state_pair
                        #     new_key = new_key.replace(old_state, new_state)
                        new_key = key.replace(old_state, new_state)
                        # print(key, '--->', new_key)
                        new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict, strict=strict)
                print(f"Restored model with key {model_key}, change key from {old_state} ---> {new_state}.")
    return model

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

def init_dataset(opts, models=None):
    if opts.phase == 'train':
        assert False, "TODO: Write the code."
    
    else:
        testset = SuPerDataset(opts, T.ToTensor(), models=models)
        testloader = DataLoader(testset, 1, shuffle=False)
        return testloader