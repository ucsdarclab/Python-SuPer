from torch.utils.data import DataLoader
import torchvision.transforms as T

from collections import OrderedDict

from super.super import SuPer
from graph.graph_encoder import DirectDeformGraph
from renderer.renderer import Pulsar

from seg.inference import *
import json

from utils.data_loader import SuPerDataset

import depth.monodepth2 as mono2

import segmentation_models_pytorch as smp

class InitNets():
    def __init__(self, opt):

        self.device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

        if opt.method in ["super", "semantic-super"]:
            self.super = SuPer(opt).to(self.device)
            self.mesh_encoder = DirectDeformGraph(opt).to(self.device)

        # Renderer.
        if opt.renderer is not None:
            if opt.renderer == "pulsar":
                self.renderer =  Pulsar(opt).to(self.device)
            # elif opt.renderer in ['grid_sample', 'warp']:
            #     self.renderer =  Pulsar(opt).to(self.device)
            # elif opt.renderer == 'matrix_transform':
            #     self.renderer =  Projector(method="direct").to(self.device)
            # elif opt.renderer == 'opencv':
            #     self.renderer =  Projector(method="opencv").to(self.device)

        # Depth model.   
        if opt.num_layers > 0:
            assert not opt.load_depth, "Choose either one: use depth estimation model or load depth."
            self.encoder = mono2.resnet_encoder.ResnetEncoder(
                                opt.num_layers, 
                                opt.weights_init == "pretrained"
                            ).to(self.device)
            # TODO: self.parameters_to_train += list(self.encoder.parameters())
            
            if opt.pretrained_encoder_checkpoint_dir is not None:
                self.encoder = self.load_checkpoints(self.encoder, 
                                    opt.pretrained_encoder_checkpoint_dir, 
                                    model_key="encoder"
                                )
        if opt.depth_model is not None:
            assert not opt.load_depth, "Choose either one: use depth estimation model or load depth."
            if opt.depth_model == 'monodepth2_stereo':
                self.depth = mono2.depth_decoder.DepthDecoder(
                                opt, 
                                self.encoder.num_ch_enc, 
                                [0]
                            ).to(self.device)

            # elif opt.depth_model == 'psm':
            #     models["depth"] = PSMNet(opt)

            else:
                assert False, "The selected depth estimation model doesn't exist."

            if opt.pretrained_depth_checkpoint_dir is not None:
                self.depth = self.load_checkpoints(self.depth, 
                                opt.pretrained_depth_checkpoint_dir, 
                                model_key="depth")

        # Segmentation model.
        # if opt.seg_model is not None:
        #     assert not opt.load_seman, "Choose either one: use segmentation model or load segmentation mask."
            
        #     if opt.seg_num_layers is not None:
        #         seg_num_layers = opt.seg_num_layers
        #     else:
        #         seg_num_layers = opt.num_layers
            
        #     if not opt.share_depth_seg_model and opt.seg_model in ["adabins", "monodepth2_stereo"]:
        #         models["seg_encoder"] = mono2.seg_resnet_encoder.ResnetEncoder(
        #                                     seg_num_layers, 
        #                                     opt.weights_init == "pretrained"
        #                                 )

        #         if opt.pretrained_seg_checkpoint_dir is not None:
        #             models["seg_encoder"] = load_checkpoints(models["seg_encoder"], 
        #                                         opt.pretrained_seg_checkpoint_dir, 
        #                                         model_key="encoder", 
        #                                         state_pairs=("encoder", "seg_encoder")
        #                                     )
        if opt.seg_model is not None:
            assert not opt.load_seg, "Choose either one: use segmentation model or load segmentation mask."

            if opt.seg_num_layers is not None:
                seg_num_layers = opt.seg_num_layers
            else:
                seg_num_layers = opt.num_layers

            if opt.seg_model == 'deeplabv3':
                self.seg = smp.DeepLabV3Plus(
                    encoder_name=f"resnet{seg_num_layers}",
                    in_channels=3,
                    classes=opt.num_classes
                ).to(self.device)

            elif opt.seg_model == 'unet':
                self.seg = smp.Unet(
                    encoder_name=f"resnet{seg_num_layers}",
                    in_channels=3,
                    classes=opt.num_classes
                ).to(self.device)

            elif opt.seg_model == 'unet++':
                self.seg = smp.UnetPlusPlus(
                    encoder_name=f"resnet{seg_num_layers}",
                    in_channels=3,
                    classes=opt.num_classes
                ).to(self.device)
            
            elif opt.seg_model == 'manet':
                self.seg = smp.MAnet(
                    encoder_name=f"resnet{seg_num_layers}",
                    in_channels=3,
                    classes=opt.num_classes
                ).to(self.device)

            if opt.pretrained_seg_checkpoint_dir is not None:
                self.seg = self.load_checkpoints(self.seg, 
                                opt.pretrained_seg_checkpoint_dir,
                                model_key='state_dict'
                            )

        # if opt.optical_flow_model == 'raft_small':
        #     models["optical_flow"] = raft_small(pretrained=True, progress=True)
        # elif opt.optical_flow_model == 'raft_large':
        #     models["optical_flow"] = raft_large(pretrained=True, progress=True)
        # elif opt.optical_flow_model == "raft_github":
        #     models["optical_flow"] = RAFT(opt)

        #     if opt.pretrained_optical_flow_checkpoint_dir is not None:
        #         models["optical_flow"] = load_checkpoints(models["optical_flow"], 
        #                                                 opt.pretrained_optical_flow_checkpoint_dir,
        #                                                 state_pairs=("module.", ""))
        #                                                 # strict=True

    def load_checkpoints(self, model, model_path, model_key=None, state_pairs=None, strict=False):
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

def init_dataset(opts):
    if opts.phase == 'test':
        testset = SuPerDataset(opts)
        testloader = DataLoader(testset, 1, shuffle=False)
        return testloader
    # else:
    #     # TODO: dataloader for train phase.