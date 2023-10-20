import os

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large, raft_small

from collections import OrderedDict

from super.super import SuPer
from super.graph_encoder import DirectDeformGraph
from renderer.renderer import Pulsar

from utils.data_loader import SuPerDataset
from utils.utils import seed_worker

import depth.monodepth2 as mono2
from depth.raft_core.raft_stereo import RAFTStereo

import segmentation_models_pytorch as smp

class InitNets():
    def __init__(self, opt):

        self.device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
        if opt.phase == "train":
            self.parameters_to_train = []

        if opt.method in ["super", "semantic-super"]:
            self.super = SuPer(opt).to(self.device)
            self.mesh_encoder = DirectDeformGraph(opt).to(self.device)

            if opt.phase == "train":
                self.parameters_to_train += list(self.super.parameters())

        # Renderer.
        if opt.renderer is not None:
            if opt.renderer == "pulsar":
                self.renderer =  Pulsar(opt).to(self.device)
            # elif opt.renderer == 'opencv':
            #     self.renderer =  Projector(method="opencv").to(self.device)

        # Depth model.
        if opt.depth_model is not None:   
            if opt.num_layers > 0:
                assert not opt.load_depth, "Choose either one: use depth estimation model or load depth."
                self.encoder = mono2.resnet_encoder.ResnetEncoder(
                                    opt.num_layers, 
                                    opt.weights_init == "pretrained"
                                ).to(self.device)
                
                if opt.pretrained_encoder_checkpoint_dir is not None:
                    self.encoder = self.load_checkpoints(self.encoder, 
                                        opt.pretrained_encoder_checkpoint_dir, 
                                        model_key="encoder"
                                    )
        
            assert not opt.load_depth, "Choose either one: use depth estimation model or load depth."
            if opt.depth_model == 'monodepth2_stereo':
                self.depth = mono2.depth_decoder.DepthDecoder(
                                opt, 
                                self.encoder.num_ch_enc, 
                                [0]
                            ).to(self.device)

                if opt.pretrained_depth_checkpoint_dir is not None:
                    self.depth = self.load_checkpoints(self.depth, 
                                    opt.pretrained_depth_checkpoint_dir, 
                                    model_key="depth")

            # elif opt.depth_model == 'psm':
            #     models["depth"] = PSMNet(opt)

            elif opt.depth_model == 'raft_stereo':
                self.depth = torch.nn.DataParallel(RAFTStereo(opt)).to(self.device)

                if opt.pretrained_depth_checkpoint_dir is not None:
                    try:
                        self.depth.load_state_dict(torch.load(opt.pretrained_depth_checkpoint_dir))
                        print("RAFTStereo checkpoint loaded successfully!")
                    except Exception as e:
                        print("Error loading RAFTStereo checkpoint:", e)

            else:
                assert False, "The selected depth estimation model doesn't exist."

        # Segmentation model.
        if hasattr(opt, 'seg_model'):
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

        if opt.optical_flow_model == 'raft_small':
            self.optical_flow = raft_small(pretrained=True, progress=True).to(self.device)
        elif opt.optical_flow_model == 'raft_large':
            self.optical_flow = raft_large(pretrained=True, progress=True).to(self.device)

    def load_checkpoints(self, model, model_path, model_key=None, state_pairs=None, strict=False):
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
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

def init_dataset(opt):
    
    testset = SuPerDataset(opt)
    testloader = DataLoader(testset, 1, shuffle=False)
    
    return testloader