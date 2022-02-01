import torch

from models import nodes, LM, pwcnet
from utils.config import *

class DLSuPer:

    def __init__(self, method, data_cost, depth_cost, arap_cost, rot_cost, corr_cost, evaluate_super):

        self.method = method
        self.evaluate_super = evaluate_super

        # self.data_cost=data_cost
        # self.arap_cost = arap_cost 
        # self.rot_cost = rot_cost 
        # self.corr_cost = corr_costs
        self.lm = LM.LM_Solver(data_cost, depth_cost, arap_cost, rot_cost, corr_cost)

        # self.matcher = pt_matching.Matcher()
    
    def init_surfels(self, points, norms, rgb, rgb_flatten, \
                    rad, conf, time_stamp, isED, valid, time, depth_id, \
                    eva_ids=None, compare_rsts=None):
        # if depth_id >= 5:
        #     # Init flow net: PWC-Net
        #     self.flow_net = PWCNet().cuda()

        #     # Load the pretrained model of PWC-Net
        #     pretrained_dict = torch.load(saved_model)
        #     # Load only optical flow part
        #     model_dict = self.flow_net.state_dict()
        #     # 1. filter out unnecessary keys
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if "flow_net" in k}
        #     # 2. overwrite entries in the existing state dict
        #     model_dict.update(pretrained_dict) 
        #     # 3. load the new state dict
        #     self.flow_net.load_state_dict(model_dict)

        #     allModel = nodes.Surfels(points[valid], norms[valid], rgb_flatten[valid], \
        #             rad[valid], conf[valid], time_stamp[valid], isED[valid], valid, evaluate_super=self.evaluate_super, eva_ids=eva_ids, compare_rsts=compare_rsts)
        #     allModel.prepareStableIndexNSwapAllModel(time, depth_id)

        try:
            print("Init surfels and ED nodes")
                    
            # Prepare the stable list and delete all useless points
            if self.evaluate_super:
                allModel = nodes.Surfels(points[valid], norms[valid], rgb_flatten[valid], \
                    rad[valid], conf[valid], time_stamp[valid], isED[valid], valid, evaluate_super=self.evaluate_super, eva_ids=eva_ids, compare_rsts=compare_rsts)
                allModel.prepareStableIndexNSwapAllModel(time, depth_id, rgb=rgb)
            else:
                allModel = nodes.Surfels(points[valid], norms[valid], rgb_flatten[valid], \
                    rad[valid], conf[valid], time_stamp[valid], isED[valid], valid, evaluate_super=self.evaluate_super)
                allModel.prepareStableIndexNSwapAllModel(time, depth_id)

            init_surfels = False
            print("Current ED num: ", allModel.ED_num, \
                "; current surfel num: ", allModel.surfel_num)

            self.init_flow_net()

        except:
            allModel = None
            init_surfels = True
            print("Initialization failed, move to the next iteration.")

        return allModel, init_surfels

    def init_flow_net(self):

        # Init flow net: PWC-Net
        self.flow_net = pwcnet.PWCNet().cuda()

        # Load the pretrained model of PWC-Net
        pretrained_dict = torch.load(saved_model)
        # Load only optical flow part
        model_dict = self.flow_net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k.partition('flow_net.')[2]: v for k, v in pretrained_dict.items() if "flow_net" in k}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.flow_net.load_state_dict(model_dict)

        # for name, param in self.flow_net.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.requires_grad)

        if freeze_optical_flow_net:
            for param in self.flow_net.parameters():
                param.requires_grad = False

    def fusion(self, allModel, points, norms, rgb, rgb_flatten, \
                    rad, conf, time_stamp, isED, valid, time, depth_id):

        # # TODO: Camera pose estimation
        # if estimate_cam_pose and depth_id>0:
        #     poses = np.eye(4)

        deform_param = self.lm.LM(allModel, points, norms, rgb, time, valid, rgb, depth_id)
                
        allModel.update(deform_param, self.lm) # commit the motion estimated by optimizor

        # fuse the input data into our reference model
        allModel.fuseInputData(points, norms, rgb_flatten, rad, conf, time_stamp, valid, isED, time, depth_id, rgb)

        if self.evaluate_super:
            allModel.prepareStableIndexNSwapAllModel(time, depth_id, rgb=rgb)
        else:
            allModel.prepareStableIndexNSwapAllModel(time, depth_id)

        allModel.surfel_num = len(allModel.points)

        print("Current ED num: ", allModel.ED_num, \
            "; current surfel num: ", allModel.surfel_num)

        return allModel
