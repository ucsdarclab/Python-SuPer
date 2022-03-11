from models import nodes, LM
import torch
import cv2

from utils.config import *
from utils.renderer import *

class DeepSuPer(torch.nn.Module):

    def __init__(self, method, phase, data_cost, depth_cost, \
        deep_depth_cost, coord_cost, arap_cost, rot_cost, corr_cost, \
        data_lambda, depth_lambda, deep_depth_lambda, coord_lambda, \
        corr_lambda, arap_lambda, rot_lambda, evaluate_super):
        super(DeepSuPer, self).__init__()

        self.method = method
        self.evaluate_super = evaluate_super

        self.lm = LM.LM_Solver(method, data_cost, depth_cost, \
            deep_depth_cost, coord_cost, arap_cost, rot_cost, corr_cost, \
            data_lambda, depth_lambda, deep_depth_lambda, coord_lambda, \
            corr_lambda, arap_lambda, rot_lambda, phase=phase)

        self.phase = phase
        if phase == "train":
            # self.renderer = pulsar.Pulsar().cuda()
            self.renderer = Pulsar2().cuda()

        # self.matcher = pt_matching.Matcher()
    
    def init_surfels(self, points, norms, rgb, rgb_flatten, \
                    rad, conf, isED, valid, \
                    eva_ids=None, compare_rsts=None):
        # if depth_id >= 5:
        allModel = nodes.Surfels(points[valid], norms[valid], rgb_flatten[valid], \
                    rad[valid], conf[valid], None, isED[valid], valid, evaluate_super=self.evaluate_super)
        init_surfels = False
        # print("Current ED num: ", allModel.ED_num, \
        #     "; current surfel num: ", allModel.surfel_num)
        allModel.renderImg = allModel.projSurfel(allModel.colors)

        try:
            # print("Init surfels and ED nodes")
                    
            # Prepare the stable list and delete all useless points
            if self.evaluate_super:
                allModel = nodes.Surfels(points[valid], norms[valid], rgb_flatten[valid], \
                    rad[valid], conf[valid], None, isED[valid], valid, evaluate_super=self.evaluate_super, eva_ids=eva_ids, compare_rsts=compare_rsts)
            else:
                allModel = nodes.Surfels(points[valid], norms[valid], rgb_flatten[valid], \
                    rad[valid], conf[valid], None, isED[valid], valid, evaluate_super=self.evaluate_super)

            init_surfels = False
            # print("Current ED num: ", allModel.ED_num, \
            #     "; current surfel num: ", allModel.surfel_num)

            allModel.renderImg = allModel.projSurfel(allModel.colors)

        except:
            allModel = None
            init_surfels = True
            print("Initialization failed, move to the next iteration.")

        return allModel, init_surfels

    def optim(self, allModel, points, norms, rgb, rgb_flatten, \
                    rad, conf, isED, valid, depth_id, rgb1, iter_id):

        # # TODO: Camera pose estimation
        # if estimate_cam_pose and depth_id>0:
        #     poses = np.eye(4)

        deform_param = self.lm.LM(allModel, points, norms, rgb, valid, rgb, depth_id)
                
        allModel.update(deform_param, self.lm) # commit the motion estimated by optimizor

        allModel.surfel_num = len(allModel.points)

        # print("Current ED num: ", allModel.ED_num, \
        #     "; current surfel num: ", allModel.surfel_num)

        allModel.renderImg = allModel.projSurfel(allModel.colors)

        # # Differentiable rendering: Pulsar.
        # renderImg, _ = self.renderer.forward(allModel)
        
        # # TODO: debug
        # # renderImg_ = renderImg.detach().numpy()
        # # cv2.imwrite(os.path.join(render_folder, "{:03d}.png".format(iter_id)), \
        # #     np.concatenate([rgb1,rgb,renderImg_], axis=1)[:,:,::-1])

        # return renderImg.cuda()

        renderImg, target = self.renderer.forward(allModel, rgb_flatten)
        return renderImg, target

        # return torch.as_tensor(allModel.renderImg, dtype=torch.float32, device=cuda0)