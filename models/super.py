from models import nodes, LM

from utils.config import *
from utils.utils import *

class SuPer:

    def __init__(self, method, data_cost, depth_cost, \
        deep_depth_cost, coord_cost, arap_cost, rot_cost, corr_cost, \
        data_lambda, depth_lambda, deep_depth_lambda, coord_lambda, \
        corr_lambda, arap_lambda, rot_lambda, evaluate_super):

        self.method = method
        self.evaluate_super = evaluate_super

        self.lm = LM.LM_Solver(method, data_cost, depth_cost, \
            deep_depth_cost, coord_cost, arap_cost, rot_cost, corr_cost, \
            data_lambda, depth_lambda, deep_depth_lambda, coord_lambda, \
            corr_lambda, arap_lambda, rot_lambda)
    
    def init_surfels(self, points, norms, rgb, rgb_flatten, \
                    rad, conf, isED, valid, time, depth_ID, \
                    eva_ids=None, compare_rsts=None):

        sfModel = nodes.Surfels(points[valid], norms[valid], rgb_flatten[valid], \
                rad[valid], conf[valid], isED[valid], valid, rgb, time, depth_ID)

        sfModel.prepareStableIndexNSwapAllModel(time, depth_ID, init=True)

        return sfModel

    def fusion(self, sfModel, points, norms, valid, colors, \
                    rad, conf, isED, rgb, time, depth_ID):

        times_ = [timeit.default_timer()]

        new_data = [points, norms, valid, \
            colors, rad, conf, isED, time]
        sfModel.ID = depth_ID
        
        sfModel.new_rgb = rgb
        sfModel.new_valid = valid

        # # TODO: Camera pose estimation
        # if estimate_cam_pose and depth_ID>0:
        #     poses = np.eye(4)

        deform_param = self.lm.LM(sfModel, new_data[0:3])
        times_.append(timeit.default_timer())
                
        sfModel.update(deform_param) # Commit the motion estimated by optimizor.
        times_.append(timeit.default_timer())

        # Fuse the input data into our reference model.
        sfModel.fuseInputData(new_data, depth_ID)
        times_.append(timeit.default_timer())

        sfModel.prepareStableIndexNSwapAllModel(time, depth_ID)
        times_.append(timeit.default_timer())

        times_ = np.array(times_)
        return sfModel, times_[1:] - times_[:-1]
