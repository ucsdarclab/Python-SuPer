from models import nodes, LM

from utils.config import *
from utils.utils import *

class SuPer:

    def __init__(self, method, \
        data_cost, corr_cost, depth_cost, arap_cost, rot_cost, \
        data_lambda, corr_lambda, depth_lambda, arap_lambda, rot_lambda, \
        evaluate_super):

        self.method = method
        self.evaluate_super = evaluate_super

        self.lm = LM.LM_Solver(method, \
            data_cost, corr_cost, depth_cost, arap_cost, rot_cost, \
            data_lambda, corr_lambda, depth_lambda, arap_lambda, rot_lambda)
    
    def init_surfels(self, new_data):
        sfModel = nodes.Surfels(new_data)
        sfModel.prepareStableIndexNSwapAllModel(new_data.time, new_data.ID, init=True)

        return sfModel

    def fusion(self, sfModel, new_data):
        times_ = [timeit.default_timer()]

        # # TODO: Camera pose estimation
        # if estimate_cam_pose:
        #     ** Estimate the camera pose here. **

        deform_param = self.lm.LM(sfModel, new_data)
        times_.append(timeit.default_timer())
                
        sfModel.update(deform_param) # Commit the motion estimated by optimizor.
        times_.append(timeit.default_timer())

        # Fuse the input data into our reference model.
        sfModel.fuseInputData(new_data)
        times_.append(timeit.default_timer())

        sfModel.prepareStableIndexNSwapAllModel(new_data.time, new_data.ID)
        times_.append(timeit.default_timer())

        times_ = np.array(times_)
        return sfModel, times_[1:] - times_[:-1]
