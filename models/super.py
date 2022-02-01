from models import nodes, LM

class SuPer:

    def __init__(self, method, data_cost, depth_cost, arap_cost, rot_cost, corr_cost, evaluate_super):

        self.method = method
        self.evaluate_super = evaluate_super

        # self.data_cost=data_cost
        # self.arap_cost = arap_cost 
        # self.rot_cost = rot_cost 
        # self.corr_cost = corr_costs
        self.lm = LM.LM_Solver(method, data_cost, depth_cost, arap_cost, rot_cost, corr_cost)

        # self.matcher = pt_matching.Matcher()
    
    def init_surfels(self, points, norms, rgb, rgb_flatten, \
                    rad, conf, time_stamp, isED, valid, time, depth_id, \
                    eva_ids=None, compare_rsts=None):
        # if depth_id >= 5:
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

        except:
            allModel = None
            init_surfels = True
            print("Initialization failed, move to the next iteration.")

        return allModel, init_surfels

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
