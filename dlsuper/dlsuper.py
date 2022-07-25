from super import nodes
import torch
import cv2

from dlsuper.deform_mesh import *

from utils.config import *
from utils.utils import *

class DeepSuPer:

    def __init__(self, method, phase, lm_params, e2e_params):
        super(DeepSuPer, self).__init__()

        self.method = method
        self.phase = phase
        self.lm_params = lm_params
        self.e2e_params = e2e_params

    def init_surfels(self, new_data, new_ED_node):
        with torch.no_grad():
            sfModel = nodes.Surfels(new_data, new_ED_node, e2e_params=self.e2e_params)
            sfModel.render_img()

            return sfModel

    def fusion(self, sfModel, new_data, new_ED_nodes, nets=None):
        is_test = self.phase == 'test'
        if is_test: times_ = [timeit.default_timer()]

        deform_param = graph_fit(self.method, self.lm_params, sfModel, new_ED_nodes, new_data, nets,
             Niter=3, detach_output=is_test)
        if is_test: times_.append(timeit.default_timer())
        
        sfModel.update(deform_param)
        if is_test: times_.append(timeit.default_timer())

        if is_test:
            sfModel.viz(new_data, new_ED_nodes)

            times_ = np.array(times_)
            return sfModel, times_[1:] - times_[:-1]
        else:
            sfModel.render_img()
            return sfModel

    # @ staticmethod
    def get_loss(self, sfModel, new_data):
        loss = 0.0

        # Surfel point-plane loss.
        if self.e2e_params['e2e_sf_point_plane'][0]:
            loss += self.e2e_params['e2e_sf_point_plane'][1] * DataLoss.autograd_forward(sfModel, new_data)

        # Photometric loss.
        if self.e2e_params['e2e_photo'][0]:
            if self.e2e_params['e2e_dy_photo'][0]:
                v, u, _, valid_indices = pcd2depth(sfModel.points, round_coords=False, valid_margin=1)
                sample_colors, _, _ = bilinear_sample(
                    [new_data.rgb[0].permute(1,2,0)], v[valid_indices], u[valid_indices])
                sample_colors = sample_colors[0].type(fl32_)

                loss += self.e2e_params['e2e_photo'][1] * \
                    nn.L1Loss()(
                        sfModel.renderer(Data(points=sfModel.points[valid_indices], colors=sample_colors)), 
                        new_data.rgb[0].permute(1,2,0))
                
            else:
                loss += self.e2e_params['e2e_photo'][1] * nn.L1Loss()(sfModel.renderImg, new_data.rgb)

        # Feature loss.
        if self.e2e_params['e2e_feat'][0]:
            v, u, _, valid_indices = pcd2depth(sfModel.points, round_coords=False, valid_margin=8)
            v /= 8.
            u /= 8.
            
            if self.e2e_params['e2e_dy_feat'][0]:
                sample_feat, _, _ = bilinear_sample(
                    [sfModel.x[0].permute(1,2,0), new_data.x[0].permute(1,2,0)],
                    v[valid_indices], u[valid_indices])
                loss += self.e2e_params['e2e_feat'][1] * nn.L1Loss()(sample_feat[0], sample_feat[1])
                
            else:
                sample_feat, _, _ = bilinear_sample(
                    [new_data.x[0].permute(1,2,0)], v[valid_indices], u[valid_indices])
                loss += self.e2e_params['e2e_feat'][1] * nn.L1Loss()(sfModel.x[valid_indices], sample_feat[0])

        return loss