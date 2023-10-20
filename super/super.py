import torch.nn as nn
from collections import OrderedDict
from super.nodes import *
from super.LM import *
from super.deform_mesh import GraphFit

from utils.utils import merge_transformation
from utils.data_loader import pred_depth, pred_seg, depth_preprocessing


class SuPer(nn.Module):

    def __init__(self, opt):
        super(SuPer, self).__init__()
        self.opt = opt
        self.sf = None

        if self.opt.use_derived_gradient:
            self.lm = LM_Solver(self.opt)
        else:
            self.graph_fit = GraphFit(self.opt)

    def forward(self, models, inputs):
        ''' Perform 1 forward pass of the SuPer pipeline.
        Input: 
            - models: super, renderer, mesh_encoder, device
            - inputs: a dict from dataloader containing the following keys:
                ['filename', 'ID', 'time', ('color', 0), 'K', 'inv_K', 'divterm', 
                'stereo_T', ('color_aug', 0), ('seg_conf', 0), ('seg', 0), ('disp', 0), ('depth', 0)]
        '''
        for key, ipt in inputs.items():                                     # Convert inputs to cuda
            if torch.is_tensor(ipt):
                if key == 'divterm': inputs[key] = ipt.item()
                elif key != "filename": inputs[key] = ipt.cuda()

        with torch.no_grad():
            if not self.opt.load_depth:                                     # Inference depth (if model is provided)
                inputs = pred_depth(self.opt, models, inputs)
            
            if hasattr(self.opt, 'seg_model'):
                if self.opt.seg_model is not None:                          # Inference seg (if model is provided)
                    assert not self.opt.load_seg
                    inputs = pred_seg(self.opt, models, inputs)

        sfdata, inputs = depth_preprocessing(self.opt, models, inputs)      # Get candidate surfels.

        if self.sf is None:                                                 # Init surfels.
            if self.opt.deform_udpate_method == 'super_edg':                    # Init ED nodes
                sfdata.ED_nodes = models.mesh_encoder(inputs, sfdata)

            self.init_surfels(models, inputs, sfdata)
            print(f"Initialized {self.sf.points.shape[0]} surfels and {self.sf.ED_nodes.points.shape[0]} ED nodes.")
            deform_param = None
        else:
            deform_param = self.fusion(models, inputs, sfdata)

        if self.opt.sf_corr and not self.opt.sf_corr_match_renderimg:
            self.sf.rgb = inputs[("color", 0)]

    def init_surfels(self, models, inputs, sfdata):
        self.sf = Surfels(self.opt, models, inputs, sfdata)
        if self.opt.phase == "test":
            self.sf.prepareStableIndexNSwapAllModel(inputs, sfdata)


    def fusion(self, models, inputs, sfdata):
        if self.opt.use_derived_gradient:
            deform_param = self.lm.LM(self.sf, inputs, sfdata)
        else:     
            deform_param = self.graph_fit(inputs, self.sf, sfdata, models)
            if deform_param is not None: deform_param = deform_param.detach()

        self.sf.update(deform_param)

        if self.opt.phase == "test":
            # Fuse the input data into our reference model.
            self.sf.fuseInputData(inputs, sfdata)
            self.sf.prepareStableIndexNSwapAllModel(inputs, sfdata)

            if self.sf.time % self.opt.save_sample_freq == 0:
                self.sf.evaluate()

        return deform_param