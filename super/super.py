import torch.nn as nn

from super.nodes import *
from super.LM import *

# from dlsuper.deform_mesh import GraphFit
from super.deform_mesh import GraphFit

from utils.data_loader import pred_depth, pred_seg, depth_preprocessing

from utils.utils import *


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
        for key, ipt in inputs.items():
            if key == 'divterm':
                inputs[key] = ipt.item()
            elif not key == 'filename':
                inputs[key] = ipt.cuda()

        with torch.no_grad():
            # Inference depth (if model is provided).
            if not self.opt.load_depth:
                inputs = pred_depth(self.opt, models, inputs)
            
            # Inference seg (if model is provided).
            # Run segmentation.
            if not self.opt.load_seg:
                inputs = pred_seg(self.opt, models, inputs)

        sfdata, inputs = depth_preprocessing(self.opt, models, inputs) # Get candidate surfels.
        sfdata.ED_nodes = models.mesh_encoder(inputs, sfdata)

        if self.sf is None: # Init surfels.
            self.init_surfels(models, inputs, sfdata)
            print("Init surfels and ED nodes.")
        else:
            self.fusion(models, inputs, sfdata)
    
    def init_surfels(self, models, inputs, sfdata):
        self.sf = Surfels(self.opt, models, inputs, sfdata)
        self.sf.prepareStableIndexNSwapAllModel(inputs, sfdata)

    def fusion(self, models, inputs, sfdata):
        if self.opt.use_derived_gradient:
            deform_param = self.lm.LM(self.sf, inputs, sfdata)
            boundary_edge_type = None
            boundary_face_type = None
        else:
            deform_param, boundary_edge_type, boundary_face_type = self.graph_fit(inputs, self.sf, sfdata, models)
            deform_param = deform_param.detach()
            # deform_param = [param.detach() for param in deform_param]
            if boundary_edge_type is not None:
                boundary_edge_type = boundary_edge_type.detach()
            if boundary_face_type is not None:
                boundary_face_type = boundary_face_type.detach()
        
        self.sf.update(deform_param, sfdata['time'], boundary_edge_type=boundary_edge_type, boundary_face_type=boundary_face_type)
        
        # Fuse the input data into our reference model.
        self.sf.fuseInputData(inputs, sfdata)
        
        self.sf.prepareStableIndexNSwapAllModel(inputs, sfdata)