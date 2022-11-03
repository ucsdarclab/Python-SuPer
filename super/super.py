import torch.nn as nn

from super.nodes import *
from super.LM import *

# from dlsuper.deform_mesh import GraphFit
from super.deform_mesh import GraphFit

from utils.config import *
from utils.utils import *


class SuPer(nn.Module):

    def __init__(self, opt):
        super(SuPer, self).__init__()
        self.opt = opt
        self.sf = None

        self.graph_fit = GraphFit(self.opt)

    def forward(self, models, inputs):
        for key, ipt in inputs.items():
            if key in ['height', 'width', 'divterm']:
                inputs[key] = ipt.item()
            elif not key == 'filename':
                inputs[key] = ipt.cuda()

        sfdata, inputs = depth_preprocessing(self.opt, models, inputs)
        sfdata.ED_nodes = models["mesh_encoder"](inputs, sfdata)

        if self.sf is None: # Init surfels.
            self.init_surfels(models, inputs, sfdata)
            print("Init surfels and ED nodes.")
        else:
            self.fusion(models, inputs, sfdata)
    
    def init_surfels(self, models, inputs, sfdata):
        self.sf = Surfels(self.opt, models, inputs, sfdata)
        self.sf.prepareStableIndexNSwapAllModel(inputs, sfdata)

    def fusion(self, models, inputs, sfdata):

        # ICP.
        deform_param, boundary_edge_type, boundary_face_type = self.graph_fit(inputs, self.sf, sfdata, models)
        deform_param = deform_param.detach()
        # deform_param = [param.detach() for param in deform_param]
        if boundary_edge_type is not None:
            boundary_edge_type = boundary_edge_type.detach()
        if boundary_face_type is not None:
            boundary_face_type = boundary_face_type.detach()
        # deform_param = self.lm.LM(sfModel, new_data)
        self.sf.update(deform_param, sfdata['time'], boundary_edge_type=boundary_edge_type, boundary_face_type=boundary_face_type)
        
        # Fuse the input data into our reference model.
        self.sf.fuseInputData(inputs, sfdata)
        
        self.sf.prepareStableIndexNSwapAllModel(inputs, sfdata)