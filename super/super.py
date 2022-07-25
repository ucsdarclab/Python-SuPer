import torch.nn as nn

from super.nodes import *
from super.LM import *

from dlsuper.deform_mesh import *

from utils.config import *
from utils.utils import *


class SuPer(nn.Module):

    def __init__(self, model_args):
        super(SuPer, self).__init__()
        self.model_args = model_args
        self.sf = None

    def forward(self, model, data):
        sfdata = depth_preprocessing(self.model_args['CamParams'], data)
        sfdata.ED_nodes = model["mesh_encoder"](sfdata)

        if self.sf is None: # Init surfels.
            self.init_surfels(model, data, sfdata)

            print("Init surfels and ED nodes.")
        else:
            self.fusion(model, data, sfdata)
    
    def init_surfels(self, model, data, sfdata):
        self.sf = Surfels(self.model_args, sfdata)
        self.sf.prepareStableIndexNSwapAllModel(model, data, sfdata)

    def fusion(self, model, data, sfdata):

        # ICP.
        deform_param = graph_fit(self.model_args, self.sf, sfdata).detach()
        # deform_param = self.lm.LM(sfModel, new_data)
        self.sf.update(deform_param, sfdata['time'])

        # Fuse the input data into our reference model.
        self.sf.fuseInputData(sfdata)
        
        self.sf.prepareStableIndexNSwapAllModel(model, data, sfdata)