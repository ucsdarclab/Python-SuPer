import numpy as np
from dataclasses import dataclass

import torch
import torchvision.transforms as T

"""
Devices.
"""
# dev = torch.device('cuda:1')

torch_version = torch.__version__

"""
Data types.
"""
fl64_ = torch.float64
fl32_ = torch.float32
int_ = torch.int
long_ = torch.int64
bool_ = torch.bool

# """
# Data normalizer for networks.
# """
# normalize = T.Compose(
#             [T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406],
#                      std=[0.229, 0.224, 0.225]),]
#             )
# normalize_only = T.Compose(
#             [T.Normalize(mean=[0.485, 0.456, 0.406],
#                      std=[0.229, 0.224, 0.225]),]
#             )

# de_normalize = T.Compose([T.Normalize(mean = [ 0., 0., 0. ],
#                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
#                         T.Normalize(mean = [ -0.485, -0.456, -0.406 ],
#                                 std = [ 1., 1., 1. ]),
#                         ])

"""
Dataset parameters
"""
@dataclass
class OldSuPerParams:
    HEIGHT: int = 480
    WIDTH: int = 640
    fx: float = 883.0
    fy: float = 883.0
    cx: float = 445.06
    cy: float = 190.24
    depth_scale: float = 1/100. # depth unit: cm -> m
    DIVTERM: float = 1./(2.*0.6*0.6)

@dataclass
class SuPerParams:
    HEIGHT: int = 480
    WIDTH: int = 640
    fx: float = 768.98551924
    fy: float = 768.98551924
    cx: float = 292.8861567
    cy: float = 291.61479526
    depth_scale: float = 1/100. # depth unit: cm -> m
    DIVTERM: float = 1./(2.*0.6*0.6)