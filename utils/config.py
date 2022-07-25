import numpy as np
from dataclasses import dataclass

import torch
import torchvision.transforms as T

"""
Devices.
"""
dev = torch.device('cuda:0')
cpu = torch.device('cpu:0')

torch_version = torch.__version__

"""
Data types.
"""
fl64_ = torch.float64
fl32_ = torch.float32
int_ = torch.int
long_ = torch.int64
bool_ = torch.bool

"""
Data normalizer for networks.
"""
normalize = T.Compose(
            [T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),]
            )
normalize_only = T.Compose(
            [T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),]
            )

de_normalize = T.Compose([T.Normalize(mean = [ 0., 0., 0. ],
                                    std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                        T.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                std = [ 1., 1., 1. ]),
                        ])

"""
Dataset parameters
"""
@dataclass
class OldSuPerParams:
    HEIGHT: int = 480
    WIDTH: int = 640
    fx: float = 768.98551924
    fy: float = 768.98551924
    cx: float = 292.8861567
    cy: float = 291.61479526
    depth_scale: float = 1/100. # depth unit: cm -> m
    # But I feel the unit of the input depth maps is mm.
    THRESHOLD_COSINE_ANGLE: float = 0.4 # C++ code: accept angle < 30 degree (0.87), 67 degree (0.4)
    THRESHOLD_DISTANCE: float = 0.2 # C++ code value: 0.2
    SQRT2: float = np.sqrt(2)
    DIVTERM: float = 1./(2.*0.6*0.6)
    CONF_TH: float = 10.0 # TODO: Select better value.
    STABLE_TH: float = 30.0
    n_neighbors: int = 4
    ED_n_neighbors: int = 8

    INP_HEIGHT: int = 1080
    INP_WIDTH: int = 1920
    K1: np.ndarray = np.array(  [[1.6264394029428900e+03, 0., 8.3440153256181497e+02],
                                [0., 1.6210282483131000e+03, 4.2607815615404297e+02],
                                [0., 0., 1. ]] ).reshape(3,3)
    K2: np.ndarray = np.array(  [[ 1.6536267010886600e+03, 0., 1.0996315512747401e+03],
                                [0., 1.6521704703888099e+03, 4.0353935560440999e+02],
                                [0., 0., 1.]] ).reshape(3,3)
    D1: np.ndarray = np.array(  [ -2.6198574218752801e-01, 9.7542413719703500e-02, 0., 0., 1.2212836507229301e+00 ]).reshape(1,5)
    D2: np.ndarray = np.array(  [ -3.3212206141363598e-01, 4.5246067056600803e-01, 0., 0., -2.7578612117510998e-01]).reshape(1,5)
    T: np.ndarray = np.array(   [ -5.8513759749420302e+00, 1.1863526038796400e-01, 8.1409249931785599e-01]).reshape(3,1)
    R: np.ndarray = np.array(   [[9.9853943863456995e-01, -1.3270688328128599e-03, -5.4011372688262498e-02],
                                [1.7410235217250401e-03, 9.9996946787925500e-01, 7.6178833265481997e-03],
                                [5.3999614150975303e-02, -7.7007920107660102e-03, 9.9851126156591397e-01]]).reshape(3,3)
    scaling_factor: int = 2
    depth_scaling: int = 10

    seg_class_num: int = 4
    max_disp: int = 96