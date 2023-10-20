# modified from: https://github.com/MengHao666/Minimal-Hand-pytorch
import time
import numpy as np
import torch
import copy

from utils.utils import *
from super.loss import *

class LM_Solver():
    def __init__(self, opt, convs=None):

        self.opt = opt
        self.device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
        
        self.losses = []
        self.lambdas = []
        if opt.sf_point_plane:
            self.losses.append(DataLoss())
            self.lambdas.append(opt.sf_point_plane_weight)
        if opt.mesh_arap:
            self.losses.append(ARAPLoss())
            self.lambdas.append(opt.mesh_arap_weight)
        if opt.mesh_rot:
            self.losses.append(RotLoss())
            self.lambdas.append(opt.mesh_rot_weight)
        # if corr_cost:
        #     self.losses.append(CorrLoss(point_loss=True))
        #     self.lambdas.append(corr_lambda)

        self.phase = opt.phase

        if self.phase == "train":
            self.convs = convs

    # Solvers.
    @staticmethod
    def Solver(A, b, method="cholesky"):
        # methods: 1) 'lu': LU linear solver.
        # 2) "cholesky": Cholesky Solver.

        # Code ref: https://github.com/DeformableFriends/NeuralTracking/blob/main/model/model.py
        if method == "lu":
            A_LU, pivots = torch.lu(A)
            x = torch.lu_solve(b, A_LU, pivots)

        elif method == "cholesky":
            U = torch.linalg.cholesky(A)
            x = torch.cholesky_solve(b, U)
        
        return x

    # Calculate the Jacobians/losses.
    def prepareCostTerm(self, sf, inputs, new_data, beta, grad=False):

        if grad:
            jtj = torch.zeros((sf.ED_nodes.param_num, sf.ED_nodes.param_num), \
                layout=torch.sparse_coo, dtype=torch.float64, device=self.device)
            jtl = torch.zeros((sf.ED_nodes.param_num, 1), dtype=torch.float64, device=self.device)
        
            for loss_term, lambda_ in zip(self.losses, self.lambdas):
                jtj_, jtl_ = loss_term.forward(lambda_, beta, inputs, new_data, grad=True)
                if jtj_ is not None:
                    jtj += jtj_
                    jtl += jtl_
            del jtj_, jtl_

            return jtj.to_dense(), jtl

        else:
            loss = []

            for loss_term, lambda_ in zip(self.losses, self.lambdas):
                loss_ = loss_term.forward(lambda_, beta, inputs, new_data)
                if loss_ is not None: loss.append(loss_)
            del loss_

            return torch.sum(torch.cat(loss))

    # LM algorithm
    def LM(self, sf, inputs, new_data, u = 10, v = 7.5, minimal_loss = 1e10):
        num_Iter = self.opt.num_optimize_iterations

        # Init quaternion and translation parameters.
        beta = torch.tensor([[1.,0.,0.,0.,0.,0.,0.]], 
                                dtype=torch.float64, 
                                device=self.device
                                ).repeat(sf.ED_nodes.num,1)
        if self.phase == "test":
            best_beta = copy.deepcopy(beta)

        jtj_diag_i = torch.arange(sf.ED_nodes.param_num, device=self.device)
        for loss_term in self.losses: 
            loss_term.prepare(sf, new_data)
        for i in range(num_Iter):     
            jtj, jtl = self.prepareCostTerm(sf, inputs, new_data, beta, grad=True) #points, norms, valid, 
            jtj[jtj_diag_i,jtj_diag_i] += u

            try:
                delta = LM_Solver.Solver(jtj, jtl).view(-1,7)
            except RuntimeError as e:
                print("\t\tSolver failed: Ill-posed system!", e)
                break
            
            beta += delta

            loss = self.prepareCostTerm(sf, inputs, new_data, beta)
            
            if self.phase == "test":
                if loss < minimal_loss: # Accept the step.
                    minimal_loss = loss
                    u /= v
                    best_beta = copy.deepcopy(beta)

                else: # Reject the step.
                    u *= v
                    beta = copy.deepcopy(best_beta)

        if self.opt.phase == "test":
            sf.logger.info(f"{inputs['ID'].item()} loss: {loss}")

        return beta