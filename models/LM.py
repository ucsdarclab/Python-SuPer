# modified from: https://github.com/MengHao666/Minimal-Hand-pytorch
import time
import numpy as np
import torch
import copy

# from utils import img_matching

from utils.config import *
from utils.utils import *
from models.loss import *

class LM_Solver():
    def __init__(self, method, data_cost, depth_cost, \
        deep_depth_cost, coord_cost, arap_cost, rot_cost, corr_cost, \
        data_lambda, depth_lambda, deep_depth_lambda, coord_lambda, \
        corr_lambda, arap_lambda, rot_lambda, phase="test"):
        
        self.losses = []
        self.lambdas = []
        if data_cost:
            self.losses.append(DataLoss())
            self.lambdas.append(data_lambda)
        if depth_cost:
            self.losses.append(DepthLoss())
            self.lambdas.append(depth_lambda)
        if arap_cost:
            self.losses.append(ARAPLoss())
            self.lambdas.append(arap_lambda)
        if rot_cost:
            self.losses.append(RotLoss())
            self.lambdas.append(rot_lambda)
        if corr_cost:
            self.losses.append(CorrLoss())
            self.lambdas.append(corr_lambda)
        if coord_cost:
            self.losses.append(CoordLoss())
            self.lambdas.append(coord_lambda)
        
        # if corr_cost or coord_cost:
        #     if method == 'super': 
        #         self.matcher = img_matching.cv2Matcher()
        #     elif method == 'dlsuper': 
        #         self.matcher = img_matching.DeepMatcher(phase)

        self.phase = phase
        self.beta_init = torch.tensor([[1.,0.,0.,0.,0.,0.,0.]], dtype=fl64_, device=dev)

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
            U = torch.cholesky(A)
            x = torch.cholesky_solve(b, U)
        
        return x

    # Calculate the Jacobians/losses.
    def prepareCostTerm(self, sfModel, new_data, beta, grad=False):

        if grad:
            jtj = torch.zeros((sfModel.param_num, sfModel.param_num), \
                layout=torch.sparse_coo, dtype=fl32_, device=dev)
            jtl = torch.zeros((sfModel.param_num, 1), dtype=fl32_, device=dev)
        
            for loss_term, lambda_ in zip(self.losses, self.lambdas):
                jtj_, jtl_ = loss_term.forward(lambda_, beta, new_data, grad=True)
                if jtj_ is not None:
                    jtj += jtj_
                    jtl += jtl_
            del jtj_, jtl_

            return jtj.to_dense(), jtl

        else:
            loss = []

            for loss_term, lambda_ in zip(self.losses, self.lambdas):
                loss_ = loss_term.forward(lambda_, beta, new_data)
                if loss_ is not None: loss.append(loss_)
            del loss_

            return torch.sum(torch.cat(loss))

    # LM algorithm
    def LM(self, sfModel, new_data, u = 50., v = 7.5, minimal_loss = 1e10, num_Iter=10):

        # Init quaternion and translation parameters.
        best_beta = self.beta_init.repeat(sfModel.ED_num,1)
        beta = copy.deepcopy(best_beta)

        jtj_diag_i = torch.arange(sfModel.param_num, device=dev)
        for loss_term in self.losses: loss_term.prepare(sfModel)
        for i in range(num_Iter):

            if debug_mode: # TODO
                start_ = timeit.default_timer()
            
            jtj, jtl = self.prepareCostTerm(sfModel, new_data, beta, grad=True) #points, norms, valid, 
            jtj[jtj_diag_i,jtj_diag_i] += u

            try:
                delta = LM_Solver.Solver(jtj, jtl).view(-1,7)
            except RuntimeError as e:
                print("\t\tSolver failed: Ill-posed system!", e)
                break
            
            beta += delta

            loss = self.prepareCostTerm(sfModel, new_data, beta)
            
            if self.phase == "train": # TODO
                minimal_loss = loss
                u /= v

            elif self.phase == "test":
                if loss < minimal_loss: # Accept the step.
                    minimal_loss = loss
                    u /= v
                    best_beta = copy.deepcopy(beta)

                else: # Reject the step.
                    u *= v
                    beta = copy.deepcopy(best_beta)

                if debug_mode: # TODO
                    stop_ = timeit.default_timer()
                    print("***Debug*** Iter: {}; time: {}s; loss: {}".format(i,np.round(stop_-start_,3),loss) )

        return beta