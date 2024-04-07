import os

import torch
import torch.nn as nn


# os.environ['CUDA_VISIBLE_DEVICES'] = "3"

"""
sinkhorn_pointcloud.py

Discrete OT : Sinkhorn algorithm for point cloud marginals.

"""
class FeatureOptimalLoss(nn.Module):
    def __init__(self, epsilon=100, niter=10):
        super(FeatureOptimalLoss, self).__init__()
        self.epsilon = epsilon
        self.niter = niter

    def forward(self, output, target,normalized=False):
        batch_size = output.shape[0]
        output = output
        if normalized==False:
            return self._sinkhorn_loss_primal(output, target, self.epsilon, batch_size, self.niter)
        else:
            Wxx =  self._sinkhorn_loss_primal(output, output, self.epsilon, batch_size, self.niter)
            Wxy =  self._sinkhorn_loss_primal(output, target, self.epsilon, batch_size, self.niter)
            Wyy =  self._sinkhorn_loss_primal(target, target, self.epsilon, batch_size, self.niter)
            return 2*Wxy-Wxx-Wyy


    def _sinkhorn_loss_primal(self, x, y, epsilon, n, niter):
        """
           Given two emprical measures with n points each with locations x and y
           outputs an approximation of the OT cost with regularization parameter epsilon
           niter is the max. number of steps in sinkhorn loop
           """
        # The Sinkhorn algorithm takes as input three variables :
        C = self._squared_distances(x, y)# Wasserstein cost function

        # both marginals are fixed with equal weights
        mu = torch.full([n, ], 1 / n).cuda()
        nu = torch.full([n, ], 1 / n).cuda()

        # Parameters of the Sinkhorn algorithm.
        rho = 1# (.5) **2          # unbalanced transport
        tau = -.8# nesterov-like acceleration

        lam = rho / (rho + epsilon)# Update exponent

        # Elementary operations .....................................................................
        def ave(u, u1):
            "Barycenter subroutine, used by kinetic acceleration through extrapolation."
            return tau * u + (1 - tau) * u1

        def M(u, v):
            "Modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

        "log-sum-exp"
        lse = lambda A: torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)# add 10^-6 to prevent NaN

        # Actual Sinkhorn loop ......................................................................
        u, v, err = 0. * mu, 0. * nu, 0.
        actual_nits = 0# to check if algorithm terminates because of threshold or max iterations reached

        for i in range(niter):
            u1 = u  # useful to check the update

            u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
            v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
            err = (u - u1).abs().sum()

            actual_nits += 1
            if (err < 1e-1).data.cpu().numpy():
                break
        U, V = u, v
        Gamma = torch.exp(M(U, V)) # Transport plan pi = diag(a)*K*diag(b)
        cost = torch.sum(Gamma * C)# Sinkhorn cost

        return cost

    def _squared_distances(self, x, y):

        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum(torch.abs(x_col - y_lin), 2)
        return c

    def _adjust_parameters(self):
        need_adjust_parameters = {"niter": [1, 10, 100],
                                  "epsilon": [10 ** 4, 10 ** 3, 10 ** 2, 10],
                                  "batch_size": "max_value"}
        return need_adjust_parameters

    def _get_parameters_info(self):
        pass

