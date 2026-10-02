import math
import torch as th
import torch.nn as nn
from config.config import cfg
from loss.safety_loss import SafetyLoss
from loss.smoothness_loss import SmoothnessLoss
from loss.guidance_loss import GuidanceLoss


class OmniRiskLoss(nn.Module):
    def __init__(self):
        super(OmniRiskLoss, self).__init__()
        self.sgm_time = cfg["sgm_time"]
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self._C, self._B, self._L, self._RJ, self._RA = self.qp_generation()
        self._RJ = self._RJ.to(self.device)
        self._RA = self._RA.to(self.device)
        self._L = self._L.to(self.device)
        self.denormalize_weight()
        self.smoothness_loss = SmoothnessLoss(self._RJ, self._RA)
        self.safety_loss = SafetyLoss(self._L)
        self.goal_loss = GuidanceLoss()

        self.last_static_safety_cost = None
        self.last_dynamic_safety_cost = None
        print("------ Actual Loss ------")
        print(f"| {'smooth':<12} = {self.smoothness_weight:6.4f} |")
        print(f"| {'safety':<12} = {self.safety_weight:6.4f} |")
        print(f"| {'goal':<12} = {self.goal_weight:6.4f} |")
        print("-------------------------")

    def qp_generation(self):
        A = th.zeros((6, 6))
        for i in range(3):
            A[2 * i, i] = math.factorial(i)
            for j in range(i, 6):
                A[2 * i + 1, j] = math.factorial(j) / math.factorial(j - i) * (self.sgm_time ** (j - i))

        H = th.zeros((6, 6))
        for i in range(3, 6):
            for j in range(3, 6):
                H[i, j] = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * (self.sgm_time ** (i + j - 5))

        Q = th.zeros((6, 6))
        for i in range(2, 6):
            for j in range(2, 6):
                Q[i, j] = (i * (i - 1)) * (j * (j - 1)) / (i + j - 3) * (self.sgm_time ** (i + j - 3))

        return self.stack_opt_dep(A, H, Q)

    def stack_opt_dep(self, A, H, Q):
        Ct = th.zeros((6, 6))
        Ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1

        _C = th.transpose(Ct, 0, 1)
        B = th.inverse(A)
        B_T = th.transpose(B, 0, 1)
        _L = B @ Ct
        _R_Jerk = _C @ (B_T) @ H @ B @ Ct
        _R_Acc = _C @ (B_T) @ Q @ B @ Ct
        return _C, B, _L, _R_Jerk, _R_Acc

    def denormalize_weight(self):
        vel_scale = cfg["vel_max_train"] / 1.0
        self.smoothness_weight = cfg["ws"] / vel_scale ** 5
        self.accele_weight = cfg["wa"] / vel_scale ** 3
        self.safety_weight = cfg["wc"]
        self.goal_weight = cfg["wg"]

    def forward(self, state, prediction, goal, map_id,
                dynamic_obstacles=None, dynamic_cylinders=None):
        Df = state.permute(0, 2, 1)
        Dp = prediction.permute(0, 2, 1)

        smoothness_cost, acceleration_cost = self.smoothness_loss(Df, Dp)
        safety_cost = self.safety_loss(Df, Dp, map_id, dynamic_obstacles, dynamic_cylinders)
        self.last_static_safety_cost = self.safety_weight * self.safety_loss.last_static_cost
        self.last_dynamic_safety_cost = self.safety_weight * self.safety_loss.last_dynamic_cost
        goal_cost = self.goal_loss(Df, Dp, goal)

        return (self.smoothness_weight * smoothness_cost,
                self.safety_weight     * safety_cost,
                self.goal_weight       * goal_cost,
                self.accele_weight     * acceleration_cost)
