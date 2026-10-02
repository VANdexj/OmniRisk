import torch.nn as nn
import torch as th


class SmoothnessLoss(nn.Module):
    def __init__(self, RJ, RA):
        super(SmoothnessLoss, self).__init__()
        self._RJ = RJ
        self._RA = RA

    def forward(self, Df, Dp):
        """
        Args:
            Dp: decision parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            Df: fixed parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
        Returns:
            cost_smooth: (batch_size) → smoothness loss
        """
        RJ = self._RJ.unsqueeze(0).expand(Dp.shape[0], -1, -1)
        RA = self._RA.unsqueeze(0).expand(Dp.shape[0], -1, -1)
        D_all = th.cat([Df, Dp], dim=2)  # dx, dy, dz will be rows

        # Reshape for matmul: (batch, 6, 1)
        dx, dy, dz = D_all[:, 0].unsqueeze(2), D_all[:, 1].unsqueeze(2), D_all[:, 2].unsqueeze(2)

        # Compute smoothness loss: dxᵀ R dx + ...
        jerk_smooth = dx.transpose(1, 2) @ RJ @ dx + dy.transpose(1, 2) @ RJ @ dy + dz.transpose(1, 2) @ RJ @ dz

        accel_smooth = dx.transpose(1, 2) @ RA @ dx + dy.transpose(1, 2) @ RA @ dy + dz.transpose(1, 2) @ RA @ dz

        return jerk_smooth.squeeze(), accel_smooth.squeeze()