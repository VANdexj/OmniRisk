import torch
import numpy as np
from config.config import cfg
from policy.primitive import LatticePrimitive


class StateTransform:
    def __init__(self):
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.goal_length = cfg['goal_length']
        # lattice → grid reorder index. grid column (h) follows image column (left→right = increasing world azimuth),
        # lattice anchor j also increases in azimuth → horizontal is **not flipped**; image row (v) top = high elevation while lattice
        # i bottom = low elevation → vertical is **flipped**. The legacy forward-only path (opposite column convention) keeps a full flip(0).
        V, H, N = (self.lattice_primitive.vertical_num,
                   self.lattice_primitive.horizon_num,
                   self.lattice_primitive.traj_num)
        if cfg['omnidirectional'] and N == V * H:
            idx = torch.arange(N)
            v, h = idx // H, idx % H
            self.lattice_reorder = (V - 1 - v) * H + h   # flip vertical only
        else:
            self.lattice_reorder = torch.arange(N).flip(0)  # legacy behavior: full reversal

    def _reorder(self, x):
        """Reorder lattice-ordered values into grid order via lattice_reorder (replaces the old .flip(0))."""
        return x[self.lattice_reorder.to(x.device)]

    def pred_to_endstate(self, endstate_pred: torch.Tensor) -> torch.Tensor:
        """
            Transform the predicted state to the body frame (Original prediction → Primitive frame → Body frame).
            endstate_pred: [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h]
            :return [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h] in body frame
        """
        B, V, H = endstate_pred.shape[0], endstate_pred.shape[2], endstate_pred.shape[3]

        # [B, 9, 3, 5] -> [B, 3, 5, 9] -> [B, 15, 9]
        endstate_pred = endstate_pred.permute(0, 2, 3, 1).reshape(B, V * H, 9)

        # lattice → grid reorder (lattice_reorder: omni flips vertical only, legacy path reverses fully)
        yaw, pitch = self.lattice_primitive.getAngleLattice()  # [N]
        yaw = self._reorder(yaw)[None, :].expand(B, -1)  # [B, N]
        pitch = self._reorder(pitch)[None, :].expand(B, -1)  # [B, N]
        Rbp = self._reorder(self.lattice_primitive.getRotation())  # [N, 3, 3]
        Rbp = Rbp[None, :, :, :].expand(B, -1, -1, -1)  # [B, N, 3, 3]

        delta_yaw = endstate_pred[:, :, 0] * self.lattice_primitive.yaw_diff  # [B, 15]
        delta_pitch = endstate_pred[:, :, 1] * self.lattice_primitive.pitch_diff
        radio = (endstate_pred[:, :, 2] + 1.0) * self.lattice_primitive.radio_range

        cos_pitch = torch.cos(pitch + delta_pitch)
        endstate_x = cos_pitch * torch.cos(yaw + delta_yaw) * radio
        endstate_y = cos_pitch * torch.sin(yaw + delta_yaw) * radio
        endstate_z = torch.sin(pitch + delta_pitch) * radio
        endstate_p = torch.stack([endstate_x, endstate_y, endstate_z], dim=-1)  # [B, 15, 3]

        # vel / acc
        endstate_vp = endstate_pred[:, :, 3:6] * self.lattice_primitive.vel_max  # [B, 15, 3]
        endstate_ap = endstate_pred[:, :, 6:9] * self.lattice_primitive.acc_max  # [B, 15, 3]

        # transform v/a into body frame
        endstate_vb = torch.matmul(Rbp, endstate_vp.unsqueeze(-1)).squeeze(-1)  # [B, 15, 3]
        endstate_ab = torch.matmul(Rbp, endstate_ap.unsqueeze(-1)).squeeze(-1)

        endstate = torch.cat([endstate_p, endstate_vb, endstate_ab], dim=-1)  # [B, 15, 9]

        endstate = endstate.permute(0, 2, 1).reshape(B, 9, V, H)  # [B, 9, 3, 5]
        return endstate

    def pred_to_endstate_cpu(self, endstate_pred: np.ndarray, lattice_id: torch.Tensor,
                             radio_range=None, vel_max=None, acc_max=None) -> np.ndarray:
        """
            Used during test:
            Numpy version of pred_to_endstate() on CPU (used in test, x10 times faster than torch on CUDA)
            :return [B; px py pz vx vy vz ax ay az] in body frame
        """
        delta_yaw = endstate_pred[:, 0] * self.lattice_primitive.yaw_diff
        delta_pitch = endstate_pred[:, 1] * self.lattice_primitive.pitch_diff
        radio_range = (self.lattice_primitive.radio_range
                       if radio_range is None else float(radio_range))
        vel_max = (self.lattice_primitive.vel_max
                   if vel_max is None else float(vel_max))
        acc_max = (self.lattice_primitive.acc_max
                   if acc_max is None else float(acc_max))
        radio = (endstate_pred[:, 2] + 1.0) * radio_range

        yaw, pitch = self.lattice_primitive.getAngleLattice(lattice_id)
        yaw, pitch = yaw.cpu().numpy(), pitch.cpu().numpy()
        endstate_x = np.cos(pitch + delta_pitch) * np.cos(yaw + delta_yaw) * radio
        endstate_y = np.cos(pitch + delta_pitch) * np.sin(yaw + delta_yaw) * radio
        endstate_z = np.sin(pitch + delta_pitch) * radio
        endstate_p = np.stack((endstate_x, endstate_y, endstate_z), axis=1)

        endstate_vp = endstate_pred[:, 3:6] * vel_max
        endstate_ap = endstate_pred[:, 6:9] * acc_max

        Rpb = self.lattice_primitive.getRotation(lattice_id).cpu().numpy()
        endstate_vb = np.matmul(Rpb, endstate_vp[:, :, np.newaxis]).squeeze(-1)
        endstate_ab = np.matmul(Rpb, endstate_ap[:, :, np.newaxis]).squeeze(-1)

        return np.concatenate((endstate_p, endstate_vb, endstate_ab), axis=1)


    def prepare_input(self, obs):
        """
            Transform the observation to the primitive frame (Body frame → Primitive frame → Body frame).
            obs: [batch; vx, vy, yz, ax, ay, az, gx, gy, gz] in body frame
            :return [batch; vx, vy, yz, ax, ay, az, gx, gy, gz; primitive_v; primitive_h] in primitive frame
        """
        B, N = obs.shape[0], self.lattice_primitive.traj_num

        # lattice → grid reorder (same as pred_to_endstate, so obs in and endstate out share one frame)
        Rbp_all = self._reorder(self.lattice_primitive.getRotation())  # shape: [N, 3, 3]

        obs = obs.view(B, 3, 3)  # [B, 3, 3]

        # expand obs and Rbp to [B, N, 3, 3]
        obs_exp = obs[:, None, :, :].expand(B, N, 3, 3)
        Rbp_exp = Rbp_all[None, :, :, :].expand(B, N, 3, 3)

        # batched frame transform
        transformed = torch.matmul(obs_exp, Rbp_exp)  # [B, N, 3, 3]

        transformed_flat = transformed.view(B, N, 9)  # [B, N, 9]
        out = transformed_flat.permute(0, 2, 1).contiguous()  # [B, 9, N]
        out = out.view(B, 9, self.lattice_primitive.vertical_num, self.lattice_primitive.horizon_num)  # [B, 9, V, H]
        return out

    def unnormalize_obs(self, vel_acc):
        vel_acc[:, 0:3] = vel_acc[:, 0:3] * self.lattice_primitive.vel_max
        vel_acc[:, 3:6] = vel_acc[:, 3:6] * self.lattice_primitive.acc_max
        return vel_acc

    def normalize_obs(self, vel_acc_goal, vel_max=None, acc_max=None,
                      goal_length=None):
        vel_max = (self.lattice_primitive.vel_max
                   if vel_max is None else float(vel_max))
        acc_max = (self.lattice_primitive.acc_max
                   if acc_max is None else float(acc_max))
        goal_length = self.goal_length if goal_length is None else float(goal_length)
        vel_acc_goal[:, 0:3] = vel_acc_goal[:, 0:3] / vel_max
        vel_acc_goal[:, 3:6] = vel_acc_goal[:, 3:6] / acc_max

        # Clamp the goal direction to unit length
        goal_norm = vel_acc_goal[:, 6:9].norm(dim=1, keepdim=True)
        vel_acc_goal[:, 6:9] = vel_acc_goal[:, 6:9] / goal_norm.clamp(min=goal_length)
        return vel_acc_goal


def rotate_body2world(rot_wb, pos_b):
    """
    Rotate pos_b from body frame to world frame using quaternion q_wb.
    rot_wb: (..., 3, 3)
    pos_b: (..., 3)
    """
    pos_w = torch.matmul(rot_wb, pos_b.unsqueeze(-1)).squeeze(-1)
    return pos_w


def transform_body2world(rot_wb, t_w, pos_b):
    """
    Transform pos_b from body frame to world frame using quaternion q_wb and t_w.
    rot_wb: (..., 3, 3)
    t_w: (..., 3)
    pos_b: (..., 3)
    """
    return rotate_body2world(rot_wb, pos_b) + t_w


def state_body2world(pos_w, rot_wb, pos_b, vel_b, acc_b):
    pos_b = transform_body2world(rot_wb, pos_w, pos_b)
    vel_b = rotate_body2world(rot_wb, vel_b)
    acc_b = rotate_body2world(rot_wb, acc_b)
    return pos_b, vel_b, acc_b


if __name__ == '__main__':
    CoordTransform = StateTransform()
