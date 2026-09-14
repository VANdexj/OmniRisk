import os
import glob
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from scipy.ndimage import distance_transform_edt
from config.config import cfg


class SafetyLoss(nn.Module):
    def __init__(self, L):
        super(SafetyLoss, self).__init__()
        self.traj_num = cfg['traj_num']
        self.map_expand_min = np.array(cfg['map_expand_min'])
        self.map_expand_max = np.array(cfg['map_expand_max'])
        self.d0 = cfg["d0"]
        self.r = cfg["r"]
        self.dynamic_weight = cfg["dynamic_weight"]
        # Anisotropic Gaussian risk field (evaluated analytically per frame from obstacle parameters, differentiable w.r.t. position/velocity):
        #   S = exp(−½·m²), m² = r_par²/σ_par² + |r_perp|²/σ_perp², obstacle velocity v_obs folded into the covariance
        self.aniso_alpha         = cfg["aniso_alpha"]          # Softplus sharpness α (applied to proj = v_rel·n_grad)
        self.aniso_perp_radius_k = cfg["aniso_perp_radius_k"]  # σ_perp = k·radius + base (lateral, narrow)
        self.aniso_perp_base     = cfg["aniso_perp_base"]
        self.aniso_par_radius_k  = cfg["aniso_par_radius_k"]   # σ_par = par_radius_k·radius + par_base + speed_k·|v_obs| (longitudinal, decoupled from σ_perp)
        self.aniso_par_base      = cfg["aniso_par_base"]
        self.aniso_par_speed_k   = cfg["aniso_par_speed_k"]    # σ_par stretch with speed
        self.aniso_sigma_z       = cfg["aniso_sigma_z"]        # vertical softening scale σ_z outside the cylinder band

        self._L = L
        self.sgm_time = cfg["sgm_time"]
        self.eval_points = 30
        self.device = self._L.device
        self.time_integral = True
        self.last_static_cost = None
        self.last_dynamic_cost = None
        self.last_pos_batch = None
        self.last_time_batch = None
        self.last_eval_points = None
        self.last_coe = None

        # SDF
        self.voxel_size = 0.2
        self.min_bounds = None  # shape: (N, 3)
        self.max_bounds = None  # shape: (N, 3)
        self.sdf_shapes = None  # shape: (N, 3)
        print("Building ESDF map...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "../", cfg["dataset_path"])
        self.sdf_maps = self.get_sdf_from_ply(data_dir)
        print("Map built!")

    def forward(self, Df, Dp, map_id, dynamic_obstacles=None, dynamic_cylinders=None):
        batch_size = Dp.shape[0]
        L = self._L.unsqueeze(0).expand(batch_size, -1, -1)
        coe = self.get_coefficient_from_derivative(Dp, Df, L)

        dt = self.sgm_time / self.eval_points
        t_list = th.linspace(dt, self.sgm_time, self.eval_points, device=self.device)
        t_list = t_list.view(1, -1, 1).expand(batch_size, -1, -1)

        pos_coe = self.get_position_from_coeff(coe, t_list)
        vel_coe = self.get_velocity_from_coeff(coe, t_list)
        pos_batch = pos_coe.reshape(-1, self.traj_num * pos_coe.shape[1], 3)
        vel_batch = vel_coe.reshape(-1, self.traj_num * vel_coe.shape[1], 3)

        time_batch = t_list.reshape(-1, self.traj_num * t_list.shape[1], 1)
        # for eval (geometric collision ground truth needs candidate sample points); no gradients, does not change training
        self.last_pos_batch = pos_batch.detach()
        self.last_time_batch = time_batch.detach()
        self.last_eval_points = pos_coe.shape[1]
        self.last_coe = coe.detach()          # (B*traj_num, 18) polynomial coefficients for dense resampling in eval
        static_cost, dynamic_cost = self.get_distance_cost_components(
            pos_batch, vel_batch, time_batch, map_id, dynamic_obstacles, dynamic_cylinders)

        if self.time_integral:
            static_cost_colli = static_cost.reshape(-1, pos_coe.shape[1]).mean(dim=-1)
        else:
            vel_norm = vel_coe.norm(dim=-1)
            line_integral_cost = (static_cost.reshape(-1, pos_coe.shape[1]) * vel_norm * dt).sum(dim=1)
            line_length = (vel_norm * dt).sum(dim=1)
            static_cost_colli = line_integral_cost / line_length

        dynamic_cost_colli = self.reduce_dynamic_cost(dynamic_cost, pos_coe.shape[1])
        self.last_static_cost = static_cost_colli
        self.last_dynamic_cost = self.dynamic_weight * dynamic_cost_colli
        return self.last_static_cost + self.last_dynamic_cost

    def get_distance_cost_components(self, pos, vel, time, map_id, dynamic_obstacles=None, dynamic_cylinders=None):
        B, N, _ = pos.shape

        sdf_maps, local_origin, local_shape = self.get_batch_sdf(pos, map_id)
        grid = (pos - local_origin.unsqueeze(1)) / self.voxel_size
        grid_point = 2.0 * grid / (local_shape - 1).unsqueeze(1) - 1.0
        grid_point = grid_point.view(B, 1, 1, N, 3)
        grid_point = th.clamp(grid_point, min=-0.99, max=0.99)

        static_dist_query = F.grid_sample(sdf_maps, grid_point, mode='bilinear', padding_mode='zeros', align_corners=True)
        static_dist_query = static_dist_query.view(B, N)

        static_cost = self.cost_function(static_dist_query)
        dynamic_cost = self.get_dynamic_risk_penalty(pos, vel, time, dynamic_obstacles, dynamic_cylinders)
        return static_cost, dynamic_cost

    def reduce_dynamic_cost(self, dynamic_cost, eval_points):
        # Equal-weight sum over sample points along the candidate Σ_i (path integral): with no threat each point costs ≈0 (Gaussian tail), so the sum is ≈0
        if dynamic_cost.numel() == 0:
            return th.zeros((dynamic_cost.shape[0],), device=dynamic_cost.device)
        return dynamic_cost.reshape(-1, eval_points).sum(dim=1)

    def get_dynamic_risk_penalty(self, pos, vel, time, dynamic_obstacles=None, dynamic_cylinders=None):
        """Anisotropic Gaussian risk field (evaluated analytically per frame, differentiable w.r.t. position/velocity): each obstacle center is extrapolated along a straight line to c+v_o·t,
        where a velocity-aligned anisotropic Gaussian S=exp(−½m²) is built (σ_par stretched along v_obs, σ_perp narrowed laterally, decoupled); its gradient ∇S is normalized
        into the "worsening axis" n_grad; per-point cost = S·Softplus(α·(v_uav−v_obs)·n_grad), heavily penalizing approach (proj>0) while escape
        (proj<0) is smoothly clipped to zero by Softplus; summed over all obstacles (multiple threats add up). Returns (B, N)."""
        B, N = pos.shape[0], pos.shape[1]
        cost = th.zeros((B, N), device=pos.device)
        if dynamic_obstacles is not None and dynamic_obstacles.numel() > 0:
            cost = cost + self._aniso_cost_sphere(pos, vel, time, dynamic_obstacles)
        if dynamic_cylinders is not None and dynamic_cylinders.numel() > 0:
            cost = cost + self._aniso_cost_cylinder(pos, vel, time, dynamic_cylinders)
        return cost

    def _aniso_cost(self, S, grad_m2, v_rel, valid):
        # Shared tail: field S (B,N,M), ∇m² (B,N,M,3), relative velocity v_rel=v_uav−v_obs (B,N,M,3), valid mask valid (B,N,M).
        # n_grad = ∇S/|∇S|, ∇S = −½S·∇m², same direction as −∇m² (toward the danger core = worsening axis); denominator clamped against the r→0 singularity.
        grad_S = -0.5 * S.unsqueeze(-1) * grad_m2                          # (B,N,M,3) ∇_p S
        n_grad = grad_S / th.linalg.norm(grad_S, dim=-1, keepdim=True).clamp(min=1e-6)
        proj = (v_rel * n_grad).sum(dim=-1)                               # (B,N,M) projection of v_rel onto the worsening axis
        cost = S * F.softplus(self.aniso_alpha * proj)                    # (B,N,M)
        cost = cost.masked_fill(~valid, 0.0)
        return cost.sum(dim=2)                                            # (B,N) sum over all obstacles

    def _aniso_cost_sphere(self, pos, vel, time, dynamic_obstacles):
        centers = dynamic_obstacles[:, :, :3]
        radii = dynamic_obstacles[:, :, 3]
        velocities = dynamic_obstacles[:, :, 4:7]
        centers_pred = centers.unsqueeze(1) + velocities.unsqueeze(1) * time.unsqueeze(2)  # c+v_o·t → (B,N,M,3)
        r = pos.unsqueeze(2) - centers_pred                               # (B,N,M,3)
        speed = th.linalg.norm(velocities, dim=-1)                        # (B,M)
        u = (velocities / speed.clamp(min=1e-6).unsqueeze(-1)).unsqueeze(1)  # (B,1,M,3) velocity axis û (speed→0 degenerates to isotropic)
        r_par = (r * u).sum(dim=-1)                                       # (B,N,M)
        sig_perp = self.aniso_perp_radius_k * radii + self.aniso_perp_base   # (B,M)
        sig_par = self.aniso_par_radius_k * radii + self.aniso_par_base + self.aniso_par_speed_k * speed  # (B,M)
        inv_pe2 = (1.0 / sig_perp.clamp(min=1e-6) ** 2).unsqueeze(1)      # (B,1,M)
        inv_pa2 = (1.0 / sig_par.clamp(min=1e-6) ** 2).unsqueeze(1)       # (B,1,M)
        r_sq = (r * r).sum(dim=-1)                                        # (B,N,M) = |r|²
        m2 = r_par ** 2 * inv_pa2 + (r_sq - r_par ** 2).clamp(min=0.0) * inv_pe2
        S = th.exp(-0.5 * m2)
        grad_m2 = 2.0 * (r_par * (inv_pa2 - inv_pe2)).unsqueeze(-1) * u + 2.0 * r * inv_pe2.unsqueeze(-1)
        v_rel = vel.unsqueeze(2) - velocities.unsqueeze(1)               # v_uav − v_obs → (B,N,M,3)
        valid = (radii > 0).unsqueeze(1).expand(-1, pos.shape[1], -1)
        return self._aniso_cost(S, grad_m2, v_rel, valid)

    def _aniso_cost_cylinder(self, pos, vel, time, dynamic_cylinders):
        centers = dynamic_cylinders[:, :, :3]
        radii = dynamic_cylinders[:, :, 3]
        heights = dynamic_cylinders[:, :, 4]
        velocities = dynamic_cylinders[:, :, 5:8]
        centers_pred = centers.unsqueeze(1) + velocities.unsqueeze(1) * time.unsqueeze(2)  # c+v_o·t → (B,N,M,3)
        r = pos.unsqueeze(2) - centers_pred                               # (B,N,M,3)
        r_xy, r_z = r[..., :2], r[..., 2]                                 # cylinders are anisotropic in the xy plane, z uses the band
        speed = th.linalg.norm(velocities[..., :2], dim=-1)              # (B,M) horizontal speed (cylinder vz=0)
        u_xy = (velocities[..., :2] / speed.clamp(min=1e-6).unsqueeze(-1)).unsqueeze(1)  # (B,1,M,2)
        r_par = (r_xy * u_xy).sum(dim=-1)                                 # (B,N,M)
        sig_perp = self.aniso_perp_radius_k * radii + self.aniso_perp_base
        sig_par = self.aniso_par_radius_k * radii + self.aniso_par_base + self.aniso_par_speed_k * speed
        inv_pe2 = (1.0 / sig_perp.clamp(min=1e-6) ** 2).unsqueeze(1)      # (B,1,M)
        inv_pa2 = (1.0 / sig_par.clamp(min=1e-6) ** 2).unsqueeze(1)       # (B,1,M)
        inv_z2 = 1.0 / max(self.aniso_sigma_z, 1e-6) ** 2
        rxy_sq = (r_xy * r_xy).sum(dim=-1)                                # (B,N,M)
        dz = (r_z.abs() - 0.5 * heights.unsqueeze(1)).clamp(min=0.0)      # (B,N,M) 0 inside the band, >0 only outside
        m2 = r_par ** 2 * inv_pa2 + (rxy_sq - r_par ** 2).clamp(min=0.0) * inv_pe2 + dz ** 2 * inv_z2
        S = th.exp(-0.5 * m2)
        grad_xy = 2.0 * (r_par * (inv_pa2 - inv_pe2)).unsqueeze(-1) * u_xy + 2.0 * r_xy * inv_pe2.unsqueeze(-1)  # (B,N,M,2)
        grad_z = (2.0 * dz * th.sign(r_z) * inv_z2).unsqueeze(-1)         # (B,N,M,1) dz=0 inside the band → 0
        grad_m2 = th.cat([grad_xy, grad_z], dim=-1)                       # (B,N,M,3)
        v_rel = vel.unsqueeze(2) - velocities.unsqueeze(1)               # (B,N,M,3)
        valid = ((radii > 0) & (heights > 0)).unsqueeze(1).expand(-1, pos.shape[1], -1)
        return self._aniso_cost(S, grad_m2, v_rel, valid)

    def cost_function(self, d):
        return th.exp(-(d - self.d0) / self.r)

    def get_coefficient_from_derivative(self, Dp, Df, L):
        coefficient = th.zeros(Dp.shape[0], 18, device=self.device)
        for i in range(3):
            d = th.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
            coe = (L @ d).squeeze()
            coefficient[:, 6 * i: 6 * (i + 1)] = coe
        return coefficient

    def get_position_from_coeff(self, coe, t):
        t_power = th.stack([th.ones_like(t), t, t ** 2, t ** 3, t ** 4, t ** 5], dim=-1).squeeze(-2)
        coe_x = coe[:, 0: 6]
        coe_y = coe[:, 6:12]
        coe_z = coe[:, 12:18]
        x = th.sum(t_power * coe_x.unsqueeze(1), dim=-1)
        y = th.sum(t_power * coe_y.unsqueeze(1), dim=-1)
        z = th.sum(t_power * coe_z.unsqueeze(1), dim=-1)
        return th.stack([x, y, z], dim=-1)

    def get_velocity_from_coeff(self, coe, t):
        t_power = th.stack([th.ones_like(t), 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4], dim=-1).squeeze(-2)
        coe_x = coe[:, 1:6]
        coe_y = coe[:, 7:12]
        coe_z = coe[:, 13:18]
        vx = th.sum(t_power * coe_x.unsqueeze(1), dim=-1)
        vy = th.sum(t_power * coe_y.unsqueeze(1), dim=-1)
        vz = th.sum(t_power * coe_z.unsqueeze(1), dim=-1)
        return th.stack([vx, vy, vz], dim=-1)

    def get_batch_sdf(self, pos, map_id):
        min_bounds = self.min_bounds[map_id]
        sdf_shapes = self.sdf_shapes[map_id]

        min_pos = pos.amin(dim=1)
        max_pos = pos.amax(dim=1)
        min_indices = ((min_pos - min_bounds) / self.voxel_size).int()
        max_indices = ((max_pos - min_bounds) / self.voxel_size).int()
        spans = max_indices - min_indices
        max_spans = spans.amax(dim=0)
        centers = (min_indices + max_indices) // 2
        min_indices = centers - max_spans // 2 - 5
        max_indices = centers + max_spans // 2 + 5

        new_min_indices = min_indices.clamp(min=0)
        underflow_amount = new_min_indices - min_indices
        min_indices = new_min_indices
        max_indices = max_indices + underflow_amount

        new_max_indices = th.minimum(max_indices, sdf_shapes.int())
        overflow_amount = max_indices - new_max_indices
        max_indices = new_max_indices
        min_indices = min_indices - overflow_amount

        if (min_indices < 0).any():
            min_underflow = th.minimum(min_indices, th.zeros_like(min_indices))
            shift = (-min_underflow).max(dim=0).values
            min_indices = min_indices + shift

        sdf_maps = th.stack([self.sdf_maps[map_idx][0, :,
                             min_idx[2]:max_idx[2],
                             min_idx[1]:max_idx[1],
                             min_idx[0]:max_idx[0]]
                             for map_idx, min_idx, max_idx in zip(map_id.tolist(), min_indices.tolist(), max_indices.tolist())
                             ])
        local_origin = min_indices * self.voxel_size + min_bounds
        local_shape = max_indices - min_indices
        return sdf_maps, local_origin, local_shape

    def get_sdf_from_ply(self, path):
        sorted_files = self.read_sorted_ply_files(path)
        sdf_maps = []
        min_bounds, max_bounds, sdf_shapes = [], [], []

        for file in sorted_files:
            pcd = o3d.io.read_point_cloud(file)
            min_bound = np.array(pcd.get_min_bound()) - self.map_expand_min
            max_bound = np.array(pcd.get_max_bound()) + self.map_expand_max
            points = np.asarray(pcd.points)
            print(f"    {os.path.basename(file)}: x=({min_bound[0] + self.map_expand_min[0]:.2f}, {max_bound[0] - self.map_expand_max[0]:.2f}), "
                  f"y=({min_bound[1] + self.map_expand_min[1]:.2f}, {max_bound[1] - self.map_expand_max[1]:.2f}), "
                  f"z=({min_bound[2] + self.map_expand_min[2]:.2f}, {max_bound[2] - self.map_expand_max[2]:.2f})")

            sdf_shape = np.ceil((max_bound - min_bound) / self.voxel_size).astype(int)
            voxel_indices = ((points - min_bound) / self.voxel_size).astype(int)

            valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < sdf_shape), axis=1)
            voxel_indices = voxel_indices[valid_mask]

            occupancy = np.zeros(sdf_shape, dtype=np.uint8)
            occupancy[tuple(voxel_indices.T)] = 1

            obstacle_mask = occupancy == 1
            free_mask = occupancy == 0

            dist_to_obstacle = distance_transform_edt(free_mask) * self.voxel_size
            dist_inside_obstacle = distance_transform_edt(obstacle_mask) * self.voxel_size
            dist_to_obstacle[obstacle_mask] = -dist_inside_obstacle[obstacle_mask]

            sdf_tensor = th.from_numpy(dist_to_obstacle).float().unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 3, 2).to(self.device)
            sdf_maps.append(sdf_tensor)
            sdf_shapes.append(sdf_tensor.shape[-3:][::-1])
            min_bounds.append(min_bound)
            max_bounds.append(max_bound)

        self.min_bounds = th.tensor(np.array(min_bounds), device=self.device).float()
        self.max_bounds = th.tensor(np.array(max_bounds), device=self.device).float()
        self.sdf_shapes = th.tensor(np.array(sdf_shapes), device=self.device).float()
        return sdf_maps

    def read_sorted_ply_files(self, path):
        ply_files = glob.glob(os.path.join(path, 'pointcloud-*.ply'))

        def extract_index(filename):
            base = os.path.basename(filename)
            number_part = base.replace('pointcloud-', '').replace('.ply', '')
            return int(number_part)

        return sorted(ply_files, key=extract_index)

    def pad_sdf_to_shape(self, sdf_map, target_shape):
        current_shape = sdf_map.shape[-3:]
        pad_sizes = [target - current for target, current in zip(target_shape[::-1], current_shape[::-1])]
        padding = [0, pad_sizes[0], 0, pad_sizes[1], 0, pad_sizes[2]]
        return F.pad(sdf_map, padding, mode='constant', value=0)
