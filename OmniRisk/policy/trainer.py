"""
Training Strategy
supervised learning, imitation learning, testing, rollout
"""
import os
import time
import atexit
from torch.nn import functional as F
from rich.progress import Progress
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config.config import cfg
from loss.loss_function import OmniRiskLoss
from policy.network import OmniRiskNetwork
from policy.dataset import OmniRiskDataset
from policy.state_transform import *


class OmniRiskTrainer:
    def __init__(
            self,
            learning_rate=0.001,
            batch_size=32,
            loss_weight=[],
            tensorboard_path=None,
            checkpoint_path=None,
            save_on_exit=False,
    ):
        self.batch_size = batch_size
        self.max_grad_norm = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_weight = loss_weight
        if save_on_exit: self._exit_func = atexit.register(self.save_model)
        # logger
        self.progress_log = Progress()
        self.tensorboard_path = self.get_next_log_path(tensorboard_path)
        self.tensorboard_log = SummaryWriter(log_dir=self.tensorboard_path)
        # params
        self.traj_num = cfg['traj_num']

        # network
        print("Loading network...")
        self.policy = OmniRiskNetwork()
        self.policy = self.policy.to(self.device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, weights_only=True)
            self.policy.load_state_dict(state_dict)
            print("Checkpoint ", checkpoint_path, " loaded successfully")
        else:
            print("Training from scratch")

        # For anchor-direction guidance: lattice node of each candidate (grid order, aligned with endstate_flat).
        st = self.policy.state_transform
        _nodes = st.lattice_primitive.lattice_pos_node                       # (N,3) lattice order
        self.anchor_nodes_grid = _nodes[st.lattice_reorder.to(_nodes.device)].to(self.device)  # (N,3) grid order

        # loss
        self.loss_fn = OmniRiskLoss()

        # optimizer
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate, fused=True)
        print("Network Loaded! Loading Dataset...")

        # dataset
        self.train_dataloader = DataLoader(OmniRiskDataset(mode='train'), batch_size=self.batch_size, shuffle=True,
                                           num_workers=16, pin_memory=True,
                                           persistent_workers=True, prefetch_factor=4)
        self.val_dataloader = DataLoader(OmniRiskDataset(mode='valid'), batch_size=self.batch_size, shuffle=False,
                                         num_workers=16, pin_memory=True,
                                         persistent_workers=True, prefetch_factor=4)
        print("Dataset Loaded!")

    def train(self, epoch, save_interval=None):
        with self.progress_log:
            total_progress = self.progress_log.add_task("Training", total=epoch)
            for self.epoch_i in range(epoch):
                self.policy.train()
                self.train_one_epoch(self.epoch_i, total_progress)
                self.policy.eval()
                self.eval_one_epoch(self.epoch_i)
                if save_interval is not None and (self.epoch_i + 1) % save_interval == 0:
                    self.progress_log.console.log("Saving model...")
                    policy_path = self.tensorboard_path + "/epoch{}.pth".format(self.epoch_i + 1, 0)
                    torch.save(self.policy.state_dict(), policy_path)
            self.progress_log.console.log("Training finished!")
            self.progress_log.remove_task(total_progress)

    def train_one_epoch(self, epoch: int, total_progress):
        one_epoch_progress = self.progress_log.add_task(f"Epoch: {epoch}", total=len(self.train_dataloader))
        inspect_interval = max(1, len(self.train_dataloader) // 16)
        traj_losses, static_score_losses, dynamic_score_losses = [], [], []
        smooth_losses, safety_losses = [], []
        goal_losses, acc_losses, start_time = [], [], time.time()

        for step, batch in enumerate(self.train_dataloader):
            (depth, mask, vel_img, pos, rot, obs_b, map_id, dyn_obs, dyn_cyl) = batch
            if depth.shape[0] != self.batch_size: continue

            self.optimizer.zero_grad()

            (trajectory_loss, static_score_loss, dynamic_score_loss,
             smooth_cost, safety_cost, goal_cost, acc_cost) = self.forward_and_compute_loss(
                depth, mask, vel_img, pos, rot, obs_b, map_id, dyn_obs, dyn_cyl,
            )

            loss = (self.loss_weight[0] * trajectory_loss
                    + self.loss_weight[1] * (static_score_loss + dynamic_score_loss))
            loss.backward()
            self.optimizer.step()

            traj_losses.append(self.loss_weight[0] * trajectory_loss.item())
            static_score_losses.append(self.loss_weight[1] * static_score_loss.item())
            dynamic_score_losses.append(self.loss_weight[1] * dynamic_score_loss.item())
            smooth_losses.append(self.loss_weight[0] * smooth_cost.item())
            safety_losses.append(self.loss_weight[0] * safety_cost.item())
            goal_losses.append(self.loss_weight[0] * goal_cost.item())
            acc_losses.append(self.loss_weight[0] * acc_cost.item())

            if step % inspect_interval == inspect_interval - 1:
                batch_fps = inspect_interval / (time.time() - start_time)
                self.progress_log.console.log(
                    f"Epoch: {epoch}, Traj: {np.mean(traj_losses):.3g}, "
                    f"StaticScore: {np.mean(static_score_losses):.3g} "
                    f"DynScore: {np.mean(dynamic_score_losses):.3g} "
                    f"Safety: {np.mean(safety_losses):.3g} "
                    f"FPS: {batch_fps:.3g}"
                )
                step_global = epoch * len(self.train_dataloader) + step
                self.tensorboard_log.add_scalar("Train/TrajLoss", np.mean(traj_losses), step_global)
                self.tensorboard_log.add_scalar("Train/StaticScoreLoss", np.mean(static_score_losses), step_global)
                self.tensorboard_log.add_scalar("Train/DynamicScoreLoss", np.mean(dynamic_score_losses), step_global)
                self.tensorboard_log.add_scalar("Detail/SmoothLoss", np.mean(smooth_losses), step_global)
                self.tensorboard_log.add_scalar("Detail/SafetyLoss", np.mean(safety_losses), step_global)
                self.tensorboard_log.add_scalar("Detail/GoalLoss", np.mean(goal_losses), step_global)
                self.tensorboard_log.add_scalar("Detail/AccelLoss", np.mean(acc_losses), step_global)
                traj_losses, static_score_losses, dynamic_score_losses = [], [], []
                smooth_losses, safety_losses = [], []
                goal_losses, acc_losses, start_time = [], [], time.time()

            self.progress_log.update(one_epoch_progress, advance=1)
            self.progress_log.update(total_progress, advance=1 / len(self.train_dataloader))

        self.progress_log.remove_task(one_epoch_progress)

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int):
        one_epoch_progress = self.progress_log.add_task(f"Eval: {epoch}", total=len(self.val_dataloader))
        traj_losses, static_score_losses, dynamic_score_losses = [], [], []

        for step, batch in enumerate(self.val_dataloader):
            (depth, mask, vel_img, pos, rot, obs_b, map_id, dyn_obs, dyn_cyl) = batch
            if depth.shape[0] != self.batch_size: continue

            (trajectory_loss, static_score_loss, dynamic_score_loss, *_) = self.forward_and_compute_loss(
                depth, mask, vel_img, pos, rot, obs_b, map_id, dyn_obs, dyn_cyl,
            )
            traj_losses.append(self.loss_weight[0] * trajectory_loss.item())
            static_score_losses.append(self.loss_weight[1] * static_score_loss.item())
            dynamic_score_losses.append(self.loss_weight[1] * dynamic_score_loss.item())
            self.progress_log.update(one_epoch_progress, advance=1)

        self.progress_log.console.log(
            f"Eval: {epoch}, Traj: {np.mean(traj_losses):.3g}, "
            f"StaticScore: {np.mean(static_score_losses):.3g}, DynScore: {np.mean(dynamic_score_losses):.3g}"
        )
        self.tensorboard_log.add_scalar("Eval/TrajLoss", np.mean(traj_losses), epoch)
        self.tensorboard_log.add_scalar("Eval/StaticScoreLoss", np.mean(static_score_losses), epoch)
        self.tensorboard_log.add_scalar("Eval/DynamicScoreLoss", np.mean(dynamic_score_losses), epoch)
        self.progress_log.remove_task(one_epoch_progress)

    def forward_and_compute_loss(self, depth, mask, vel_img, pos, rot, obs_b, map_id,
                                 dynamic_obstacles, dynamic_cylinders):
        (depth, mask, vel_img, pos, rot, obs_b, map_id,
         dynamic_obstacles, dynamic_cylinders) = [
            x.to(self.device) for x in [
                depth, mask, vel_img, pos, rot, obs_b, map_id,
                dynamic_obstacles, dynamic_cylinders]
        ]

        # 1. pre-process
        goal_w, start_vel_w, start_acc_w = state_body2world(pos, rot, obs_b[:, 6:9], obs_b[:, 0:3], obs_b[:, 3:6])
        start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)

        # 2. forward propagation (two-head: static endstate+score, dynamic score)
        endstate, static_score, dynamic_score = self.policy.inference(depth, mask, vel_img, obs_b)

        # 3. post-process [B, 9, V, H] -> [B*V*H, 9]
        endstate_flat = endstate.permute(0, 2, 3, 1).reshape(self.batch_size * self.traj_num, 9)
        static_score_flat  = static_score.reshape(self.batch_size * self.traj_num)
        dynamic_score_flat = dynamic_score.reshape(self.batch_size * self.traj_num)

        pos_expanded       = pos.repeat_interleave(self.traj_num, dim=0)
        rot_expanded       = rot.repeat_interleave(self.traj_num, dim=0)
        start_state_w      = start_state_w.repeat_interleave(self.traj_num, dim=0)
        goal_w             = goal_w.repeat_interleave(self.traj_num, dim=0)

        end_pos_w, end_vel_w, end_acc_w = state_body2world(
            pos_expanded, rot_expanded,
            endstate_flat[:, 0:3],
            endstate_flat[:, 3:6],
            endstate_flat[:, 6:9]
        )
        end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)

        smooth_cost, safety_cost, goal_cost, acc_cost = self.loss_fn(
            start_state_w, end_state_w, goal_w, map_id,
            dynamic_obstacles, dynamic_cylinders,
        )

        # split static / dynamic safety (each B*traj_num, already weighted)
        static_safety  = self.loss_fn.last_static_safety_cost
        dynamic_safety = self.loss_fn.last_dynamic_safety_cost

        # Anchor-direction guidance (replaces global-goal guidance on endstate): each candidate extends toward its own lattice node.
        # With an omnidirectional lattice, global-goal guidance forces rear-facing anchors to zero length and, via the shared
        # head, drags down front anchors (radio channel tanh saturates at -1, the whole net collapses). Per-anchor directions
        # keep all 36 anchor targets consistent; goal-seeking is left to score-based selection.
        anchor_goal_w = pos_expanded + torch.bmm(
            rot_expanded, self.anchor_nodes_grid.repeat(self.batch_size, 1).unsqueeze(-1)
        ).squeeze(-1)
        anchor_goal_cost = self.loss_fn.goal_weight * self.loss_fn.goal_loss(
            start_state_w.permute(0, 2, 1), end_state_w.permute(0, 2, 1), anchor_goal_w)

        # endstate loss: smooth + static safety + anchor guidance + acc (no global goal).
        endstate_traj_cost = smooth_cost + static_safety + anchor_goal_cost + acc_cost
        trajectory_loss = endstate_traj_cost.mean()

        # static_score regresses the full static cost including the global goal, used as the selection criterion;
        # dynamic_score regresses log1p-compressed dynamic safety (regressing the heavy-tailed exp barrier directly
        # pushes forward/sideways score gaps below the resolution floor). Inference argmin(static+λ·dynamic) then moves toward the goal.
        static_score_target = smooth_cost + static_safety + goal_cost + acc_cost
        static_score_loss  = F.smooth_l1_loss(static_score_flat,  static_score_target.detach())
        dynamic_score_loss = F.smooth_l1_loss(dynamic_score_flat, torch.log1p(dynamic_safety).detach())

        return (
            trajectory_loss,
            static_score_loss,
            dynamic_score_loss,
            smooth_cost.mean(),
            safety_cost.mean(),
            anchor_goal_cost.mean(),   # Detail/GoalLoss tracks endstate anchor guidance (decreasing = no collapse)
            acc_cost.mean(),
        )

    def save_model(self):
        if hasattr(self, "epoch_i"):
            self.progress_log.console.log("Saving model...")
            policy_path = self.tensorboard_path + "/epoch{}.pth".format(self.epoch_i + 1, 0)
            torch.save(self.policy.state_dict(), policy_path)
            atexit.unregister(self._exit_func)

    def get_next_log_path(self, base_path):
        nums = [int(name.split("_")[1])
                for name in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, name)) and name.startswith("run_") and name.split("_")[1].isdigit()]
        next_n = max(nums, default=-1) + 1
        next_path = os.path.join(base_path, f"run_{next_n}")
        os.makedirs(next_path, exist_ok=False)
        print("record tensorboard log to ", next_path)
        return next_path
