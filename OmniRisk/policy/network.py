"""
OmniRisk Network
forward, prediction, pre-processing, post-processing
"""

import torch
from torch import nn
import numpy as np
from policy.models.backbone import StaticBackbone, DynamicBackbone
from policy.models.head import ScoreHead
from policy.state_transform import *


class OmniRiskNetwork(nn.Module):

    def __init__(
            self,
            observation_dim=9,  # 9: v_xyz, a_xyz, goal_xyz
            output_dim=10,  # 10: x_pva, y_pva, z_pva, static_score
            hidden_state=64,
    ):
        super(OmniRiskNetwork, self).__init__()
        self.state_transform = StateTransform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Shared depth backbone feeds BOTH heads (gradient reaches it from each).
        self.image_backbone = StaticBackbone(hidden_state)
        # Dynamic-head-only encoder for mask(1)+velocity-image(3); never touches static head.
        self.dynamic_backbone = DynamicBackbone(hidden_state, in_channels=4)
        self.state_backbone = nn.Sequential()
        # Static head: depth feature + obs → endstate(9) + static score(1).
        self.static_head = ScoreHead(hidden_state + observation_dim, output_dim)
        # Dynamic head: depth feature + dynamic feature + obs → dynamic score(1).
        self.dynamic_head = ScoreHead(2 * hidden_state + observation_dim, 1)

    def forward(self, depth: torch.Tensor, mask: torch.Tensor,
                vel_img: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
            forward propagation of neural network (two-head)
            returns: endstate, static_score, dynamic_score
        """
        depth_feature = self.image_backbone(depth)
        dyn_feature = self.dynamic_backbone(torch.cat((mask, vel_img), 1))
        obs_feature = self.state_backbone(obs)

        # static head — depth only, so endstate is independent of mask/velocity
        static_in = torch.cat((obs_feature, depth_feature), 1)
        static_out = self.static_head(static_in)
        endstate = torch.tanh(static_out[:, :9])  # [batch, 9, vertical_num, horizon_num]
        static_score = torch.nn.functional.softplus(static_out[:, 9])  # [batch, V, H]

        # dynamic head — shares depth feature, adds mask+velocity feature
        dyn_in = torch.cat((obs_feature, depth_feature, dyn_feature), 1)
        dynamic_score = torch.nn.functional.softplus(self.dynamic_head(dyn_in)[:, 0])  # [batch, V, H]
        return endstate, static_score, dynamic_score

    def inference(self, depth: torch.Tensor, mask: torch.Tensor,
                  vel_img: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
            For network training:
            (1) normalize the input state and transform to primitive frame
            (2) forward propagation
            (3) convert the prediction to endstate in body frame.
            obs: current state in the body frame.
            return: (endstate in body frame, static_score, dynamic_score)
        """
        obs = self.state_transform.normalize_obs(obs)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, static_score, dynamic_score = self.forward(depth, mask, vel_img, obs)
        endstate = self.state_transform.pred_to_endstate(endstate_pred)
        return endstate, static_score, dynamic_score

    def print_grad(self, grad):
        print("grad of hook: ", grad)
