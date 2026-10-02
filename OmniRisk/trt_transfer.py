"""
    Convert the OmniRisk model to TensorRT
    prepare:
        1 pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
        2 git clone https://github.com/NVIDIA-AI-IOT/torch2trt
          cd torch2trt
          python setup.py install
"""

import os
import argparse
import time
import numpy as np
import torch
from torch2trt import torch2trt
from config.config import cfg
from inference_inputs import forward_with_contiguous_inputs
from policy.network import OmniRiskNetwork


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=60, help="epoch number")
    parser.add_argument("--dir", type=str, default='omnirisk_trt.pth', help="output file name")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = base_dir + "/saved/run_{}/epoch{}.pth".format(args.trial, args.epoch)

    print("Loading Network...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(weight, weights_only=True)
    policy = OmniRiskNetwork()
    policy.load_state_dict(state_dict)
    policy = policy.to(device)
    policy.eval()

    # The inputs should be consistent with training (4-in/3-out two-head net)
    H, W = cfg["image_height"], cfg["image_width"]  # 96, 384
    depth = np.zeros(shape=[1, 1, H, W], dtype=np.float32)
    mask = np.zeros(shape=[1, 1, H, W], dtype=np.float32)
    vel_img = np.zeros(shape=[1, 3, H, W], dtype=np.float32)
    obs = np.zeros(shape=[1, 9, cfg["vertical_num"], cfg["horizon_num"]], dtype=np.float32)
    depth_in = torch.from_numpy(depth).to(device)
    mask_in = torch.from_numpy(mask).to(device)
    vel_in = torch.from_numpy(vel_img).to(device)
    obs_in = torch.from_numpy(obs).to(device)

    print("TensorRT Transfer...")
    model_trt = torch2trt(policy, [depth_in, mask_in, vel_in, obs_in], fp16_mode=True)


    print("Evaluation...")
    # Warm Up...
    forward_with_contiguous_inputs(model_trt, depth_in, mask_in, vel_in, obs_in)
    forward_with_contiguous_inputs(policy, depth_in, mask_in, vel_in, obs_in)
    torch.cuda.synchronize()

    # Validate with non-zero deployment-like values. Zero C-contiguous inputs do
    # not expose torch2trt 0.5.0's temporary contiguous-buffer lifetime bug.
    torch.manual_seed(0)
    depth_in.uniform_(0.05, 0.95)
    mask_in.bernoulli_(0.1)
    vel_in.normal_()
    vel_in.mul_(mask_in)
    obs_in.normal_(std=0.1)

    # PyTorch Latency
    torch_start = time.time()
    endstate, static, dynamic = forward_with_contiguous_inputs(
        policy, depth_in, mask_in, vel_in, obs_in)
    torch.cuda.synchronize()
    torch_end = time.time()

    # TensorRT Latency
    trt_start = time.time()
    endstate_trt, static_trt, dynamic_trt = forward_with_contiguous_inputs(
        model_trt, depth_in, mask_in, vel_in, obs_in)
    torch.cuda.synchronize()
    trt_end = time.time()

    noncontiguous_inputs = tuple(
        value.transpose(-2, -1).contiguous().transpose(-2, -1)
        for value in (depth_in, mask_in, vel_in, obs_in))
    noncontiguous_outputs = forward_with_contiguous_inputs(
        model_trt, *noncontiguous_inputs)
    layout_error = max(
        torch.max(torch.abs(reference - candidate)).item()
        for reference, candidate in zip(
            (endstate_trt, static_trt, dynamic_trt), noncontiguous_outputs))

    zero_mask = torch.zeros_like(mask_in)
    zero_velocity = torch.zeros_like(vel_in)
    endstate_no_mask, static_no_mask, _ = forward_with_contiguous_inputs(
        model_trt, depth_in, zero_mask, zero_velocity, obs_in)
    static_head_mask_error = max(
        torch.max(torch.abs(endstate_trt - endstate_no_mask)).item(),
        torch.max(torch.abs(static_trt - static_no_mask)).item())
    endstate_zero_depth, static_zero_depth, _ = forward_with_contiguous_inputs(
        model_trt, torch.zeros_like(depth_in), mask_in, vel_in, obs_in)
    depth_sensitivity = max(
        torch.mean(torch.abs(endstate_trt - endstate_zero_depth)).item(),
        torch.mean(torch.abs(static_trt - static_zero_depth)).item())
    if layout_error > 1e-6:
        raise RuntimeError(f"contiguous/non-contiguous output mismatch: {layout_error:.6g}")
    if static_head_mask_error > 1e-6:
        raise RuntimeError(f"mask changed the static head: {static_head_mask_error:.6g}")
    if depth_sensitivity <= 1e-4:
        raise RuntimeError(f"TensorRT depth sensitivity is too small: {depth_sensitivity:.6g}")

    # Transfer Error
    endstate_error = torch.mean(torch.abs(endstate - endstate_trt))
    static_error = torch.mean(torch.abs(static - static_trt))
    dynamic_error = torch.mean(torch.abs(dynamic - dynamic_trt))
    max_transfer_error = max(
        endstate_error.item(), static_error.item(), dynamic_error.item())
    if max_transfer_error > 0.1:
        raise RuntimeError(f"PyTorch/TensorRT MAE exceeds 0.1: {max_transfer_error:.6g}")

    print(f"Torch Latency: {1000 * (torch_end - torch_start):.3f} ms, "
          f"TensorRT Latency: {1000 * (trt_end - trt_start):.3f} ms, "
          f"Transfer Endstate Error: {endstate_error.item():.6f}, "
          f"Transfer Static Score Error: {static_error.item():.6f}, "
          f"Transfer Dynamic Score Error: {dynamic_error.item():.6f}, "
          f"Layout Max Error: {layout_error:.6f}, "
          f"Static-head Mask Error: {static_head_mask_error:.6f}, "
          f"Depth Sensitivity: {depth_sensitivity:.6f}")
    torch.save(model_trt.state_dict(), args.dir)
