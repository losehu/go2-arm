"""Separate physical-failure cost from time-integrated locomotion shaping."""
import math
import torch
from leg_action import LEG_ACTION_CONTROL

EPISODE_OBJECTIVE = {
    "revision": "nonnegative_shaping_physical_failure_v9",
    "stage1_shaping_floor": 0.,
    "physical_failure_cost_per_event": 5.,
    "timeout_failure_cost": 0.,
    "reference_initial_target_scale_rad": .35,
    "reference_initial_filter_alpha": .2,
    "reference_initial_action_std": .6,
}


def stage1_reward(raw_shaping, physically_terminated, positive_factor=1.):
    # All task, posture, slip, participation and clearance terms enter raw_shaping.
    # A terminal event is charged AFTER the floor and is not multiplied by dt.
    return raw_shaping.clamp_min(EPISODE_OBJECTIVE["stage1_shaping_floor"]) * positive_factor - (
        physically_terminated.to(raw_shaping.dtype) * EPISODE_OBJECTIVE["physical_failure_cost_per_event"])


def initial_leg_std(device):
    cfg = EPISODE_OBJECTIVE
    old_alpha = cfg["reference_initial_filter_alpha"]
    alpha = LEG_ACTION_CONTROL["leg_filter_alpha"]
    # Match the old filter's stationary target-noise variance, not the raw action variance.
    filter_ratio = math.sqrt((old_alpha / (2 - old_alpha)) / (alpha / (2 - alpha)))
    scales = torch.tensor(LEG_ACTION_CONTROL["scale_rad_hip_thigh_calf"] * 4, device=device)
    return cfg["reference_initial_action_std"] * cfg["reference_initial_target_scale_rad"] / scales * filter_ratio
