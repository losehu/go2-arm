"""One-sided penalty for excessive hip abduction in FL, FR, RL, RR order."""
import math

import torch


HIP_SPLAY_REWARD = {
    "revision": "outward_hip_abduction_v1",
    "straight_limit_deg": 12.0,
    "lateral_and_turn_allowance_deg": 6.0,
    "excess_scale_deg": 8.0,
    "weight_per_second": 3.0,
}


def hip_splay(hip_position, commands):
    """Return cost, signed outward angles and per-leg excess indicators."""
    outward = hip_position * hip_position.new_tensor([1.0, -1.0, 1.0, -1.0])
    lateral_or_turn = (commands[:, 1:3] / commands.new_tensor([0.6, 0.8])).norm(dim=-1).clamp(max=1.0)
    limit = math.radians(HIP_SPLAY_REWARD["straight_limit_deg"]) + math.radians(
        HIP_SPLAY_REWARD["lateral_and_turn_allowance_deg"]
    ) * lateral_or_turn
    excess = (outward - limit[:, None]) / math.radians(HIP_SPLAY_REWARD["excess_scale_deg"])
    normalized = excess.clamp(min=0.0, max=3.0)
    cost = HIP_SPLAY_REWARD["weight_per_second"] * (0.5 * normalized + normalized.square()).mean(dim=-1)
    return cost, torch.rad2deg(outward), excess > 0.0
