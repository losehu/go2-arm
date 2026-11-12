"""Two-sided soft hip band, expressed relative to gravity during body roll."""
import math

import torch


HIP_SPLAY_REWARD = {
    "revision": "two_sided_gravity_aligned_hip_v10",
    "straight_limit_deg": 12.0,
    "lateral_and_turn_allowance_deg": 6.0,
    "excess_scale_deg": 8.0,
    "weight_per_second": 3.0,
    "straight_inward_limit_deg": -2.0,
    "inward_lateral_allowance_deg": 3.0,
    "inward_excess_scale_deg": 6.0,
    "inward_weight_per_second": 3.0,
    "hip_x_m": .1934,
    "leg_lateral_speed_scale_mps": .8,
}


def inward_hip_cost(aligned_deg, commands):
    cfg = HIP_SPLAY_REWARD
    x = commands.new_tensor([cfg['hip_x_m'], cfg['hip_x_m'], -cfg['hip_x_m'], -cfg['hip_x_m']])
    demand = ((commands[:, 1:2] + commands[:, 2:3] * x) /
              cfg['leg_lateral_speed_scale_mps']).abs().clamp(max=1.)
    lower = cfg['straight_inward_limit_deg'] - cfg['inward_lateral_allowance_deg'] * demand
    inward = ((lower - aligned_deg) / cfg['inward_excess_scale_deg']).clamp(0., 2.)
    return cfg['inward_weight_per_second'] * inward.square().mean(-1), inward > 0.


def hip_splay(hip_position, commands, body_roll=None, include_inward=True):
    """Return cost, raw hip angles, outer/inner flags and gravity-aligned angles."""
    cfg = HIP_SPLAY_REWARD
    side = hip_position.new_tensor([1., -1., 1., -1.])
    outward = hip_position * side
    aligned = outward if body_roll is None else (hip_position + body_roll[:, None]) * side
    x = commands.new_tensor([cfg['hip_x_m'], cfg['hip_x_m'], -cfg['hip_x_m'], -cfg['hip_x_m']])
    lateral_or_turn = ((commands[:, 1:2] + commands[:, 2:3] * x) /
                       cfg['leg_lateral_speed_scale_mps']).abs().clamp(max=1.)
    limit = math.radians(HIP_SPLAY_REWARD["straight_limit_deg"]) + math.radians(
        HIP_SPLAY_REWARD["lateral_and_turn_allowance_deg"]
    ) * lateral_or_turn
    excess = (aligned - limit) / math.radians(HIP_SPLAY_REWARD["excess_scale_deg"])
    normalized = excess.clamp(min=0.0, max=3.0)
    cost = HIP_SPLAY_REWARD["weight_per_second"] * (0.5 * normalized + normalized.square()).mean(dim=-1)
    inward_cost, inward = inward_hip_cost(torch.rad2deg(aligned), commands)
    if include_inward:
        cost += inward_cost
    return cost, torch.rad2deg(outward), excess > 0., inward, torch.rad2deg(aligned)
