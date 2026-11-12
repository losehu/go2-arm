"""Penalize persistent under-use of a foot without imposing a gait clock."""
import math

import torch


FOOT_PARTICIPATION = {
    "revision": "per_foot_load_and_touchdown_v1",
    "load_time_constant_s": .35,
    "warmup_s": .70,
    "minimum_load_share": .12,
    "touchdown_grace_s": .80,
    "touchdown_penalty_ramp_s": .40,
    "load_weight_per_s": 6.,
    "touchdown_weight_per_s": 4.,
    "max_normal_force_n": 300.,
}


class FootParticipation:
    def __init__(self, num_envs, device):
        self.load = torch.zeros((num_envs, 4), device=device)
        self.since_touchdown = torch.zeros_like(self.load)
        self.active_time = torch.zeros(num_envs, device=device)

    def reset(self, ids):
        self.load[ids] = 0.
        self.since_touchdown[ids] = 0.
        self.active_time[ids] = 0.

    def step(self, normal_force, contact, touchdown, moving, dt):
        cfg = FOOT_PARTICIPATION
        active = moving[:, None]
        load = normal_force.clamp(0., cfg["max_normal_force_n"]) * contact
        alpha = -math.expm1(-dt / cfg["load_time_constant_s"])
        self.load.copy_(torch.where(active, self.load.lerp(load, alpha), 0.))
        self.active_time.copy_(torch.where(moving, self.active_time + dt, 0.))
        delay_cap = cfg["touchdown_grace_s"] + cfg["touchdown_penalty_ramp_s"]
        self.since_touchdown.copy_(torch.where(
            active & ~touchdown, (self.since_touchdown + dt).clamp_max(delay_cap), 0.
        ))
        shares = self.load / self.load.sum(dim=-1, keepdim=True).clamp_min(1.)
        deficit = (1. - shares / cfg["minimum_load_share"]).clamp_min(0.)
        overdue = ((self.since_touchdown - cfg["touchdown_grace_s"]) /
                   cfg["touchdown_penalty_ramp_s"]).clamp(0., 1.)
        ready = self.active_time >= cfg["warmup_s"]
        cost = cfg["load_weight_per_s"] * deficit.square().mean(dim=-1)
        cost += cfg["touchdown_weight_per_s"] * overdue.square().mean(dim=-1)
        return cost * ready, shares, ready
