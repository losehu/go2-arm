"""Bounded touchdown reward based on measured swing time and clearance."""
import torch

AIR_TIME_REWARD = {
    "revision": "measured_swing_touchdown_v2",
    "minimum_swing_s": .08,
    "full_swing_s": .18,
    "minimum_clearance_m": .015,
    "full_clearance_m": .04,
    "per_foot_max": .009,
    "per_step_max": .018,
}


class AirTimeReward:
    def __init__(self, num_envs, device):
        self.time = torch.zeros((num_envs, 4), device=device)
        self.height = torch.zeros_like(self.time)
        self.seen_contact = torch.zeros_like(self.time, dtype=torch.bool)
        self.had_support = torch.zeros_like(self.seen_contact)

    def reset(self, ids):
        self.time[ids] = 0
        self.height[ids] = 0
        self.seen_contact[ids] = False
        self.had_support[ids] = False

    def step(self, contact, clearance, moving, dt):
        cfg = AIR_TIME_REWARD
        active = moving[:, None]
        touchdown = contact & self.seen_contact & self.had_support & active
        duration = ((self.time - cfg["minimum_swing_s"]) /
                    (cfg["full_swing_s"] - cfg["minimum_swing_s"])).clamp(0., 1.)
        lift = ((self.height - cfg["minimum_clearance_m"]) /
                (cfg["full_clearance_m"] - cfg["minimum_clearance_m"])).clamp(0., 1.)
        qualified_touchdown = touchdown & (duration > 0.) & (lift > 0.)
        bonus = (cfg["per_foot_max"] * (duration * lift * touchdown).sum(dim=-1)).clamp_max(cfg["per_step_max"])
        self.seen_contact |= contact
        swing = active & ~contact & self.seen_contact
        self.time.copy_(torch.where(swing, (self.time + dt).clamp_max(cfg["full_swing_s"]), 0.))
        self.height.copy_(torch.where(swing, torch.maximum(self.height, clearance), 0.))
        self.had_support.copy_(swing & (self.had_support | contact.any(dim=-1, keepdim=True)))
        return bonus, qualified_touchdown
