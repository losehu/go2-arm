"""Dense clearance feedback during measured swing, without a clock or target phase."""
import torch

SWING_CLEARANCE = {
    "revision": "measured_swing_dense_clearance_v8",
    "target_peak_min_m": .025,
    "target_peak_max_m": .035,
    "speed_for_max_target_mps": .8,
    "endpoint_ramp_s": .06,
    "speed_scale_mps": .15,
    "shortfall_weight_per_s": 2.,
    "high_lift_start_m": .06,
    "high_lift_scale_m": .03,
    "high_lift_weight_per_s": .5,
}


class SwingClearance:
    def __init__(self, n, device):
        self.airtime = torch.zeros(n, 4, device=device)
        self.seen = torch.zeros(n, 4, dtype=torch.bool, device=device)
        self.supported = torch.zeros_like(self.seen)

    def reset(self, ids):
        self.airtime[ids] = 0; self.seen[ids] = False; self.supported[ids] = False

    def step(self, contact, height, foot_speed, commands, moving, dt):
        cfg = SWING_CLEARANCE
        self.seen |= contact
        swing = ~contact & self.seen & moving[:, None]
        self.airtime.copy_(torch.where(swing, self.airtime + dt, 0.).clamp_max(2.))
        self.supported.copy_(swing & (self.supported | contact.any(-1, keepdim=True)))
        demand = commands[:, :2].norm(dim=-1)
        target = cfg["target_peak_min_m"] + (cfg["target_peak_max_m"] - cfg["target_peak_min_m"]) * (
            demand / cfg["speed_for_max_target_mps"]).clamp(0., 1.)
        # Speed weighting leaves slow liftoff/landing transitions inexpensive.
        # The airtime ramp also suppresses one-frame contact chatter.
        weight = (self.airtime / cfg["endpoint_ramp_s"]).clamp(0., 1.)
        weight *= (foot_speed[..., :2].norm(dim=-1) / cfg["speed_scale_mps"]).clamp(0., 1.)
        weight *= swing & self.supported
        shortfall = ((target[:, None] - height) / target[:, None]).clamp(0., 1.).square()
        excessive = ((height - cfg["high_lift_start_m"]) / cfg["high_lift_scale_m"]).clamp(0., 2.).square()
        cost = (cfg["shortfall_weight_per_s"] * weight * shortfall +
                cfg["high_lift_weight_per_s"] * weight * excessive).mean(-1)
        return cost, weight, target
