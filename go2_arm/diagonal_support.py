"""Shape the order of measured support pairs without prescribing gait phase."""
import torch


DIAGONAL_SUPPORT = {
    "revision": "measured_diagonal_time_normalized_v2",
    "confirm_support_s": .04,
    "minimum_switch_s": .12,
    "switch_reward_per_second": .16,
    "maximum_switch_time_credit_s": .5,
    "same_side_grace_s": .06,
    "same_side_ramp_s": .12,
    "same_side_cost_per_s": 3.,
    "no_switch_grace_s": .75,
    "no_switch_ramp_s": .45,
    "no_switch_cost_per_s": 3.,
}


def support_pairs(contact):
    """Exact two-foot supports in FL, FR, RL, RR order, for Torch or NumPy."""
    fl, fr, rl, rr = (contact[..., i] for i in range(4))
    diagonal_a = fl & rr & ~fr & ~rl
    diagonal_b = fr & rl & ~fl & ~rr
    left = fl & rl & ~fr & ~rr
    right = fr & rr & ~fl & ~rl
    return diagonal_a, diagonal_b, left, right


class DiagonalSupport:
    def __init__(self, num_envs, device):
        self.last_pair = torch.zeros(num_envs, dtype=torch.int8, device=device)
        self.candidate = torch.zeros_like(self.last_pair)
        self.dwell = torch.zeros(num_envs, device=device)
        self.since_pair = torch.zeros_like(self.dwell)
        self.side_time = torch.zeros_like(self.dwell)

    def reset(self, ids):
        for state in (self.last_pair, self.candidate, self.dwell, self.since_pair, self.side_time):
            state[ids] = 0

    def step(self, contact, moving, dt):
        cfg = DIAGONAL_SUPPORT
        a, b, left, right = support_pairs(contact)
        pair = a.to(self.candidate.dtype) - b.to(self.candidate.dtype)
        pair = torch.where(moving, pair, 0)
        same_side = moving & (left | right)

        self.dwell.copy_(torch.where(
            pair != 0, torch.where(pair == self.candidate, self.dwell + dt, dt), 0.
        ).clamp_max(cfg["confirm_support_s"]))
        self.candidate.copy_(pair)
        elapsed = self.since_pair + dt
        confirmed = (pair != 0) & (self.dwell >= cfg["confirm_support_s"])
        first = confirmed & (self.last_pair == 0)
        switch = confirmed & (self.last_pair != 0) & (pair != self.last_pair) & (elapsed >= cfg["minimum_switch_s"])
        accepted = first | switch
        self.last_pair.copy_(torch.where(moving, torch.where(accepted, pair, self.last_pair), 0))
        self.since_pair.copy_(torch.where(moving & ~accepted, elapsed, 0.).clamp_max(
            cfg["no_switch_grace_s"] + cfg["no_switch_ramp_s"]))
        self.side_time.copy_(torch.where(same_side, self.side_time + dt, 0.).clamp_max(
            cfg["same_side_grace_s"] + cfg["same_side_ramp_s"]))

        side_excess = ((self.side_time - cfg["same_side_grace_s"]) / cfg["same_side_ramp_s"]).clamp(0., 1.)
        stale = ((self.since_pair - cfg["no_switch_grace_s"]) / cfg["no_switch_ramp_s"]).clamp(0., 1.)
        cost = cfg["same_side_cost_per_s"] * side_excess.square() + cfg["no_switch_cost_per_s"] * stale.square()
        # Accepted opposite pairs earn credit for elapsed time, not a fixed payment
        # for each event. Faster switching therefore does not increase reward rate.
        reward = cfg["switch_reward_per_second"] * switch * elapsed.clamp_max(
            cfg["maximum_switch_time_credit_s"]) - dt * cost
        return reward, switch, same_side, cost
