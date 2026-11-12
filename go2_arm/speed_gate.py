"""Require completed high-speed episodes in each signed axis before attitude expansion."""
import math
import torch

SPEED_GATE = {
    "revision": "signed_full_speed_episodes_v8",
    "labels": ["vx+", "vx-", "vy+", "vy-", "wz+", "wz-"],
    "full_limits": [1., .6, .8],
    "minimum_command_fraction": .8,
    "minimum_valid_steps_per_episode": 100,
    "minimum_episodes_per_direction": 64,
    "ema_episode_scale": 128.,
    "success_threshold": .7,
}


class FullSpeedGate:
    def __init__(self, n, device):
        # count, three normalized squared errors, gain, pitch/roll squared error
        self.sums = torch.zeros(n, 6, 7, device=device)
        self.episodes = torch.zeros(6, dtype=torch.long, device=device)
        self.ema = torch.zeros(6, device=device)

    def clear(self):
        self.sums.zero_(); self.episodes.zero_(); self.ema.zero_()

    def record(self, command, normalized, gain, pitch_error, roll_error, valid):
        axis = command.abs().argmax(-1)
        active = command.ne(0.).sum(-1) == 1
        magnitude = command.gather(1, axis[:, None])[:, 0]
        caps = command.new_tensor(SPEED_GATE["full_limits"])[axis]
        eligible = valid & active & (magnitude.abs() >= caps * SPEED_GATE["minimum_command_fraction"])
        signed_axis = axis * 2 + (magnitude < 0).long()
        values = torch.cat((torch.ones_like(magnitude[:, None]), normalized.square(),
                            gain.gather(1, axis[:, None]), pitch_error.square()[:, None],
                            roll_error.square()[:, None]), -1)
        rows = torch.arange(len(command), device=command.device)
        self.sums[rows, signed_axis] += values * eligible[:, None]

    def finish(self, ids, physically_complete):
        values = self.sums[ids]
        n = values[..., 0]
        eligible = n >= SPEED_GATE["minimum_valid_steps_per_episode"]
        mean = values[..., 1:] / n.clamp_min(1)[..., None]
        passed = physically_complete[:, None] & (mean[..., :3] <= 1.).all(-1)
        passed &= (mean[..., 3] >= .7) & (mean[..., 3] <= 1.3)
        passed &= (mean[..., 4:] < math.radians(3.) ** 2).all(-1)
        count = eligible.sum(0)
        batch = (passed & eligible).sum(0) / count.clamp_min(1)
        decay = torch.exp(-count / SPEED_GATE["ema_episode_scale"])
        self.ema.copy_(decay * self.ema + (1 - decay) * batch)
        self.episodes += count
        self.sums[ids] = 0

    def ready(self):
        return bool(((self.episodes >= SPEED_GATE["minimum_episodes_per_direction"]) &
                     (self.ema >= SPEED_GATE["success_threshold"])).all())

    def metrics(self):
        result = {"full_speed_gate_ready": self.ready()}
        for label, n, rate in zip(SPEED_GATE["labels"], self.episodes.tolist(), self.ema.tolist()):
            result[f"full_speed_{label}_episodes"] = n
            result[f"full_speed_{label}_success_ema"] = rate
        return result
