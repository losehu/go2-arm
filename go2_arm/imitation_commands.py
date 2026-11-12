"""Expert-frame-centered commands, then a continuous mixture of full-envelope commands."""
import torch
from base_commands import apply_velocity_deadzone

IMITATION_COMMANDS = {
    "revision": "expert_frames_to_omni_mixture_v8",
    "use_imitation_commands": True,
    "randomize_imitation_commands": True,
    "resample_s": 5.,
    "fixed_jitter": [.15, .10, .15],
    "full_limits": [1., .6, .8],
    "expert_only_updates": 1000,
    "omni_ramp_updates": 8000,
    "maximum_omni_fraction": .8,
    "omni_stand_fraction": .1,
    "omni_single_axis_fraction": .5,
    "retains_expert_fraction": .2,
}


def omni_fraction(iteration):
    cfg = IMITATION_COMMANDS
    return cfg["maximum_omni_fraction"] * min(1., max(0., (
        iteration - cfg["expert_only_updates"]) / cfg["omni_ramp_updates"]))


class ImitationCommands:
    def __init__(self, n, device):
        self.device = device
        self.clip = torch.zeros(n, dtype=torch.long, device=device)
        self.frame = torch.zeros_like(self.clip)
        self.age = torch.zeros(n, device=device)
        self.fraction = 0.
        self.reference = None

    def configure(self, frames, lengths, signatures):
        # Equal mass for each demonstrated direction, then equal mass per clip.
        self.reference = frames
        self.lengths = lengths
        self.offsets = lengths.cumsum(0) - lengths
        _, inverse, counts = torch.unique(signatures, dim=0, return_inverse=True, return_counts=True)
        weights = counts[inverse].float().reciprocal()
        self.weights = weights / weights.sum()

    def reset(self, ids):
        if self.reference is None:
            return
        self.clip[ids] = torch.multinomial(self.weights, len(ids), replacement=True)
        self.frame[ids] = (torch.rand(len(ids), device=self.device) * self.lengths[self.clip[ids]]).long()
        self.age[ids] = 0

    def advance(self, dt):
        if self.reference is not None:
            self.age += dt

    def sample(self, ids, dt):
        if self.reference is None:
            raise RuntimeError("Expert command bank is not configured")
        cfg = IMITATION_COMMANDS
        index = (self.frame[ids] + (self.age[ids] / dt).long()) % self.lengths[self.clip[ids]]
        reference = self.reference[self.offsets[self.clip[ids]] + index]
        limits = reference.new_tensor(cfg["full_limits"])
        jitter = reference.new_tensor(cfg["fixed_jitter"])
        expert = reference + (2 * torch.rand_like(reference) - 1) * jitter
        # The sampling envelope is fixed from the first update. Only the mixture changes.
        free = (2 * torch.rand_like(reference) - 1) * limits
        axis = torch.randint(0, 3, (len(ids),), device=self.device)
        single = torch.rand(len(ids), device=self.device) < cfg["omni_single_axis_fraction"]
        free[single] *= torch.nn.functional.one_hot(axis[single], 3)
        free[torch.rand(len(ids), device=self.device) < cfg["omni_stand_fraction"]] = 0
        use_free = torch.rand(len(ids), device=self.device) < self.fraction
        command = torch.where(use_free[:, None], free, expert).clamp(-limits, limits)
        return apply_velocity_deadzone(command), reference, use_free
