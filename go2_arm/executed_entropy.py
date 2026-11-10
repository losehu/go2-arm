"""Exploration regularizer for the existing action clamp, without changing PPO likelihoods.

Raw actions still follow the same Gaussian and are stored by PPO before the
environment clamps them.  Only the entropy bonus accounts for that clamp.
The executed law has a continuous density inside [-1, 1] and two endpoint
atoms.  Its entropy is measured against Lebesgue measure plus those atoms;
it is bounded above by log(4) per action, unlike raw Gaussian entropy.
"""
import math
import torch
from rsl_rl.modules.distribution import GaussianDistribution


def clipped_normal_entropy(mean, std, low=-1.0, high=1.0):
    """Exact mixed-measure entropy of clamp(Normal(mean, std), low, high)."""
    a = (low - mean) / std
    b = (high - mean) / std
    left = torch.special.ndtr(a)
    right = torch.special.ndtr(-b)
    inside = (torch.special.ndtr(b) - left).clamp_min(0.0)
    phi_a = torch.exp(-0.5 * a.square()) / math.sqrt(2 * math.pi)
    phi_b = torch.exp(-0.5 * b.square()) / math.sqrt(2 * math.pi)
    # The clamp only stabilizes log(0); zero-mass atoms contribute zero.
    atoms = -left * left.clamp_min(1e-30).log() - right * right.clamp_min(1e-30).log()
    interior = inside * (std.log() + 0.5 * math.log(2 * math.pi * math.e))
    interior = interior + 0.5 * (a * phi_a - b * phi_b)
    return atoms + interior


class ExecutedEntropyGaussian(GaussianDistribution):
    """Same samples, raw log-probability, KL and deterministic actions as the parent."""

    @property
    def entropy(self):
        return clipped_normal_entropy(self.mean, self.std).sum(dim=-1)

    @property
    def raw_entropy(self):
        return self._distribution.entropy().sum(dim=-1)
