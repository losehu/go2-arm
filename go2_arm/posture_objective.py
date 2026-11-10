"""Attitude precision shaping with informative tails away from the target.

The acceptance tolerance stays three degrees. The coefficient is a reward
choice to test, not a claim of proven convergence or whole-body acceptance.
"""
import math
import torch

REVISION = "posture_log_tail_v1"
TAIL_WEIGHT = 0.5

def posture_objective(pitch_error, roll_error):
    squared_error = (pitch_error.square() + roll_error.square()) / math.radians(3.0)**2
    return torch.exp(-squared_error) - TAIL_WEIGHT * torch.log1p(squared_error)
