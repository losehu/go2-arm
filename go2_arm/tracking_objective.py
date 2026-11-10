"""Per-axis tracking error tail; existing command tolerances stay unchanged."""
import torch

REVISION = 'tracking_log_tail_v1'
TAIL_WEIGHT = 0.5


def tracking_error_objective(normalized_error):
    planar = 4 * torch.exp(-0.5 * normalized_error[..., :2].square().sum(-1))
    yaw = 2 * torch.exp(-0.5 * normalized_error[..., 2].square())
    tail = TAIL_WEIGHT * torch.log1p(normalized_error.square()).sum(-1)
    return planar + yaw - tail
