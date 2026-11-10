"""Soft cost for sustained crouching below the locomotion height band."""
import torch

BASE_HEIGHT_SPEC = {
    "revision": "one_sided_low_base_huber_v1",
    "lower_band_m": .285,
    "huber_transition_m": .015,
    "weight": 2.,
}


def low_base_cost(height):
    deficit = (BASE_HEIGHT_SPEC["lower_band_m"] - height).clamp_min(0.)
    scaled = deficit / BASE_HEIGHT_SPEC["huber_transition_m"]
    return BASE_HEIGHT_SPEC["weight"] * torch.where(scaled <= 1., scaled.square(), 2.*scaled - 1.)
