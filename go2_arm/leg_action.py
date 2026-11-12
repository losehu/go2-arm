"""Shared leg target envelope and filter for training and policy playback."""
import torch

LEG_ACTION_CONTROL = {
    "revision": "per_joint_envelope_filter_v8",
    "scale_rad_hip_thigh_calf": [.35, .55, .65],
    "leg_filter_alpha": .35,
    "other_filter_alpha": .2,
    "clamp_to_soft_joint_limits": True,
    "kp": 45.,
    "kd": 1.5,
}


def filter_action(filtered, action):
    alpha = action.new_full((action.shape[-1],), LEG_ACTION_CONTROL["other_filter_alpha"])
    alpha[:12] = LEG_ACTION_CONTROL["leg_filter_alpha"]
    return filtered + alpha * (action - filtered)


def leg_targets(nominal, filtered, limits):
    scale = filtered.new_tensor(LEG_ACTION_CONTROL["scale_rad_hip_thigh_calf"] * 4)
    return (nominal + scale * filtered[..., :12]).clamp(limits[..., 0], limits[..., 1])
