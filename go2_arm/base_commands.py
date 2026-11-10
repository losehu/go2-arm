"""Per-axis deadzone shared by training, inference, and AMP command matching."""
import torch

VELOCITY_DEADZONE = .10  # m/s for vx and vy, rad/s for yaw.


def apply_velocity_deadzone(command):
    return torch.where(command.abs() > VELOCITY_DEADZONE, command, 0.)


def advance_base_command(previous, target, max_step):
    target = apply_velocity_deadzone(target)
    slew = previous + (target-previous).clamp(-max_step, max_step)
    # A zero target takes effect immediately. Keep the hidden slew state
    # separate so an accelerating command can move across the deadzone.
    slew = torch.where(target == 0., 0., slew)
    return slew, apply_velocity_deadzone(slew)
