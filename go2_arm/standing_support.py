"""Measured zero-command support, with a settling ramp and no gait phase."""
import math
import torch

STANDING_SUPPORT = {
    'revision': 'zero_command_clearance_contact_load_v11',
    'settle_grace_s': .25,
    'settle_ramp_s': .50,
    'load_time_constant_s': .20,
    'clearance_deadband_m': .003,
    'clearance_scale_m': .03,
    'clearance_weight_per_second': 2.,
    'contact_force_target_n': 5.,
    'contact_weight_per_second': 2.,
    'minimum_normal_load_share': .12,
    'load_share_weight_per_second': 2.,
    'application': 'bounded discount of positive shaping; physical failure remains separate',
}


class StandingSupport:
    def __init__(self, n, device):
        self.zero_age = torch.zeros(n, device=device)
        self.load = torch.zeros(n, 4, device=device)

    def reset(self, ids):
        self.zero_age[ids] = 0.
        self.load[ids] = 0.

    def step(self, commands, clearance, normal_force, dt):
        cfg = STANDING_SUPPORT
        stopped = commands[:, :3].eq(0.).all(-1)
        self.zero_age.copy_(torch.where(stopped, self.zero_age + dt, 0.))
        # Include only true near-ground support. Tiny spurious distant forces
        # must not make a raised foot count as planted or load bearing.
        near = clearance < .020
        force = normal_force.clamp_min(0.) * near
        alpha = -math.expm1(-dt / cfg['load_time_constant_s'])
        self.load.copy_(torch.where(stopped[:, None], self.load.lerp(force, alpha), 0.))
        shares = self.load / self.load.sum(-1, keepdim=True).clamp_min(1.)
        blend = ((self.zero_age - cfg['settle_grace_s']) / cfg['settle_ramp_s']).clamp(0., 1.)
        height = ((clearance - cfg['clearance_deadband_m']) /
                  cfg['clearance_scale_m']).clamp(0., 3.)
        missing = (1. - force / cfg['contact_force_target_n']).clamp(0., 1.)
        underload = (1. - shares / cfg['minimum_normal_load_share']).clamp(0., 1.)
        cost = blend * (
            cfg['clearance_weight_per_second'] * height.square().mean(-1) +
            cfg['contact_weight_per_second'] * missing.square().mean(-1) +
            cfg['load_share_weight_per_second'] * underload.square().mean(-1))
        ready = self.zero_age >= cfg['settle_grace_s'] + cfg['settle_ramp_s']
        contact = (force > 1.) & near
        return cost, ready, contact, shares
