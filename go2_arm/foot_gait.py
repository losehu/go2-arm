"""Foot collision-sphere kinematics and phase/contact/clearance gait shaping."""
import torch


COORDINATED_TROT = {
    "revision": "coordinated_trot_contact_v2",
    "frequency_hz": [5.0 / 3.0, 2.65],
    "cadence_ramp_normalized_translation": [0.35, 0.80],
    "translation_scale_mps": [1.0, 0.60],
    "stance_fraction": 0.60,
    "swing_peak_m": 0.055,
    "contact_height_m": 0.020,
    "slip_scale_mps": 0.20,
    "slip_cost_weight": 4.0,
    "paired_contact_weight": 4.0,
    "clearance_cost_weight": 6.0,
}


def foot_centers(link_position, link_quaternion, com_position, com_velocity, angular_velocity):
    # URDF collision spheres on FL/FR/RL/RR calf: xyz=(-.002,0,-.213), r=.022.
    local = torch.zeros_like(link_position)
    local[..., 0] = -.002
    local[..., 2] = -.213
    qv = link_quaternion[..., 1:]
    cross = 2. * torch.linalg.cross(qv, local, dim=-1)
    rotated = local + link_quaternion[..., :1]*cross + torch.linalg.cross(qv, cross, dim=-1)
    position = link_position + rotated
    velocity = com_velocity + torch.linalg.cross(angular_velocity, position-com_position, dim=-1)
    return position, velocity


def gait_terms(phase, walk_blend, foot_force, clearance, velocity):
    leg_phase = (phase[:, None] + phase.new_tensor([0., .5, .5, 0.])) % 1.
    swing = (leg_phase >= .5).to(phase.dtype) * walk_blend[:, None]
    stance = 1. - swing
    # A smooth 4 cm arc, measured above the bottom of the collision sphere.
    target_clearance = .04 * torch.sin(torch.pi*((leg_phase-.5)*2).clamp(0., 1.)) * swing
    no_contact = torch.exp(-foot_force.square()/100.)
    support = (1.-no_contact)*torch.exp(-velocity.square().sum(-1)/.04)
    swing_score = no_contact*torch.exp(-(clearance-target_clearance).square()/.0004)
    score = (stance*support+swing*swing_score).mean(-1)
    clearance_cost = (swing*((target_clearance-clearance).clamp_min(0.)/.04).square()).mean(-1)
    return score, clearance_cost, target_clearance, swing


def trot_frequency(commands):
    low, high = COORDINATED_TROT["frequency_hz"]
    start, stop = COORDINATED_TROT["cadence_ramp_normalized_translation"]
    scale = commands.new_tensor(COORDINATED_TROT["translation_scale_mps"])
    demand = (commands[:, :2] / scale).norm(dim=-1)
    blend = ((demand - start) / (stop - start)).clamp(0.0, 1.0)
    return low + (high - low) * blend


def trot_phase(phase):
    leg_phase = (phase[:, None] + phase.new_tensor([0.0, 0.5, 0.5, 0.0])) % 1.0
    duty = COORDINATED_TROT["stance_fraction"]
    swing_progress = ((leg_phase - duty) / (1.0 - duty)).clamp(0.0, 1.0)
    return leg_phase, swing_progress


def trot_placement_time(phase, commands):
    leg_phase, swing_progress = trot_phase(phase)
    duty = COORDINATED_TROT["stance_fraction"]
    swing_ease = swing_progress.square() * (3.0 - 2.0 * swing_progress)
    # During support the foot stays approximately stationary in world coordinates.
    stride_phase = torch.where(leg_phase < duty, duty / 2.0 - leg_phase,
                               duty * (swing_ease - 0.5))
    return stride_phase / trot_frequency(commands)[:, None]


def planted_foot_cost(foot_force, clearance, velocity):
    contact = (foot_force > 1.0) & (clearance < COORDINATED_TROT["contact_height_m"])
    slip = velocity[:, :, :2].square().sum(dim=-1)
    loss = COORDINATED_TROT["slip_cost_weight"] * (
        contact.float() * (slip / COORDINATED_TROT["slip_scale_mps"] ** 2).clamp(max=4.0)
    ).mean(dim=-1)
    return contact, slip, loss


def coordinated_trot_terms(phase, walk_blend, foot_force, clearance, velocity):
    leg_phase, swing_progress = trot_phase(phase)
    swing = leg_phase >= COORDINATED_TROT["stance_fraction"]
    blend = walk_blend[:, None]
    peak = COORDINATED_TROT["swing_peak_m"]
    arc = 16.0 * (swing_progress * (1.0 - swing_progress)).square()
    target = peak * arc * blend
    contact, slip, slip_loss = planted_foot_cost(foot_force, clearance, velocity)
    contact_confidence = (1.0 - torch.exp(-foot_force.square() / 100.0)) * contact.float()
    planted = contact_confidence * torch.exp(-slip / 0.04)
    swing_score = (1.0 - contact_confidence) * torch.exp(-(clearance - target).square() / 0.0004)
    gait = ((1.0 - swing.float() * blend) * planted + swing.float() * blend * swing_score).mean(dim=-1)

    # Reward both swing feet leaving together while both opposite feet carry load.
    # The arc weights the middle of swing and leaves a transition interval.
    agreement = torch.where(swing, 1.0 - contact_confidence, contact_confidence).prod(dim=-1)
    pair_loss = COORDINATED_TROT["paired_contact_weight"] * (1.0 - agreement) * arc.amax(dim=-1) * walk_blend
    clearance_loss = COORDINATED_TROT["clearance_cost_weight"] * (
        arc * blend * ((target - clearance).clamp_min(0.0) / peak).square()
    ).mean(dim=-1)
    return gait, pair_loss + clearance_loss + slip_loss, target, swing, contact
