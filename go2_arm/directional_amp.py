"""Direction-matched AMP using short histories of measured whole-body motion."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from base_commands import VELOCITY_DEADZONE

from stage1_amp import (AMP_RAMP_UPDATES, AMP_REWARD_PER_SECOND, COMMAND_SCALE,
                        LEG_ORDER, Stage1AMP, foot_positions, motion_features)

FEATURES = 59
HISTORY_LAGS = (10, 5, 0)
HISTORY_FRAMES = max(HISTORY_LAGS) + 1
SPEED_TEMPERATURE = .025
HISTORY_COMMAND_TOLERANCE = .10
MAX_SPEED_DISTANCE = .30
TOE_X = -.002
SPEED_BAND_EDGES = (.33, .66)
GUIDED_MIXED_FRACTION = .80
GUIDED_JITTER = .035
UNMATCHED_MIXED_FALLBACK_FRACTION = .80


def command_signature(commands):
    thresholds = VELOCITY_DEADZONE / commands.new_tensor(COMMAND_SCALE)
    return torch.where(commands.abs() > thresholds, commands.sign(), 0.)


def matched_references(commands, clip_commands, clip_signatures):
    same_direction = (command_signature(commands)[:, None] == clip_signatures).all(dim=-1)
    distance = (commands[:, None] - clip_commands).square().sum(dim=-1)
    return same_direction & (distance <= MAX_SPEED_DISTANCE**2), distance


def clip_probabilities(commands, clip_commands, clip_signatures):
    allowed, distance = matched_references(commands, clip_commands, clip_signatures)
    supported = allowed.any(dim=-1)
    # Unsupported commands are excluded before training or scoring the AMP prior.
    logits = torch.where(supported[:, None], (-distance / SPEED_TEMPERATURE).masked_fill(~allowed, -torch.inf), 0.)
    return torch.softmax(logits, dim=-1), supported


def guide_mixed_targets(commands, caps, clip_commands, clip_signatures, generator=None):
    """Guide some targets to nearby qualified speeds while keeping the requested direction."""
    signatures = command_signature(commands)
    selected = (signatures.count_nonzero(dim=-1) > 1) & (
        torch.rand(len(commands), device=commands.device, generator=generator) < GUIDED_MIXED_FRACTION)
    ids = torch.where(selected)[0]
    target = commands[ids]
    center = (clip_signatures * (clip_signatures*clip_commands).clamp_min(
        VELOCITY_DEADZONE / commands.new_tensor(COMMAND_SCALE) + GUIDED_JITTER + .0001)).clamp(-caps, caps)
    # Reserve room for jitter before deciding whether a center is admissible.
    eligible = ((center-clip_commands).norm(dim=-1) <= MAX_SPEED_DISTANCE - math.sqrt(3)*GUIDED_JITTER)
    same = (signatures[ids, None] == clip_signatures).all(dim=-1)
    distance = (target[:, None] - center).square().sum(dim=-1).masked_fill(~(same & eligible), torch.inf)
    nearest = distance.argmin(dim=-1)
    active = clip_signatures[nearest] != 0
    noise = 2*torch.rand(target.shape, device=commands.device, generator=generator)-1
    adjusted = (center[nearest] + GUIDED_JITTER*noise*active).clamp(-caps, caps)
    result = commands.clone()
    result[ids] = torch.where((same & eligible).any(dim=-1, keepdim=True), adjusted, target)
    return result


def fallback_mixed_targets(commands, clip_commands, clip_signatures, generator=None):
    """Reduce some unsupported mixed targets to their strongest matching axes."""
    mixed = command_signature(commands).count_nonzero(dim=-1) > 1
    selected = mixed & (torch.rand(len(commands), device=commands.device, generator=generator)
                        < UNMATCHED_MIXED_FALLBACK_FRACTION)
    ids = torch.where(selected)[0]
    allowed, _ = matched_references(commands[ids], clip_commands, clip_signatures)
    ids = ids[~allowed.any(dim=-1)]
    target = commands[ids]
    # Targets are already normalized by each axis limit. Ties keep vx, vy, wz order.
    order = target.abs().argsort(dim=-1, descending=True, stable=True)
    weakest = order[:, 2:]
    strongest = order[:, :1]
    two_axes = target.scatter(1, weakest, 0.)
    one_axis = torch.zeros_like(target).scatter(1, strongest, target.gather(1, strongest))
    two_matches, _ = matched_references(two_axes, clip_commands, clip_signatures)
    one_matches, _ = matched_references(one_axis, clip_commands, clip_signatures)
    result = commands.clone()
    result[ids] = torch.where(two_matches.any(dim=-1, keepdim=True), two_axes,
                              torch.where(one_matches.any(dim=-1, keepdim=True), one_axis, target))
    return result


def down_in_base(q_xyzw):
    x, y, z, w = q_xyzw.unbind(-1)
    return torch.stack((2 * (w*y-x*z), -2 * (y*z+w*x), 2 * (x*x+y*y)-1), dim=-1)


def foot_velocity(q, qd, feet, linear_b, angular_b):
    # Symmetric finite difference of the same collision-sphere FK on both data sources.
    delta = .0005*qd
    relative_speed = (foot_positions(q + delta, TOE_X) - foot_positions(q - delta, TOE_X)) / .001
    return relative_speed + linear_b[:, None] + torch.linalg.cross(
        angular_b[:, None].expand_as(feet), feet, dim=-1)


def whole_body_features(q, qd, linear_b, angular_b, height, down_b):
    feet = foot_positions(q, TOE_X)
    feet_speed = foot_velocity(q, qd, feet, linear_b, angular_b)
    ground_clearance = height[:, None] - (feet * down_b[:, None]).sum(dim=-1) - .022
    return torch.cat((motion_features(q, qd, feet), linear_b, .5*angular_b, 4.*(height[:, None]-.31),
                      4.*ground_clearance, .5*feet_speed.flatten(1)), dim=-1)


class DirectionalAMP(Stage1AMP):
    def __init__(self, directory, env):
        self.device = env.device
        self.command_scale = torch.tensor(COMMAND_SCALE, device=self.device)
        self.motion_size = FEATURES * len(HISTORY_LAGS)
        if not math.isclose(env.step_dt, .02, abs_tol=1e-7):
            raise ValueError("Directional AMP requires a 50 Hz policy step")

        revision = json.loads((directory / "revision.json").read_text())
        upstream = sorted(p for p in directory.glob("go2_*.txt") if not p.name.startswith("go2_recorded_"))
        if len(upstream) != 17:
            raise ValueError("Expected all 17 upstream reference clips")
        recordings = directory / "recording_manifest.json"
        recorded = []
        if recordings.is_file():
            report = json.loads(recordings.read_text())
            if report["source"] != "fixed_command_free_base_physx_policy_recording":
                raise ValueError("Unexpected supplemental clip provenance")
            if report.get("linear_velocity_frame") != "root_link" or report.get("swing_quality_revision") != "measured_swing_v1":
                raise ValueError("Re-record supplemental clips with root-link velocity and measured swing screening")
            recorded = report["clips"]
            for entry in recorded:
                name = entry["file"]
                if Path(name).name != name or not name.startswith("go2_recorded_"):
                    raise ValueError(f"Invalid recording file name: {name}")
        pairs, clips, means, signatures = [], [], [], []
        sources = [(p, None) for p in upstream] + [(directory / c["file"], c) for c in recorded]
        for path, entry in sources:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if entry is not None and digest != entry["sha256"]:
                raise ValueError(f"Supplemental clip hash mismatch: {path.name}")
            clip = json.loads(raw)
            a = torch.as_tensor(np.asarray(clip["Frames"], dtype=np.float32), device=self.device)
            if a.ndim != 2 or a.shape[1] != 61 or len(a) < HISTORY_FRAMES or not torch.isfinite(a).all():
                raise ValueError(f"Invalid reference frames: {path.name}")
            if not math.isclose(clip["FrameDuration"], env.step_dt, abs_tol=1e-7):
                raise ValueError(f"Unexpected reference interval: {path.name}")
            features = whole_body_features(a[:, 7:19], a[:, 37:49], a[:, 31:34], a[:, 34:37], a[:, 2],
                                           down_in_base(a[:, 3:7]))
            count = len(features) - max(HISTORY_LAGS)
            pairs.append(torch.cat([features[max(HISTORY_LAGS)-lag:][:count] for lag in HISTORY_LAGS], dim=-1))
            measured = a[:, [31, 32, 36]].mean(dim=0)
            means.append(measured / self.command_scale)
            if entry is not None:
                label = a.new_tensor(entry["command"]) / self.command_scale
            elif path.name == "go2_stance.txt":
                label = a.new_zeros(3)
            else:
                group = path.stem.removeprefix("go2_").removesuffix("_faster").removesuffix("_fast")
                axis, sign = {"forward": (0, 1), "backward": (0, -1), "left": (1, 1), "right": (1, -1),
                              "turn_left": (2, 1), "turn_right": (2, -1)}[group]
                label = a.new_zeros(3)
                label[axis] = sign
            signatures.append(command_signature(label))
            clips.append({"file": path.name, "frames": len(a), "sha256": digest,
                          "source": "upstream" if entry is None else "physx_recording",
                          "direction": signatures[-1].tolist()})

        self.expert_pairs = torch.cat(pairs)
        self.lengths = torch.tensor([len(pair) for pair in pairs], device=self.device)
        self.offsets = self.lengths.cumsum(0) - self.lengths
        self.clip_commands = torch.stack(means)
        self.clip_signatures = torch.stack(signatures)
        self.rng = torch.Generator(device=self.device).manual_seed(61109)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(61109)
            self.discriminator = nn.Sequential(nn.Linear(self.motion_size + 3, 256), nn.LeakyReLU(.2),
                                               nn.Linear(256, 128), nn.LeakyReLU(.2), nn.Linear(128, 1)).to(self.device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

        self.history = torch.zeros((env.num_envs, HISTORY_FRAMES, FEATURES), device=self.device)
        self.commands = torch.zeros((env.num_envs, HISTORY_FRAMES, 3), device=self.device)
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.cursor = -1
        self.metadata = {
            **revision, "revision": "direction_speed_mixed_fallback_v5",
            "leg_order": list(LEG_ORDER), "frame_dt_s": env.step_dt,
            "history_lags": list(HISTORY_LAGS), "feature_size": FEATURES,
            "physical_command_deadzone": VELOCITY_DEADZONE, "speed_temperature": SPEED_TEMPERATURE,
            "max_normalized_speed_distance": MAX_SPEED_DISTANCE, "linear_velocity_frame": "root_link",
            "coverage_speed_band_edges": list(SPEED_BAND_EDGES), "guided_mixed_target_fraction": GUIDED_MIXED_FRACTION,
            "unmatched_mixed_fallback_fraction": UNMATCHED_MIXED_FALLBACK_FRACTION,
            "unmatched_mixed_fallback_axis_counts": [2, 1],
            "guided_jitter_normalized": GUIDED_JITTER, "clips": clips, "expert_transitions": len(self.expert_pairs),
            "recorded_clips": len(recorded), "reward_max_per_second": AMP_REWARD_PER_SECOND,
            "ramp_updates": AMP_RAMP_UPDATES, "uses_reference_state_resets": False,
            "keeps_existing_gait_reward": False,
        }

    @staticmethod
    def observe(env):
        data = env.robot.data
        height = data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        return whole_body_features(data.joint_pos[:, env.legs], data.joint_vel[:, env.legs],
                                   data.root_link_lin_vel_b, data.root_link_ang_vel_b, height, data.projected_gravity_b)

    def clear_history(self):
        self.count.zero_()

    def record(self, features, commands, done):
        self.cursor = (self.cursor + 1) % HISTORY_FRAMES
        command = commands / self.command_scale
        self.history[:, self.cursor] = features
        self.commands[:, self.cursor] = command
        self.count.copy_(torch.where(done, 0, (self.count + 1).clamp_max(HISTORY_FRAMES)))
        indices = [(self.cursor-lag) % HISTORY_FRAMES for lag in HISTORY_LAGS]
        signature = command_signature(command)
        direction_stable = (command_signature(self.commands) == signature[:, None]).all(dim=(1, 2))
        speed_stable = (self.commands - command[:, None]).abs().amax(dim=(1, 2)) <= HISTORY_COMMAND_TOLERANCE
        allowed, _ = matched_references(command, self.clip_commands, self.clip_signatures)
        supported = allowed.any(dim=-1)
        valid = (self.count == HISTORY_FRAMES) & direction_stable & speed_stable & supported
        return torch.cat((self.history[:, indices].flatten(1), command), dim=-1), valid

    def coverage(self, rollout, valid):
        commands = torch.cat([pair[:, self.motion_size:] for pair in rollout])
        sign = (command_signature(commands) + 1).long()
        signature = sign[:, 0]*9 + sign[:, 1]*3 + sign[:, 2]
        speed = torch.bucketize(commands.abs().amax(dim=-1), commands.new_tensor(SPEED_BAND_EDGES))
        index = signature*3 + speed
        counts = torch.stack((torch.bincount(index, minlength=81),
                              torch.bincount(index[torch.cat(valid)], minlength=81)), dim=-1)
        result = {}
        for i, (total, matched) in enumerate(counts.tolist()):
            if total:
                key = f"{i//27-1:+d},{(i//9)%3-1:+d},{(i//3)%3-1:+d}:band{i%3}"
                result[key] = {"samples": total, "matched": matched}
        return result

    @staticmethod
    def verify_foot_kinematics(env):
        from isaaclab.utils.math import quat_apply_inverse
        data = env.robot.data
        world_pos, world_speed = env.foot_kinematics()
        root_q = data.root_link_quat_w[:, None].expand(-1, 4, -1)
        actual_pos = quat_apply_inverse(root_q, world_pos - data.root_link_pos_w[:, None])
        actual_speed = quat_apply_inverse(root_q, world_speed)
        q, qd = data.joint_pos[:, env.legs], data.joint_vel[:, env.legs]
        predicted_pos = foot_positions(q, TOE_X)
        predicted_speed = foot_velocity(q, qd, predicted_pos, data.root_link_lin_vel_b, data.root_link_ang_vel_b)
        position_error = float((actual_pos - predicted_pos).norm(dim=-1).max())
        velocity_error = float((actual_speed - predicted_speed).norm(dim=-1).max())
        mixed_error = float((actual_speed - predicted_speed -
                             (data.root_com_lin_vel_b-data.root_link_lin_vel_b)[:, None]).norm(dim=-1).max())
        if not math.isfinite(position_error) or position_error > .0005:
            raise RuntimeError(f"AMP foot sphere position mismatch: {position_error:g} m")
        if not math.isfinite(velocity_error) or velocity_error > .006:
            raise RuntimeError(f"AMP foot sphere velocity mismatch: {velocity_error:g} m/s")
        return position_error, velocity_error, mixed_error

    def sample_expert(self, commands):
        weights, _ = clip_probabilities(commands, self.clip_commands, self.clip_signatures)
        clip = torch.multinomial(weights, 1, generator=self.rng).squeeze(-1)
        frame = (torch.rand(len(clip), device=self.device, generator=self.rng) * self.lengths[clip]).long()
        return self.expert_pairs[self.offsets[clip] + frame]
