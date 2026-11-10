"""Direction-matched AMP using short histories of measured whole-body motion."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from stage1_amp import (AMP_RAMP_UPDATES, AMP_REWARD_PER_SECOND, COMMAND_SCALE,
                        LEG_ORDER, Stage1AMP, foot_positions, motion_features)

FEATURES = 59
HISTORY_LAGS = (10, 5, 0)
HISTORY_FRAMES = max(HISTORY_LAGS) + 1
DIRECTION_DEADBAND = .08  # In normalized command coordinates.
SPEED_TEMPERATURE = .025
HISTORY_COMMAND_TOLERANCE = .10


def command_signature(commands):
    return torch.where(commands.abs() >= DIRECTION_DEADBAND, commands.sign(), 0.)


def clip_probabilities(commands, clip_commands, clip_signatures):
    allowed = (command_signature(commands)[:, None] == clip_signatures).all(dim=-1)
    supported = allowed.any(dim=-1)
    distance = (commands[:, None] - clip_commands).square().sum(dim=-1)
    # Unsupported commands are excluded from the AMP reward and the discriminator update.
    logits = torch.where(supported[:, None], (-distance / SPEED_TEMPERATURE).masked_fill(~allowed, -torch.inf), 0.)
    return torch.softmax(logits, dim=-1), supported


def down_in_base(q_xyzw):
    x, y, z, w = q_xyzw.unbind(-1)
    return torch.stack((2 * (w*y-x*z), -2 * (y*z+w*x), 2 * (x*x+y*y)-1), dim=-1)


def whole_body_features(q, qd, linear_b, angular_b, height, down_b):
    feet = foot_positions(q)
    # The same FK and velocity calculation runs for both 50 Hz reference
    # recordings and measured simulation state. No contact labels are inferred.
    relative_speed = (foot_positions(q + .001*qd) - foot_positions(q - .001*qd)) / .002
    feet_speed = relative_speed + linear_b[:, None] + torch.linalg.cross(
        angular_b[:, None].expand_as(feet), feet, dim=-1)
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
            **revision, "revision": "direction_matched_whole_body_history_v2",
            "leg_order": list(LEG_ORDER), "frame_dt_s": env.step_dt,
            "history_lags": list(HISTORY_LAGS), "feature_size": FEATURES,
            "direction_deadband": DIRECTION_DEADBAND, "speed_temperature": SPEED_TEMPERATURE,
            "clips": clips, "expert_transitions": len(self.expert_pairs),
            "recorded_clips": len(recorded), "reward_max_per_second": AMP_REWARD_PER_SECOND,
            "ramp_updates": AMP_RAMP_UPDATES, "uses_reference_state_resets": False,
            "keeps_existing_gait_reward": False,
        }

    @staticmethod
    def observe(env):
        data = env.robot.data
        height = data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        return whole_body_features(data.joint_pos[:, env.legs], data.joint_vel[:, env.legs],
                                   data.root_lin_vel_b, data.root_ang_vel_b, height, data.projected_gravity_b)

    def clear_history(self):
        self.count.zero_()

    def record(self, features, commands, done):
        self.cursor = (self.cursor + 1) % HISTORY_FRAMES
        command = commands / self.command_scale
        self.history[:, self.cursor] = features
        self.commands[:, self.cursor] = command
        self.count.copy_(torch.where(done, 0, (self.count + 1).clamp_max(HISTORY_FRAMES)))
        indices = [(self.cursor-lag) % HISTORY_FRAMES for lag in HISTORY_LAGS]
        stable = (self.commands[:, indices] - command[:, None]).abs().amax(dim=(1, 2)) <= HISTORY_COMMAND_TOLERANCE
        supported = (command_signature(command)[:, None] == self.clip_signatures).all(dim=-1).any(dim=-1)
        valid = (self.count == HISTORY_FRAMES) & stable & supported
        return torch.cat((self.history[:, indices].flatten(1), command), dim=-1), valid

    def sample_expert(self, commands):
        weights, _ = clip_probabilities(commands, self.clip_commands, self.clip_signatures)
        clip = torch.multinomial(weights, 1, generator=self.rng).squeeze(-1)
        frame = (torch.rand(len(clip), device=self.device, generator=self.rng) * self.lengths[clip]).long()
        return self.expert_pairs[self.offsets[clip] + frame]
