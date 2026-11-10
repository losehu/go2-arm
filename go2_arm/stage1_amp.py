"""Command-conditioned motion prior for folded Go2 stage-one PPO.

Inspired by ak1raljl/amp_go2 at the revision recorded in amp_reference.
The prior uses leg motion only. Reference poses are never written into PhysX.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


LEG_ORDER = ("FL", "FR", "RL", "RR")
AMP_REWARD_PER_SECOND = 2.0
AMP_RAMP_UPDATES = 100
COMMAND_SCALE = (1.0, 0.6, 0.8)
FEATURE_SIZE = 36
MOTION_PAIR_SIZE = 2 * FEATURE_SIZE


def foot_positions(q):
    """Go2 foot frame positions in the base frame, in FL/FR/RL/RR order."""
    hip, thigh, calf = q.reshape(-1, 4, 3).unbind(-1)
    side = q.new_tensor([1, -1, 1, -1])
    front = q.new_tensor([1, 1, -1, -1])
    lateral = 0.0955 * side
    x = -0.213 * (thigh.sin() + (thigh + calf).sin())
    z = -0.213 * (thigh.cos() + (thigh + calf).cos())
    return torch.stack(
        (
            x + 0.1934 * front,
            lateral * hip.cos() - z * hip.sin() + 0.0465 * side,
            lateral * hip.sin() + z * hip.cos(),
        ),
        dim=-1,
    )


def motion_features(q, qd):
    # Reconstruct both expert and policy foot positions with the same geometry.
    # The recorded foot frames have an offset relative to this URDF's frames.
    return torch.cat((q, 0.1 * qd, 4.0 * foot_positions(q).flatten(1)), dim=-1)


class Stage1AMP:
    def __init__(self, directory: Path, env):
        self.device = env.device
        self.command_scale = torch.tensor(COMMAND_SCALE, device=self.device)
        revision = json.loads((directory / "revision.json").read_text())
        if not math.isclose(env.step_dt, 0.02, abs_tol=1e-7):
            raise ValueError("AMP reference pairs require a 50 Hz policy step")
        pairs = []
        clip_velocities = []
        clips = []
        for path in sorted(directory.glob("go2_*.txt")):
            raw = path.read_bytes()
            clip = json.loads(raw)
            frames = np.asarray(clip["Frames"], dtype=np.float32)
            if (frames.ndim != 2 or frames.shape[0] < 2 or frames.shape[1] != 61
                    or not np.isfinite(frames).all()
                    or not math.isclose(float(clip["FrameDuration"]), env.step_dt, abs_tol=1e-7)):
                raise ValueError(f"Incompatible AMP reference clip: {path.name}")
            q = torch.from_numpy(frames[:, 7:19]).to(self.device)
            qd = torch.from_numpy(frames[:, 37:49]).to(self.device)
            features = motion_features(q, qd)
            pairs.append(torch.cat((features[:-1], features[1:]), dim=-1))
            clip_velocities.append(frames[:, [31, 32, 36]].mean(axis=0))
            clips.append({"file": path.name, "frames": len(frames), "sha256": hashlib.sha256(raw).hexdigest()})
        if len(clips) != 17:
            raise ValueError("Expected all 17 pinned Go2 reference clips")
        self.expert_pairs = torch.cat(pairs)
        self.lengths = torch.tensor([len(pair) for pair in pairs], device=self.device)
        self.offsets = self.lengths.cumsum(0) - self.lengths
        self.clip_commands = torch.tensor(np.stack(clip_velocities), device=self.device) / self.command_scale

        # A separate generator keeps discriminator sampling out of PPO's RNG.
        self.rng = torch.Generator(device=self.device).manual_seed(61109)
        # Seed discriminator initialization without perturbing the rollout RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(61109)
            self.discriminator = nn.Sequential(
                nn.Linear(MOTION_PAIR_SIZE + 3, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
            ).to(self.device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.metadata = {
            **revision, "revision": "command_conditioned_leg_motion_v1",
            "leg_order": list(LEG_ORDER), "frame_dt_s": env.step_dt,
            "clips": clips, "expert_transitions": len(self.expert_pairs),
            "reward_max_per_second": AMP_REWARD_PER_SECOND,
            "ramp_updates": AMP_RAMP_UPDATES,
            "uses_reference_state_resets": False,
            "keeps_existing_gait_reward": not env.cfg.phase_free_amp,
        }

    @staticmethod
    def observe(env):
        return motion_features(
            env.robot.data.joint_pos[:, env.legs],
            env.robot.data.joint_vel[:, env.legs],
        )

    @staticmethod
    def validate_foot_frames(env):
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        robot = env.robot.data
        calves = [env.robot.body_names.index(f"{leg}_calf") for leg in LEG_ORDER]
        calf_position = robot.body_link_pos_w[:, calves]
        offset = calf_position.new_tensor([0.0, 0.0, -0.213]).expand_as(calf_position)
        foot_world = calf_position + quat_apply(robot.body_link_quat_w[:, calves], offset)
        relative = foot_world - robot.root_link_pos_w[:, None, :]
        root_q = robot.root_link_quat_w[:, None, :].expand(-1, len(calves), -1)
        actual = quat_apply_inverse(root_q, relative)
        expected = foot_positions(robot.joint_pos[:, env.legs])
        error = float((actual - expected).norm(dim=-1).max())
        if not math.isfinite(error) or error > 0.0005:
            raise RuntimeError(f"AMP foot frames disagree with PhysX by {error:g} m")
        return error

    def transition(self, before, after, commands):
        return torch.cat((before, after, commands / self.command_scale), dim=-1)

    def reward(self, transition, done, new_updates, dt):
        score = self.discriminator(transition).squeeze(-1)
        style = (1.0 - 0.25 * (score - 1.0).square()).clamp_min(0.0)
        weight = AMP_REWARD_PER_SECOND * min(1.0, new_updates / AMP_RAMP_UPDATES)
        return torch.where(done, 0.0, style * weight * dt)

    def sample_expert(self, commands):
        # Pick a near-speed clip in normalized vx/vy/wz coordinates.
        distance = (commands[:, None, :] - self.clip_commands).square().sum(dim=-1)
        weights = torch.softmax(-distance / 0.08, dim=-1)
        clip = torch.multinomial(weights, 1, generator=self.rng).squeeze(-1)
        frame = (torch.rand(len(clip), device=self.device, generator=self.rng) * self.lengths[clip]).long()
        return self.expert_pairs[self.offsets[clip] + frame]

    def update(self, rollout, valid_masks):
        samples = torch.cat(rollout)[torch.cat(valid_masks)]
        if not len(samples):
            return {"valid_transitions": 0}
        if not torch.isfinite(samples).all():
            raise RuntimeError("Non-finite AMP policy transitions")
        stats = []
        for _ in range(4):
            ids = torch.randint(len(samples), (min(1024, len(samples)),), device=self.device, generator=self.rng)
            policy = samples[ids]
            commands = policy[:, MOTION_PAIR_SIZE:]
            expert = self.sample_expert(commands).requires_grad_()
            expert_score = self.discriminator(torch.cat((expert, commands), dim=-1))
            policy_score = self.discriminator(policy)
            grad, = torch.autograd.grad(expert_score.sum(), expert, create_graph=True)
            penalty = 5.0 * grad.square().sum(dim=-1).mean()
            loss = 0.5 * ((expert_score - 1.0).square().mean() + (policy_score + 1.0).square().mean())
            self.optimizer.zero_grad(set_to_none=True)
            (loss + penalty).backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            stats.append(torch.stack((loss.detach(), penalty.detach(), expert_score.detach().mean(), policy_score.detach().mean())))
        mean = torch.stack(stats).mean(dim=0).tolist()
        return {
            "valid_transitions": len(samples), "discriminator_loss": mean[0],
            "gradient_penalty": mean[1], "expert_score": mean[2], "policy_score": mean[3],
        }

    def save(self):
        return {"discriminator": self.discriminator.state_dict(), "optimizer": self.optimizer.state_dict()}
