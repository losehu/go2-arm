"""Record qualified 50 Hz motion clips from a frozen policy in free-base PhysX."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--suite", choices=("all", "slow", "mixed", "repair"), default="all")
parser.add_argument("--lift-assist-mm", type=float, default=0., help="Optional teacher-only foot lift after measured liftoff")
parser.add_argument("--front-lift-scale", type=float, default=1., help="Scale teacher-only assist on the front feet")
parser.add_argument("--assist-paired", action="store_true", help="Use measured diagonal partner liftoff to assist the weaker side")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.path.insert(0, str(args.package.resolve()))
app = AppLauncher(args).app

import os
import traceback
import numpy as np
import torch
from cooperative_ppo import make_ppo, observations
from duet_env import DuetCfg, DuetEnv
from foot_gait import foot_centers
from gait_metrics import measured_swing


def mirrored_legs(values):
    swap = values.reshape(-1, 4, 3)[:, [1, 0, 3, 2]]
    return (swap * values.new_tensor([-1, 1, 1])).flatten(1)


def mirrored_observation(obs):
    flipped = obs.clone()
    flipped[:, [1, 3, 5, 7, 46, 47, 49]] *= -1
    for start in (9, 21, 33):
        flipped[:, start:start + 12] = mirrored_legs(obs[:, start:start + 12])
    phase_width = obs.shape[-1] - 69
    if phase_width == 8:
        for start in (50, 54):
            flipped[:, start:start + 4] = obs[:, start:start + 4][:, [1, 0, 3, 2]]
    for start in (50 + phase_width, 56 + phase_width, 62 + phase_width):
        flipped[:, start + torch.tensor([0, 3, 5], device=obs.device)] *= -1
    return flipped


def main():
    torch.set_num_threads(2)
    torch.manual_seed(61221)
    candidates = {"slow_lateral": (0., .12, 0.)} if args.suite != "mixed" else {}
    if args.suite in ("all","slow"):
        for speed in (.14,.16,.18,.20):
            candidates[f"slow_lateral_{round(100*speed)}"] = (0., speed, 0.)
    if args.suite == "repair":
        candidates.update(forward_diagonal_3=(.60, .30, 0.), backward_diagonal_3=(-.60, .30, 0.),
                          forward_diagonal_bridge=(.55, .25, 0.), backward_diagonal_bridge=(-.55, .25, 0.),
                          forward_turn_1=(.65, 0., .14), backward_turn_1=(-.65, 0., .14),
                          lateral_counterturn_0=(0., .23, -.20), lateral_turn_1=(0., .37, .14))
    elif args.suite != "slow":
        for name, direction in (("forward", 1), ("backward", -1)):
            for tier, (vx, vy, wz) in enumerate(((.25, .15, .20), (.65, .11, .14), (.18, .37, .14), (.17, .11, .53))):
                candidates[f"{name}_diagonal_turn_{tier}"] = (direction*vx, vy, wz)
                candidates[f"{name}_diagonal_counterturn_{tier}"] = (direction*vx, vy, -wz)
            for tier, (vx, vy) in enumerate(((.25, .15), (.65, .11), (.18, .37), (.60, .30))):
                candidates[f"{name}_diagonal_{tier}"] = (direction*vx, vy, 0.)
            for tier, (vx, wz) in enumerate(((.30, .25), (.65, .14), (.17, .53))):
                candidates[f"{name}_turn_{tier}"] = (direction*vx, 0., wz)
        for tier, (vy, wz) in enumerate(((.23, .20), (.37, .14), (.10, .53))):
            candidates[f"lateral_turn_{tier}"] = (0., vy, wz)
            candidates[f"lateral_counterturn_{tier}"] = (0., vy, -wz)
    cases = [(name + suffix, (vx, direction * vy, direction * wz), direction < 0)
             for name, (vx, vy, wz) in candidates.items()
             for suffix, direction in (("_left", 1), ("_right", -1))]
    n = len(cases)
    commands = torch.tensor([case[1] for case in cases], device=args.device)
    mirror = torch.tensor([case[2] for case in cases], device=args.device)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    assert saved["manifest"]["loco_obs_dim"] in (69, 77)
    assert saved["training_stage"] == 1

    class Recorder(DuetEnv):
        def _sample_base_commands(self, ids):
            self.command_targets[ids, :3] = commands[ids]
            self.command_targets[ids, 3] = 0
            self.command_age[ids] = 0
            self.command_period[ids] = 1e9
            if hasattr(self, "guidance_target"):
                self.guidance_target[ids] = 0

        def _get_dones(self):
            term, trunc = super()._get_dones()
            if getattr(self, "recording", False):
                self.failed |= term | trunc
            return term, trunc

    cfg = DuetCfg()
    if hasattr(cfg, "phase_free_amp"):
        cfg.phase_free_amp = saved["manifest"].get("phase_free_amp", False)
    if hasattr(cfg, "coordinated_trot"):
        cfg.coordinated_trot = "coordinated_trot" in saved["manifest"]
    if hasattr(cfg, "directional_amp"):
        cfg.directional_amp = saved["manifest"].get("directional_amp", False)
    cfg.scene.num_envs = n
    cfg.scene.env_spacing = 20.
    cfg.sim.device = args.device
    cfg.seed = 61221
    env = Recorder(cfg)
    env.reset()
    env.grip_target.fill_(.5)
    assert not env.robot.is_fixed_base and abs(env.step_dt - .02) < 1e-7
    loco = make_ppo(observations(env.get_loco_observation()), 12, args.device)
    loco.load(saved["loco"], load_cfg=None, strict=True)
    loco.actor.eval()
    data = env.robot.data
    feet = [env.robot.body_names.index(f"{leg}_calf") for leg in ("FL", "FR", "RL", "RR")]
    env.failed = torch.zeros(n, dtype=torch.bool, device=env.device)
    env.recording = True
    frames, clearance_samples, contacts, slip_samples = [], [], [], []
    zero = torch.zeros(n, 8, device=env.device)
    time_airborne = torch.zeros((n, 4), device=env.device)
    assist_side = torch.tensor([True,False,True,False],device=env.device)[None] ^ mirror[:,None]
    front_lift = commands.new_tensor([args.front_lift_scale,args.front_lift_scale,1.,1.])
    if not .5 <= args.front_lift_scale <= 2.:
        raise ValueError("Front lift scale must be within 0.5–2")
    if not 0. <= args.lift_assist_mm <= 65.:
        raise ValueError("Lift assist must be within 0–65 mm")
    with torch.inference_mode():
        for step in range(400):
            env.prepare_arm(zero)
            obs = env.get_loco_observation()
            obs[mirror] = mirrored_observation(obs[mirror])
            actions = loco.actor(observations(obs))
            actions[mirror] = mirrored_legs(actions[mirror])
            if args.lift_assist_mm:
                force_now = env.contact.data.net_forces_w[:, env.feet].norm(dim=-1)
                grounded = force_now > 1.
                time_airborne = torch.where(grounded, 0., time_airborne + env.step_dt)
                signal = time_airborne
                if args.assist_paired:
                    signal = time_airborne[:, [3,2,1,0]] * assist_side
                ramp = (signal/.04).clamp(0., 1.) * ((.26-signal)/.07).clamp(0., 1.)
                lift = .001*args.lift_assist_mm*front_lift*ramp*grounded.any(dim=-1, keepdim=True)
                hip, thigh, calf = data.joint_pos[:, env.legs].reshape(n, 4, 3).unbind(dim=-1)
                knee = thigh + calf
                dx_thigh, dx_calf = -.213*(thigh.cos()+knee.cos()), -.213*knee.cos()
                dz_thigh, dz_calf = .213*(thigh.sin()+knee.sin())*hip.cos(), .213*knee.sin()*hip.cos()
                det = dx_thigh*dz_calf-dx_calf*dz_thigh
                scale = lift*det/(det.square()+.002**2)
                correction = torch.stack((torch.zeros_like(lift), -dx_calf*scale, dx_thigh*scale), dim=-1)
                actions = actions + (correction.flatten(1)/.35).clamp(-.65, .65)
            env.step(torch.cat((actions, zero[:, :7]), dim=-1))
            if step < 100:
                continue
            pos, vel = foot_centers(data.body_link_pos_w[:, feet], data.body_link_quat_w[:, feet],
                                    data.body_com_pos_w[:, feet], data.body_com_lin_vel_w[:, feet],
                                    data.body_com_ang_vel_w[:, feet])
            clear = pos[:, :, 2] - env.scene.env_origins[:, None, 2] - .022
            force = env.contact.data.net_forces_w[:, env.feet].norm(dim=-1)
            contact = (force > 1.) & (clear < .02)
            root_pos = data.root_link_pos_w - env.scene.env_origins
            quat_xyzw = data.root_link_quat_w[:, [1, 2, 3, 0]]
            # The 61-column AMP layout includes toe fields. Features reconstruct
            # toes from joint and root kinematics for both recorded and upstream clips.
            blank_toes = torch.zeros(n, 12, device=env.device)
            frame = torch.cat((root_pos, quat_xyzw, data.joint_pos[:, env.legs], blank_toes,
                               data.root_link_lin_vel_b, data.root_link_ang_vel_b,
                               data.joint_vel[:, env.legs], blank_toes), dim=-1)
            frames.append(frame.cpu().numpy())
            clearance_samples.append(clear.cpu().numpy())
            contacts.append(contact.cpu().numpy())
            slip_samples.append((contact & (vel[:, :, :2].norm(dim=-1) > .1)).cpu().numpy())

    frames = np.stack(frames)
    clearance_samples = np.stack(clearance_samples)
    contacts = np.stack(contacts)
    slip_samples = np.stack(slip_samples)
    args.output.mkdir(parents=True, exist_ok=False)
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    results = {}
    accepted_frames = {}
    for i, (name, command, is_mirrored) in enumerate(cases):
        c = contacts[:, i]
        a = c[:, 0] & c[:, 3] & ~c[:, 1] & ~c[:, 2]
        b = c[:, 1] & c[:, 2] & ~c[:, 0] & ~c[:, 3]
        rmse = np.sqrt(((frames[:, i][:, [31, 32, 36]] - command)**2).mean(axis=0))
        p95 = np.percentile(clearance_samples[:, i], 95, axis=0)
        diagonal = float((a | b).mean())
        slip = float(slip_samples[:, i].sum() / max(1, c.sum()))
        swing = measured_swing(c, clearance_samples[:, i], env.step_dt)
        qualified_swing = all(count >= 3 and peak is not None and peak >= .015 and low <= .25
                              for count, peak, low in zip(swing["swing_count"], swing["swing_peak_median_m"],
                                                          swing["swing_peak_below_1cm_fraction"]))
        passed = (qualified_swing and not bool(env.failed[i]) and max(rmse) <= .06 and diagonal >= .20
                  and bool(a.any() and b.any()) and slip <= .15)
        results[name] = dict(command=list(command), mirrored_teacher=is_mirrored, passed=passed,
                             physical_failure=bool(env.failed[i]), velocity_rmse=rmse.tolist(),
                             clearance_p95_m=p95.tolist(), exclusive_diagonal=diagonal, slip=slip, **swing)
        if passed:
            accepted_frames[name] = frames[:, i]
    # Use paired signs only so additional demonstrations do not bias one side.
    clips = []
    for name in candidates:
        pair = [name + "_left", name + "_right"]
        if not all(p in accepted_frames for p in pair):
            continue
        for p in pair:
            path = args.output / f"go2_recorded_{p}.txt"
            path.write_text(json.dumps({"FrameDuration": .02, "Frames": accepted_frames[p].round(6).tolist()}))
            clips.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "command": results[p]["command"]})
    report = {"source": "fixed_command_free_base_physx_policy_recording",
              "linear_velocity_frame": "root_link", "swing_quality_revision": "measured_swing_v1",
              "teacher_checkpoint_sha256": sha, "teacher_lift_assist_mm": args.lift_assist_mm, "teacher_assist_paired": args.assist_paired, "teacher_front_lift_scale": args.front_lift_scale, "seed": 61221, "clips": clips, "results": results}
    (args.output / "recording_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"clips": len(clips), "cases": len(results)}), flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
os._exit(0)
