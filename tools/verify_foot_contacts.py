"""Exercise foot-only, calf-only and simultaneous contacts in PhysX."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.path.insert(0, str(args.package.resolve()))
app = AppLauncher(args).app

import torch
from duet_env import DuetCfg, DuetEnv


def main():
    cfg = DuetCfg()
    cfg.scene.num_envs = 3
    cfg.scene.env_spacing = 4.
    cfg.sim.device = args.device
    cfg.phase_free_amp = True
    env = DuetEnv(cfg)
    env.reset()
    assert not env.robot.is_fixed_base
    root = env.robot.data.default_root_state.clone()
    root[:, :3] += env.scene.env_origins
    root[:, 2] = env.scene.env_origins[:, 2] + root.new_tensor([.302, .217, .175])
    joint = env.nominal.clone()
    joint[1:, env.legs] = joint.new_tensor([0., 1.3, -2.4]*4)
    joint[1, env.legs[:3]] = joint.new_tensor([0., -.228846, -1.830357])
    joint[2, env.legs[:3]] = joint.new_tensor([0., -.050962, -1.557143])
    env.robot.write_root_pose_to_sim(root[:, :7])
    env.robot.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
    env.robot.write_joint_state_to_sim(joint, torch.zeros_like(joint))
    env.scene.write_data_to_sim()
    peak = torch.zeros((3, 4), device=env.device)
    simultaneous = torch.zeros(3, dtype=torch.bool, device=env.device)
    for _ in range(8):
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        foot = env.foot_contact_forces()
        peak = torch.maximum(peak, foot)
        calf = env.contact.data.net_forces_w[:, env.calf_sensors[0]].norm(dim=-1)
        simultaneous |= (foot[:, 0]>1.) & (calf>.5)
    termination, _ = env._get_dones()
    # The calf-only pose may have other feet planted. The affected front foot
    # must still report zero support. Both simultaneous signals remain visible.
    result = {"cases": ["foot_only", "calf_only", "simultaneous"],
              "front_left_foot_peak_force_n": peak[:, 0].tolist(),
              "front_left_calf_contact": env.calf_contact[:, 0].tolist(),
              "physical_termination": termination.tolist(),
              "front_left_simultaneous_same_step": simultaneous.tolist(),
              "mass_kg": float(env.robot.root_physx_view.get_masses()[0].sum())}
    print(json.dumps(result), flush=True)
    assert env.calf_contact[:, 0].tolist() == [False, True, True]
    assert termination.tolist() == [False, True, True]
    assert (peak[:, 0] > 1.).tolist() == [True, False, True]
    assert simultaneous.tolist() == [False, False, True]


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
os._exit(0)
