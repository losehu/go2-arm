"""Check recorded calf-cylinder clearance above the flat reference ground."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package", type=Path, required=True)
parser.add_argument("--recordings", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.package.resolve()))
from robot_constants import DEFAULT_Q

model = mujoco.MjModel.from_xml_path(str((args.package / "robot/scene.xml").resolve()))
data = mujoco.MjData(model)
bodies = [model.body(f"{leg}_calf").id for leg in ("FL", "FR", "RL", "RR")]
calves = np.flatnonzero(np.isin(model.geom_bodyid, bodies) & (model.geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER))
assert len(calves) == 8
radius, half_length = model.geom_size[calves, :2].T
report = json.loads((args.recordings / "recording_manifest.json").read_text())
minimum = {}
for entry in report["clips"]:
    name = entry["file"]
    frames = json.loads((args.recordings / name).read_text())["Frames"]
    clearance = []
    for frame in frames:
        data.qpos[:] = DEFAULT_Q
        data.qpos[:3] = frame[:3]
        data.qpos[3:7] = [frame[6], *frame[3:6]]
        data.qpos[7:19] = frame[7:19]
        mujoco.mj_kinematics(model, data)
        axis_z = data.geom_xmat[calves, 8]
        lowest = (data.geom_xpos[calves, 2] - half_length*np.abs(axis_z)
                  - radius*np.sqrt(np.maximum(0., 1.-axis_z**2)))
        clearance.append(lowest.min())
    minimum[name] = min(clearance)
print(json.dumps({"min_calf_clearance_m": min(minimum.values()), "per_clip_min_calf_clearance_m": minimum,
                  "ground_plane_m": 0., "poses": sum(len(json.loads((args.recordings/e["file"]).read_text())["Frames"])
                                                    for e in report["clips"])}, indent=2))
if min(minimum.values()) <= .003:
    raise RuntimeError("A supplemental clip comes within 3 mm of the ground on a calf cylinder")
