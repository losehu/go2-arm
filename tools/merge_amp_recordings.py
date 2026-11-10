"""Combine qualified root-link recordings, preferring earlier inputs on duplicates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SOURCE = "fixed_command_free_base_physx_policy_recording"
SCREEN = "measured_swing_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    clips, results, payloads = {}, {}, {}
    for directory in args.inputs:
        report = json.loads((directory / "recording_manifest.json").read_text())
        if (report["source"] != SOURCE or report.get("linear_velocity_frame") != "root_link"
                or report.get("swing_quality_revision") != SCREEN):
            raise ValueError(f"Incompatible recordings: {directory}")
        for entry in report["clips"]:
            name = entry["file"]
            if Path(name).name != name or not name.startswith("go2_recorded_") or not name.endswith(".txt"):
                raise ValueError(f"Invalid file name: {name}")
            if name in clips:
                continue
            key = Path(name).stem.removeprefix("go2_recorded_")
            result = report["results"][key]
            if not result["passed"] or result["command"] != entry["command"]:
                raise ValueError(f"Unqualified recording: {name}")
            path = directory / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Hash mismatch: {path}")
            clips[name] = {**entry, "teacher_checkpoint_sha256": report["teacher_checkpoint_sha256"],
                           "teacher_lift_assist_mm": report.get("teacher_lift_assist_mm", 0.),
                           "teacher_assist_paired": report.get("teacher_assist_paired", False),
                           "teacher_front_lift_scale": report.get("teacher_front_lift_scale", 1.)}
            results[key] = result
            payloads[name] = path
    if not clips:
        raise ValueError("No qualified clips")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, path in payloads.items():
        shutil.copyfile(path, args.output / name)
    report = {"source": SOURCE, "linear_velocity_frame": "root_link", "swing_quality_revision": SCREEN,
              "clips": list(clips.values()), "results": results}
    (args.output / "recording_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Qualified clips: {len(clips)}")


if __name__ == "__main__":
    main()
