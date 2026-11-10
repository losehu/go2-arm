"""Populate the ignored runtime data from an existing, hash-matching model bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "go2_arm"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install(path, data, expected):
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest(path) != expected:
            raise ValueError(f"Existing runtime file differs: {path}")
        return
    path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path,
                        help="Existing model package containing robot/ and path_bank.npz")
    parser.add_argument("--recordings", type=Path, help="Install locally qualified supplemental recordings")
    parser.add_argument("--skip-reference", action="store_true", help="Install only the data needed for inference")
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    manifest = json.loads((ROOT / "asset_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        src = bundle / name
        if not src.is_file() or digest(src) != expected:
            raise ValueError(f"Missing or incompatible bundle file: {src}")
        dest = PACKAGE / name
        if dest.exists() and digest(dest) != expected:
            raise ValueError(f"Existing runtime file differs: {dest}")
    for name in manifest["files"]:
        dest = PACKAGE / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle / name, dest)
    if args.skip_reference:
        print("Robot and trajectory files verified for inference.")
        return

    ref = manifest["amp_reference"]
    clips = PACKAGE / "amp_reference"
    upstream = f"https://raw.githubusercontent.com/ak1raljl/amp_go2/{ref['commit']}/datasets/go2_motion"
    for name, expected in ref["clips"].items():
        dest = clips / name
        src = bundle / "amp_reference" / name
        if dest.exists() and digest(dest) == expected:
            continue
        if src.is_file():
            data = src.read_bytes()
        else:
            with urlopen(f"{upstream}/{name}", timeout=30) as response:
                data = response.read()
        install(dest, data, expected)
    revision = {k: ref[k] for k in ("repository", "commit", "date")}
    (clips / "revision.json").write_text(json.dumps(revision, indent=2) + "\n")
    recording_dir = args.recordings if args.recordings else bundle / "amp_reference"
    recording_manifest = recording_dir / "recording_manifest.json"
    if args.recordings or recording_manifest.is_file():
        report = json.loads(recording_manifest.read_text())
        if (report["source"] != "fixed_command_free_base_physx_policy_recording"
                or report.get("linear_velocity_frame") != "root_link"
                or report.get("swing_quality_revision") != "measured_swing_v1"):
            raise ValueError("Supplemental clips require root-link velocity and measured-swing screening. Re-record them and pass --recordings.")
        for entry in report["clips"]:
            name = entry["file"]
            if Path(name).name != name or not name.startswith("go2_recorded_"):
                raise ValueError(f"Invalid recording file name: {name}")
            install(clips / name, (recording_dir / name).read_bytes(), entry["sha256"])
        (clips / "recording_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Supplemental recordings verified: {len(report['clips'])}")
    print(f"Robot and trajectory files verified with {len(ref['clips'])} AMP reference clips.")


if __name__ == "__main__":
    main()
