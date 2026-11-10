"""Run one local PhysX policy worker and its interactive Tk/MuJoCo panel."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

PACKAGE = Path(__file__).resolve().parent


def wait_until_ready(worker, ipc):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if worker.poll() is not None:
            raise RuntimeError(f"Policy worker exited. See {ipc / 'worker.log'}")
        state_path = ipc / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("status") == "error":
                raise RuntimeError(state["message"])
            if state.get("status") == "paused" and "qpos" in state:
                return
        time.sleep(0.25)
    raise TimeoutError(f"Policy worker did not become ready. See {ipc / 'worker.log'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu", help="PhysX device, e.g. cpu or cuda:0")
    parser.add_argument("--panel-python", default=sys.executable, help="Python with Tk, MuJoCo and Pillow")
    parser.add_argument("--kit-args", help="Additional Isaac Sim arguments")
    parser.add_argument("--self-test", action="store_true", help="Verify rendering and controls in a hidden Tk window")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    if not (PACKAGE / "robot/scene.xml").is_file():
        parser.error("Import the robot bundle with tools/prepare_assets.py first")
    previews = PACKAGE.parent / "runs" / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    ipc = Path(tempfile.mkdtemp(prefix=f"{checkpoint.stem}_", dir=previews))
    command = {"seq": 0, "base": [0, 0, 0], "grip": 0.5, "paused": True,
               "arm_mode": "joint_position", "arm_revision": 0, "reset_revision": 0, "shutdown": False}
    (ipc / "command.json").write_text(json.dumps(command))
    print(f"Preview state and logs: {ipc}", flush=True)

    worker_env = os.environ.copy()
    worker_env.pop("DISPLAY", None)
    worker_env.pop("WAYLAND_DISPLAY", None)
    worker_env.setdefault("OMP_NUM_THREADS", "2")
    worker_args = [sys.executable, str(PACKAGE / "live_controller.py"), "--headless",
                   "--device", args.device, "--checkpoint", str(checkpoint), "--ipc", str(ipc)]
    if args.kit_args:
        worker_args.extend(["--kit_args", args.kit_args])
    panel_args = [args.panel_python, str(PACKAGE / "control_panel.py"), "--ipc", str(ipc)]
    if args.self_test:
        panel_args.append("--self_test")

    with (ipc / "worker.log").open("w") as log:
        worker = subprocess.Popen(worker_args, env=worker_env, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_until_ready(worker, ipc)
        subprocess.run(panel_args, check=True)
        if args.self_test and not (ipc / "panel_test.json").is_file():
            raise RuntimeError("Panel self-test did not produce a result")
    finally:
        try:
            worker.wait(timeout=3)
        except subprocess.TimeoutExpired:
            worker.terminate()
            try:
                worker.wait(timeout=3)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()


if __name__ == "__main__":
    main()
