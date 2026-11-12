"""Launch a zero-command PhysX worker and full-range live panel for one checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from probe_jobs import ROOT, environment
from verify_bundle import verify


def publish(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--panel-python', default=sys.executable)
    parser.add_argument('--start-paused', action='store_true')
    parser.add_argument('--self-test', action='store_true', help='Hidden panel control/render verification')
    args = parser.parse_args()
    verify()
    checkpoint = args.checkpoint.resolve(strict=True)
    previews = ROOT / 'runs/previews'
    previews.mkdir(parents=True, exist_ok=True)
    ipc = Path(tempfile.mkdtemp(prefix=checkpoint.stem + '_', dir=previews))
    command = {'seq': 0, 'base': [0., 0., 0.], 'grip': .5, 'paused': True,
               'arm_mode': 'joint_position', 'arm_revision': 0, 'reset_revision': 0, 'shutdown': False}
    publish(ipc / 'command.json', command)
    with (ipc / 'worker.log').open('w') as log:
        worker = subprocess.Popen([sys.executable, str(ROOT / 'go2_arm/live_controller.py'),
                                   '--headless', '--device', args.device, '--checkpoint', str(checkpoint),
                                   '--ipc', str(ipc)], env=environment(), stdout=log,
                                  stderr=subprocess.STDOUT)
    launch = {'ipc': str(ipc), 'checkpoint': str(checkpoint),
              'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'worker_pid': worker.pid}
    publish(ipc / 'launch.json', launch)
    print(json.dumps(launch), flush=True)
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if worker.poll() is not None:
                raise RuntimeError(f'Worker exited; inspect {ipc / "worker.log"}')
            try:
                state = json.loads((ipc / 'state.json').read_text())
            except (OSError, ValueError):
                time.sleep(.25)
                continue
            if state.get('status') == 'error':
                raise RuntimeError(state['message'])
            if state.get('status') == 'paused' and 'qpos' in state:
                break
            time.sleep(.25)
        else:
            raise TimeoutError(f'Worker startup timeout; inspect {ipc / "worker.log"}')
        command.update(seq=1, paused=args.start_paused)
        publish(ipc / 'command.json', command)
        panel_env = os.environ.copy()
        panel_env.update(MUJOCO_GL='egl', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        # Optional compatibility for Ubuntu system Tk with an Isaac/Conda Python.
        libraries = ['/usr/lib/x86_64-linux-gnu/libtcl8.6.so', '/usr/lib/x86_64-linux-gnu/libtk8.6.so']
        if all(Path(x).is_file() for x in libraries) and Path('/usr/share/tcltk/tk8.6').is_dir():
            existing = panel_env.get('LD_PRELOAD', '')
            panel_env['LD_PRELOAD'] = ':'.join(libraries + ([existing] if existing else []))
            panel_env.setdefault('TCL_LIBRARY', '/usr/share/tcltk/tcl8.6')
            panel_env.setdefault('TK_LIBRARY', '/usr/share/tcltk/tk8.6')
        panel_command = [args.panel_python, str(ROOT / 'scripts/full_range_panel.py'), '--ipc', str(ipc)]
        if args.self_test:
            panel_command.append('--self_test')
        with (ipc / 'panel.log').open('w') as log:
            panel = subprocess.Popen(panel_command, env=panel_env, stdout=log, stderr=subprocess.STDOUT)
            launch['panel_pid'] = panel.pid
            publish(ipc / 'launch.json', launch)
            try:
                code = panel.wait()
            except BaseException:
                panel.terminate()
                panel.wait()
                raise
        if code:
            raise RuntimeError(f'Panel exited {code}; inspect {ipc / "panel.log"}')
        if args.self_test:
            result = json.loads((ipc / 'panel_test.json').read_text())
            ranges = json.loads((ipc / 'panel_ranges.json').read_text())
            assert result['state_rendered'] and ranges['actual_widget_ranges'] == [[-1., 1.], [-.6, .6], [-.8, .8]]
            print(json.dumps({'panel_test': result, 'ranges': ranges}), flush=True)
    finally:
        if worker.poll() is None:
            try:
                command = json.loads((ipc / 'command.json').read_text())
                command.update(seq=command.get('seq', 0) + 1, shutdown=True)
                publish(ipc / 'command.json', command)
                worker.wait(timeout=5)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                worker.terminate()
                worker.wait()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
