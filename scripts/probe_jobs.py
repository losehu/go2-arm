"""Shared independent-probe commands; existing results are never overwritten."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
KIT = '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 --/plugins/omni.tbb.globalcontrol/maxThreadCount=2'


def environment(threads=2):
    env = os.environ.copy()
    env.pop('DISPLAY', None)
    env.pop('WAYLAND_DISPLAY', None)
    env.update(OMNI_KIT_ACCEPT_EULA='YES', OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def probe_command(tool, checkpoint, output, seed=61119, cases=None):
    command = [sys.executable, str(ROOT / 'tools' / tool), '--checkpoint', str(checkpoint),
               '--package', str(ROOT / 'go2_arm'), '--output', str(output), '--seed', str(seed),
               '--headless', '--device', 'cpu', '--kit_args', KIT]
    for case in cases or []:
        command.extend(['--case', case])
    return command


def run_probe(command, output):
    output = Path(output)
    if (output / 'result.json').is_file():
        print('Existing result, skipped:', output, flush=True)
        return 0
    if output.exists():
        raise RuntimeError(f'Inspect incomplete probe and logs before retrying: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    record = Path(str(output) + '.launch.json')
    # A launch record with no result also needs diagnosis, rather than automatic duplication.
    if record.exists():
        raise RuntimeError(f'Previous probe launch needs diagnosis: {record}')
    with Path(str(output) + '.log').open('w') as log:
        process = subprocess.Popen(command, env=environment(), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        record.write_text(json.dumps({'pid': process.pid, 'argv': command}, indent=2))
        code = process.wait()
    Path(str(output) + '.exitcode').write_text(str(code) + '\n')
    return code
