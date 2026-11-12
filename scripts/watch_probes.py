"""Run the scheduled 1000/3000/9000 CPU probes for one fresh training run."""
import argparse
import fcntl
import json
from pathlib import Path
import time

from probe_jobs import probe_command, run_probe


def trainer_matches(pid, ticks, run):
    try:
        process = Path('/proc') / str(pid)
        state = (process / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = (process / 'cmdline').read_bytes().split(b'\0')
        return state[0] != 'Z' and int(state[19]) == ticks and str(run).encode() in argv
    except (OSError, ValueError):
        return False


def watch(args):
    run = args.run_dir.resolve()
    run.parent.mkdir(parents=True, exist_ok=True)
    with (run.parent / ('.' + run.name + '.probes.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for step in [1000, 3000, 9000]:
            if step >= args.updates:
                continue
            checkpoint = run / f'model_{step}.pt'
            while not checkpoint.is_file():
                if any((run / name).exists() for name in ['train.exitcode', 'health_failure.json', 'intentional_stop.json']):
                    raise RuntimeError(f'Training ended before planned checkpoint {step}')
                if not trainer_matches(args.trainer_pid, args.trainer_start_ticks, run):
                    raise RuntimeError('Trainer identity no longer matches this run')
                time.sleep(10)
            output = run / f'style_probes/model_{step}'
            cases = ['stop', 'vx_+0.26', 'vx_-0.26', 'vy_+0.23', 'vy_-0.23'] if step < 9000 else None
            jobs = [(probe_command('direction_probe.py', checkpoint, output, cases=cases), output)]
            for seed in ([61119, 61120] if step == 3000 else [61119]):
                suffix = '_seed61120' if seed == 61120 else ''
                output = run / f'standing_probes/model_{step}{suffix}'
                jobs.append((probe_command('stop_transition_probe.py', checkpoint, output, seed=seed), output))
            for command, output in jobs:
                code = run_probe(command, output)
                if code:
                    return code
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--trainer-pid', type=int, required=True)
    parser.add_argument('--trainer-start-ticks', type=int, required=True)
    args = parser.parse_args()
    raise SystemExit(watch(args))
