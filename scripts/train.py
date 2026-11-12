"""Portable supervisor for the unchanged v11 training entry point."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from probe_jobs import ROOT, environment, probe_command, run_probe
from verify_bundle import verify


def checkpoint_report(path, expected_iteration):
    import torch
    torch.set_num_threads(1)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(x) for x in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite(x) for x in value)
        return True
    assert finite(saved), 'Nonfinite checkpoint'
    assert saved['iteration'] == expected_iteration
    assert not saved['manifest']['is_fixed_base'] and saved['manifest']['home_consistency_verified']
    return {'checkpoint': str(path), 'iteration': saved['iteration'], 'finite': True,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--updates', type=int, default=20000)
    parser.add_argument('--stage1-budget', type=int, default=40000)
    parser.add_argument('--stage2-budget', type=int, default=40000)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Print the exact command without starting a job')
    args = parser.parse_args()
    if args.updates <= 0 or args.num_envs <= 0:
        parser.error('updates and num-envs must be positive')
    if args.updates > args.stage1_budget + args.stage2_budget:
        parser.error('updates exceeds the full two-stage budget')
    if args.stage1_budget < args.updates < args.stage1_budget + args.stage2_budget:
        parser.error('Use a bounded stage1 trial or the full two-stage budget')
    run = args.run_dir.resolve()
    verify()
    command = [sys.executable, str(ROOT / 'go2_arm/train_duet.py'), '--headless', '--device', args.device,
               '--num_envs', str(args.num_envs), '--amp', '--phase_free_amp', '--directional_amp',
               '--hip_splay_penalty', '--stage1_iterations', str(args.stage1_budget),
               '--stage2_iterations', str(args.stage2_budget), '--run_dir', str(run), '--kit_args',
               '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4']
    if args.updates <= args.stage1_budget:
        command += ['--stop_after_updates', str(args.updates)]
    if args.smoke:
        command.append('--smoke')
    if args.dry_run:
        print(json.dumps({'argv': command, 'scheduled_probes': not args.smoke,
                          'expected_final_iteration': args.updates - 1}, indent=2))
        return 0
    if run.exists():
        parser.error('Run directory already exists; use a fresh directory. Resume is not validated.')
    run.parent.mkdir(parents=True, exist_ok=True)
    watcher = None
    watcher_log = None
    with Path(str(run) + '.log').open('x') as log:
        trainer = subprocess.Popen(command, env=environment(4), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        ticks = int(Path(f'/proc/{trainer.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
        launch = {'pid': trainer.pid, 'start_ticks': ticks, 'argv': command}
        Path(str(run) + '.launch.json').write_text(json.dumps(launch, indent=2))
        print(json.dumps(launch), flush=True)
        try:
            if not args.smoke:
                deadline = time.monotonic() + 120
                while not run.is_dir() and trainer.poll() is None and time.monotonic() < deadline:
                    time.sleep(.25)
                if run.is_dir() and trainer.poll() is None:
                    watcher_log = (run / 'style_probe_watcher.log').open('x')
                    watcher = subprocess.Popen([sys.executable, str(ROOT / 'scripts/watch_probes.py'),
                                                '--run-dir', str(run), '--updates', str(args.updates),
                                                '--trainer-pid', str(trainer.pid), '--trainer-start-ticks', str(ticks)],
                                               env=environment(), stdin=subprocess.DEVNULL,
                                               stdout=watcher_log, stderr=subprocess.STDOUT)
                    (run / 'style_probe_watcher.pid').write_text(str(watcher.pid))
            code = trainer.wait()
        except BaseException:
            if trainer.poll() is None:
                trainer.terminate()
                trainer.wait()
            if watcher is not None and watcher.poll() is None:
                watcher.terminate()
                watcher.wait()
            raise
        finally:
            if watcher_log is not None:
                watcher_log.close()
    Path(str(run) + '.supervisor_exitcode').write_text(str(code) + '\n')
    if code:
        if watcher is not None and watcher.poll() is None:
            watcher.wait(timeout=30)
        return code
    result = json.loads((run / 'result.json').read_text())
    final = checkpoint_report(Path(result['checkpoint']), result['completed_iterations'] - 1)
    final['completed_iterations'] = result['completed_iterations']
    (run / 'final_checkpoint_validation.json').write_text(json.dumps(final, indent=2))
    if result['completed_iterations'] != args.updates:
        raise RuntimeError('Training ended before requested budget; inspect stage gate and result.json')
    if watcher is not None:
        watcher_code = watcher.wait()
        (run / 'style_probe_watcher.exitcode').write_text(str(watcher_code) + '\n')
        if watcher_code:
            return watcher_code
    if args.smoke:
        return 0
    checkpoint = Path(final['checkpoint'])
    output = run / 'direction_probe'
    jobs = [(probe_command('direction_probe.py', checkpoint, output), output)]
    for seed in [61119, 61120]:
        output = run / f'final_standing_probe_seed{seed}'
        jobs.append((probe_command('stop_transition_probe.py', checkpoint, output, seed=seed), output))
    for command, output in jobs:
        code = run_probe(command, output)
        if code:
            return code
    print('Training and scheduled diagnostic processes completed; independent acceptance still requires review.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
