"""Recompute true continuous-contact time without modifying original probe evidence."""
import argparse
import json
from pathlib import Path


def analyze(directory):
    report = json.loads((directory / 'result.json').read_text())
    traces = json.loads((directory / 'traces.json').read_text())
    assert len(traces) == len(report['cases'])
    results = []
    for case, trace in zip(report['cases'], traces):
        for phase, start, end in [('stop', 6., 14.), ('restop', 18., 24.)]:
            first = onset = confirmed = None
            for point in trace:
                if not start <= point['t'] < end:
                    continue
                if not all(point['contact']):
                    first = None
                    continue
                if first is None:
                    first = point['t']
                if point['t'] - first >= .5 - 1e-8:
                    onset = max(0., first - start)
                    confirmed = point['t'] - start
                    break
            results.append({'case': case['case'], 'policy': case['policy'], 'phase': phase,
                            'onset_s': onset, 'confirmed_after_s': confirmed})
    return {'checkpoint_sha256': report['sha256'], 'seed': report['seed'],
            'criterion': 'at least .50 seconds elapsed continuous all-four contact', 'segments': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.probe_dir)
    output = args.probe_dir / 'settle_reanalysis.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'settled': sum(x['onset_s'] is not None for x in result['segments']),
                      'segments': len(result['segments'])}))
