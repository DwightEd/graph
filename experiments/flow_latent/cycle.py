"""Run the predeclared three iterations; retain every version and evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def run_cycle(capture, prefix, seeds):
    logs = prefix.parent / (prefix.name + '_cycles')
    logs.mkdir(exist_ok=True)
    for seed in seeds:
        with (logs / f'seed{seed}.log').open('x') as log:
            for version in (1, 2, 3):
                suffix = '' if seed == 42 else f'_seed{seed}'
                output = prefix.parent / f'{prefix.name}_v{version}{suffix}'
                command = [sys.executable, '-m', 'experiments.flow_latent.run',
                           '--capture', str(capture), '--output', str(output),
                           '--version', str(version), '--seed', str(seed)]
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                subprocess.run([sys.executable, '-m', 'experiments.flow_latent.evaluate',
                                '--output', str(output)], stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads((output / 'results.json').read_text())
                summary = {name: round(row['auroc'], 6) for name, row in result['metrics'].items()}
                print(f'COMPLETE v{version} seed={seed} AUC={summary}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    args = parser.parse_args()
    run_cycle(args.capture, args.prefix, args.seeds)
