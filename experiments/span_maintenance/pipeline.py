"""One command for the cached pilot, two fixed variants and post-score audits."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def invoke(stage, *arguments):
    command = [sys.executable, '-m', 'experiments.span_maintenance.' + stage, *map(str, arguments)]
    print('RUN', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    os.environ['PYTHONPATH'] = os.pathsep.join((str(root), str(root / 'teaching/state_audit/src')))
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    first = Path(str(args.output_prefix) + '_v1')
    second = Path(str(args.output_prefix) + '_v2')
    invoke('run', '--output', first)
    invoke('evaluate', '--output', first)
    invoke('audit', '--output', first)
    invoke('boundary', '--previous', first, '--output', second)
    invoke('evaluate', '--output', second)
    invoke('audit', '--output', second)
    invoke('payload', '--output', second)


if __name__ == '__main__':
    main()
