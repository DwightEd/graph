"""Run the fixed twelve-answer maintenance and source-control pilot."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory to preserve previous measurements.')
    subprocess.run([sys.executable, '-m', 'unittest',
        'experiments.span_source_control.test_control', '-v'], check=True)
    for stage in ('measure', 'finite', 'memory', 'evaluate', 'diagnostics', 'report'):
        subprocess.run([sys.executable, '-m', f'experiments.span_source_control.{stage}',
            '--output', str(args.output)], check=True)


if __name__ == '__main__':
    main()
