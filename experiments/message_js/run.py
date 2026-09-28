"""One command: capture, unlabeled score/calibration, then evaluate."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    stages = (('capture', 'capture_complete.json'), ('score', 'scores_frozen.json'),
              ('evaluate', 'evaluation.json'))
    for stage, marker in stages:
        if (args.output / marker).exists():
            print('Reusing completed stage:', stage, flush=True)
            continue
        subprocess.run([sys.executable, '-m', 'experiments.message_js.' + stage,
                        '--output', str(args.output)], check=True)
    subprocess.run([sys.executable, '-m', 'experiments.message_js.report',
                    '--output', str(args.output)], check=True)
    subprocess.run([sys.executable, '-m', 'experiments.message_js.verify',
                    '--output', str(args.output)], check=True)


if __name__ == '__main__':
    main()
