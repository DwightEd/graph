"""Reuse completed phases; capture and evaluate the unsupervised native operator."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('outputs/message_js_20260928_v1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (args.base / 'scores_frozen.json').exists():
        raise FileNotFoundError('First run experiments.message_js.run to create the base measurements.')
    stages = (('operator_capture', 'capture_complete.json'), ('operator_score', 'scores_frozen.json'),
              ('evaluate', 'evaluation.json'), ('report', 'CASE_BROWSER.html'),
              ('operator_verify', 'verification.json'), ('score_verify', 'score_verification.json'))
    for stage, marker in stages:
        if (args.output / marker).exists():
            print('Reusing completed phase:', stage, flush=True)
            continue
        command = [sys.executable, '-m', 'experiments.message_js.' + stage, '--output', str(args.output)]
        if stage == 'operator_capture':
            command.extend(['--base', str(args.base)])
            if (args.output / 'manifest.json').exists():
                command.append('--resume')
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
