"""Run the fixed 48-answer route comparison and separate posthoc diagnostics."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--operator', type=Path, default=Path('outputs/message_operator_20260928_v2'))
    parser.add_argument('--model', default='/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct')
    args = parser.parse_args()
    stages = [
        ('route_complement.capture', 'capture_complete.json', ['--operator', str(args.operator), '--model', args.model]),
        ('route_complement.score', 'scores_frozen.json', []),
        ('message_js.evaluate', 'evaluation.json', []),
        ('route_complement.verify', 'verification.json', []),
        ('message_js.score_verify', 'score_verification.json', []),
        ('message_js.report', 'CASE_BROWSER.html', []),
        ('route_complement.diagnose', 'diagnostics/summary.json', []),
        ('route_complement.head_null', 'diagnostics/head_identity_null.json', []),
        ('route_complement.head_profiles', 'diagnostics/unlabeled_head_profiles.json', []),
        ('route_complement.browser', 'diagnostics/HEAD_BROWSER.html', []),
    ]
    for module, marker, extra in stages:
        if (args.output / marker).exists():
            print('Complete; preserving:', module, flush=True)
            continue
        subprocess.run([sys.executable, '-m', 'experiments.' + module,
                        '--output', str(args.output), *extra], check=True)


if __name__ == '__main__':
    main()
