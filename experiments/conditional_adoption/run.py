"""Reproduce the frozen eight-answer native-response research pilot.

Use a fresh output directory. The cohort is exploratory and this command does
not run the three-task benchmark or fit a detector with hallucination labels.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .readout import file_hash


SOURCE_FIRST = Path('outputs/native_support_ragtruth_all/source_first_v1')


def command(module, *arguments):
    subprocess.run([sys.executable, '-m', f'experiments.conditional_adoption.{module}',
                    *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--relay', action='store_true', help='also measure current/past/both source amplification')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshots = args.output / 'executed_code'
    snapshots.mkdir()
    hashes = {}
    for path in sorted(Path(__file__).parent.glob('*.py')):
        (snapshots / path.name).write_bytes(path.read_bytes())
        hashes[str(path.resolve())] = file_hash(path)
    started = time.time()
    command('capture', '--prepare-only', '--output', args.output / 'inputs')
    if args.prepare_only:
        (args.output / 'RUN.json').write_text(json.dumps(dict(status='PREPARED', hashes=hashes), indent=2) + '\n')
        return
    command('capture', '--inputs', args.output / 'inputs' / 'INPUTS.json', '--output', args.output / 'capture')
    command('readout', '--capture', args.output / 'capture', '--output', args.output / 'scores')
    command('evaluate', '--scores', args.output / 'scores', '--source-root', SOURCE_FIRST)
    if args.relay:
        command('relay', '--inputs', args.output / 'inputs' / 'INPUTS.json',
                '--capture', args.output / 'capture', '--output', args.output / 'relay')
        command('readout', '--relay', '--capture', args.output / 'relay', '--output', args.output / 'relay_scores')
        command('evaluate', '--scores', args.output / 'relay_scores', '--source-root', SOURCE_FIRST)
    (args.output / 'RUN.json').write_text(json.dumps(dict(status='DONE', seconds=time.time() - started,
        hashes=hashes, source_label_fit=False, uses_entropy_for_error=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
