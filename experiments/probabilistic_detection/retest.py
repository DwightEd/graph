"""One-command test of the trained legacy conditional detector, not risk_response."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

from state_audit.storage import read_json, write_json


TASKS = ('QA', 'Summary', 'Data2txt')
FROZEN_FILES = ('models.joblib', 'selection.json', 'readouts.joblib',
                'readout_selection.json', 'detector_selection.json')


def check_inputs(args):
    """Require complete official test coverage and source separation before scoring."""
    manifest = read_json(args.cache / 'manifest.json')
    (Path(manifest['dataset']) / 'response.jsonl').stat()
    inventory = {}
    for task in args.tasks:
        for filename in FROZEN_FILES:
            (args.frozen / task / filename).stat()
        (args.frozen / 'packs' / f'{task}_test.npz').stat()
        metadata = read_json(args.frozen / 'packs' / f'{task}_test.json')
        records = metadata['records']
        actual = [row['id'] for row in records]
        expected = {row['id'] for row in manifest['records']
                    if row['task'] == task and row['split'] == 'test'}
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f'{task}: test pack does not cover the complete official task')
        selection = read_json(args.frozen / task / 'selection.json')
        occupied = set(selection['fit_sources']) | set(selection['dev_sources'])
        if occupied & set(metadata['sources']):
            raise ValueError(f'{task}: test sources overlap fitting or development')
        for row in records:
            (args.cache / row['directory'] / 'response.json').stat()
        inventory[task] = dict(answers=len(records), sources=len(metadata['sources']),
                              tokens=max(row['packed_stop'] for row in records))
    return inventory


def prepare_output(args, inventory):
    args.output.mkdir(parents=True, exist_ok=False)
    packs = args.output / 'packs'
    packs.mkdir()
    for task in args.tasks:
        destination = args.output / task
        destination.mkdir()
        for filename in FROZEN_FILES:
            shutil.copy2(args.frozen / task / filename, destination / filename)
        for suffix in ('json', 'npz'):
            filename = f'{task}_test.{suffix}'
            (packs / filename).symlink_to((args.frozen / 'packs' / filename).resolve())
    write_json(args.output / 'run.json', dict(
        method='legacy_supervised_conditional_readout_and_baselines',
        risk_response_detector=False, retrained=False, new_llm_forwards=0,
        frozen=str(args.frozen), cache=str(args.cache), tasks=inventory,
        bootstrap=args.bootstrap, threads=args.threads))


def summarize(output, tasks):
    summary = {}
    for task in tasks:
        metrics = read_json(output / task / 'test_metrics.json')
        summary[task] = {method: {name: metrics[method][name] for name in
            ('auroc', 'ap', 'token_fpr', 'token_recall', 'normal_answer_false_alarm')}
            for method in ('selected_readout', 'trees', 'source_refine')}
    write_json(output / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--frozen', type=Path,
                        default=Path('outputs/probabilistic_detection_20260928_full'))
    parser.add_argument('--cache', type=Path,
                        default=Path('outputs/native_support_ragtruth_all/source_first_v1'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--bootstrap', type=int, default=300)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check', action='store_true', help='Check inputs only; write nothing')
    args = parser.parse_args(argv)
    args.frozen = args.frozen.resolve()
    args.cache = args.cache.resolve()
    inventory = check_inputs(args)
    print('Complete test inputs:', inventory, flush=True)
    if args.check:
        return
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        args.output = Path('outputs') / f'probabilistic_test_{stamp}'
    args.output = args.output.resolve()
    prepare_output(args, inventory)

    from threadpoolctl import threadpool_limits
    from .run import score_task, evaluate_task
    with threadpool_limits(limits=args.threads):
        for task in args.tasks:
            score_task(args, task)
        write_json(args.output / 'SCORING_FREEZE.json', dict(tasks=args.tasks,
                   status='all_predictions_frozen_before_evaluation'))
        for task in args.tasks:
            evaluate_task(args, task)
    summary = summarize(args.output, args.tasks)
    write_json(args.output / 'complete.json', dict(status='complete', tasks=inventory))
    print('Legacy detector AUROC:', {task: values['selected_readout']['auroc']
          for task, values in summary.items()}, flush=True)
    print('Results:', args.output, flush=True)


if __name__ == '__main__':
    main()
