"""Reproduce the frozen scalar baseline without fitting the retired graph models.

The historical baseline uses unit/source means and an offline route window.
It remains a comparison, not the new independent token measurement method.
"""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from .data import load_inputs
from .scalar import fit_ranks, rank_scores, scalar_scores


TASKS = ('QA', 'Summary', 'Data2txt')
PACKS = Path('outputs/probabilistic_detection_20260928_full/packs')
HISTORICAL = Path('outputs/unsupervised_graph_20260928')


def fit_reference(pack):
    """Only scalar measurements and the pre-existing unlabeled partition enter."""
    raw = scalar_scores(pack)
    fit = {name: value[~pack['development']] for name, value in raw.items()}
    reference = fit_ranks(fit)
    dev = {name: value[pack['development']] for name, value in raw.items()}
    values = rank_scores(dev, reference)['fixed_unsupervised']
    return reference, float(np.quantile(values, .95, method='higher'))


def reproduce(packs, historical, task):
    """Compare all cached partitions bit for bit; never read annotation arrays."""
    train, _ = load_inputs(packs, task, 'train')
    reference, threshold = fit_reference(train)
    values = rank_scores(scalar_scores(train), reference)['fixed_unsupervised']
    partitions = {'fit_scores': values[~train['development']],
                  'development_scores': values[train['development']]}
    test, metadata = load_inputs(packs, task, 'test')
    partitions['test_scores'] = rank_scores(scalar_scores(test), reference)['fixed_unsupervised']
    errors = {}
    for name, current in partitions.items():
        with np.load(historical / task / (name + '.npz'), allow_pickle=False) as saved:
            original = saved['fixed_unsupervised']
        np.testing.assert_array_equal(current, original)
        errors[name] = dict(tokens=len(current), max_absolute_error=float(np.abs(current - original).max()))
    original_threshold = read_json(historical / task / 'selection.json')['mixed_thresholds']['fixed_unsupervised']
    if threshold != original_threshold:
        raise ValueError(f'{task}: frozen threshold changed: {threshold} != {original_threshold}')
    return dict(partitions=errors, threshold=threshold, test_answers=len(metadata['records']),
                labels_accessed=False, new_model_forwards=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packs', type=Path, default=PACKS)
    parser.add_argument('--historical', type=Path, default=HISTORICAL)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = {task: reproduce(args.packs, args.historical, task) for task in TASKS}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, dict(status='pass', tasks=report,
        formula='.75 * historical_midrank(pair_unit_source) + .25 * historical_midrank(route_window)',
        historical_aggregation_preserved=True, labels_accessed=False))
    print('Fixed baseline reproduced exactly for all three tasks.', flush=True)


if __name__ == '__main__':
    main()
