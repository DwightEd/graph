"""Unlabeled reference scaling and calibration; never imports annotations."""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from .measure import normalize, propagate, branch_alpha, branch_relative_js

METHODS = ('attention_js', 'influence_js', 'read_use_js', 'prompt_deficit',
           'combined', 'propagated', 'permuted', 'entropy', 'surprisal',
           'raw_route', 'raw_route_offline_mean', 'route_combined',
           'relative_influence_js', 'relative_propagated')


def derive_branch_readout(directory):
    with np.load(directory / 'readouts.npz') as saved:
        count = saved['measured'].shape[2]
        divergence = saved['measured'][:, :, :, 1].reshape(1024, count)
        prompt = int(saved['prompt_length'])
    derivative = np.load(directory / 'derivative.npy', mmap_mode='r').reshape(1024, count, -1)
    alpha = np.empty((1024, count))
    for head in range(1024):
        alpha[head] = branch_alpha(normalize(np.abs(derivative[head])), prompt)
    relative = branch_relative_js(divergence, alpha)
    np.savez_compressed(directory / 'branch.npz', alpha=alpha, relative_js=relative)


def valid_tokens(record, output, old):
    if record['kind'] == 'observer':
        return np.load(old / 'static' / (record['key'] + '.npz'))['valid']
    samples = Path(read_json(output / 'manifest.json')['samples'])
    with np.load(samples / record['trace']) as trace:
        return ~trace['special_mask'][int(trace['prompt_length']):]


def fit_cdf(values, weights):
    valid = np.isfinite(values)
    order = np.argsort(values[valid], kind='stable')
    ordered = values[valid][order]
    cumulative = np.r_[0., np.cumsum(weights[valid][order])]
    if len(ordered):
        cumulative /= cumulative[-1]
    return ordered, cumulative


def percentile(values, fitted):
    ordered, cumulative = fitted
    if not len(ordered):
        return np.full(values.shape, np.nan)
    left = np.searchsorted(ordered, values, side='left')
    right = np.searchsorted(ordered, values, side='right')
    result = (cumulative[left] + cumulative[right]) / 2
    return np.where(np.isfinite(values), result, np.nan)


def task_reference(records, task, output, old):
    arrays, weights = [], []
    for row in records:
        if row['role'] != 'fit' or row['task'] != task:
            continue
        valid = valid_tokens(row, output, old)
        data = np.load(output / row['key'] / 'readouts.npz')['measured']
        relative = np.load(output / row['key'] / 'branch.npz')['relative_js'][:, valid]
        selected = data[:, :, valid, :4].reshape(1024, valid.sum(), 4)
        arrays.append(np.concatenate((selected, relative[:, :, None]), -1))
        weights.append(np.full(valid.sum(), 1 / valid.sum()))
    joined, weight = np.concatenate(arrays, 1), np.concatenate(weights)
    return [[fit_cdf(joined[head, :, field], weight) for field in range(5)]
            for head in range(1024)]


def score_answer(output, row, reference, route_reference):
    directory = output / row['key']
    with np.load(directory / 'readouts.npz') as saved:
        raw = saved['measured']
        observed = saved['confidence']
        prompt = int(saved['prompt_length'])
    count = raw.shape[2]
    relative = np.load(directory / 'branch.npz')['relative_js']
    raw = np.concatenate((raw.reshape(1024, count, -1)[:, :, :4], relative[:, :, None]), -1)
    ranks = np.empty((1024, count, 5), dtype=np.float32)
    inherited, shuffled = np.empty((1024, count)), np.empty((1024, count))
    relative_inherited = np.empty((1024, count))
    derivative = np.load(directory / 'derivative.npy', mmap_mode='r').reshape(1024, count, -1)
    for head in range(1024):
        for field in range(5):
            ranks[head, :, field] = percentile(raw[head, :, field], reference[head][field])
        # Missing relay branch omits this component, with its mask retained.
        combined = np.nanmean(ranks[head, :, 1:4], -1)
        influence = normalize(np.abs(derivative[head]))
        inherited[head] = propagate(combined, influence, prompt)
        shuffled[head] = propagate(combined, influence, prompt, permuted=True)
        relative_combined = np.nanmean(ranks[head][:, [4, 2, 3]], -1)
        relative_inherited[head] = propagate(relative_combined, influence, prompt)
    score = {}
    for field, name in enumerate(METHODS[:4]):
        values = np.nanquantile(raw[:, :, field], .9, axis=0)
        # Undefined JS at the first positions remains explicitly missing.
        score[name] = values
    score['combined'] = np.quantile(np.nanmean(ranks[:, :, 1:4], -1), .9, axis=0)
    score['propagated'] = np.quantile(inherited, .9, axis=0)
    score['permuted'] = np.quantile(shuffled, .9, axis=0)
    score['relative_influence_js'] = np.nanquantile(relative, .9, axis=0)
    score['relative_propagated'] = np.quantile(relative_inherited, .9, axis=0)
    score['entropy'], score['surprisal'] = observed[:, 0], observed[:, 1]
    if row['kind'] == 'observer':
        # Access only fixed unlabeled baselines, never selected_scores or targets.
        with np.load(Path(row['root']) / row['directory'] / 'scores.npz') as baseline:
            for name in ('raw_route', 'raw_route_offline_mean'):
                score[name] = baseline[name]
        route_rank = percentile(score['raw_route'], route_reference)
        score['route_combined'] = .5 * (route_rank + score['propagated'])
    else:
        for name in ('raw_route', 'raw_route_offline_mean', 'route_combined'):
            score[name] = np.full(count, np.nan)
    np.savez_compressed(directory / 'scores.npz', **score, head_percentiles=ranks,
                        head_propagated=inherited, head_permuted=shuffled,
                        head_relative_propagated=relative_inherited)


def calibrate(output, records, old):
    thresholds = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        scores = {method: [] for method in METHODS}
        weights = []
        for row in records:
            if row['role'] != 'dev' or row['task'] != task:
                continue
            valid = valid_tokens(row, output, old)
            data = np.load(output / row['key'] / 'scores.npz')
            weights.append(np.full(valid.sum(), 1 / valid.sum()))
            for name in METHODS:
                scores[name].append(data[name][valid])
        weight = np.concatenate(weights)
        thresholds[task] = {}
        for name, arrays in scores.items():
            ordered, cumulative = fit_cdf(np.concatenate(arrays), weight)
            thresholds[task][name] = float(ordered[np.searchsorted(cumulative[1:], .95)])
    write_json(output / 'thresholds.json', thresholds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old', type=Path, default=Path('outputs/transport_topology_cases_20260928'))
    args = parser.parse_args()
    read_json(args.output / 'capture_complete.json')
    records = read_json(args.output / 'manifest.json')['records']
    for row in records:
        if not (args.output / row['key'] / 'branch.npz').exists():
            derive_branch_readout(args.output / row['key'])
        print('branch readout', row['key'], flush=True)
    for task in ('QA', 'Summary', 'Data2txt'):
        reference = task_reference(records, task, args.output, args.old)
        route_values, route_weights = [], []
        for row in records:
            if row['role'] == 'fit' and row['task'] == task:
                valid = valid_tokens(row, args.output, args.old)
                with np.load(Path(row['root']) / row['directory'] / 'observations.npz') as baseline:
                    route_values.append(baseline['raw_route'][valid])
                route_weights.append(np.full(valid.sum(), 1 / valid.sum()))
        route_reference = fit_cdf(np.concatenate(route_values), np.concatenate(route_weights))
        for row in records:
            if row['task'] == task:
                if not (args.output / row['key'] / 'scores.npz').exists():
                    score_answer(args.output, row, reference, route_reference)
                print('scored', row['key'], flush=True)
    calibrate(args.output, records, args.old)
    write_json(args.output / 'scores_frozen.json', dict(status='all scores and thresholds frozen',
        labels_read=False, methods=METHODS, main='propagated', answers=len(records),
        missing_js='NaN; evaluated coverage reported; omitted from component mean when unavailable',
        threshold='source-balanced dev mixture quantile .95, no normal filtering'))


if __name__ == '__main__':
    main()
