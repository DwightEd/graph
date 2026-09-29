"""Boundary diagnostics, pooling contamination and selected real-prefix checks."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .boundary import attention_layer, punctuation_boundaries
from .run import read_json, write_json
from .state import FIELDS, maintain_layer


def boundary_diagnostics(record, directory):
    with np.load(directory / 'states.npz') as saved:
        state = saved['state'].mean((0, 1))
    cue = punctuation_boundaries(record['response']['token_text'])
    report = dict(key=record['key'], boundaries=int(cue.sum()), tokens=len(cue))
    for name in ('prompt_refresh', 'continuation', 'history_mass', 'age', 'distance_excess'):
        values = state[:, FIELDS.index(name)]
        report[name + '_at_hint'] = float(values[cue].mean()) if cue.any() else None
        report[name + '_away'] = float(values[~cue].mean())
    if cue.any() and (~cue).any():
        report['punctuation_proxy_auc'] = float(roc_auc_score(cue, 1 - state[:, FIELDS.index('continuation')]))
    return report


def pooling_contamination(record, directory, positions):
    if record['dataset'] != 'ragtruth' or record['original']['kind'] != 'observer':
        return []
    rows = positions[positions.key == record['key']].sort_values('position')
    gold = rows.gold.to_numpy()
    with np.load(directory / 'weights.npz') as saved:
        kernels = {name: saved[name].copy() for name in ('state', 'dependency')}
    if (directory / 'boundary_weights.npz').exists():
        with np.load(directory / 'boundary_weights.npz') as saved:
            kernels.update({name: saved[name].copy() for name in saved.files})
    count = len(gold)
    uniform = np.zeros((count, count))
    for target in range(count):
        start, stop = max(0, target - 7), min(count, target + 9)
        uniform[target, start:stop] = 1 / (stop - start)
    kernels['offline16'] = uniform
    result = []
    for name, kernel in kernels.items():
        wrong_mass = kernel @ (gold == 1)
        for index, row in enumerate(rows.itertuples()):
            result.append(dict(key=record['key'], position=int(row.position), gold=int(row.gold),
                position_group=row.position_group, method=name, wrong_token_weight=float(wrong_mass[index]),
                self_weight=float(kernel[index, index])))
    return result


def real_prefix_checks(records, output):
    results = []
    selected = {'15604', '00006', 'gsm8k-49'}
    for row in records:
        if row['key'] not in selected:
            continue
        attention = attention_layer(row, 14)
        prompt = len(row['prompt'])
        for target in range(attention.shape[1]):
            assert not np.any(attention[:, target, prompt + target:] != 0)
        prefix = min(37, attention.shape[1])
        trimmed = attention[:, :prefix, :prompt + prefix - 1]
        observed, weights, _ = maintain_layer(trimmed, prompt, row['response']['answer_ids'][:prefix])
        with np.load(output / row['key'] / 'states.npz') as saved:
            actual = saved['state'][14, :, :prefix]
        error = float(np.max(np.abs(observed - actual)))
        np.testing.assert_allclose(observed, actual, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-12)
        results.append(dict(key=row['key'], layer=14, prefix=prefix, max_float32_error=error, future_keys=0))
    return results


def make_plot(output, positions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)
    for axis, key in zip(axes, ('15604', '00006', '00013')):
        rows = positions[positions.key == key].sort_values('position')
        axis.plot(rows.position, rows.continuation, label='reading retention')
        axis.plot(rows.position, rows.prompt_refresh, label='prompt address refresh')
        axis.plot(rows.position, rows.signed_stability, label='signed margin-profile overlap', alpha=.7)
        for row in rows.itertuples():
            if row.gold == 1:
                axis.axvspan(row.position - .5, row.position + .5, color='red', alpha=.08)
        axis.set(title=key + ' (red = known erroneous tokens; unlabeled positions not presumed correct)',
            xlabel='Answer token index', ylim=(0, 1))
        axis.legend(loc='upper right')
    figure.savefig(output / 'STATE_TRACES.svg')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = read_json(args.output / 'manifest.json')['records']
    positions = pd.read_csv(args.output / 'positions.csv', dtype={'key': str})
    boundary = [boundary_diagnostics(row, args.output / row['key']) for row in records]
    contamination = [item for row in records for item in pooling_contamination(row, args.output / row['key'], positions)]
    pd.DataFrame(boundary).to_csv(args.output / 'boundary_diagnostics.csv', index=False)
    frame = pd.DataFrame(contamination)
    frame.to_csv(args.output / 'pooling_contamination.csv', index=False)
    summary = frame.groupby(['method', 'position_group'])[['wrong_token_weight', 'self_weight']].mean().reset_index()
    summary.to_csv(args.output / 'pooling_summary.csv', index=False)
    checks = real_prefix_checks(records, args.output)
    write_json(args.output / 'audit.json', dict(status='complete', real_prefix_checks=checks,
        boundary_proxy='punctuation cue, not annotated semantic boundary',
        contamination='gold consulted only for post-scoring audit; no labels used by kernels'))
    make_plot(args.output, positions)
    print('audited', len(records), 'answers;', len(checks), 'real prefix checks', flush=True)


if __name__ == '__main__':
    main()
