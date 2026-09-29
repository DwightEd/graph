"""Compact numeric audit and standalone paired propagation figure."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.span_maintenance.boundary import punctuation_boundaries
from .evaluate import FEATURES


def boundary_summary(tokens):
    result = {}
    for field in ('continuation', 'address_reuse', 'signed_reuse'):
        means = []
        for key in dict.fromkeys(row['key'] for row in tokens):
            selected = [row for row in tokens if row['key'] == key]
            boundary = punctuation_boundaries([row['text'] for row in selected])
            values = np.array([float(row[field]) for row in selected])
            inside = ~boundary
            inside[0] = False
            means.append((values[inside].mean(), values[boundary].mean()))
        result[field] = dict(zip(('inside', 'after_boundary'), np.mean(means, axis=0)))
    return result


def validate_rankings(output, tokens):
    with (output / 'units.csv').open() as stream:
        units = list(csv.DictReader(stream))
    summary = read_json(output / 'summary.json')
    for cohort, group in summary.items():
        selected = [row for row in units if row['cohort'] == cohort and int(row['gold']) >= 0]
        labels = np.array([int(row['gold']) for row in selected])
        for feature in FEATURES:
            values = np.array([float(row[feature]) for row in selected])
            # Independent pairwise definition, including exact ties.
            differences = values[labels == 1, None] - values[labels == 0][None]
            auc = np.mean((differences > 0) + .5 * (differences == 0))
            np.testing.assert_allclose(auc, group['features'][feature]['auroc_high_is_error'], atol=1e-12)
    gsm = [row for row in tokens if row['key'].startswith('gsm8k-')]
    assert all(int(row['token_gold']) == -1 for row in gsm)
    return len(summary) * len(FEATURES)


def plot_pairs(output, records):
    cases = [('00005', 0, 'Correct caps'), ('00006', 0, 'Incorrect headdress'),
             ('00013', 0, 'Correct grilling time'), ('00012', 0, 'Incorrect onion time'),
             ('gsm8k-243', 0, 'Correct base 20'), ('gsm8k-49', 0, 'Incorrect base 24')]
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for axis, (key, index, title) in zip(axes.flat, cases):
        meta = read_json(output / key / f'propagation_diagnostic_{index}.json')
        target = meta['probe']['target']
        values = np.load(output / key / f'propagation_diagnostic_{index}.npz')['effect_margin_slope']
        stop = min(len(values), target + 61)
        axis.plot(np.arange(target + 1, stop) - target, values[target + 1:stop], color='#126D91')
        boundaries = np.flatnonzero(punctuation_boundaries(records[key]['response']['token_text']))
        for position in boundaries[(boundaries > target) & (boundaries < stop)]:
            axis.axvline(position - target, color='#aaaaaa', alpha=.4, linewidth=.7)
        axis.axhline(0, color='#444444', linewidth=.6)
        axis.set_title(f'{key}: {title}\nsource={meta["audits"][0]["text"]}; current slope={values[target]:.3f}')
        axis.set_xlabel('Tokens after intervened receiver')
        axis.set_ylabel('Signed margin response')
    figure.suptitle('Native downstream propagation; grey lines: punctuation boundaries\nDifferent target rivals: compare structure, not factual support magnitude', fontsize=12)
    figure.savefig(output / 'propagation_pairs.png', dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with (args.output / 'tokens.csv').open() as stream:
        tokens = list(csv.DictReader(stream))
    records = {row['key']: row for row in read_json(args.output / 'manifest.json')['records']}
    ranking_checks = validate_rankings(args.output, tokens)
    audits = []
    for path in args.output.glob('*/propagation_*.json'):
        result = read_json(path)
        for audit in result['audits']:
            audits.append(dict(key=path.parent.name, measurement=path.stem,
                target=result['probe']['target'], **audit))
    write_json(args.output / 'all_intervention_audits.json', audits)
    write_json(args.output / 'verification.json', dict(status='complete', reviewer='same agent',
        ranking_checks=ranking_checks, per_token_rows=len(tokens), intervention_conditions=len(audits),
        max_reconstruction=max(row['reconstruction'] for row in audits),
        max_effect_before_receiver=max(row['earlier_effect'] for row in audits),
        gsm_token_labels='all unknown; only step labels evaluated',
        boundary_answer_equal_means=boundary_summary(tokens)))
    plot_pairs(args.output, records)
    print('verified', ranking_checks, 'rankings and', len(audits), 'intervention conditions', flush=True)


if __name__ == '__main__':
    main()
