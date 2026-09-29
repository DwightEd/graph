"""Report frozen contrasts without fitting, ranking labels or choosing a winner."""
import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json


def read_run(directory, name):
    with np.load(directory / f'{name}.npz') as data:
        return {key: data[key] for key in data.files}


def first_difference(left, right):
    common = min(len(left), len(right))
    changed = np.flatnonzero(left[:common] != right[:common])
    if len(changed):
        return int(changed[0])
    return common if len(left) != len(right) else None


def summarize_case(case, directory):
    runs = {name: read_run(directory, name) for name in (
        'clamp_zero', 'clamp_minus', 'clamp_plus', 'control_plus',
        'force_zero', 'force_plus', 'greedy_zero', 'greedy_plus', 'greedy_force')}
    zero, plus, minus = (runs[name] for name in ('clamp_zero', 'clamp_plus', 'clamp_minus'))
    source = plus['margin'] - zero['margin']
    forced = runs['force_zero']['margin'] - zero['margin']
    interaction = runs['force_plus']['margin'] - runs['force_zero']['margin'] - source
    delta_kv = plus['kv'] - minus['kv']
    norm = np.linalg.norm(zero['kv'], axis=-1)
    relative_kv = np.linalg.norm(delta_kv, axis=-1) / np.maximum(norm, 1e-12)
    before_layer = float(np.max(np.abs(delta_kv[:, :case['probe']['layer'] + 1])))
    assert before_layer < 1e-5, ('source changed upstream layer KV', case['name'], before_layer)
    assert abs(forced[0]) < 1e-5, ('feedback changed preceding prediction', case['name'])
    np.savez_compressed(directory / 'responses.npz', source_plus=source,
        source_slope=plus['margin'] - minus['margin'], first_token_effect=forced,
        source_token_interaction=interaction, kv_relative_slope=relative_kv)
    selected = sorted({0, 1, 8, 19, 27, len(source) - 1})
    aligned = [dict(target=case['target'] + lag, lag=lag,
        original_margin=float(zero['margin'][lag]), source_plus=float(source[lag]),
        source_slope=float(plus['margin'][lag] - minus['margin'][lag]),
        first_token_effect=float(forced[lag]), interaction=float(interaction[lag])) for lag in selected]
    return dict(aligned=aligned, source_first_margin=float(zero['margin'][0]),
        source_first_slope=float(plus['margin'][0] - minus['margin'][0]),
        control_first_effect=float(runs['control_plus']['margin'][0] - zero['margin'][0]),
        source_masses=[float(run['mass']) for run in (minus, zero, plus)],
        upstream_kv_max_change=before_layer,
        max_kv_relative_slope=float(relative_kv.max()),
        greedy_source_first_difference=first_difference(runs['greedy_zero']['tokens'], runs['greedy_plus']['tokens']),
        greedy_forced_first_difference=first_difference(runs['greedy_zero']['tokens'], runs['greedy_force']['tokens']))


def plot_responses(output, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    for case, axis in zip(cases, axes.flat):
        data = read_run(output / case['name'], 'responses')
        positions = case['target'] + np.arange(len(data['source_plus']))
        axis.plot(positions, data['source_plus'], label='source event (+0.5)')
        axis.plot(positions, data['first_token_effect'], label='swap first numeral')
        axis.plot(positions, data['source_token_interaction'], label='interaction', linestyle='--')
        axis.axhline(0, color='gray', linewidth=.5)
        axis.set_title(case['name'])
        axis.set_xlabel('Answer token index (fixed continuation)')
        axis.set_ylabel('Change in logit(24) - logit(20)')
    axes[0, 0].legend(fontsize=8)
    figure.savefig(output / 'feedback_responses.png', dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    complete = read_json(args.output / 'complete.json')
    cases = read_json(args.output / 'plan.json')['cases']
    summaries = {case['name']: summarize_case(case, args.output / case['name']) for case in cases}
    role_effect = summaries['role_repaired']['source_first_margin'] - summaries['wrong_base']['source_first_margin']
    source_interaction = summaries['role_repaired']['source_first_slope'] - summaries['wrong_base']['source_first_slope']
    error = max(run['full_recompute_max_error'] for runs in complete['runs'].values()
                for run in runs if 'full_recompute_max_error' in run)
    result = dict(cases=summaries, role_repair_first_margin_change=role_effect,
        role_repair_source_slope_interaction=source_interaction, max_full_recompute_error=error,
        trajectories=sum(len(runs) for runs in complete['runs'].values()),
        prediction_positions=sum(run['tokens'] for runs in complete['runs'].values() for run in runs),
        answers=2, independent_problems=1,
        metadata_note='v1 complete.json independent_cases=2 counts answers, not independent problems',
        scope='manual exposed-case diagnostic; one problem, 2 answers, 4 prefixes; no detector AUROC')
    write_json(args.output / 'summary.json', result)
    plot_responses(args.output, cases)
    print(result)


if __name__ == '__main__':
    main()
