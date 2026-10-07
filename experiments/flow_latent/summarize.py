"""Aggregate every predeclared run; no metric-based model selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from experiments.token_backtrace.grounded_projection_data import write_json


def collect(prefix):
    runs = []
    for version in (1, 2, 3):
        for seed in (42, 43, 44):
            suffix = '' if seed == 42 else f'_seed{seed}'
            output = prefix.parent / f'{prefix.name}_v{version}{suffix}'
            result = json.loads((output / 'results.json').read_text())
            diagnostics = json.loads((output / 'diagnostics.json').read_text())
            runs.append(dict(version=version, seed=seed, output=str(output),
                metrics=result['metrics'], within_answer=result['within_answer'],
                comparisons=result['comparisons'], diagnostics=diagnostics))
    return runs


def aggregate(runs):
    summary = {}
    for version in (1, 2, 3):
        group = [run for run in runs if run['version'] == version]
        summary[f'v{version}'] = {}
        for name in ('node', 'native', 'rewired', 'nll'):
            metrics = {}
            for metric in ('auroc', 'ap', 'onset_hits', 'normal_alarm'):
                values = [run['metrics'][name][metric] for run in group]
                metrics[metric] = dict(mean=float(np.mean(values)), std=float(np.std(values, ddof=1)),
                                       values=values)
            summary[f'v{version}'][name] = metrics
    return summary


def figure(summary, output):
    colors = ('#737e8a', '#167c9d', '#d48734')
    fig, axis = plt.subplots(figsize=(6.2, 3.5), layout='constrained')
    positions = np.arange(3)
    for index, name in enumerate(('node', 'native', 'rewired')):
        rows = [summary[f'v{version}'][name]['auroc'] for version in (1, 2, 3)]
        axis.bar(positions + (index - 1) * .23, [row['mean'] for row in rows],
                 yerr=[row['std'] for row in rows], width=.22, color=colors[index],
                 capsize=3, label=name)
    axis.set(xticks=positions, xticklabels=['Direct transport', 'Rooted paths', 'Remove history'],
             ylabel='Token AUROC (3 seeds)', ylim=(0, 1))
    axis.axhline(.5, color='#777', linestyle='--', linewidth=.7)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(frameon=False, loc='upper left')
    fig.savefig(output / 'comparison.svg')
    fig.savefig(output / 'comparison.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    args = parser.parse_args()
    runs = collect(args.prefix)
    summary = aggregate(runs)
    output = args.prefix.parent / (args.prefix.name + '_cycles')
    write_json(output / 'summary.json', dict(runs=runs, aggregate=summary,
        model_selected=False, scope='17 previously studied regression answers; exploratory'))
    figure(summary, output)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
