"""Plot exact alarm positions and separate locally annotated natural pairs."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.role_free_flow.diagnostics import read_local_spans

COLORS = {'TP':'#419d78', 'FP':'#d95b58', 'FN':'#a785c6', 'TN':'#e7e9ec'}


def plot_locations(output):
    with (output/'token_audit.csv').open() as stream:
        records = list(csv.DictReader(stream))
    keys = list(dict.fromkeys(row['key'] for row in records))
    methods = ('raw_route_offline_mean', 'source_route_fixed', 'association_fused')
    figure, axes = plt.subplots(len(keys), 1, figsize=(14, 14), sharex=True, constrained_layout=True)
    for axis, key in zip(axes, keys):
        for index, method in enumerate(methods):
            rows = [row for row in records if row['key']==key and row['method']==method]
            for category, color in COLORS.items():
                positions = [int(row['position']) for row in rows if row['status']==category]
                axis.scatter(positions, np.full(len(positions), index), marker='s', s=16, c=color, linewidths=0)
        axis.set(yticks=range(3), yticklabels=('Norm route', 'Old fixed fusion', 'Head association fusion'),
                 ylim=(-.6, 2.6), title='Answer '+key)
        axis.grid(axis='x', alpha=.2)
    axes[-1].set_xlabel('Original answer token index (zero-based)')
    figure.legend(handles=[Patch(color=color, label=name) for name, color in COLORS.items()],
                  loc='outside upper center', ncols=4)
    figure.savefig(output/'token_positions.png', dpi=150)
    figure.savefig(output/'token_positions.svg')
    plt.close(figure)


def plot_natural(output):
    manifest = read_json(output/'manifest.json')
    lookup = {row['trace']: row for row in manifest['records'] if row['role']=='natural'}
    local = read_local_spans(Path('outputs/role_free_flow_20260928'), Path('experiments/path_conflict/paired_cases.json'))
    threshold = read_json(output/'thresholds.json')['QA']['association']
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    token_rows = []
    for axis, (case, group) in zip(axes, local.groupby('case')):
        for side, selected in group.groupby('side'):
            row = lookup[selected.trace.iloc[0]]
            with np.load(output/row['key']/'scores.npz') as saved:
                values = saved['association'][selected.position.to_numpy()]
            with np.load(Path(manifest['samples'])/row['trace']) as saved:
                pieces = saved['token_text'][int(saved['prompt_length']):]
            axis.plot(selected.position, values, 'o-', label=side)
            for position, score in zip(selected.position, values):
                token_rows.append(dict(case=case, side=side, key=row['key'], position=int(position),
                    text=str(pieces[position]), score=float(score), threshold=threshold, alarm=bool(score>threshold)))
        axis.axhline(threshold, color='black', linestyle='--', label='Frozen threshold')
        axis.set(title=case, xlabel='Original answer token index', ylabel='Conditional association score')
        axis.legend()
    figure.savefig(output/'natural_local_scores.png', dpi=150)
    plt.close(figure)
    write_json(output/'natural_token_scores.json', token_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot_locations(args.output)
    plot_natural(args.output)


if __name__ == '__main__':
    main()
