"""Publication artifact from all corrected groups, including the matched node control."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(summary, output):
    groups = [summary['corrected']['aggregate'][f'v{v}'] for v in (1, 2, 3)]
    groups.append(summary['high_fidelity']['aggregate'])
    positions = np.arange(4)
    figure, axis = plt.subplots(figsize=(7.6, 4.2), layout='constrained')
    for index, (name, label, color) in enumerate((
            ('node', 'node (fewer parameters)', '#737e8a'),
            ('native', 'native graph', '#167c9d'),
            ('rewired', 'matched rewiring', '#d48734'))):
        rows = [group[name]['auroc'] for group in groups]
        axis.bar(positions + (index - 1.5) * .19, [row['mean'] for row in rows],
                 yerr=[row['std'] for row in rows], width=.18, color=color,
                 capsize=3, label=label)
    control = summary['capacity_matched_control']['aggregate']['auroc']
    axis.bar(positions[-1] + 1.5 * .19, control['mean'], yerr=control['std'],
             width=.18, color='#8457a4', capsize=3, label='lag pair (matched capacity)')
    axis.set(xticks=positions, xticklabels=['Direct content', 'Rooted paths', 'Remove history', 'Head16 / PCA128'],
             ylabel='Token AUROC (mean ± initialization seed SD)', ylim=(0, 1))
    axis.axhline(.5, color='#777', linestyle='--', linewidth=.7)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(frameon=False, loc='upper left', fontsize=8)
    figure.savefig(output / 'comparison.svg')
    svg = output / 'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    figure.savefig(output / 'comparison.png', dpi=200)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text()), args.output)
