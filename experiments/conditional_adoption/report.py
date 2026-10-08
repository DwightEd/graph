"""Export reproducible raw-score figures for the completed exploratory pilot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(root, output):
    output.mkdir(parents=True, exist_ok=False)
    experiments = [('scores', 'source_candidate_regret', 'Signed response'),
        ('scores', 'source_margin_rejection', 'Native rival margin'),
        ('scores', 'source_local_countervote', 'Head countervote'),
        ('elasticity_scores', 'source_loss_elasticity', 'Loss elasticity (exploratory)'),
        ('relay_scores', 'regret_both', 'Current + past'),
        ('relay_elasticity_scores', 'loss_elasticity_both', 'Relay elasticity (exploratory)'),
        ('scores', 'old_source_route_native_fixed', 'Legacy fixed'),
        ('scores', 'old_source_unit_detector', 'Legacy dev-selected unit')]
    summary = []
    for folder, method, label in experiments:
        report = json.loads((root / folder / 'evaluation_v2.json').read_text())
        result = report['methods'][method]
        summary.append(dict(folder=folder, method=method, label=label,
            auroc=result['all_token']['auroc'], ap=result['all_token']['ap'],
            macro_within_answer_auroc=result['macro_within_answer_auroc'],
            oracle_operating_point=result['operating_point'], provenance=result['provenance']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    positions = np.arange(len(summary))
    for axis, field, title in zip(axes, ('auroc', 'macro_within_answer_auroc'),
                                  ('Pooled token AUROC', 'Macro AUROC within 3 error answers')):
        axis.barh(positions, [row[field] for row in summary], color='#416a8a')
        axis.axvline(.5, color='gray', linestyle='--', linewidth=1)
        axis.set_xlim(0, 1)
        axis.set_title(title)
        axis.grid(axis='x', alpha=.2)
    axes[0].set_yticks(positions, [row['label'] for row in summary])
    axes[0].invert_yaxis()
    fig.suptitle('Exploratory historical QA cohort: 8 answers / 1139 tokens / 77 error tokens', fontsize=10)
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=220)
    fig.savefig(output / 'comparison.pdf')
    plt.close(fig)
    token_sets = {folder: json.loads((root / folder / 'tokens_v2.json').read_text())
                  for folder in ('scores', 'elasticity_scores', 'relay_scores')}
    for identity in ('12219', '12297', '15604'):
        fig, axes = plt.subplots(3, 1, figsize=(10, 5.7), sharex=True)
        for axis, (folder, method, title) in zip(axes, [
                ('scores', 'source_candidate_regret', 'Raw joint source response'),
                ('elasticity_scores', 'source_loss_elasticity', 'Loss elasticity (post-pilot)'),
                ('relay_scores', 'regret_both', 'Finite L15 current + past response')]):
            rows = [row for row in token_sets[folder] if row['id'] == identity]
            x = [row['target'] for row in rows]
            y = [row['scores'][method] for row in rows]
            axis.plot(x, y, color='#305a7c', linewidth=.8)
            for row in rows:
                if row['label']:
                    axis.axvspan(row['target']-.5, row['target']+.5, color='#dd816c', alpha=.14, linewidth=0)
            axis.axhline(0, color='gray', linewidth=.6)
            axis.set_ylabel('Raw score')
            axis.set_title(title, fontsize=9)
        axes[-1].set_xlabel('Original response token position')
        fig.suptitle(f'{identity}: official error spans shaded; no span averaging', fontsize=10)
        fig.tight_layout()
        fig.savefig(output / f'trace_{identity}.png', dpi=220)
        plt.close(fig)
    (output / 'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.root, args.output)


if __name__ == '__main__':
    main()
