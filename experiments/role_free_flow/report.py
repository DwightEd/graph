"""Export measured tables and figures, keeping original and replay provenance distinct."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .run import write_json


def verify_and_summarize(output):
    inventory = pd.read_csv(output / 'inventory.csv')
    summaries, checks = [], []
    for trace in ('00012', '00013'):
        rows = pd.read_csv(output / 'messages' / (trace + '.csv'))
        final = rows[rows.layer == 32]
        with np.load(output / 'messages' / (trace + '.npz')) as saved:
            numeric = saved['stats']
            np.testing.assert_allclose(numeric[..., 0] - numeric[..., 1], numeric[..., 2], atol=1e-5)
            np.testing.assert_allclose(numeric[..., 0] + numeric[..., 1], numeric[..., 3], atol=1e-5)
            expected = saved['embedding_margin'] + rows.groupby('position')[['attention_direct', 'mlp_direct']].sum().sum(axis=1).to_numpy()
            np.testing.assert_allclose(expected, final.cumulative_direct, atol=2e-5)
        checks.append(dict(trace=trace, tokens=len(final),
            max_frozen_margin_reconstruction_error=float((final.frozen_scale_state_margin - final.cumulative_direct).abs().max()),
            median_frozen_margin_reconstruction_error=float((final.frozen_scale_state_margin - final.cumulative_direct).abs().median()),
            max_attention_projection_roundoff=float(rows.attention_roundoff.abs().max())))
        for position in range(123, 134):
            selected = rows[rows.position == position]
            fields = ['prompt_positive', 'prompt_negative_magnitude', 'prompt_net',
                      'history_net', 'special_net', 'mlp_direct']
            total = selected[fields].sum().to_dict()
            summaries.append(dict(trace=trace, position=position, **total))
    pd.DataFrame(summaries).to_csv(output / 'message_neighborhood_totals.csv', index=False)
    write_json(output / 'verification.json', dict(original_generation_checks=inventory.to_dict('records'),
        full_graph_readout_checks=checks, signed_sums_and_telescoping='passed',
        max_original_replay_logit_error=float(inventory.max_logit_error.max()),
        max_original_replay_attention_error=float(inventory.max_attention_error.max()),
        max_fp32_lens_vs_bf16_native_logit_error=float(inventory.fp32_candidate_logit_max_error.max()),
        review='same-agent numeric verification; no independent reviewer', new_llm_forwards=0))


def plot_trajectories(output, report):
    table = pd.read_csv(output / 'events.csv')
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for column, (name, positive, negative, onset) in enumerate([
            ('Headwear', '00005.npz', '00006.npz', 35),
            ('Onion stage', '00013.npz', '00012.npz', 126)]):
        for trace, label, color in [(positive, 'Locally supported', '#1967a5'),
                                    (negative, 'Locally unsupported', '#c34b32')]:
            rows = table[(table.trace == trace) & (table.position >= onset - 3) & (table.position <= onset + 7)]
            axes[0, column].plot(rows.position - onset, rows.entropy_nats / np.log(2), marker='o', label=label, color=color)
            axes[1, column].plot(rows.position - onset, rows.chosen_margin_final, marker='o', color=color)
        axes[0, column].set_title(name + ' (different natural prefixes)')
        axes[0, column].set_ylabel('Native entropy (bits)')
        axes[1, column].set_ylabel('Chosen vs best alternative logit')
        axes[1, column].set_xlabel('Token position relative to reviewed onset')
        for row in range(2):
            axes[row, column].axvline(0, color='gray', linestyle=':', alpha=.7)
        axes[1, column].axhline(0, color='gray', linewidth=.7)
    axes[0, 0].legend(fontsize=8)
    figure.savefig(report / 'natural_trajectories.png', dpi=180)
    figure.savefig(report / 'natural_trajectories.svg')
    plt.close(figure)


def plot_writes(output, report):
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    for axis, trace, title in zip(axes, ('00013', '00012'),
                                ('Supported: over vs in', 'Unsupported: for vs on')):
        table = pd.read_csv(output / 'messages' / (trace + '.csv'))
        rows = table[table.position == 126]
        for offset, field, label, color in [(-.25, 'prompt_net', 'Prompt direct write', '#1967a5'),
                (0, 'history_net', 'History direct write', '#de9951'),
                (.25, 'mlp_direct', 'MLP direct write', '#688b68')]:
            axis.bar(rows.layer + offset, rows[field], width=.24, label=label, color=color)
        axis.set_title(title + ' | frozen final RMS scale; direct, not causal')
        axis.set_ylabel('Logit margin contribution')
        axis.axhline(0, color='gray', linewidth=.7)
        axis.set_xlabel('Layer (1-based)')
    axes[0].legend(fontsize=8, ncol=3)
    figure.savefig(report / 'onion_direct_writes.png', dpi=180)
    figure.savefig(report / 'onion_direct_writes.svg')
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)
    verify_and_summarize(args.output)
    plot_trajectories(args.output, args.report)
    plot_writes(args.output, args.report)
