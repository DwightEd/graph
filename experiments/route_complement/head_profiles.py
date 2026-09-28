"""Describe discovered head patterns using separate, unlabelled fit trajectories."""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import valid_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.output / 'manifest.json')
    profiles = []
    for row in manifest['records']:
        if row['role'] != 'fit':
            continue
        directory = args.output / row['key']
        with np.load(directory / 'measurements.npz') as saved:
            mass = saved['mass']
            prompt, count = int(saved['prompt_length']), len(saved['token_ids'])
        with np.load(directory / 'head_features.npz') as saved:
            cancellation = saved['values'][..., 2:4]
        valid = valid_tokens(row, args.output, Path('outputs/transport_topology_cases_20260928'))
        proportions = mass / np.maximum(mass.sum(1, keepdims=True), 1e-30)
        # Index the token axis before averaging; physical layer/head axes stay fixed.
        source_share = proportions[:, 0][:, :, valid].mean(-1)
        history_share = proportions[:, 1][:, :, valid].mean(-1)
        attention = np.load(Path(manifest['base']) / row['key'] / 'attention.npy', mmap_mode='r')
        lag = (prompt - 1 + np.arange(count))[:, None] - np.arange(prompt, prompt+count-1)[None]
        recent = (lag >= 0) & (lag < 4)
        local_shares = []
        for layer in range(32):
            history = attention[layer, :, :, prompt:]
            total = history.sum(-1)
            share = np.divide((history * recent).sum(-1), total, out=np.full_like(total, np.nan), where=total > 0)
            local_shares.append(np.nanmean(share[:, valid], axis=1))
        profiles.append(np.stack((source_share, history_share, np.stack(local_shares),
                                  cancellation[..., 0][:, :, valid].mean(-1), cancellation[..., 1][:, :, valid].mean(-1)), -1))
    mean = np.mean(profiles, axis=0)
    directory = args.output / 'diagnostics'
    first = np.load(directory / '14315_headwear_scope.npz')['head_auc'][..., 3]
    second = np.load(directory / '14375_onion_stage.npz')['head_auc'][..., 3]
    groups = dict(error_higher_cancellation=(first > .75) & (second > .75),
                  error_lower_cancellation=(first < .25) & (second < .25), all_heads=np.ones((32, 32), dtype=bool))
    names = ('source_norm_share', 'history_norm_share', 'last4_attention_within_history', 'source_cancel', 'history_cancel')
    summary = {}
    for name, selected in groups.items():
        summary[name] = dict(heads=int(selected.sum()), physical_heads=np.argwhere(selected).tolist(),
            unlabeled_fit_means={field: float(mean[selected][:, index].mean()) for index, field in enumerate(names)})
    np.savez_compressed(directory / 'unlabeled_head_profiles.npz', values=mean, fields=names)
    write_json(directory / 'unlabeled_head_profiles.json', dict(posthoc=True, fit_sources=len(profiles), groups=summary,
        labels='Local labels identify reporting groups only; independent fit trajectories provide descriptions. No detector changes.'))


if __name__ == '__main__':
    main()
