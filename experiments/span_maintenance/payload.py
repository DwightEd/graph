"""Audit content-vector changes separately from stable addresses; RAG cache only."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .run import read_json, write_json


def adjacent_change(values):
    """Last axes [token, vector]; both-zero vectors are unmeasured, not stable."""
    previous, current = values[..., :-1, :], values[..., 1:, :]
    denominator = np.linalg.norm(previous, axis=-1) + np.linalg.norm(current, axis=-1)
    result = np.full(values.shape[:-1], np.nan)
    result[..., 1:] = np.divide(np.linalg.norm(current - previous, axis=-1), denominator,
        out=np.full(denominator.shape, np.nan), where=denominator > 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = read_json(args.output / 'manifest.json')['records']
    rows = []
    for row in records:
        if row['dataset'] != 'ragtruth':
            continue
        key = row['key']
        vectors = np.load(Path('outputs/route_complement_20260928') / key / 'head_vectors.npy', mmap_mode='r')
        # [layer, source/history/template, head, target, 128]; before W_O, not residual coordinates.
        change = adjacent_change(vectors.astype(float))
        native = np.load(Path('outputs/message_operator_20260928_v2') / key / 'operator.npz')
        np.testing.assert_array_equal(native['token_ids'], row['response']['answer_ids'])
        hidden_change = adjacent_change(native['final_state'].astype(float))
        directory = args.output / key
        np.savez_compressed(directory / 'payload_changes.npz', head_value_change=change,
            final_hidden_change=hidden_change, groups=['source', 'history', 'template'])
        for target in range(1, len(row['response']['answer_ids'])):
            item = dict(key=key, position=target, final_hidden_change=float(hidden_change[target]))
            for group, name in enumerate(('source', 'history', 'template')):
                measured = change[:, group, :, target]
                valid = np.isfinite(measured)
                item[name + '_value_change'] = float(measured[valid].mean()) if valid.any() else np.nan
                item[name + '_valid_heads'] = int(valid.sum())
            rows.append(item)
    positions = pd.read_csv(args.output / 'positions.csv', dtype={'key': str})
    joined = pd.DataFrame(rows).merge(positions[['key', 'position', 'gold', 'cohort', 'position_group']], on=['key', 'position'])
    joined.to_csv(args.output / 'payload_positions.csv', index=False)
    fields = ['source_value_change', 'history_value_change', 'template_value_change', 'final_hidden_change']
    summary = joined.groupby(['cohort', 'position_group'])[fields].mean().reset_index()
    summary.to_csv(args.output / 'payload_summary.csv', index=False)
    write_json(args.output / 'payload_audit.json', dict(status='complete', answers=12,
        labels_used_for_scoring=False, new_scores=False, hidden_dimension=4096, head_value_dimension=128,
        limitation='source groups reuse prior format masks; these are changes, not semantic novelty or causal truth'))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
