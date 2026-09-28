"""Verify all native head axes and independently recompute the norm routes."""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.native_support.dual_state.scoring import window_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.output / 'manifest.json')
    sources = {role: {row['source_id'] for row in manifest['records'] if row['role'] == role}
               for role in ('fit', 'dev', 'regression', 'natural')}
    roles = list(sources)
    for index, role in enumerate(roles):
        for other in roles[index+1:]:
            assert not sources[role] & sources[other]
    rows, total_tokens = [], 0
    for row in manifest['records']:
        directory = args.output / row['key']
        with np.load(directory / 'measurements.npz') as measured:
            token_ids = measured['token_ids']
            count, prompt = len(token_ids), int(measured['prompt_length'])
            mass = measured['mass'].astype(np.float64)
            assert float(measured['replay_error']) < .0005
        with np.load(Path(manifest['base']) / row['key'] / 'readouts.npz') as original:
            np.testing.assert_array_equal(token_ids, original['token_ids'])
        values = np.load(directory / 'values.npy', mmap_mode='r')
        factors = np.load(directory / 'head_vectors.npy', mmap_mode='r')
        gram = np.load(directory / 'head_gram.npy', mmap_mode='r')
        assert values.shape == (32, 8, prompt+count-1, 128)
        assert factors.shape == (32, 3, 32, count, 128)
        assert gram.shape == (32, 3, count, 32, 32)
        for array in (values, factors, gram):
            for layer in array:
                assert np.isfinite(layer).all()
        net = np.sqrt(np.maximum(np.diagonal(gram, axis1=-2, axis2=-1), 0)).transpose(0, 1, 3, 2)
        # Triangle inequality across the exact same message set.
        assert np.all(net <= mass + 1e-4 * np.maximum(mass, 1))
        raw = np.zeros(count)
        integrated = np.zeros(count)
        for layer in range(32):
            raw += (mass[layer, 1].sum(0) - mass[layer, 0].sum(0)) / mass[layer].sum((0, 1)) / 32
            integrated += (net[layer, 1].sum(0) - net[layer, 0].sum(0)) / net[layer].sum((0, 1)) / 32
            # Complete head geometry, not a sketch. Scale accounts for FP32 bmm.
            assert np.allclose(gram[layer], gram[layer].transpose(0, 1, 3, 2), atol=1e-5, rtol=1e-5)
        with np.load(directory / 'scores.npz') as score:
            np.testing.assert_allclose(raw, score['raw_route'], atol=1e-10)
            np.testing.assert_allclose(integrated, score['net_route'], atol=2e-6)
            np.testing.assert_allclose(window_mean(raw, 16, offline=True), score['raw_route_offline_mean'], atol=1e-10)
            comparison = {}
            if row['kind'] == 'observer':
                comparison = dict(old_route_max_absolute_difference=float(np.abs(raw-score['original_cached_route']).max()),
                                  old_route_mean_absolute_difference=float(np.abs(raw-score['original_cached_route']).mean()))
        rows.append(dict(key=row['key'], tokens=count, **comparison))
        total_tokens += count
    write_json(args.output / 'verification.json', dict(status='passed', answers=len(rows), tokens=total_tokens,
        source_intersections_empty=True, native_axes_checked=True, routes_independently_recomputed=True,
        same_agent_numeric_verification=True, rows=rows))
    print(dict(status='passed', answers=len(rows), tokens=total_tokens), flush=True)


if __name__ == '__main__':
    main()
