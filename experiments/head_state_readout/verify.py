"""Audit retained axes, source exclusion, references and stored score identities."""

import argparse
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from .features import BLOCKS

DIMENSIONS = dict(geometry_head=4096, geometry_gram=31744, js=5120,
                  jacobian_head=6144, tangent=12288, fisher=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.output/'manifest.json')
    role_sources = {role: {row['source_id'] for row in manifest['records'] if row['role']==role}
                    for role in ('fit', 'dev', 'regression', 'natural')}
    for first, sources in role_sources.items():
        for second, others in role_sources.items():
            if first != second:
                assert not sources & others
    count, neighbor_checks = 0, 0
    for row in manifest['records']:
        directory = args.output/row['key']
        with np.load(directory/'context.npz') as context:
            ids = context['token_ids']
        with np.load(Path(manifest['route_base'])/row['key']/'measurements.npz') as original:
            np.testing.assert_array_equal(ids, original['token_ids'])
        feature_directory = directory
        if not (feature_directory/'geometry_head.npy').exists():
            feature_directory = Path(manifest['previous'])/row['key']
        for block in BLOCKS:
            value = np.load(feature_directory/(block+'.npy'), mmap_mode='r')
            assert value.shape == (len(ids), DIMENSIONS[block]), (row['key'], block)
            assert np.isfinite(value).all() if block != 'js' else not np.isinf(value).any()
        reference_path = args.output/(row['task']+'_reference.json')
        if not reference_path.exists():
            reference_path = Path(manifest['previous'])/(row['task']+'_reference.json')
        reference = read_json(reference_path)
        sources = np.asarray(reference['sources'])
        with np.load(directory/'scores.npz') as scores:
            for path in directory.glob('*_neighbors.npz'):
                method = path.name.removesuffix('_neighbors.npz')
                with np.load(path) as saved:
                    neighbors = saved['indices']
                    assert neighbors.shape[0] == len(ids)
                    assert not (sources[neighbors] == row['source_id']).any()
                    for source in np.unique(sources):
                        if source != row['source_id']:
                            assert (sources[neighbors] == source).sum(axis=1).min() == 8
                            assert (sources[neighbors] == source).sum(axis=1).max() == 8
                    recomputed = saved['query_radius']/np.maximum(saved['neighbor_radius'], 1e-8)
                    np.testing.assert_allclose(recomputed, scores[method], rtol=1e-12, atol=1e-12)
                    neighbor_checks += len(ids)
            for name in scores.files:
                if row['kind']=='observer' or name not in ('state_fused','transition_fused','source_route_fixed',
                    'association_fused','association_gate','transition_gate','unsmoothed_fused',
                    'raw_backbone','token_backbone','token_source_fused'):
                    assert np.isfinite(scores[name]).all(), (row['key'], name)
        count += len(ids)
    write_json(args.output/'verification.json', dict(status='passed', answers=len(manifest['records']),
        tokens=count, checked_neighbor_rows=neighbor_checks, source_exclusion=True,
        full_coordinate_dimensions=DIMENSIONS, raw_axes_ids_checked=True,
        same_agent_verification=True, note='Structural and score-identity audit; not an independent replication of the entire statistical algorithm.'))
    print('verified', count, 'tokens;', neighbor_checks, 'neighbour rows')


if __name__ == '__main__':
    main()
