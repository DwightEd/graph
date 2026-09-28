"""Check identities, branches, reference exclusion and unchanged controls."""
import argparse
from pathlib import Path
import numpy as np
from experiments.decision_risk_flow.data import read_json, write_json
from .measure import FIELDS
from .score import BASELINES, VIEWS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest = read_json(args.output/'manifest.json')
    assert (args.output/'scores_frozen.json').stat().st_mtime <= (args.output/'evaluation.json').stat().st_mtime
    sources = {role:{r['source_id'] for r in manifest['records'] if r['role']==role}
               for role in ('fit','dev','regression','natural')}
    for first in sources:
        for second in sources:
            if first!=second:
                assert not sources[first]&sources[second]
    tokens = neighbors = 0
    for row in manifest['records']:
        directory = args.output/row['key']
        feature = np.load(directory/'relations.npz')
        context = np.load(directory/'context.npz')
        original = np.load(Path(manifest['route_base'])/row['key']/'measurements.npz')
        np.testing.assert_array_equal(context['token_ids'],original['token_ids'])
        assert feature['values'].shape==(32,32,len(context['token_ids']),len(FIELDS))
        assert feature['fields'].tolist()==read_json(args.output/'features_complete.json')['fields']
        assert not np.isinf(feature['values']).any()
        # History-free first prediction cannot define a path disagreement.
        assert np.isnan(feature['values'][:,:,0,:10]).all()
        assert np.isfinite(feature['values'][:,:,1:,:]).all()
        scores = np.load(directory/'scores.npz')
        previous = np.load(Path(manifest['previous'])/row['key']/'scores.npz')
        for name in BASELINES:
            np.testing.assert_array_equal(scores[name],previous[name])
        module_path = directory/'module_ranks.npz'
        if module_path.exists():
            modules = np.load(module_path)['values']
            np.testing.assert_array_equal(scores['omnibus_max'],modules.max(-1))
            np.testing.assert_array_equal(scores['omnibus_mean'],modules.mean(-1))
        reference = np.asarray(read_json(args.output/(row['task']+'_reference.json'))['sources'])
        for view in VIEWS:
            saved = np.load(directory/(view+'_neighbors.npz'))
            assert not (reference[saved['indices']]==row['source_id']).any()
            np.testing.assert_allclose(scores[view+'_conditional'],
                saved['query_radius']/np.maximum(saved['neighbor_radius'],1e-8),rtol=1e-12,atol=1e-12)
            neighbors += len(saved['indices'])
        for name in ('binding_conditional','relation_given_adoption'):
            path = directory/(name+'_neighbors.npz')
            if path.exists():
                saved = np.load(path)
                assert not (reference[saved['indices']]==row['source_id']).any()
                np.testing.assert_allclose(scores[name],saved['query_radius']/np.maximum(saved['neighbor_radius'],1e-8),
                                           rtol=1e-12,atol=1e-12)
                neighbors += len(saved['indices'])
        tokens += len(context['token_ids'])
    capture = read_json(args.output/'capture_complete.json')
    max_error = max(max(row['layer_value_relative_errors']) for row in capture['verification'].values())
    assert max_error<.0005
    result = dict(status='passed',answers=len(manifest['records']),tokens=tokens,
        neighbor_rows=neighbors,source_exclusion=True,baseline_arrays_identical=True,
        prompt_replay_max_relative_error=max_error,same_agent_verification=True)
    write_json(args.output/'verification.json',result)
    print(result)


if __name__ == '__main__':
    main()
