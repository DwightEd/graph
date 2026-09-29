"""Audit preserved baselines, exact message axes, fit isolation and reported changes."""
import argparse
from pathlib import Path
import numpy as np
from experiments.decision_risk_flow.data import read_json,write_json
from experiments.context_response.restore import cached_baselines
from .model import combine_overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest = read_json(args.output/'manifest.json')
    records = manifest['records']
    frozen = read_json(args.output/'scores_frozen.json')
    original = cached_baselines({r['key'] for r in records if r['cached']})
    groups = {role:{r['pair'] for r in records if r['dataset']=='gsm8k' and r['role']==role}
              for role in ('fit','dev','case')}
    assert not groups['fit']&groups['dev']
    assert not (groups['fit']|groups['dev'])&groups['case']
    axes,answers,tokens,max_bound = [],0,0,0.
    for row in records:
        key = row['key']
        count = len(row['response']['answer_ids'])
        coverage = np.zeros(count,dtype=int)
        for unit in row['response']['units']:
            coverage[unit['start']:unit['stop']] += 1
        np.testing.assert_array_equal(coverage,np.ones(count,dtype=int))
        with np.load(args.output/key/'observations.npz') as observation:
            np.testing.assert_array_equal(observation['token_id'],row['response']['answer_ids'])
            for name in observation.files:
                assert np.isfinite(observation[name]).all(),(key,name)
        with np.load(args.output/key/'scores.npz') as scores:
            if row['cached']:
                valid = np.isfinite(original[key])
                np.testing.assert_array_equal(scores['base'][valid],original[key][valid])
            if row['role']!='case':
                continue
            for name in frozen['methods']:
                assert scores[name].shape==(count,)
                assert np.isfinite(scores[name]).all()
                if name in ('clipped','anchored_flow','uniform_tv','shuffled_tv','matched_shuffle_tv'):
                    max_bound = max(max_bound,float(np.max(abs(scores[name]-scores['base']))))
            assert max_bound<=.1+1e-12
        with np.load(args.output/key/'edges.npz') as edge:
            assert edge['energy'].shape==(1024,count)
            assert edge['head_overlap'].shape==(1024,count-1)
            for field in ('edges','shuffled'):
                assert np.all((edge[field]>=0)&(edge[field]<=1+1e-6))
            np.testing.assert_allclose(edge['edges'],combine_overlap(edge['energy'],edge['head_overlap']),atol=0,rtol=0)
            if frozen.get('primary_unchanged'):
                with np.load(args.output/key/'matched_control.npz') as control:
                    np.testing.assert_array_equal(np.sort(control['edges']),np.sort(edge['edges']))
                with np.load(Path(manifest['previous'])/key/'scores.npz') as previous:
                    with np.load(args.output/key/'scores.npz') as current:
                        for name in previous.files:
                            np.testing.assert_array_equal(current[name],previous[name])
            axes.append(key)
        answers += 1
        tokens += count
    write_json(args.output/'audit.json',dict(status='passed',cases=answers,tokens=tokens,
        official_baselines_exact=len(original),full_head_axes=axes,max_correction=max_bound,
        gsm_problem_counts={k:len(v) for k,v in groups.items()},
        max_dual_gap=max(g for row in frozen['dual_gaps'].values() for g in row.values()),
        note='same-agent separate mathematical checks, no claim of external review'))
    print('audit passed',answers,tokens,len(original),max_bound,flush=True)


if __name__=='__main__':
    main()
