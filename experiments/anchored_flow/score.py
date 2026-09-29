"""Preserve the original source/route anchor and freeze bounded corrections."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from experiments.decision_risk_flow.data import read_json,write_json
from experiments.context_response.restore import cached_baselines,original_threshold,ORIGINAL
from experiments.native_support.dual_state.scoring import window_mean
from .model import score_observations

METHODS = ('base','token_observation','clipped','anchored_flow','uniform_tv','shuffled_tv')


def unit_average(values,units):
    result = np.empty(len(values),dtype=float)
    for unit in units:
        start,stop = unit['start'],unit['stop']
        result[start:stop] = values[start:stop].mean()
    return result


def read_observations(row,output):
    saved = np.load(output/row['key']/'observations.npz')
    np.testing.assert_array_equal(saved['token_id'],row['response']['answer_ids'])
    source = .5*(saved['source_local'].astype(float)+saved['source_full'].astype(float))
    route = saved['raw_route'].astype(float)
    raw = dict(pair=unit_average(source,row['response']['units']),token=source,
        route=window_mean(route,16,offline=True),raw_route=route)
    if row['cached']:
        # Re-associating means changes exact ties at stored empirical quantiles.
        # Use the original observations so the previous anchor is bit-identical.
        with np.load(Path(row['cached'])/'scores.npz') as previous:
            raw['pair'] = previous['source_pair_unit_mean'].copy()
            raw['route'] = previous['raw_route_offline_mean'].copy()
    return raw


def rank(value,reference):
    left = np.searchsorted(reference,value,'left')
    right = np.searchsorted(reference,value,'right')
    return (left+right)/(2*len(reference))


def anchored_observations(raw,references):
    source = rank(raw['pair'],references['pair'])
    route = rank(raw['route'],references['route'])
    local_source = rank(raw['token'],references['pair'])
    local_route = rank(raw['raw_route'],references['route'])
    return .75*source+.25*route,.75*local_source+.25*local_route


def fit_gsm(rows,raw):
    reference = {}
    for name in ('pair','route'):
        values = np.concatenate([raw[r['key']][name] for r in rows if r['role']=='fit'])
        reference[name] = np.quantile(values,np.linspace(0,1,10001))
    values = []
    for row in rows:
        if row['role']=='dev':
            base,_ = anchored_observations(raw[row['key']],reference)
            values.extend(base[a:b].mean() for a,b in row['step_ranges'])
    return reference,float(np.quantile(values,.95,method='higher'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    read_json(args.output/'capture_complete.json')
    read_json(args.output/'edges_complete.json')
    records = read_json(args.output/'manifest.json')['records']
    raw = {r['key']:read_observations(r,args.output) for r in records}
    references,thresholds = {},{}
    for task in ('QA','Summary','Data2txt'):
        fitted = joblib.load(ORIGINAL/task/'models.joblib')['ranks']
        references[task] = {name:fitted[name] for name in ('pair','route')}
        thresholds[task] = original_threshold(task)
    gsm = [r for r in records if r['dataset']=='gsm8k']
    references['GSM8K'],thresholds['GSM8K'] = fit_gsm(gsm,raw)
    originals = cached_baselines({r['key'] for r in records if r['cached']})
    gaps,checks = {},{}
    for row in records:
        key = row['key']
        base,local = anchored_observations(raw[key],references[row['task']])
        if row['cached']:
            valid = np.isfinite(originals[key])
            checks[key] = float(np.max(np.abs(base[valid]-originals[key][valid])))
            np.testing.assert_allclose(base[valid],originals[key][valid],atol=1e-12,rtol=0)
        if row['role']=='case':
            edge = np.load(args.output/key/'edges.npz')
            scores,gaps[key] = score_observations(base,local,edge['edges'],edge['shuffled'])
        else:
            scores = dict(base=base)
        np.savez_compressed(args.output/key/'scores.npz',**scores)
        np.savez_compressed(args.output/key/'observations_ranked.npz',**raw[key],base=base,local=local)
    joblib.dump(references,args.output/'references.joblib')
    write_json(args.output/'thresholds.json',thresholds)
    write_json(args.output/'scores_frozen.json',dict(status='complete',methods=METHODS,primary='anchored_flow',
        labels_used=False,baseline_max_errors=checks,dual_gaps=gaps,
        calibration='all candidates retain original base threshold; GSM uses 8 unlabeled dev problems step95',
        references='RAG original full-fit rank references; GSM 8 disjoint fit problems',
        caveat='not equal actual FPR; no score-specific threshold tuning; exposed development cases'))
    print('scores frozen; exact old-baseline matches',checks,flush=True)


if __name__=='__main__':
    main()
