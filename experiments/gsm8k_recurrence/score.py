"""Freeze attention recurrence scores using only disjoint unlabeled questions."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.message_js.score import fit_cdf, percentile
from experiments.context_response.recurrence import propagate
from .measure import write_json

METHODS = ('source_only','base','recurrence','within_step','distance_within','shuffled_within','step_mean')


def question_weights(rows, arrays):
    counts = Counter(row['problem'] for row in rows)
    return np.concatenate([np.full(len(value),1/(len(value)*counts[row['problem']]))
                           for row,value in zip(rows,arrays)])


def fit_reference(rows, values):
    arrays = [values[row['id']] for row in rows]
    return fit_cdf(np.concatenate(arrays), question_weights(rows, arrays))


def covariates(row):
    return np.column_stack((np.ones(row['tokens']),
        np.full(row['tokens'],np.log(row['prompt_tokens'])),np.log1p(np.arange(row['tokens']))))


def restrict_steps(edges, steps):
    return [np.where((steps[lag:]==steps[:-lag]) & (steps[lag:]>=0),edge,0.)
            for lag,edge in enumerate(edges,1)]


def step_scores(value, ranges, tail=True):
    result = []
    for start,stop in ranges:
        selected = value[start:stop]
        size = max(1,int(np.ceil(.2*len(selected)))) if tail else len(selected)
        result.append(np.sort(selected)[-size:].mean())
    return np.array(result)


def fit_scales(rows, arrays):
    selected = [r for r in rows if r['role']=='fit']
    design = np.concatenate([covariates(r) for r in selected])
    raw = {r['id']:arrays[r['id']]['raw'] for r in rows}
    weights = question_weights(selected,[raw[r['id']] for r in selected])
    target = np.concatenate([raw[r['id']] for r in selected])
    coefficients = np.linalg.lstsq(design*np.sqrt(weights[:,None]),target*np.sqrt(weights),rcond=None)[0]
    residual = {r['id']:raw[r['id']]-covariates(r)@coefficients for r in rows}
    scales = dict(raw=fit_reference(selected,raw),residual=fit_reference(selected,residual))
    for mode in ('native','shuffled'):
        for lag in range(1,9):
            values = {r['id']:arrays[r['id']][f'head_{lag}'].mean(0) if mode=='native'
                      else arrays[r['id']][f'shuffled_{lag}'] for r in selected}
            scales[f'{mode}_{lag}'] = fit_reference(selected,values)
    return coefficients, scales, residual


def score_record(data, residual, scales):
    base = percentile(residual,scales['residual'])
    edges, shuffled = [], []
    for lag in range(1,9):
        for mode,destination in (('native',edges),('shuffled',shuffled)):
            raw = data[f'head_{lag}'].mean(0) if mode=='native' else data[f'shuffled_{lag}']
            rank = percentile(raw,scales[f'{mode}_{lag}'])
            destination.append(np.where(rank>=.95,rank,0.))
    within = restrict_steps(edges,data['step_id'])
    distance = restrict_steps([np.full(len(edge),np.exp(-lag/32))
                               for lag,edge in enumerate(edges,1)],data['step_id'])
    values = dict(source_only=percentile(data['raw'],scales['raw']),base=base)
    graphs = dict(recurrence=edges,within_step=within,distance_within=distance,
                  shuffled_within=restrict_steps(shuffled,data['step_id']))
    for name,graph in graphs.items():
        values[name] = propagate(base,graph)[0]
    steps = {name:step_scores(value,data['step_ranges']) for name,value in values.items()}
    steps['step_mean'] = step_scores(base,data['step_ranges'],tail=False)
    return values,steps,edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest = json.loads((args.output/'manifest.json').read_text())
    rows = manifest['records']
    arrays = {}
    for row in rows:
        with np.load(args.output/(row['id']+'.npz')) as saved:
            arrays[row['id']] = {name:saved[name] for name in saved.files}
    coefficients,scales,residual = fit_scales(rows,arrays)
    (args.output/'scores').mkdir()
    all_steps = {}
    for row in rows:
        key = row['id']
        token,steps,edges = score_record(arrays[key],residual[key],scales)
        all_steps[key] = steps
        np.savez_compressed(args.output/'scores'/(key+'.npz'),
            **{'token_'+k:v for k,v in token.items()},**steps,
            **{f'edge_{lag}':value for lag,value in enumerate(edges,1)})
    dev = [r for r in rows if r['role']=='dev']
    thresholds = {}
    for name in METHODS:
        cdf = fit_reference(dev,{r['id']:all_steps[r['id']][name] for r in dev})
        thresholds[name] = float(cdf[0][np.searchsorted(cdf[1][1:],.95)])
    np.savez_compressed(args.output/'fit.npz',coefficients=coefficients,
        **{name+'_values':cdf[0] for name,cdf in scales.items()},
        **{name+'_cdf':cdf[1] for name,cdf in scales.items()})
    write_json(args.output/'thresholds.json',thresholds)
    write_json(args.output/'scores_frozen.json',dict(status='complete',main='within_step',methods=METHODS,
        labels_used=False,parameters=dict(lags=8,hops=3,gate=.95,step_top_fraction=.2),
        seed='question-equal unlabeled fit residual ranks; length trend linear regression',
        threshold='question-equal unlabeled dev step mixture95 strict gt',
        runtime='CPU reuse of existing attention; no model forward'))
    print('frozen',len(rows),'answers',flush=True)


if __name__=='__main__':
    main()
