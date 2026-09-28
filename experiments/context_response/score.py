"""Freeze signed response, graph controls and unlabeled reference thresholds."""
import argparse
from pathlib import Path
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.head_state_readout.density import fit_scale,standardize,distances,local_ratios
from experiments.head_state_readout.score import reference_indices,reference_matrix,OLD
from experiments.message_js.score import valid_tokens,fit_cdf,percentile
from experiments.source_relation.refine import fit_rank

BASELINES = ('source_route_fixed','association','association_fused','relation_tail','omnibus_max_fused')
VIEWS = {'signed_tail':(0,1),'relative_tail':(4,5),'specific_tail':(6,7),'sham_tail':(2,3)}


def view_values(output,rows):
    views = {name:{} for name in VIEWS}
    for row in rows:
        values = np.load(output/row['key']/'responses.npz')['values']
        for name,indices in VIEWS.items():
            selected = values[...,indices].transpose(2,0,1,3).reshape(values.shape[2],-1)
            views[name][row['key']] = selected if name=='specific_tail' else -selected
    return views


def coordinate_ranks(values,rows,output):
    fit = [row for row in rows if row['role']=='fit']
    arrays,weights = [],[]
    for row in fit:
        valid = valid_tokens(row,output,OLD)
        arrays.append(values[row['key']][valid])
        weights.append(np.full(valid.sum(),1/valid.sum()))
    reference,weight = np.concatenate(arrays),np.concatenate(weights)
    fitted = [fit_cdf(reference[:,i],weight) for i in range(reference.shape[1])]
    return {key:np.stack([percentile(value[:,i],cdf) for i,cdf in enumerate(fitted)],-1)
            for key,value in values.items()}


def metric_distances(values,selected):
    fitted = fit_scale(reference_matrix(values,selected))
    normalized = {key:standardize(value,fitted) for key,value in values.items()}
    reference = reference_matrix(normalized,selected)
    return {key:distances(value,reference) for key,value in normalized.items()}


def conditional_scores(output,rows,values,scores):
    selected = reference_indices(output,rows)
    source_ids = np.concatenate([np.repeat(row['source_id'],len(selected[row['key']]))
                                for row in rows if row['key'] in selected])
    context = {row['key']:np.load(output/row['key']/'context.npz')['values'] for row in rows}
    metric,condition = metric_distances(values,selected),metric_distances(context,selected)
    for row in rows:
        key = row['key']
        ratio,neighbors,radius,scale = local_ratios(metric[key],reference_matrix(metric,selected),
            condition[key],reference_matrix(condition,selected),np.repeat(row['source_id'],len(metric[key])),source_ids)
        scores[key]['response_conditional'] = ratio
        np.savez_compressed(output/key/'neighbors.npz',indices=np.stack(neighbors),radius=radius,scale=scale)
    write_json(output/(rows[0]['task']+'_reference.json'),dict(sources=source_ids.tolist(),
        identities=[(row['key'],int(i)) for row in rows if row['key'] in selected for i in selected[row['key']]]))


def score_task(output,previous,rows):
    scores = {}
    for row in rows:
        with np.load(previous/row['key']/'scores.npz') as saved:
            scores[row['key']] = {name:saved[name] for name in BASELINES}
    views = view_values(output,rows)
    for name,values in views.items():
        ranks = coordinate_ranks(values,rows,output)
        for key,value in ranks.items():
            scores[key][name] = np.quantile(value,.9,axis=-1)
    conditional_scores(output,rows,views['signed_tail'],scores)
    for name in ('signed_tail','specific_tail','response_conditional'):
        ranked = fit_rank(scores,name,rows,output)
        for key in scores:
            scores[key][name+'_fused'] = .75*scores[key]['source_route_fixed']+.25*ranked[key]
    return scores


def calibrate(output,rows,scores):
    result = {}
    for name in next(iter(scores.values())):
        values,weights = [],[]
        for row in rows:
            if row['role']=='dev':
                valid = valid_tokens(row,output,OLD)
                values.append(scores[row['key']][name][valid])
                weights.append(np.full(valid.sum(),1/valid.sum()))
        ordered,cumulative = fit_cdf(np.concatenate(values),np.concatenate(weights))
        result[name] = float(ordered[np.searchsorted(cumulative[1:],.95)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    read_json(args.output/'features_complete.json')
    manifest = read_json(args.output/'manifest.json')
    thresholds = {}
    for task in ('QA','Summary','Data2txt'):
        rows = [row for row in manifest['records'] if row['task']==task]
        scores = score_task(args.output,Path(manifest['previous']),rows)
        thresholds[task] = calibrate(args.output,rows,scores)
        for key,value in scores.items():
            np.savez_compressed(args.output/key/'scores.npz',**value)
        print('scored',task,flush=True)
    write_json(args.output/'thresholds.json',thresholds)
    write_json(args.output/'scores_frozen.json',dict(status='complete',methods=list(next(iter(scores.values()))),
        main='signed_tail',labels_used=False,head_selection=False,
        target='actual token vs fixed strongest alternative margin',
        operator='first-order mass-preserving context-completion response, not fact probability',
        calibration='source-equal unlabeled dev mixture95; no normal-FPR guarantee'))


if __name__=='__main__':
    main()
