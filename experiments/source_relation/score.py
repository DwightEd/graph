"""Freeze label-free source-relation scores before evaluation."""
import argparse
from pathlib import Path
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import fit_cdf, percentile, valid_tokens
from experiments.head_state_readout.density import distances, fit_scale, standardize, local_ratios
from experiments.head_state_readout.score import reference_indices, reference_matrix, OLD

VIEWS = dict(direct=(0,4), relation=(1,5), state=(2,6), shuffled=(3,7), excess=(8,9))
BASELINES = ('source_route_fixed', 'association_fused', 'association', 'raw_route_offline_mean')


def finite_cdf(values, weights):
    valid = np.isfinite(values)
    return fit_cdf(values[valid], weights[valid])


def task_scores(output, previous, rows, task):
    selected = reference_indices(output, rows)
    ref_rows = [row for row in rows if row['key'] in selected]
    sources = np.concatenate([np.repeat(row['source_id'], len(selected[row['key']])) for row in ref_rows])
    raw = {row['key']: np.load(output / row['key'] / 'relations.npz')['values'] for row in rows}
    context = {row['key']: np.load(previous / row['key'] / 'context.npz')['values'] for row in rows}
    fitted = fit_scale(reference_matrix(context, selected))
    context = {key: standardize(value, fitted) for key,value in context.items()}
    reference = reference_matrix(context, selected)
    context = {key: distances(value, reference) for key,value in context.items()}
    ref_context = reference_matrix(context, selected)
    scores = {row['key']: {} for row in rows}
    metrics = {}
    for view, indices in VIEWS.items():
        values = {key: value[...,indices].transpose(2,0,1,3).reshape(value.shape[2],-1)
                  for key,value in raw.items()}
        ref_raw = reference_matrix(values, selected)
        fitted = fit_scale(ref_raw)
        normalized = {key: standardize(value, fitted) for key,value in values.items()}
        reference = reference_matrix(normalized, selected)
        metric = {key: distances(value, reference) for key,value in normalized.items()}
        metrics[view] = metric
        ref_metric = reference_matrix(metric, selected)
        for row in rows:
            key = row['key']
            score, neighbors, radius, scale = local_ratios(metric[key], ref_metric, context[key],
                ref_context, np.repeat(row['source_id'],len(metric[key])), sources)
            scores[key][view+'_conditional'] = score
            np.savez_compressed(output / key / (view+'_neighbors.npz'), indices=np.stack(neighbors),
                                query_radius=radius, neighbor_radius=scale)
        # High-tail ranks retain each head until final scalar readout. Fit source
        # weights equal, rather than allowing long answers to dominate ranks.
        fit_values = np.concatenate([values[r['key']][valid_tokens(r,output,OLD)] for r in ref_rows])
        fit_weights = np.concatenate([np.full(valid_tokens(r,output,OLD).sum(),
            1/valid_tokens(r,output,OLD).sum()) for r in ref_rows])
        cdfs = [finite_cdf(fit_values[:,i],fit_weights) for i in range(fit_values.shape[1])]
        for key,value in values.items():
            ranks = np.stack([percentile(value[:,i],cdf) for i,cdf in enumerate(cdfs)],-1)
            ranks[~np.isfinite(value)] = np.nan
            # At the first choice no generated context exists: no observed
            # disagreement; use neutral rank instead of silently calling it high.
            ranks[np.isnan(ranks).all(-1)] = .5
            scores[key][view+'_tail'] = np.nanquantile(ranks,.9,axis=-1)
    # Relation anomaly conditional on direct JS, to check incremental information.
    direct = metrics['direct']
    unit = np.median(reference_matrix(direct, selected)[reference_matrix(direct, selected)>1e-10])
    context_unit = np.median(ref_context[ref_context>1e-10])
    matching = {key: .5*(value/unit+context[key]/context_unit) for key,value in direct.items()}
    metric = metrics['relation']
    for row in rows:
        key = row['key']
        value, _, _, _ = local_ratios(metric[key],reference_matrix(metric,selected), matching[key],
            reference_matrix(matching,selected),np.repeat(row['source_id'],len(metric[key])),sources)
        scores[key]['relation_incremental'] = value
        with np.load(previous / key / 'scores.npz') as saved:
            scores[key].update({name:saved[name] for name in BASELINES})
    for name in ('relation_conditional','relation_tail','relation_incremental','excess_tail'):
        values, weights = [], []
        for row in ref_rows:
            valid = valid_tokens(row,output,OLD)
            values.append(scores[row['key']][name][valid])
            weights.append(np.full(valid.sum(),1/valid.sum()))
        cdf = finite_cdf(np.concatenate(values),np.concatenate(weights))
        for score in scores.values():
            score[name+'_fused'] = .75*score['source_route_fixed']+.25*percentile(score[name],cdf)
    write_json(output / (task+'_reference.json'), dict(sources=sources.tolist(),
        identities=[(r['key'],int(t)) for r in ref_rows for t in selected[r['key']]], labels_used=False))
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path,default=Path('outputs/head_state_readout_20260928_v3'))
    args = parser.parse_args()
    read_json(args.output / 'features_complete.json')
    manifest = read_json(args.output / 'manifest.json')
    assert not (args.output / 'scores_frozen.json').exists()
    manifest['previous'] = str(args.previous.resolve())
    write_json(args.output / 'manifest.json',manifest)
    thresholds = {}
    for task in ('QA','Summary','Data2txt'):
        rows = [r for r in manifest['records'] if r['task']==task]
        for row in rows:
            (args.output / row['key'] / 'context.npz').symlink_to((args.previous / row['key'] / 'context.npz').resolve())
        scores = task_scores(args.output,args.previous,rows,task)
        methods = list(next(iter(scores.values())))
        for key,score in scores.items():
            np.savez_compressed(args.output / key / 'scores.npz',**score)
        thresholds[task] = {}
        for name in methods:
            values,weights = [],[]
            for row in rows:
                if row['role'] != 'dev':
                    continue
                valid = valid_tokens(row,args.output,OLD)
                values.append(scores[row['key']][name][valid])
                weights.append(np.full(valid.sum(),1/valid.sum()))
            ordered,cumulative = finite_cdf(np.concatenate(values),np.concatenate(weights))
            thresholds[task][name] = float(ordered[np.searchsorted(cumulative[1:],.95)])
        print('scored',task,flush=True)
    write_json(args.output / 'thresholds.json',thresholds)
    write_json(args.output / 'scores_frozen.json',dict(methods=methods, main='relation_conditional',
        labels_used=False, head_selection=False, status='complete',
        measurement=manifest.get('measurement','source_relation_backward_js'),
        threshold='4 dev sources/task equal-source unlabeled mixture 95th percentile',
        interpretation='descriptive same-head source-context path discrepancy; not factuality probability',
        design_scope='repeatedly exposed development cases; no independent confirmation'))


if __name__ == '__main__':
    main()
