"""Condition complete head adoption states on native source relations."""
import argparse
from pathlib import Path
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.head_state_readout.density import distances, fit_scale, standardize, local_ratios
from experiments.head_state_readout.score import reference_indices, reference_matrix, block_distances, OLD
from experiments.message_js.score import fit_cdf, percentile, valid_tokens

NEW = ('binding_conditional','binding_symmetric','agreement_rank')


def scaled_metric(raw,selected):
    fitted = fit_scale(reference_matrix(raw,selected))
    values = {key:standardize(value,fitted) for key,value in raw.items()}
    reference = reference_matrix(values,selected)
    return {key:distances(value,reference) for key,value in values.items()}


def unit(metric,selected):
    reference = reference_matrix(metric,selected)
    return np.median(reference[reference>1e-10])


def fit_rank(scores,name,rows,output):
    values,weights = [],[]
    for row in rows:
        if row['role']=='fit':
            valid = valid_tokens(row,output,OLD)
            values.append(scores[row['key']][name][valid])
            weights.append(np.full(valid.sum(),1/valid.sum()))
    fitted = fit_cdf(np.concatenate(values),np.concatenate(weights))
    return {key:percentile(value[name],fitted) for key,value in scores.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first',type=Path,required=True)
    parser.add_argument('--states',type=Path,default=Path('outputs/head_state_readout_20260928_v1'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.first/'manifest.json')
    manifest.update(previous=str(args.first.resolve()),states=str(args.states.resolve()))
    write_json(args.output/'manifest.json',manifest)
    for name in ('sources','capture_complete.json','features_complete.json'):
        (args.output/name).symlink_to((args.first/name).resolve())
    for row in manifest['records']:
        directory = args.output/row['key']
        directory.mkdir()
        for path in (args.first/row['key']).iterdir():
            if path.name not in ('scores.npz',):
                (directory/path.name).symlink_to(path.resolve())
    thresholds = read_json(args.first/'thresholds.json')
    for task in ('QA','Summary','Data2txt'):
        rows = [r for r in manifest['records'] if r['task']==task]
        selected = reference_indices(args.output,rows)
        reference_sources = np.concatenate([np.repeat(r['source_id'],len(selected[r['key']]))
            for r in rows if r['key'] in selected])
        blocks = {name:block_distances(args.states,rows,selected,name)[0]
                  for name in ('geometry_head','js','jacobian_head')}
        reading = blocks['geometry_head']
        adoption = {key:.5*(blocks['js'][key]+blocks['jacobian_head'][key]) for key in reading}
        raw = {r['key']:np.load(args.first/r['key']/'relations.npz')['values'] for r in rows}
        relation = scaled_metric({key:v[...,[1,5]].transpose(2,0,1,3).reshape(v.shape[2],-1)
                                  for key,v in raw.items()},selected)
        context = scaled_metric({r['key']:np.load(args.first/r['key']/'context.npz')['values'] for r in rows},selected)
        ru,au,bu,cu = [unit(v,selected) for v in (reading,adoption,relation,context)]
        matching = {key:(reading[key]/ru+relation[key]/bu+context[key]/cu)/3 for key in reading}
        reverse_matching = {key:(reading[key]/ru+adoption[key]/au+context[key]/cu)/3 for key in reading}
        scores = {}
        for row in rows:
            key = row['key']
            with np.load(args.first/key/'scores.npz') as previous:
                scores[key] = {name:previous[name] for name in previous.files}
            values = []
            for name,metric,condition in [('binding_conditional',adoption,matching),
                                           ('relation_given_adoption',relation,reverse_matching)]:
                value,neighbors,radius,scale = local_ratios(metric[key],reference_matrix(metric,selected),
                    condition[key],reference_matrix(condition,selected),
                    np.repeat(row['source_id'],len(metric[key])),reference_sources)
                values.append(value)
                scores[key][name] = value
                np.savez_compressed(args.output/key/(name+'_neighbors.npz'),indices=np.stack(neighbors),
                                    query_radius=radius,neighbor_radius=scale)
            scores[key]['binding_symmetric'] = np.sqrt(values[0]*values[1])
        association_rank = fit_rank(scores,'association',rows,args.output)
        relation_rank = fit_rank(scores,'relation_incremental',rows,args.output)
        for key in scores:
            scores[key]['agreement_rank'] = np.sqrt(association_rank[key]*relation_rank[key])
        for name in NEW:
            rank = fit_rank(scores,name,rows,args.output)
            for key in scores:
                scores[key][name+'_fused'] = .75*scores[key]['source_route_fixed']+.25*rank[key]
        methods = list(next(iter(scores.values())))
        new_methods = [name for name in methods if name not in thresholds[task]]
        for name in new_methods:
            values,weights = [],[]
            for row in rows:
                if row['role']=='dev':
                    valid = valid_tokens(row,args.output,OLD)
                    values.append(scores[row['key']][name][valid])
                    weights.append(np.full(valid.sum(),1/valid.sum()))
            ordered,cumulative = fit_cdf(np.concatenate(values),np.concatenate(weights))
            thresholds[task][name] = float(ordered[np.searchsorted(cumulative[1:],.95)])
        for key,value in scores.items():
            np.savez_compressed(args.output/key/'scores.npz',**value)
        reference = read_json(args.first/(task+'_reference.json'))
        reference['distance_units'] = dict(reading=float(ru),adoption=float(au),relation=float(bu),context=float(cu))
        write_json(args.output/(task+'_reference.json'),reference)
        print('refined',task,flush=True)
    write_json(args.output/'thresholds.json',thresholds)
    write_json(args.output/'scores_frozen.json',dict(methods=methods,main='binding_conditional',status='complete',
        labels_used=False, head_selection=False, primary_condition='equal reading/relation/generation context',
        fusion_scope='source baseline unavailable on natural traces: NaN retained, not a negative prediction',
        design_scope='second exposed-case development iteration, not independent confirmation'))


if __name__ == '__main__':
    main()
