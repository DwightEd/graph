"""One label-free calibration for complementary internal mechanism modules."""
import argparse
from pathlib import Path
import numpy as np
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.message_js.score import fit_cdf, percentile, valid_tokens
from experiments.head_state_readout.score import OLD
from .refine import fit_rank


def two_sided(output,rows):
    values = {}
    for row in rows:
        raw = np.load(output/row['key']/'relations.npz')['values']
        values[row['key']] = raw[...,[1,5]].transpose(2,0,1,3).reshape(raw.shape[2],-1)
    fit = [r for r in rows if r['role']=='fit']
    samples = np.concatenate([values[r['key']][valid_tokens(r,output,OLD)] for r in fit])
    weights = np.concatenate([np.full(valid_tokens(r,output,OLD).sum(),1/valid_tokens(r,output,OLD).sum()) for r in fit])
    cdfs = [fit_cdf(samples[:,i],weights) for i in range(samples.shape[1])]
    result = {}
    for key,value in values.items():
        ranks = np.stack([percentile(value[:,i],cdf) for i,cdf in enumerate(cdfs)],-1)
        ranks[np.isnan(ranks).all(-1)] = .5
        result[key] = np.nanquantile(np.abs(2*ranks-1),.9,axis=-1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first',type=Path,required=True)
    parser.add_argument('--forward',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.forward/'manifest.json')
    manifest.update(previous=str(args.forward.resolve()),backward=str(args.first.resolve()),
                    measurement='joint_internal_module_calibration')
    write_json(args.output/'manifest.json',manifest)
    for name in ('sources','capture_complete.json','features_complete.json'):
        (args.output/name).symlink_to((args.forward/name).resolve())
    for row in manifest['records']:
        directory = args.output/row['key']
        directory.mkdir()
        for path in (args.forward/row['key']).iterdir():
            if path.name!='scores.npz':
                (directory/path.name).symlink_to(path.resolve())
    thresholds = read_json(args.forward/'thresholds.json')
    for task in ('QA','Summary','Data2txt'):
        rows = [r for r in manifest['records'] if r['task']==task]
        tails = two_sided(args.output,rows)
        scores = {}
        for row in rows:
            key = row['key']
            with np.load(args.forward/key/'scores.npz') as saved:
                scores[key] = {name:saved[name] for name in saved.files}
            scores[key]['backward_incremental'] = np.load(args.first/key/'scores.npz')['relation_incremental']
            scores[key]['forward_two_sided'] = tails[key]
        names = ('association','backward_incremental','relation_tail','forward_two_sided')
        ranks = {name:fit_rank(scores,name,rows,args.output) for name in names}
        for key in scores:
            matrix = np.stack([ranks[name][key] for name in names],-1)
            scores[key]['omnibus_max'] = matrix.max(-1)
            scores[key]['omnibus_mean'] = matrix.mean(-1)
            np.savez_compressed(args.output/key/'module_ranks.npz',names=names,values=matrix)
        for name in ('omnibus_max','omnibus_mean','forward_two_sided'):
            ranked = fit_rank(scores,name,rows,args.output)
            for key in scores:
                scores[key][name+'_fused'] = .75*scores[key]['source_route_fixed']+.25*ranked[key]
        methods = list(next(iter(scores.values())))
        for name in methods:
            if name in thresholds[task]:
                continue
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
        (args.output/(task+'_reference.json')).symlink_to((args.forward/(task+'_reference.json')).resolve())
        print('combined',task,flush=True)
    write_json(args.output/'thresholds.json',thresholds)
    write_json(args.output/'scores_frozen.json',dict(status='complete',methods=methods,main='omnibus_max',
        labels_used=False,head_selection=False,source_specific_rules=False,
        calibration='new maximum/mean calibrated on unlabeled dev mixture, not OR of separate thresholds',
        module_names=names,design_scope='exposed-case development; not independent confirmation'))


if __name__ == '__main__':
    main()
