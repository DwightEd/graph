"""Independent ranking checks, finite-effect accuracy, and hidden-path integrity."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from experiments.decision_risk_flow.data import read_json,write_json


def check_rankings(output):
    evaluation = read_json(output/'evaluation.json')
    rows = list(csv.DictReader((output/'units.csv').open()))
    count = 0
    for cohort,methods in evaluation['cohorts'].items():
        selected = [r for r in rows if r['cohort']==cohort and int(r['gold'])>=0]
        y = np.array([int(r['gold']) for r in selected])
        positives = y.sum()
        for name,result in methods.items():
            score = np.array([float(r[name]) for r in selected])
            assert np.isfinite(score).all()
            ranks = rankdata(score)
            auc = (ranks[y==1].sum()-positives*(positives+1)/2)/(positives*(len(y)-positives))
            assert abs(auc-result['auroc'])<1e-12,(cohort,name)
            alarm = score>result['threshold']
            assert int((alarm&(y==0)).sum())==result['fp']
            assert int((alarm&(y==1)).sum())==result['tp']
            count += 1
    return count


def finite_summary(rows):
    result = {}
    for dose in sorted({r['dose'] for r in rows}):
        selected = [r for r in rows if r['dose']==dose]
        predicted = np.array([r['predicted'] for r in selected])
        observed = np.array([r['observed'] for r in selected])
        available = np.abs(predicted)>1e-4
        relative = np.abs(observed[available]-predicted[available])/np.abs(predicted[available])
        result[str(dose)] = dict(count=len(selected),nontrivial=int(available.sum()),
            sign_agree=int((np.sign(observed[available])==np.sign(predicted[available])).sum()),
            median_relative_error=float(np.median(relative)) if len(relative) else None,
            max_absolute_error=float(np.max(np.abs(observed-predicted))))
    return result


def interaction_summary(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[(row['key'],row['token'],row['dose'])][row['treatment']] = row
    result = []
    for (key,token,dose),group in groups.items():
        if dose==0:
            continue
        effect = {name:row['observed'] for name,row in group.items()}
        joint = effect['joint']-effect['positive']-effect['negative']
        control = effect['control_joint']-effect['positive']-effect['control']
        result.append(dict(key=key,token=token,dose=dose,gamma=joint,control_gamma=control,
            normalized_gamma=abs(joint)/max(abs(effect['positive'])+abs(effect['negative']),1e-6),
            normalized_control_gamma=abs(control)/max(abs(effect['positive'])+abs(effect['control']),1e-6)))
    return result


def check_paths(output,rows):
    paths = []
    for row in rows:
        with np.load(output/row['state_file']) as saved:
            delta = saved['delta_state']
            lens = saved['delta_lens']
            assert delta.shape==(32,4096)
            assert np.isfinite(delta).all()
            assert np.max(np.abs(delta[:row['layer']]),initial=0)==0
            assert abs(float(lens[-1])-row['observed'])<5e-5
            if row['dose']==0:
                assert np.max(np.abs(delta))==0
            active = lens[row['layer']:]
            active = active[np.abs(active)>1e-4]
            flips = int((np.sign(active[1:])!=np.sign(active[:-1])).sum())
            if row['dose']==.25:
                paths.append(dict(key=row['key'],treatment=row['treatment'],layer=row['layer'],head=row['head'],
                    target=row['target_text'],competitor=row['competitor_text'],layer_sign_changes=flips,
                    final_effect=row['observed'],block_text=row['block_text']))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--reweighted',type=Path,required=True)
    parser.add_argument('--dense',type=Path,required=True)
    parser.add_argument('--centered',type=Path)
    parser.add_argument('--exchange',type=Path)
    args = parser.parse_args()
    counts = {str(p):check_rankings(p) for p in (args.source,args.reweighted,args.dense/'weighted')}
    if args.centered is not None:
        counts[str(args.centered)] = check_rankings(args.centered)
    finite = read_json(args.source/'finite.json')['rows']
    interactions = read_json(args.dense/'interactions.json')['rows']
    paths = check_paths(args.dense,interactions)
    manifest = read_json(args.source/'manifest.json')
    reuse_checks = 0
    for row in manifest['records']:
        if row['dataset']!='ragtruth':
            continue
        old = np.load(args.reweighted/row['key']/'scores.npz')
        dense = np.load(args.dense/'weighted'/row['key']/'scores.npz')
        for name in old.files:
            np.testing.assert_array_equal(old[name],dense[name])
            reuse_checks += 1
    result = dict(status='passed',independent_rank_checks=counts,
        unchanged_rag_arrays=reuse_checks,finite=finite_summary(finite),
        targeted_finite=finite_summary(interactions),interactions=interaction_summary(interactions),
        paths=paths,scope='same-agent independent numeric checks, not an external scientific review')
    destination = args.dense
    if args.exchange is not None:
        exchanged = read_json(args.exchange/'interactions.json')['rows']
        result['exchange_finite'] = finite_summary(exchanged)
        result['exchange_paths'] = check_paths(args.exchange,exchanged)
        result['exchange_interactions'] = interaction_summary(exchanged)
        destination = args.exchange
    write_json(destination/'audit.json',result)
    print('audited',sum(counts.values()),'rankings and',len(interactions),'full hidden paths',flush=True)


if __name__=='__main__':
    main()
