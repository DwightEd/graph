"""Evaluate frozen scores and enumerate improvements and regressions per position."""
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from experiments.decision_risk_flow.data import read_json,write_json
from experiments.constraint_uptake.evaluate import label_tables,rag_units,step_units,metrics,first_error
from .model import BOUND


def write_csv(path,rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_units(records,output,methods):
    cases = [r for r in records if r['role']=='case']
    truth,local_truth,gsm = label_tables(cases)
    rows = []
    for record in cases:
        count = len(record['response']['answer_ids'])
        row = dict(record,answer=record['response']['answer_ids'],positions=list(range(count)),
            text=record['response']['token_text'])
        with np.load(output/record['key']/'scores.npz') as scores:
            if row['dataset']=='ragtruth':
                units = rag_units(row,scores,methods,truth,local_truth)
            else:
                units = step_units(row,scores,methods,gsm)
        for unit in units:
            unit['task'] = record['task']
        rows.extend(units)
    return rows


def paired_order_changes(rows,name):
    positive = [r for r in rows if r['gold']==1]
    negative = [r for r in rows if r['gold']==0]
    before = np.subtract.outer([r['base'] for r in positive],[r['base'] for r in negative])
    after = np.subtract.outer([r[name] for r in positive],[r[name] for r in negative])
    difference = .5*(np.sign(after)-np.sign(before))
    return dict(improved_pairs=float(np.maximum(difference,0).sum()),
        worsened_pairs=float(np.maximum(-difference,0).sum()),pairs=int(difference.size))


def summarize(rows,name,thresholds):
    known = [r for r in rows if r['gold']>=0]
    result = metrics(rows,name,0.)
    scores = np.array([r[name] for r in known])
    y = np.array([r['gold'] for r in known])
    alarm = np.array([r[name]>thresholds[r['task']] for r in known])
    old = np.array([r['base']>thresholds[r['task']] for r in known])
    result.update(tp=int((alarm&(y==1)).sum()),fp=int((alarm&(y==0)).sum()),
        fn=int((~alarm&(y==1)).sum()),threshold='unchanged task baseline threshold',
        gained_tp=int((alarm&~old&(y==1)).sum()),lost_tp=int((~alarm&old&(y==1)).sum()),
        added_fp=int((alarm&~old&(y==0)).sum()),removed_fp=int((~alarm&old&(y==0)).sum()),
        **paired_order_changes(known,name))
    if len(set(y))==2:
        positive = y.sum()
        rank_auc = (rankdata(scores)[y==1].sum()-positive*(positive+1)/2)/(positive*(len(y)-positive))
        assert abs(rank_auc-result['auroc'])<1e-12
    return result


def historical_gsm(rows):
    root = Path('outputs/gsm8k_states_20260929_v1')
    threshold = read_json(root/'thresholds.json')['attention_base']
    compared = []
    for row in rows:
        with np.load(root/'scores'/(row['key']+'.npz')) as saved:
            value = float(saved['attention_base'][row['position']])
        compared.append(dict(row,old_attention=value))
    return dict(metrics=metrics(compared,'old_attention',threshold),
        first_error=first_error(compared,'old_attention',threshold),
        scope='same six answers; historical single-layer attention with its original dev calibration')


def correction_reachability(rows,thresholds):
    positive = [r for r in rows if r['gold']==1]
    missed = [r for r in positive if r['base']<=thresholds[r['task']]]
    unreachable = [r for r in missed if r['base']+BOUND<=thresholds[r['task']]]
    return dict(positive=len(positive),baseline_fn=len(missed),unreachable_fn=len(unreachable),
        maximum_possible_tp=len(positive)-len(unreachable),bound=BOUND,
        rule='base + bound <= threshold cannot alarm, independent of message graph or optimizer',
        positions=[dict(key=r['key'],position=r['position'],text=r['text'],
                        required_increase=thresholds[r['task']]-r['base']) for r in unreachable])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    frozen = read_json(args.output/'scores_frozen.json')
    records = read_json(args.output/'manifest.json')['records']
    thresholds = read_json(args.output/'thresholds.json')
    methods = frozen['methods']
    rows = make_units(records,args.output,methods)
    groups = {name:[r for r in rows if r['cohort']==name] for name in ('rag_official','rag_local','gsm_step')}
    result = {group:{name:summarize(items,name,thresholds) for name in methods} for group,items in groups.items()}
    cases = {key:{name:summarize([r for r in rows if r['key']==key],name,thresholds) for name in methods}
             for key in dict.fromkeys(r['key'] for r in rows)}
    predictions = {name:first_error(groups['gsm_step'],name,thresholds['GSM8K']) for name in methods}
    errors,changes = [],[]
    for row in rows:
        if row['gold']<0:
            continue
        old = row['base']>thresholds[row['task']]
        for name in methods:
            alarm = row[name]>thresholds[row['task']]
            item = dict(key=row['key'],cohort=row['cohort'],position=row['position'],gold=row['gold'],
                text=row['text'],method=name,base=row['base'],score=row[name],
                threshold=thresholds[row['task']],old_alarm=bool(old),new_alarm=bool(alarm))
            if alarm!=bool(row['gold']):
                errors.append(dict(item,status='FN' if row['gold'] else 'FP'))
            if alarm!=old:
                changes.append(dict(item,change=('gained_tp' if row['gold'] else 'added_fp') if alarm
                    else ('lost_tp' if row['gold'] else 'removed_fp')))
    write_csv(args.output/'units.csv',rows)
    write_csv(args.output/'errors.csv',errors)
    if changes:
        write_csv(args.output/'changes.csv',changes)
    write_json(args.output/'evaluation.json',dict(status='complete',cohorts=result,cases=cases,
        first_error=predictions,primary=frozen['primary'],thresholds=thresholds,
        historical_gsm=historical_gsm(groups['gsm_step']),
        reachability={name:correction_reachability(items,thresholds) for name,items in groups.items()},
        verification='all two-class AUCs independently rank-recomputed; labels read after frozen scores'))
    for cohort,values in result.items():
        print(cohort,{name:(v['auroc'],v['tp'],v['fp']) for name,v in values.items()},flush=True)
    print('first errors',predictions[frozen['primary']],flush=True)


if __name__=='__main__':
    main()
