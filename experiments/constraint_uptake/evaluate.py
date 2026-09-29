"""Evaluate frozen pilot rankings, preserving local/official/step label scopes."""
import argparse
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.decision_risk_flow.data import inputs,labels,read_json,write_json
from experiments.role_free_flow.diagnostics import read_local_spans
from experiments.gsm8k_recurrence.evaluate import step_label


def label_tables(records):
    official = [r['original'] for r in records if r['dataset']=='ragtruth' and r['original']['kind']=='observer']
    truth = labels(official)
    local = read_local_spans(Path('outputs/role_free_flow_20260928'),Path('experiments/path_conflict/paired_cases.json'))
    local_truth = {(r.trace,int(r.position)):int(r.local_label) for r in local.itertuples()}
    gsm_manifest = read_json('outputs/gsm8k_states_20260929_v1/manifest.json')
    gsm = {row['id']:row for row in read_json(gsm_manifest['data'])}
    return truth,local_truth,gsm


def rag_units(row,scores,methods,truth,local_truth):
    result = []
    official = row['original']['kind']=='observer'
    if official:
        response = inputs(row['original'])[1]
        valid = [b>a and token not in response['special_ids']
                 for (a,b),token in zip(response['offsets'],row['answer'])]
    for index,position in enumerate(row['positions']):
        if official:
            gold = int(truth[row['key']][position]) if valid[position] else -1
        else:
            gold = local_truth.get((row['original']['trace'],position),-1)
        result.append(dict(key=row['key'],pair=row['pair'],cohort='rag_official' if official else 'rag_local',
            position=position,gold=gold,first_error=-1,text=row['text'][position],
            **{name:float(scores[name][index]) for name in methods}))
    return result


def step_units(row,scores,methods,gsm):
    result = []
    positions = np.array(row['positions'])
    record = gsm[row['key']]
    for step,(start,end) in enumerate(row['step_ranges']):
        selected = (positions>=start)&(positions<end)
        result.append(dict(key=row['key'],pair=row['pair'],cohort='gsm_step',position=step,
            gold=step_label(step,record['label']),first_error=record['label'],text=record['steps'][step],
            **{name:float(scores[name][selected].mean()) for name in methods}))
    return result


def metrics(rows,name,threshold):
    selected = [r for r in rows if r['gold']>=0]
    y = np.array([r['gold'] for r in selected])
    score = np.array([r[name] for r in selected])
    alarm = score>threshold
    both = len(np.unique(y))==2
    return dict(known=len(y),positive=int(y.sum()),unknown=len(rows)-len(selected),
        auroc=float(roc_auc_score(y,score)) if both else None,
        ap=float(average_precision_score(y,score)) if y.sum() else None,
        tp=int((alarm&(y==1)).sum()),fp=int((alarm&(y==0)).sum()),
        fn=int((~alarm&(y==1)).sum()),threshold=threshold)


def first_error(rows,name,threshold):
    result = []
    for key in dict.fromkeys(r['key'] for r in rows):
        steps = [r for r in rows if r['key']==key]
        alarm = [r['position'] for r in steps if r[name]>threshold]
        prediction = min(alarm) if alarm else -1
        result.append(dict(key=key,gold=steps[0]['first_error'],predicted=prediction,
            argmax=int(np.argmax([r[name] for r in steps]))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    frozen = read_json(args.output/'scores_frozen.json')
    records = read_json(args.output/'manifest.json')['records']
    truth,local_truth,gsm = label_tables(records)
    methods = frozen['methods']
    thresholds = {name:.5 for name in methods}
    thresholds.update(negative_margin=0.,nll=float(np.log(2)))
    rows = []
    for record in records:
        scores = np.load(args.output/record['key']/'scores.npz')
        if record['dataset']=='ragtruth':
            rows.extend(rag_units(record,scores,methods,truth,local_truth))
        else:
            rows.extend(step_units(record,scores,methods,gsm))
    groups = ('rag_official','rag_local','gsm_step')
    result = {group:{name:metrics([r for r in rows if r['cohort']==group],name,thresholds[name]) for name in methods}
              for group in groups}
    pairs = {}
    for pair in dict.fromkeys(r['pair'] for r in rows):
        pairs[pair] = {name:metrics([r for r in rows if r['pair']==pair],name,thresholds[name]) for name in methods}
    predictions = {name:first_error([r for r in rows if r['cohort']=='gsm_step'],name,thresholds[name]) for name in methods}
    errors = []
    for row in rows:
        if row['gold']<0:
            continue
        for name in methods:
            alarm = row[name]>thresholds[name]
            if alarm!=bool(row['gold']):
                errors.append(dict(key=row['key'],cohort=row['cohort'],position=row['position'],
                    text=row['text'],method=name,status='FN' if row['gold'] else 'FP',
                    score=row[name],threshold=thresholds[name]))
    for file,table in (('units.csv',rows),('errors.csv',errors)):
        with (args.output/file).open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    write_json(args.output/'evaluation.json',dict(status='complete',cohorts=result,pairs=pairs,
        first_error=predictions,threshold_scope='fixed exploratory operating points, no dev calibration',
        primary=frozen['primary'],no_independent_generalization_claim=True))
    print({group:{name:value['auroc'] for name,value in values.items()} for group,values in result.items()},flush=True)


if __name__=='__main__':
    main()
