"""First-error evaluation and exact supervised/unsupervised failure locations."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from experiments.gsm8k_recurrence.measure import write_json
from experiments.gsm8k_recurrence.evaluate import metrics,step_label


def make_rows(output,manifest,methods):
    original = {r['id']:r for r in json.loads(Path(manifest['data']).read_text())}
    result = []
    for row in manifest['records']:
        record = original[row['id']]
        gold = record['label']
        with np.load(output/'scores'/(row['id']+'.npz')) as saved:
            for name in methods:
                for step,score in enumerate(saved[name]):
                    result.append(dict(id=row['id'],problem=row['problem'],generator=row['generator'],
                        role=row['role'],method=name,step=step,label=step_label(step,gold),first_error=gold,
                        score=float(score),text=record['steps'][step]))
    return result


def within_answer(rows):
    values = []
    for key in dict.fromkeys(r['id'] for r in rows):
        group = [r for r in rows if r['id']==key and r['label']>=0]
        y = [r['label'] for r in group]
        if len(set(y))==2:
            values.append(roc_auc_score(y,[r['score'] for r in group]))
    return dict(mean=float(np.mean(values)),answers=len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output
    manifest = json.loads((output/'manifest.json').read_text())
    frozen = json.loads((output/'scores_frozen.json').read_text())
    thresholds = json.loads((output/'thresholds.json').read_text())
    rows = make_rows(output,manifest,frozen['methods'])
    result, errors = {},[]
    for name in frozen['methods']:
        result[name] = {}
        for role in ('fit','dev','evaluation'):
            selected = [r for r in rows if r['method']==name and r['role']==role]
            value = metrics(selected,thresholds[name])
            value['within_answer'] = within_answer(selected)
            result[name][role] = value
            if role!='evaluation':continue
            for row in selected:
                label = row['label']
                alarm = row['score']>thresholds[name]
                if (label==1 and not alarm) or (label==0 and alarm):
                    errors.append(dict(**row,status='FN' if label else 'FP',threshold=thresholds[name]))
        # Independent rank definition of AUROC on the known evaluation steps.
        known = [r for r in rows if r['method']==name and r['role']=='evaluation' and r['label']>=0]
        y = np.array([r['label'] for r in known])
        ranks = rankdata([r['score'] for r in known])
        positive = y.sum()
        auc = (ranks[y==1].sum()-positive*(positive+1)/2)/(positive*(len(y)-positive))
        assert abs(auc-result[name]['evaluation']['step_auroc'])<1e-12
    for file,items in [('steps.csv',rows),('errors.csv',errors)]:
        with (output/file).open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    write_json(output/'evaluation.json',dict(status='complete',metrics=result,
        primary=frozen['main'],supervised_diagnostics=frozen['supervised_diagnostics'],
        steps='first error and earlier/all-correct known, later unknown',
        verification='all evaluation AUROCs independently rank-recomputed, same agent'))
    print({name:result[name]['evaluation']['step_auroc'] for name in result},flush=True)


if __name__=='__main__':
    main()
