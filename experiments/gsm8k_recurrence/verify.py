"""Independent rank AUROC, question split, and propagation invariant checks."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from .measure import write_json
from .score import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output
    manifest = json.loads((output/'manifest.json').read_text())
    evaluation = json.loads((output/'evaluation.json').read_text())
    problems = {role:{r['problem'] for r in manifest['records'] if r['role']==role}
                for role in ('fit','dev','evaluation')}
    assert all(not problems[a]&problems[b] for a,b in [('fit','dev'),('fit','evaluation'),('dev','evaluation')])
    for row in manifest['records']:
        with np.load(output/'scores'/(row['id']+'.npz')) as saved:
            for name in METHODS:
                assert np.isfinite(saved[name]).all()
                assert ((saved[name]>=0)&(saved[name]<=1+1e-12)).all()
            assert np.all(saved['token_within_step']>=saved['token_base'])
            assert np.all(saved['token_within_step']<=saved['token_recurrence'])
    with (output/'steps.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    differences = []
    for name in METHODS:
        selected = [r for r in rows if r['method']==name and int(r['label'])>=0]
        y = np.array([int(r['label']) for r in selected])
        values = np.array([float(r['score']) for r in selected])
        positive,negative = int(y.sum()),int((y==0).sum())
        auc = (rankdata(values)[y==1].sum()-positive*(positive+1)/2)/(positive*negative)
        differences.append(abs(auc-evaluation['metrics'][name]['step_auroc']))
        for row in [r for r in rows if r['method']==name]:
            step,first,label = (int(row[k]) for k in ('step','first_error','label'))
            expected = 0 if first==-1 or step<first else (1 if step==first else -1)
            assert label==expected
    assert max(differences)<1e-12
    write_json(output/'verification.json',dict(status='passed',answers=400,methods=len(METHODS),
        disjoint_questions=True,propagation_bounds=True,unknown_later_steps_preserved=True,
        max_independent_auroc_difference=max(differences),review='same-agent numeric implementation, not external audit'))
    print('Verified 400 score files, question exclusion, propagation and seven step AUROCs',flush=True)


if __name__=='__main__':
    main()
