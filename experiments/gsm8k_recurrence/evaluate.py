"""Evaluate only certified pre-error steps and the first error; later steps unknown."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .measure import write_json
from .score import METHODS


def step_label(index, first_error):
    if first_error<0 or index<first_error:
        return 0
    return 1 if index==first_error else -1


def metrics(rows, threshold):
    known = [r for r in rows if r['label']>=0]
    labels = np.array([r['label'] for r in known])
    score = np.array([r['score'] for r in known])
    positives, negatives = int(labels.sum()), int((labels==0).sum())
    groups = defaultdict(list)
    for row in rows:
        groups[row['id']].append(row)
    incorrect,correct,located,accepted,early,missed = 0,0,0,0,0,0
    localization = []
    for group in groups.values():
        gold = group[0]['first_error']
        alarm = [r['step'] for r in group if r['score']>threshold]
        predicted = min(alarm) if alarm else -1
        if gold<0:
            correct += 1
            accepted += predicted<0
        else:
            incorrect += 1
            located += predicted==gold
            early += 0<=predicted<gold
            missed += predicted<0 or predicted>gold
            localization.append(int(np.argmax([r['score'] for r in group]))==gold)
    error_acc = located/incorrect if incorrect else None
    correct_acc = accepted/correct if correct else None
    harmonic = None
    if error_acc is not None and correct_acc is not None:
        harmonic = 2*error_acc*correct_acc/(error_acc+correct_acc) if error_acc+correct_acc else 0.
    return dict(known_steps=len(known), first_error_steps=positives, correct_steps=negatives,
        unknown_later_steps=sum(r['label']<0 for r in rows),
        step_auroc=float(roc_auc_score(labels,score)) if positives and negatives else None,
        step_ap=float(average_precision_score(labels,score)) if positives else None,
        first_error_step_recall=float(np.mean(score[labels==1]>threshold)) if positives else None,
        correct_step_fpr=float(np.mean(score[labels==0]>threshold)) if negatives else None,
        erroneous_answers=incorrect,correct_answers=correct,first_error_exact=located,
        early_alarms=early,no_or_late_alarms=missed,correct_answers_accepted=accepted,
        erroneous_answer_accuracy=error_acc,correct_answer_accuracy=correct_acc,
        processbench_harmonic=harmonic,
        error_answer_argmax_localization=float(np.mean(localization)) if localization else None)


def evaluate_rows(output, manifest, originals, thresholds):
    rows, predictions = [], []
    for record in manifest['records']:
        if record['role']!='evaluation':
            continue
        key = record['id']
        original = originals[key]
        first = original['label']
        with np.load(output/'scores'/(key+'.npz')) as saved:
            for name in METHODS:
                alarm = np.flatnonzero(saved[name]>thresholds[name])
                predicted = int(alarm[0]) if len(alarm) else -1
                predictions.append(dict(id=key,problem=record['problem'],generator=record['generator'],
                    method=name,first_error=first,predicted=predicted,exact=predicted==first))
                for step,value in enumerate(saved[name]):
                    rows.append(dict(id=key,problem=record['problem'],generator=record['generator'],
                        method=name,step=step,label=step_label(step,first),first_error=first,
                        score=float(value),alarm=bool(value>thresholds[name]),text=original['steps'][step]))
    return rows,predictions


def pair_audit(output, manifest, originals):
    groups = defaultdict(list)
    for record in manifest['records']:
        if record['role']!='evaluation':
            continue
        key = record['id']
        first = originals[key]['label']
        with np.load(output/(key+'.npz')) as measured, np.load(output/'scores'/(key+'.npz')) as scores:
            steps = measured['step_id']
            labels = np.array([step_label(int(s),first) if s>=0 else -1 for s in steps])
            for lag in range(1,9):
                similarity = measured[f'head_{lag}'].mean(0)
                for index,value in enumerate(similarity):
                    left,right = index,index+lag
                    if min(labels[left],labels[right])<0:
                        category = 'unknown_endpoint'
                    elif steps[left]==steps[right]:
                        category = 'same_step_first_error' if labels[left] else 'same_step_correct'
                    else:
                        category = 'cross_into_first_error' if labels[right] else 'cross_step_correct'
                    groups[category].append((float(value),float(scores[f'edge_{lag}'][index])>0))
    return {key:dict(pairs=len(values),mean_affinity=float(np.mean([v[0] for v in values])),
                     active_edge_fraction=float(np.mean([v[1] for v in values]))) for key,values in groups.items()}


def bootstrap_difference(rows):
    lookup = {name:defaultdict(list) for name in ('base','within_step','distance_within')}
    for row in rows:
        if row['method'] in lookup and row['label']>=0:
            lookup[row['method']][row['problem']].append((row['label'],row['score']))
    problems = sorted(lookup['base'])
    rng = np.random.default_rng(429)
    differences = {name:[] for name in ('base','distance_within')}
    for _ in range(300):
        chosen = rng.choice(problems,len(problems),replace=True)
        auc = {}
        for name in lookup:
            value = np.array([pair for problem in chosen for pair in lookup[name][problem]])
            auc[name] = roc_auc_score(value[:,0],value[:,1])
        for name in differences:
            differences[name].append(auc['within_step']-auc[name])
    return {name:np.quantile(values,[.025,.975]).tolist() for name,values in differences.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    frozen = json.loads((args.output/'scores_frozen.json').read_text())
    manifest = json.loads((args.output/'manifest.json').read_text())
    thresholds = json.loads((args.output/'thresholds.json').read_text())
    originals = {r['id']:r for r in json.loads(Path(manifest['data']).read_text())}
    rows,predictions = evaluate_rows(args.output,manifest,originals,thresholds)
    results = {name:metrics([r for r in rows if r['method']==name],thresholds[name]) for name in METHODS}
    generators = {}
    for generator in sorted({r['generator'] for r in rows}):
        generators[generator] = {name:metrics([r for r in rows if r['method']==name and r['generator']==generator],
                                              thresholds[name]) for name in METHODS}
    for filename,values in (('steps.csv',rows),('predictions.csv',predictions)):
        with (args.output/filename).open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write_json(args.output/'evaluation.json',dict(metrics=results,by_generator=generators,
        split_answer_counts=dict(Counter(r['role'] for r in manifest['records'])),
        split_problem_counts={role:len({r['problem'] for r in manifest['records'] if r['role']==role})
                              for role in ('fit','dev','evaluation')},
        pair_audit=pair_audit(args.output,manifest,originals),
        within_step_auc_delta95=bootstrap_difference(rows),
        primary=frozen['main'],scope='single-layer attention-only transfer on historically exposed ProcessBench',
        labels='first error positive; earlier/all-correct steps negative; later unknown; no token truth invented'))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    main()
