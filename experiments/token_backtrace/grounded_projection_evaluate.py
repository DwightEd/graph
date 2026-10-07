"""Evaluation only: predictions and protocol are frozen before annotations enter."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from state_audit.dataset import Example
from experiments.native_support.ragtruth import aligned_labels
from .grounded_projection_data import write_json


BASES = ('node', 'static', 'nll', 'graph', 'flat', 'rewired', 'edge_increment')


def empirical_tail(reference, values):
    rank = np.searchsorted(np.sort(reference), values, side='right')
    return -np.log1p(-(rank + .5) / (len(reference) + 1))


def calibrated_scores(cases, scores):
    references, thresholds = {}, {}
    for task in ('QA', 'Summary', 'Data2txt'):
        reference = [scores[c['id']] for c in cases if c['cohort'] == 'reference' and c['task'] == task]
        references[task] = {name: np.concatenate([r[name] for r in reference]) for name in BASES}
    for case in cases:
        values, reference = scores[case['id']], references[case['task']]
        for name in ('graph', 'flat', 'rewired'):
            values['joint_' + name] = (empirical_tail(reference['node'], values['node'])
                                      + empirical_tail(reference[name], values[name]))
    for task, reference in references.items():
        rows = [scores[c['id']] for c in cases if c['cohort'] == 'reference' and c['task'] == task]
        thresholds[task] = {name: float(np.quantile(np.concatenate([r[name] for r in rows]), .95))
                           for name in BASES + ('joint_graph', 'joint_flat', 'joint_rewired')}
    return thresholds


def load_annotations(inputs):
    wanted = {c['id']: c for c in inputs['cases'] if c['cohort'] != 'reference'}
    result = {}
    for line in (Path(inputs['dataset']) / 'response.jsonl').open():
        row = json.loads(line)
        identity = str(row['id'])
        if identity not in wanted:
            continue
        case, response = wanted[identity], wanted[identity]['response']
        assert row['response'] == response['text']
        assert str(row['source_id']) == case['source_id'] and row['labels'] is not None
        example = Example(identity, case['source_id'], '', response=row['response'], labels=row['labels'])
        encoded = dict(input_ids=response['answer_ids'], offset_mapping=response['offsets'])
        result[identity] = aligned_labels(example, encoded, response['special_ids'])
    assert set(result) == set(wanted)
    return result


def metrics(cases, scores, annotations, thresholds, name):
    truths, values, alarms = [], [], []
    onsets, onset_hits, normal_answers, normal_alarm = 0, 0, 0, 0
    for case in cases:
        annotation = annotations[case['id']]
        valid = np.asarray(annotation['valid_tokens'], bool)
        truth = np.asarray(annotation['labels'])[valid]
        risk = scores[case['id']][name][valid]
        alarm = risk > thresholds[case['task']][name]
        onset = np.asarray(annotation['span_onsets'], bool)[valid]
        truths.append(truth)
        values.append(risk)
        alarms.append(alarm)
        onsets += int(onset.sum())
        onset_hits += int((onset & alarm).sum())
        normal_answers += not truth.any()
        normal_alarm += int(not truth.any() and alarm.any())
    truth, risk, alarm = map(np.concatenate, (truths, values, alarms))
    tp, fp = int((truth & alarm).sum()), int(((1 - truth) & alarm).sum())
    return dict(auroc=float(roc_auc_score(truth, risk)), ap=float(average_precision_score(truth, risk)),
        tokens=len(truth), wrong=int(truth.sum()), tp=tp, fp=fp,
        recall=tp / max(int(truth.sum()), 1), precision=tp / max(tp + fp, 1),
        onset_hits=onset_hits, onsets=onsets, normal_answers=normal_answers, normal_alarm=normal_alarm)


def bootstrap(cases, scores, annotations, first, second, draws=1000):
    source_ids = sorted({c['source_id'] for c in cases})
    groups = {}
    for source in source_ids:
        rows = [c for c in cases if c['source_id'] == source]
        truth, a, b = [], [], []
        for c in rows:
            valid = np.asarray(annotations[c['id']]['valid_tokens'], bool)
            truth.append(np.asarray(annotations[c['id']]['labels'])[valid])
            a.append(scores[c['id']][first][valid])
            b.append(scores[c['id']][second][valid])
        groups[source] = tuple(map(np.concatenate, (truth, a, b)))
    generator = np.random.default_rng(42)
    differences = []
    for _ in range(draws):
        sampled = [groups[s] for s in generator.choice(source_ids, len(source_ids), replace=True)]
        truth, a, b = [np.concatenate([row[j] for row in sampled]) for j in range(3)]
        if 0 < truth.sum() < len(truth):
            differences.append(roc_auc_score(truth, a) - roc_auc_score(truth, b))
    return dict(first=first, second=second, draws=len(differences),
                auroc_delta_ci95=np.quantile(differences, [.025, .975]).tolist())


def case_report(case, scores, annotation, thresholds):
    truth = np.asarray(annotation['labels'], bool)
    valid = np.asarray(annotation['valid_tokens'], bool)
    onset = np.asarray(annotation['span_onsets'], bool)
    row = dict(id=case['id'], task=case['task'], source_id=case['source_id'],
               tokens=int(valid.sum()), wrong=int((truth & valid).sum()))
    for name in BASES + ('joint_graph', 'joint_flat', 'joint_rewired'):
        alarm = (scores[name] > thresholds[name]) & valid
        row[name] = dict(tp=int((alarm & truth).sum()), fp=int((alarm & ~truth).sum()),
            onset_hits=int((alarm & onset).sum()), onsets=int(onset.sum()))
    row['wrong_spans'] = []
    for span in annotation['character_spans']:
        offsets = np.asarray(case['response']['offsets'])
        mask = (offsets[:, 0] < span['end']) & (offsets[:, 1] > span['start']) & valid
        indices = np.where(mask)[0]
        alarm = scores['joint_graph'] > thresholds['joint_graph']
        row['wrong_spans'].append(dict(text=case['response']['text'][span['start']:span['end']],
            token_start=int(indices[0]), token_end=int(indices[-1]) + 1,
            detected=int((alarm & mask).sum()), tokens=int(mask.sum())))
    top = np.argsort(scores['joint_graph'] + np.where(valid, 0, -1e20))[-10:][::-1]
    row['top_tokens'] = [dict(index=int(i), text=case['response']['token_text'][i], wrong=bool(truth[i]),
                             score=float(scores['joint_graph'][i])) for i in top]
    return row


def evaluate(output):
    inputs = json.loads((output / 'inputs.json').read_text())
    assert json.loads((output / 'execution.json').read_text())['status'] == 'DONE'
    cases = inputs['cases']
    paths = [output / c['id'] / 'scores.npz' for c in cases]
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_json(output / 'frozen_scores.json', frozen)
    scores = {c['id']: dict(np.load(output / c['id'] / 'scores.npz')) for c in cases}
    for case in cases:
        assert np.array_equal(scores[case['id']]['token_id'], case['response']['answer_ids'])
        assert all(np.isfinite(v).all() for v in scores[case['id']].values())
    thresholds = calibrated_scores(cases, scores)
    write_json(output / 'thresholds.json', thresholds)
    annotations = load_annotations(inputs)
    write_json(output / 'evaluation_annotations.json', annotations)
    regression = [c for c in cases if c['cohort'] != 'reference']
    names = BASES + ('joint_graph', 'joint_flat', 'joint_rewired')
    result = dict(scope='reused natural regression pilot, not independent held-out confirmation',
        metrics={n: metrics(regression, scores, annotations, thresholds, n) for n in names},
        comparisons=[bootstrap(regression, scores, annotations, 'joint_graph', n)
                     for n in ('joint_rewired', 'joint_flat', 'node')],
        cases=[case_report(c, scores[c['id']], annotations[c['id']], thresholds[c['task']]) for c in regression])
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [c for c in regression if c['task'] == task]
        if sum(sum(annotations[c['id']]['labels']) for c in rows):
            result.setdefault('tasks', {})[task] = {n: metrics(rows, scores, annotations, thresholds, n) for n in names}
    write_json(output / 'results.json', result)
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    evaluate(parser.parse_args().output)
