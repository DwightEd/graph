"""Gold is read only here, after immutable score and threshold files exist."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from experiments.token_backtrace.grounded_projection_data import write_json
from experiments.token_backtrace.grounded_projection_evaluate import (
    load_annotations, metrics, bootstrap)
from .data import digest_files


NAMES = ('node', 'native', 'rewired', 'nll')


def within_answer_auc(cases, scores, annotations, name):
    values = []
    for case in cases:
        annotation = annotations[case['id']]
        valid = np.asarray(annotation['valid_tokens'], bool)
        truth = np.asarray(annotation['labels'])[valid]
        if 0 < truth.sum() < len(truth):
            values.append(roc_auc_score(truth, scores[case['id']][name][valid]))
    return dict(mean=float(np.mean(values)), mixed_answers=len(values))


def detailed_case(case, scores, annotation, thresholds, names=NAMES):
    valid = np.asarray(annotation['valid_tokens'], bool)
    truth = np.asarray(annotation['labels'], bool)
    offsets = np.asarray(case['response']['offsets'])
    row = dict(id=case['id'], source_id=case['source_id'], task=case['task'],
               tokens=int(valid.sum()), wrong=int((truth & valid).sum()), methods={})
    for name in names:
        alarm = (scores[name] > thresholds[name]) & valid
        onsets = np.asarray(annotation['span_onsets'], bool)
        row['methods'][name] = dict(tp=int((alarm & truth).sum()), fp=int((alarm & ~truth).sum()),
            onset_hits=int((alarm & onsets).sum()), onsets=int(onsets.sum()))
    row['wrong_spans'] = []
    for span in annotation['character_spans']:
        mask = (offsets[:, 0] < span['end']) & (offsets[:, 1] > span['start']) & valid
        row['wrong_spans'].append(dict(text=case['response']['text'][span['start']:span['end']],
            tokens=int(mask.sum()), hits={name: int((mask & (scores[name] > thresholds[name])).sum())
                                         for name in names}))
    return row


def write_visualization(output, cases, scores, annotations, thresholds, names=NAMES):
    parts = ['<!doctype html><meta charset="utf-8"><title>Flow latent token pilot</title>',
        '<style>body{max-width:1100px;margin:30px auto;font:16px/1.9 sans-serif}'
        '.alarm{background:#ffc970}.wrong{text-decoration:underline;text-decoration-color:#c11}'
        'article{border-top:1px solid #ccc;white-space:pre-wrap}</style>',
        '<h1>Flow latent：逐 token 回归结果</h1><p>橙底：报警；红下划线：标注错误。'
        '使用全部原始 token；标注仅在评分后用于展示。已有回归例，非独立确认。</p>']
    for case in cases:
        identity = case['id']
        truth = np.asarray(annotations[identity]['labels'], bool)
        parts.append(f'<h2>{identity} — {html.escape(case["task"])}</h2>')
        for name in names:
            alarm = scores[identity][name] > thresholds[case['task']][name]
            parts.append(f'<h3>{name}</h3><article>')
            for index, token in enumerate(case['response']['token_text']):
                classes = ('alarm ' if alarm[index] else '') + ('wrong' if truth[index] else '')
                value = float(scores[identity][name][index])
                parts.append(f'<span class="{classes}" title="t={index}, score={value:.4f}">'
                             f'{html.escape(token)}</span>')
            parts.append('</article>')
    (output / 'tokens.html').write_text(''.join(parts))


def evaluate(output):
    inputs = json.loads((output / 'inputs.json').read_text())
    protocol = json.loads((output / 'protocol.json').read_text())
    names = NAMES + (('lag_pair',) if protocol.get('lag_pair_control') else ())
    assert json.loads((output / 'execution.json').read_text())['status'] == 'DONE'
    frozen = json.loads((output / 'frozen_scores.json').read_text())
    assert digest_files([Path(name) for name in frozen]) == frozen, 'Scores changed after freeze'
    cases = [case for case in inputs['cases'] if case['cohort'] == 'regression']
    scores = {case['id']: dict(np.load(output / case['id'] / 'scores.npz')) for case in cases}
    for case in cases:
        assert np.array_equal(scores[case['id']]['token_id'], case['response']['answer_ids'])
    thresholds = json.loads((output / 'thresholds.json').read_text())
    annotations = load_annotations(dict(inputs, cases=cases))
    write_json(output / 'evaluation_annotations.json', annotations)
    result = dict(scope='previously studied exploratory regression, not independent confirmation',
        metrics={name: metrics(cases, scores, annotations, thresholds, name) for name in names},
        within_answer={name: within_answer_auc(cases, scores, annotations, name) for name in names},
        comparisons=[bootstrap(cases, scores, annotations, 'native', name) for name in (('node', 'rewired', 'lag_pair') if 'lag_pair' in names else ('node', 'rewired'))],
        cases=[detailed_case(case, scores[case['id']], annotations[case['id']], thresholds[case['task']], names)
               for case in cases])
    result['tasks'] = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        subset = [case for case in cases if case['task'] == task]
        result['tasks'][task] = {name: metrics(subset, scores, annotations, thresholds, name) for name in names}
    write_json(output / 'results.json', result)
    write_visualization(output, cases, scores, annotations, thresholds, names)
    print(json.dumps(result['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    evaluate(parser.parse_args().output)
