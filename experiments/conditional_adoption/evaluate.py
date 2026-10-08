"""Evaluate already frozen scores; official labels never enter the readout.

The normal-token quantile below is an explicitly oracle diagnostic operating
point, not an independently calibrated deployment threshold. Historical paired
cases are exploratory; they cannot establish a generalization claim.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.native_support.evaluate import ranking
from experiments.native_support.ragtruth_benchmark.data import annotations
from .readout import file_hash


ORACLE_NORMAL_QUANTILE = .95
LEGACY_FIXED = Path('outputs/supervised_local_transport_20261008/unlabeled')
LEGACY_PACKS = Path('outputs/probabilistic_detection_20260928_full/packs')


def legacy_fixed_rows(selected):
    """Read a frozen historical control; do not refit or alter its scores."""
    result = {}
    for split in sorted({record['split'] for record in selected}):
        metadata = json.loads((LEGACY_PACKS / f'QA_{split}.json').read_text())
        lookup = {record['id']: record for record in metadata['records']}
        with np.load(LEGACY_PACKS / f'QA_{split}.npz') as packed, \
                np.load(LEGACY_FIXED / f'{split}_scores.npz') as saved:
            for record in selected:
                if record['split'] != split:
                    continue
                original = lookup[record['id']]
                region = slice(original['packed_start'], original['packed_stop'])
                result[record['id']] = dict(token_id=packed['token_id'][region],
                    target=packed['target'][region],
                    score=saved['source_route_native_huber'][region])
    return result


def spans(labels):
    labels = np.asarray(labels, dtype=bool)
    starts = np.flatnonzero(labels & ~np.r_[False, labels[:-1]])
    stops = np.flatnonzero(labels & ~np.r_[labels[1:], False]) + 1
    return list(zip(starts.tolist(), stops.tolist()))


def diagnostic_operating_point(rows, method):
    normal = np.concatenate([row['scores'][method][row['labels'] == 0] for row in rows])
    cutoff = float(np.quantile(normal, ORACLE_NORMAL_QUANTILE))
    counts = dict(tp=0, fp=0, fn=0, tn=0, normal_answers=0, normal_answer_alarms=0,
                  error_spans=0, spans_hit=0, spans_fully_detected=0,
                  error_answers=0, first_error_detected=0, first_error_detected_without_prior_fp=0)
    per_answer = []
    for row in rows:
        label, score = row['labels'], row['scores'][method]
        alarm = score > cutoff
        positive = label == 1
        tp, fp, fn, tn = (int((alarm & positive).sum()), int((alarm & ~positive).sum()),
                         int((~alarm & positive).sum()), int((~alarm & ~positive).sum()))
        for key, value in zip(('tp', 'fp', 'fn', 'tn'), (tp, fp, fn, tn)):
            counts[key] += value
        intervals = spans(label)
        counts['error_spans'] += len(intervals)
        counts['spans_hit'] += sum(bool(alarm[start:stop].any()) for start, stop in intervals)
        counts['spans_fully_detected'] += sum(bool(alarm[start:stop].all()) for start, stop in intervals)
        if positive.any():
            first = int(np.flatnonzero(positive)[0])
            counts['error_answers'] += 1
            counts['first_error_detected'] += int(alarm[first])
            counts['first_error_detected_without_prior_fp'] += int(alarm[first] and not alarm[:first].any())
        else:
            counts['normal_answers'] += 1
            counts['normal_answer_alarms'] += int(alarm.any())
        per_answer.append(dict(id=row['id'], tp=tp, fp=fp, fn=fn, tn=tn,
                               missed_targets=row['targets'][positive & ~alarm].tolist(),
                               false_positive_targets=row['targets'][~positive & alarm].tolist()))
    return dict(oracle_diagnostic=True, calibrated=False, cutoff=cutoff,
                rule='score > label-derived 95th percentile of evaluated normal tokens',
                counts=counts, per_answer=per_answer)


def load_rows(score_root, source_root):
    frozen_path = score_root / 'frozen_scores.json'
    frozen = json.loads(frozen_path.read_text())
    if file_hash(score_root / 'scores.npz') != frozen['score_sha256']:
        raise ValueError('scores changed after freezing')
    source_manifest = json.loads((source_root / 'manifest.json').read_text())
    lookup = {record['id']: record for record in source_manifest['records']}
    selected = [lookup[record['id']] for record in frozen['records']]
    truth = annotations(source_root, source_manifest, selected)
    fixed = legacy_fixed_rows(selected)
    rows, token_table = [], []
    with np.load(score_root / 'scores.npz', allow_pickle=False) as saved:
        for record in frozen['records']:
            identity = record['id']
            annotation = truth[identity]
            token_ids = saved[f'{identity}/actual_id']
            if not np.array_equal(token_ids, annotation['token_ids']):
                raise ValueError(f'{identity}: official tokens differ from scored tokens')
            if record['source_id'] != lookup[identity]['source_id']:
                raise ValueError(f'{identity}: source mismatch')
            labels = np.asarray(annotation['labels'], dtype=np.int64)
            valid = np.asarray(annotation['valid_tokens'], dtype=bool)
            if labels.shape != token_ids.shape or valid.shape != labels.shape:
                raise ValueError(f'{identity}: missing token-level coverage')
            scores = {name: saved[f'{identity}/{name}'][valid] for name in record['score_names']}
            directory = source_root / lookup[identity]['directory']
            response = json.loads((directory / 'response.json').read_text())
            with np.load(directory / 'observations.npz') as baseline:
                if not np.array_equal(baseline['token_id'], token_ids):
                    raise ValueError(f'{identity}: source baseline token mismatch')
                scores['old_source_local_gap'] = baseline['source_local'][valid]
                scores['old_source_full_gap'] = baseline['source_full'][valid]
            with np.load(directory / 'selected_scores.npz') as baseline:
                if not np.array_equal(baseline['token_id'], token_ids):
                    raise ValueError(f'{identity}: selected baseline token mismatch')
                scores['old_source_unit_detector'] = baseline['selected_detector'][valid]
            targets = np.flatnonzero(valid)
            old = fixed[identity]
            if not np.array_equal(old['target'], targets) or not np.array_equal(old['token_id'], token_ids[valid]):
                raise ValueError(f'{identity}: legacy fixed control token mismatch')
            scores['old_source_route_native_fixed'] = old['score']
            row = dict(id=identity, source_id=record['source_id'], labels=labels[valid], scores=scores, targets=targets)
            rows.append(row)
            for index, target in enumerate(targets):
                token_table.append(dict(id=identity, source_id=record['source_id'], target=int(target),
                    token_id=int(token_ids[target]), text=response['token_text'][target], label=int(labels[target]),
                    scores={name: float(values[index]) for name, values in scores.items()}))
    return frozen, rows, token_table


def run(score_root, source_root, report_name='evaluation.json', token_report_name='tokens.json'):
    if (score_root / report_name).exists() or (score_root / token_report_name).exists():
        raise ValueError('evaluation artifacts exist; use new report names to preserve prior results')
    frozen, rows, tokens = load_rows(score_root, source_root)
    labels = np.concatenate([row['labels'] for row in rows])
    names = list(rows[0]['scores'])
    if any(list(row['scores']) != names for row in rows):
        raise ValueError('readout columns differ across answers')
    methods = {}
    native_winner = np.concatenate([row['scores']['native_actual_rejection'] <= 0 for row in rows])
    for name in names:
        scores = np.concatenate([row['scores'][name] for row in rows])
        per_answer = {row['id']: ranking(row['labels'], row['scores'][name]) for row in rows}
        strict_labels, strict_scores = [], []
        for row in rows:
            stop = int(np.flatnonzero(row['labels'])[0]) + 1 if row['labels'].any() else len(row['labels'])
            strict_labels.append(row['labels'][:stop])
            strict_scores.append(row['scores'][name][:stop])
        methods[name] = dict(all_token=ranking(labels, scores), per_answer=per_answer,
            actual_is_native_winner=ranking(labels[native_winner], scores[native_winner]),
            actual_is_not_native_winner=ranking(labels[~native_winner], scores[~native_winner]),
            strict_first=ranking(np.concatenate(strict_labels), np.concatenate(strict_scores)),
            operating_point=diagnostic_operating_point(rows, name))
    result = dict(status='DONE', primary=frozen['primary'], official_token_labels=True,
        new_detector_uses_natural_label_fit=False, independently_calibrated_threshold=False,
        selection='eight historical QA answers, four sources; exploratory mechanism cohort',
        valid_tokens=len(labels), positive_tokens=int(labels.sum()), answers=len(rows),
        positive_answers=sum(bool(row['labels'].any()) for row in rows),
        source_count=len({row['source_id'] for row in rows}), methods=methods,
        frozen_score_sha256=frozen['score_sha256'], evaluator_sha256=file_hash(__file__))
    for method in result['methods'].values():
        within = [answer['auroc'] for answer in method['per_answer'].values() if answer['auroc'] is not None]
        method['macro_within_answer_auroc'] = float(np.mean(within)) if within else None
    for name, method in result['methods'].items():
        method['provenance'] = dict(natural_label_parameter_fit=False,
            natural_label_method_selection=name == 'old_source_unit_detector',
            post_pilot_design=bool(frozen.get('exploratory_post_pilot')) and 'elasticity' in name,
            note='legacy detector selected using natural development labels' if name == 'old_source_unit_detector' else
                 'legacy unlabeled reference includes source 15521 (two current answers)' if name == 'old_source_route_native_fixed' else
                 'no natural label fitting; historical cohort and exploratory design remain exposed')
    result['provenance_scope'] = 'new detector only; legacy baseline method selection is reported separately'
    (score_root / report_name).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (score_root / token_report_name).write_text(json.dumps(tokens, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--report-name', default='evaluation.json')
    parser.add_argument('--token-report-name', default='tokens.json')
    args = parser.parse_args()
    result = run(args.scores, args.source_root, args.report_name, args.token_report_name)
    print(json.dumps({name: method['all_token'] for name, method in result['methods'].items()}), flush=True)


if __name__ == '__main__':
    main()
