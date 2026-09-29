"""Freeze unlabeled thresholds, then report complete detection and localization."""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.context_response.restore import cached_baselines
from experiments.constraint_uptake.evaluate import label_tables, rag_units, step_units
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.probabilistic_detection.evaluation import source_bootstrap
from .capture import METHODS

SCORE_METHODS = ('legacy_token_source', *METHODS)


def legacy_token_source(row):
    if row['dataset'] == 'ragtruth' and row['original']['kind'] == 'observer':
        directory = Path(row['original']['root']) / row['original']['directory']
    else:
        directory = Path('outputs/anchored_flow_20260929_v1') / row['key']
    with np.load(directory / 'observations.npz') as saved:
        np.testing.assert_array_equal(saved['token_id'], row['response']['answer_ids'])
        return .5 * (saved['source_local'].astype(float) + saved['source_full'].astype(float))

def weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind='stable')
    total = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(total, quantile * total[-1])]])


def calibration_values(row, values):
    if row['dataset'] == 'gsm8k':
        return np.asarray([values[start:stop].mean() for start, stop in row['step_ranges']])
    offsets = np.asarray(row['response']['offsets'])
    valid = (offsets[:, 1] > offsets[:, 0]) & ~np.isin(
        row['response']['answer_ids'], row['response']['special_ids'])
    selected = values[valid]
    assert np.isfinite(selected).all(), ('invalid calibration score', row['key'])
    return selected


def freeze(output, records):
    read_json(output / 'capture_complete.json')
    original = cached_baselines({row['key'] for row in records
                                 if row['dataset'] == 'ragtruth' and row['original']['kind'] == 'observer'})
    arrays = {}
    for row in records:
        with np.load(output / row['key'] / 'scores.npz') as saved:
            scores = {name: saved[name] for name in METHODS}
        scores['legacy_token_source'] = legacy_token_source(row)
        if row['key'] in original:
            scores['base'] = original[row['key']]
        else:
            with np.load(Path('outputs/anchored_flow_20260929_v1') / row['key'] / 'scores.npz') as saved:
                scores['base'] = saved['base']
        scores['base_matched'] = scores['base']
        arrays[row['key']] = scores
    thresholds = {}
    old = read_json('outputs/anchored_flow_20260929_v1/thresholds.json')
    for task in dict.fromkeys(row['task'] for row in records):
        development = [row for row in records if row['role'] == 'dev' and row['task'] == task]
        thresholds[task] = dict(base=old[task])
        for name in (*SCORE_METHODS, 'base_matched'):
            groups = [calibration_values(row, arrays[row['key']][name]) for row in development]
            weights = np.concatenate([np.full(len(values), 1 / len(values)) for values in groups])
            thresholds[task][name] = weighted_quantile(np.concatenate(groups), weights, .95)
    write_json(output / 'thresholds.json', thresholds)
    write_json(output / 'scores_frozen.json', dict(methods=['base', 'base_matched', *SCORE_METHODS],
        primary='reset_cad_tail', natural_labels_used=False,
        threshold='task/source-equal unlabeled dev mixture95, except historical base'))
    return arrays, thresholds


def labelled_rows(records, arrays):
    selected = [row for row in records if row['role'] in ('case', 'heldout')]
    truth, local, gsm = label_tables(selected)
    result = []
    methods = ('base', 'base_matched', *SCORE_METHODS)
    for record in selected:
        count = len(record['response']['answer_ids'])
        row = dict(record, answer=record['response']['answer_ids'], positions=list(range(count)),
                   text=record['response']['token_text'])
        if record['dataset'] == 'ragtruth':
            units = rag_units(row, arrays[row['key']], methods, truth, local)
            membership = np.full(count, -1, dtype=int)
            for index, span in enumerate(record['response']['units']):
                membership[span['start']:span['stop']] = index
            assert (membership >= 0).all()
            for unit in units:
                unit['unit_id'] = int(membership[unit['position']])
        else:
            units = step_units(row, arrays[row['key']], methods, gsm)
            for unit in units:
                unit['unit_id'] = unit['position']
        for unit in units:
            unit.update(task=record['task'], partition=record['role'])
        result.extend(units)
    return result


def summarize(rows, method, thresholds):
    known = [row for row in rows if row['gold'] >= 0]
    labels = np.array([row['gold'] for row in known])
    scores = np.array([row[method] for row in known])
    alarm = np.array([row[method] > thresholds[row['task']][method] for row in known])
    both = len(set(labels)) == 2
    auc = float(roc_auc_score(labels, scores)) if both else None
    if both:
        positive, negative = int(labels.sum()), int((labels == 0).sum())
        independent = (rankdata(scores)[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative)
        assert abs(auc - independent) < 1e-12
    result = dict(known=len(known), positives=int(labels.sum()), auroc=auc,
        ap=float(average_precision_score(labels, scores)) if labels.sum() else None,
        tp=int((alarm & (labels == 1)).sum()), fp=int((alarm & (labels == 0)).sum()),
        fn=int((~alarm & (labels == 1)).sum()))
    return result


def localization(rows, method, thresholds):
    normal, firsts, onset_hits = [], [], []
    within_numerator, within_denominator = 0., 0
    answers = []
    for key in dict.fromkeys(row['key'] for row in rows):
        selected = sorted((row for row in rows if row['key'] == key), key=lambda row: row['position'])
        known = [row for row in selected if row['gold'] >= 0]
        y = np.array([row['gold'] for row in known])
        values = np.array([row[method] for row in known])
        alarms = [row['position'] for row in known if row[method] > thresholds[row['task']][method]]
        all_alarms = [row['position'] for row in selected if row[method] > thresholds[row['task']][method]]
        errors = [row['position'] for row in known if row['gold'] == 1]
        if len(set(y)) == 2:
            pairs = int(y.sum()) * int((y == 0).sum())
            within_numerator += pairs * roc_auc_score(y, values)
            within_denominator += pairs
        if errors:
            firsts.append(errors[0] in alarms)
            truth = {row['position']: row['gold'] for row in selected}
            starts = [position for position in errors if truth.get(position - 1, 0) != 1]
            onset_hits.extend(position in alarms for position in starts)
        elif len(known) == len(selected):
            normal.append(bool(alarms))
        answers.append(dict(key=key, gold_first=errors[0] if errors else -1,
            predicted_first=min(alarms) if alarms else -1,
            predicted_first_all_positions=min(all_alarms) if all_alarms else -1,
            unknown=sum(row['gold'] < 0 for row in selected)))
    return dict(within_answer_auroc=within_numerator / within_denominator if within_denominator else None,
        first_error_hits=int(sum(firsts)), error_answers=len(firsts),
        first_alarm_exact_hits=sum(row['gold_first'] >= 0 and row['gold_first'] == row['predicted_first'] for row in answers),
        onset_hits=int(sum(onset_hits)), onsets=len(onset_hits),
        normal_answers=len(normal), normal_answer_false_alarms=int(sum(normal)), answers=answers)


def within_unit(rows, method):
    numerator, denominator = 0., 0
    for key, unit in {(row['key'], row['unit_id']) for row in rows}:
        selected = [row for row in rows if row['key'] == key and row['unit_id'] == unit and row['gold'] >= 0]
        labels = np.array([row['gold'] for row in selected])
        if len(set(labels)) == 2:
            pairs = int(labels.sum()) * int((labels == 0).sum())
            numerator += pairs * roc_auc_score(labels, [row[method] for row in selected])
            denominator += pairs
    return numerator / denominator if denominator else None


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(output):
    records = read_json(output / 'manifest.json')['records']
    arrays, thresholds = freeze(output, records)
    rows = labelled_rows(records, arrays)
    methods = ('base', 'base_matched', *SCORE_METHODS)
    groups = {cohort: [row for row in rows if row['partition'] == 'case' and row['cohort'] == cohort]
              for cohort in ('rag_official', 'rag_local', 'gsm_step')}
    groups.update({f'heldout_{task}': [row for row in rows if row['partition'] == 'heldout' and row['task'] == task]
                   for task in ('QA', 'Summary', 'Data2txt')})
    summary = {group: {method: dict(summarize(selected, method, thresholds),
        within_unit_auroc=within_unit(selected, method),
        **localization(selected, method, thresholds)) for method in methods} for group, selected in groups.items()}
    intervals = {}
    for group, selected in groups.items():
        if not group.startswith('heldout_'):
            continue
        known = [row for row in selected if row['gold'] >= 0]
        source_ids = sorted({str(row['pair']) for row in known})
        pack = dict(labels=np.array([row['gold'] for row in known]),
                    source_index=np.array([source_ids.index(str(row['pair'])) for row in known]))
        if len(set(pack['labels'])) == 2:
            intervals[group] = source_bootstrap(pack,
                np.array([row['reset_cad_tail'] for row in known]), np.array([row['base'] for row in known]))
    write_json(output / 'summary.json', summary)
    write_json(output / 'bootstrap.json', intervals)
    write_csv(output / 'units.csv', rows)
    errors = []
    for row in rows:
        if row['gold'] < 0:
            continue
        for method in ('base', 'base_matched', 'reset_cad_tail'):
            alarm = row[method] > thresholds[row['task']][method]
            if alarm != bool(row['gold']):
                errors.append(dict(key=row['key'], partition=row['partition'], position=row['position'],
                    text=row['text'], method=method, gold=row['gold'], score=row[method],
                    threshold=thresholds[row['task']][method], status='FN' if row['gold'] else 'FP'))
    write_csv(output / 'errors.csv', errors)
    write_json(output / 'evaluation_complete.json', dict(status='complete', primary='reset_cad_tail',
        methods=methods, rows=len(rows), labels_after_freeze=True, active_jobs=False,
        scope='historical cases plus source-disjoint subset of previously exposed official test'))
    print({group: {name: (values[name]['auroc'], values[name]['tp'], values[name]['fp'])
                  for name in ('base', 'base_matched', 'reset_cad_tail', 'full_cad_tail')}
           for group, values in summary.items()}, flush=True)
