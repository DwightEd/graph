"""Recompute case metrics directly from official character spans after evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .data import inputs, read_json, write_json


def independent_metrics(target, scores, threshold):
    positive = target == 1
    negative = ~positive
    alarm = scores > threshold
    auroc, average_precision = None, None
    if positive.any():
        order = np.argsort(-scores, kind='stable')
        ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
        true_positive = np.cumsum(positive[order])[ends]
        precision = true_positive / (ends + 1)
        average_precision = float(np.sum(np.diff(np.r_[0, true_positive]) * precision) / positive.sum())
        if negative.any():
            ranks = rankdata(scores)
            auroc = float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2)
                          / (positive.sum() * negative.sum()))
    return dict(auroc=auroc, ap=average_precision, false_alarms=int(alarm[negative].sum()),
        detected_errors=int(alarm[positive].sum()), normal_tokens=int(negative.sum()),
        error_tokens=int(positive.sum()))


def official_targets(record, official):
    _, response = inputs(record)
    row = official[record['id']]
    assert row['response'] == response['text'], record['id']
    offsets = np.asarray(response['offsets'])
    target = np.zeros(len(offsets), dtype=int)
    for span in row['labels']:
        overlap = (offsets[:, 0] < span['end']) & (offsets[:, 1] > span['start'])
        target[overlap] = 1
    valid = (offsets[:, 1] > offsets[:, 0]) & ~np.isin(response['answer_ids'], response['special_ids'])
    return target, valid, response


def verify(output):
    assert read_json(output / 'evaluation.json')['status'] == 'complete'
    assert read_json(output / 'scores_frozen.json')['status'] == 'all_predictions_frozen'
    records = read_json(output / 'manifest.json')['records']
    sources = {role: {r['source_id'] for r in records if r['role'] == role}
               for role in ('fit', 'dev', 'regression')}
    assert not (sources['fit'] & sources['dev'] or sources['fit'] & sources['regression']
                or sources['dev'] & sources['regression'])
    root = Path(records[0]['root'])
    dataset = Path(read_json(root / 'manifest.json')['dataset'])
    official = {}
    wanted = {r['id'] for r in records if r['role'] in ('dev', 'regression')}
    with (dataset / 'response.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if str(row['id']) in wanted:
                official[str(row['id'])] = row
    calibration = read_json(output / 'calibration.json')
    claimed = {(r['id'], r['method']): r for r in read_json(output / 'cases/metrics.json')}
    replay_errors, dev_normal, dev_maxima, checked = [], {}, {}, 0
    valid_tokens = 0
    for record in records:
        with np.load(output / 'responses' / f"{record['key']}.npz") as response:
            _, original = inputs(record)
            assert np.array_equal(response['token_ids'], original['answer_ids'])
            assert np.isfinite(response['features']).all()
            replay_errors.extend(response['replay_errors'])
        if record['role'] not in ('dev', 'regression'):
            continue
        target, valid, _ = official_targets(record, official)
        with np.load(output / 'static' / f"{record['key']}.npz") as cached:
            assert np.array_equal(cached['valid'], valid)
        with np.load(output / 'scores' / f"detector_{record['key']}.npz") as saved:
            for method in saved.files:
                scores = saved[method][valid]
                assert np.isfinite(scores).all()
                if record['role'] == 'dev':
                    key = record['task'], method
                    dev_normal.setdefault(key, []).extend(scores[target[valid] == 0])
                    if not target[valid].any():
                        dev_maxima.setdefault(key, []).append(float(scores.max()))
                else:
                    threshold = calibration['token_5pct'][record['task']][method]
                    values = independent_metrics(target[valid], scores, threshold)
                    for key, value in values.items():
                        expected = claimed[record['id'], method][key]
                        assert (value is None and expected is None) or np.isclose(value, expected, atol=1e-12, rtol=0), (record['id'], method, key)
                    checked += 1
        if record['role'] == 'regression':
            valid_tokens += int(valid.sum())
    for (task, method), values in dev_normal.items():
        assert calibration['token_5pct'][task][method] == np.quantile(values, .95, method='higher')
        maxima = dev_maxima.get((task, method), [])
        expected = float(np.quantile(maxima, .95, method='higher')) if maxima else None
        assert calibration['answer_5pct'][task][method] == expected
    maxima = np.max(replay_errors, axis=0)
    assert maxima[0] <= .005 and maxima[1] <= .0005
    result = dict(status='passed', audit_scope='same-agent independent numeric recomputation; not fresh review',
        checked_case_method_pairs=checked, regression_valid_tokens=valid_tokens,
        source_counts={key: len(value) for key, value in sources.items()},
        source_intersections_empty=True, official_labels_recomputed=True,
        all_answer_token_ids_and_features_checked=len(records),
        dev_thresholds_recomputed=True, max_logit_error=float(maxima[0]), max_state_relative_error=float(maxima[1]))
    write_json(output / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    verify(parser.parse_args().output)
