"""Fit on source OOF features, calibrate on dev, freeze cases, then read gold."""
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from experiments.probabilistic_detection.cases import build_case, render_html
from experiments.native_support.ragtruth_benchmark.data import annotations
from .data import inputs, labels, read_json, write_json

VIEWS = ('confidence', 'attention', 'native', 'kernel', 'topology', 'shuffled',
         'probe', 'probe_kernel', 'new_method')


def columns_for(view, names):
    prefixes = ['confidence_']
    if view != 'confidence' and view != 'probe':
        prefixes += ['attention_']
    if view in ('native', 'kernel', 'topology', 'shuffled', 'probe_kernel', 'new_method'):
        prefixes += ['choice_']
    if view in ('kernel', 'topology', 'shuffled', 'probe_kernel', 'new_method'):
        prefixes += ['kernel_']
    if view in ('topology', 'new_method'):
        prefixes += ['graph_']
    if view == 'shuffled':
        prefixes += ['shuffled_']
    if view in ('probe', 'probe_kernel', 'new_method'):
        prefixes += ['probe_']
    return np.array([i for i, name in enumerate(names) if name.startswith(tuple(prefixes))])


def load_features(output, records):
    parts, regions, masks, source_weights, folds = [], {}, [], [], []
    start = 0
    for record in records:
        with np.load(output / 'static' / f"{record['key']}.npz") as data:
            confidence, valid = data['confidence'], data['valid']
        with np.load(output / 'responses' / f"{record['key']}.npz") as data:
            raw, raw_names = data['features'], data['names'].tolist()
        with np.load(output / 'scores' / f"probe_{record['key']}.npz") as data:
            probes = np.stack((data['rank8'], data['mlp9']), -1)
        # Signed log compression prevents one large energy from setting the scale.
        raw = np.sign(raw) * np.log1p(np.abs(raw))
        common = np.concatenate((confidence, raw), -1)
        parts.append(np.concatenate((np.broadcast_to(common, (5, *common.shape)), probes), -1))
        regions[record['key']] = slice(start, start + len(valid))
        masks.append(valid)
        source_weights.append(valid / max(int(valid.sum()), 1))
        folds.extend([record['fold']] * len(valid))
        start += len(valid)
    names = [f'confidence_{i}' for i in range(8)] + raw_names + ['probe_rank8', 'probe_mlp9']
    return np.concatenate(parts, 1), names, regions, np.concatenate(masks), np.concatenate(source_weights), np.array(folds)


def partition_mask(records, regions, role, size):
    mask = np.zeros(size, dtype=bool)
    for record in records:
        if record['role'] == role:
            mask[regions[record['key']]] = True
    return mask


def predict_folds(model, scaler, features, indices):
    return np.mean([model.predict_proba(scaler.transform(fold[:, indices]))[:, 1]
                    for fold in features], axis=0)


def fit_readouts(features, names, target, fit_mask, dev_mask, weights, folds, output):
    fitted, predictions, decisions = {}, {}, {}
    rows = np.flatnonzero(fit_mask)
    oof = features[folds[rows], rows]
    for view in VIEWS:
        indices = columns_for(view, names)
        scaler = StandardScaler().fit(oof[:, indices])
        train = scaler.transform(oof[:, indices])
        best = None
        for regularization in (.01, .1, 1.):
            model = LogisticRegression(C=regularization, max_iter=2000, tol=1e-6)
            model.fit(train, target[fit_mask], sample_weight=weights[fit_mask] / weights[fit_mask].mean())
            score = predict_folds(model, scaler, features[:, dev_mask], indices)
            loss = log_loss(target[dev_mask], score, sample_weight=weights[dev_mask], labels=[0, 1])
            if best is None or loss < best[0]:
                best = (loss, model, regularization)
        loss, model, regularization = best
        predictions[view] = predict_folds(model, scaler, features, indices)
        fitted[view] = dict(model=model, scaler=scaler, columns=indices)
        decisions[view] = dict(C=regularization, dev_bce=loss, features=len(indices))
    joblib.dump(fitted, output / 'models' / 'readouts.joblib')
    write_json(output / 'models' / 'readout_selection.json', decisions)
    return predictions


def metrics(target, scores, threshold):
    normal, error = target == 0, target == 1
    alarm = scores > threshold
    return dict(auroc=float(roc_auc_score(target, scores)) if normal.any() and error.any() else None,
        ap=float(average_precision_score(target, scores)) if error.any() else None,
        false_alarms=int(alarm[normal].sum()), normal_tokens=int(normal.sum()),
        token_fpr=float(alarm[normal].mean()) if normal.any() else None,
        detected_errors=int(alarm[error].sum()), error_tokens=int(error.sum()),
        recall=float(alarm[error].mean()) if error.any() else None, threshold=threshold)


def calibrate(records, regions, target, valid, predictions):
    thresholds, answer_thresholds, counts = {}, {}, {}
    for task in ('QA', 'Summary', 'Data2txt'):
        rows = [r for r in records if r['role'] == 'dev' and r['task'] == task]
        mask = np.zeros(len(valid), dtype=bool)
        normal_answers = []
        for record in rows:
            region = regions[record['key']]
            mask[region] = True
            if not target[region][valid[region]].any():
                normal_answers.append(region)
        normal = mask & valid & (target == 0)
        thresholds[task] = {name: float(np.quantile(score[normal], .95, method='higher'))
                            for name, score in predictions.items()}
        answer_thresholds[task] = {
            name: float(np.quantile([score[region][valid[region]].max() for region in normal_answers],
                                   .95, method='higher')) if normal_answers else None
            for name, score in predictions.items()}
        counts[task] = dict(answers=len(rows), normal_answers=len(normal_answers), normal_tokens=int(normal.sum()))
    return dict(token_5pct=thresholds, answer_5pct=answer_thresholds, calibration_counts=counts,
                caveat='Only 8 development sources/task; tail calibration is exploratory, not a population FPR guarantee.')


def report_cases(output, records, regions, valid, predictions, calibration):
    regression = [r for r in records if r['role'] == 'regression']
    root = Path(regression[0]['root'])
    truth = annotations(root, read_json(root / 'manifest.json'), regression)
    cases, table, token_rows = [], [], []
    shown = ('probe', 'native', 'kernel', 'probe_kernel', 'new_method', 'shuffled')
    for record in regression:
        _, response = inputs(record)
        region = regions[record['key']]
        selected = valid[region]
        targets = np.flatnonzero(selected)
        scores = {name: value[region][selected] for name, value in predictions.items()}
        thresholds = calibration['token_5pct'][record['task']]
        case = build_case(dict(record, partition='test'), response, truth[record['id']], targets,
            np.array(response['answer_ids'])[selected], {name: scores[name] for name in shown}, thresholds, 8)
        case.update(partition='source_excluded_regression', in_sample=False,
            partition_note='本轮训练/开发来源全部排除；历史已暴露，仅作回归测试。',
            historical_partition=record['partition'])
        cases.append(case)
        target = np.array(truth[record['id']]['labels'])[selected]
        for name, score in scores.items():
            entry = dict(id=record['id'], task=record['task'], method=name,
                         **metrics(target, score, thresholds[name]))
            entry['normal_answer_any_alarm'] = bool(np.any(score > thresholds[name])) if not target.any() else None
            answer_threshold = calibration['answer_5pct'][record['task']][name]
            entry['answer_budget_threshold'] = answer_threshold
            entry['any_alarm_at_answer_budget'] = bool(np.any(score > answer_threshold)) if answer_threshold is not None else None
            table.append(entry)
        for i, token in enumerate(targets):
            begin, end = response['offsets'][token]
            token_rows.append(dict(id=record['id'], token=int(token), text=response['text'][begin:end],
                gold=int(target[i]), **{name: float(score[i]) for name, score in scores.items()}))
    case_output = output / 'cases'
    case_output.mkdir(exist_ok=True)
    report = dict(cases=cases, methods=list(predictions), calibration=calibration, exposed_regression=True)
    write_json(case_output / 'report.json', report)
    write_json(case_output / 'metrics.json', table)
    (case_output / 'index.html').write_text(render_html(report).replace('概率检测逐例回归', '原生响应拓扑：逐例实测'))
    with (case_output / 'tokens.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(token_rows[0]))
        writer.writeheader()
        writer.writerows(token_rows)
    return cases, table


def evaluate(args):
    if (args.output / 'evaluation.json').exists():
        raise FileExistsError('Evaluation already completed; preserve the frozen run.')
    records = read_json(args.output / 'manifest.json')['records']
    features, names, regions, valid, weights, folds = load_features(args.output, records)
    truth = labels([r for r in records if r['role'] in ('fit', 'dev')])
    target = np.full(len(valid), -1)
    for record in records:
        if record['key'] in truth:
            target[regions[record['key']]] = truth[record['key']]
    fit_mask = partition_mask(records, regions, 'fit', len(valid)) & valid
    dev_mask = partition_mask(records, regions, 'dev', len(valid)) & valid
    with threadpool_limits(limits=4):
        predictions = fit_readouts(features, names, target, fit_mask, dev_mask, weights, folds, args.output)
    calibration = calibrate(records, regions, target, valid, predictions)
    write_json(args.output / 'calibration.json', calibration)
    for record in records:
        region = regions[record['key']]
        np.savez_compressed(args.output / 'scores' / f"detector_{record['key']}.npz",
                            **{name: score[region] for name, score in predictions.items()})
    write_json(args.output / 'scores_frozen.json', dict(status='all_predictions_frozen',
        methods=list(predictions), records=[r['key'] for r in records],
        regression_labels_used_for_training_or_calibration=False))
    cases, table = report_cases(args.output, records, regions, valid, predictions, calibration)
    dev_results = {}
    for task in ('QA', 'Summary', 'Data2txt'):
        mask = np.zeros(len(valid), dtype=bool)
        for record in records:
            if record['role'] == 'dev' and record['task'] == task:
                mask[regions[record['key']]] = True
        mask &= valid
        dev_results[task] = {name: metrics(target[mask], score[mask], calibration['token_5pct'][task][name])
                             for name, score in predictions.items()}
    write_json(args.output / 'evaluation.json', dict(status='complete', development=dev_results,
        regression=table, complete_cases=len(cases), official_spans=sum(len(c['spans']) for c in cases),
        interpretation='small supervised source-excluded regression; not full test or evidence of generalization'))
    print(args.output / 'cases' / 'index.html', flush=True)
