"""只评价已经冻结的预测；沿用原项目的 RAGTruth 身份、offset 与七种口径。"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..evaluate import Ranking, label_views, scoped_metrics
from ..evaluation_data import EvaluationBinding, read_sources
from .data import write_json


SCORES = ('offline_span', 'singleton', 'history_minus_prompt', 'entropy', 'position')


def match_spans(predicted, gold, threshold=.5):
    """最大一对一匹配，先最大化命中数量，再最大化IoU；一个预测不能命中多个金标。"""
    if not len(predicted) or not len(gold):
        return []
    overlap = np.maximum(0, np.minimum(predicted[:, None, 1], gold[None, :, 1])
                         - np.maximum(predicted[:, None, 0], gold[None, :, 0]))
    union = np.maximum(predicted[:, None, 1], gold[None, :, 1]) - np.minimum(predicted[:, None, 0], gold[None, :, 0])
    iou = overlap / union
    allowed = iou >= threshold
    weight = allowed * (1 + iou / (max(iou.shape) + 1))
    rows, columns = linear_sum_assignment(-weight)
    return [(int(row), int(column)) for row, column in zip(rows, columns) if allowed[row, column]]


def span_metrics(blocks):
    predicted_count = gold_count = exact_count = matched_count = 0
    boundary_errors, false_after, normal_alarms = [], [], []
    covered_after = eligible_after = 0
    for block in blocks:
        predicted, gold = block['predicted_spans'], block['gold_spans']
        predicted_count += len(predicted)
        gold_count += len(gold)
        matched = match_spans(predicted, gold, .5)
        matched_count += len(matched)
        exact_count += len(match_spans(predicted, gold, 1.))
        for p, g in matched:
            boundary_errors.append(abs(predicted[p] - gold[g]).tolist())
        error = block['views']['all_error'][0]
        alarm = np.zeros(len(error), bool)
        for start, end in predicted:
            alarm[start:end] = True
        if not error.any():
            normal_alarms.append(bool(alarm.any()))
        after_mask = np.zeros(len(error), bool)
        for _, end in gold:
            after_mask[end:min(len(error), end + 8)] = True
        after_mask &= ~error
        eligible_after += int(after_mask.sum())
        after_mask &= np.isfinite(block['scores']['offline_span'])
        covered_after += int(after_mask.sum())
        false_after.extend(alarm[after_mask].tolist())

    def measures(hits):
        precision = hits / predicted_count if predicted_count else 0.
        recall = hits / gold_count if gold_count else None
        f1 = 2 * hits / (predicted_count + gold_count) if predicted_count + gold_count else None
        return dict(precision=precision, recall=recall, f1=f1, true_positive=hits)

    return dict(unit='response-token intervals, not exact original character boundaries',
                predicted=predicted_count, gold=gold_count, exact=measures(exact_count), iou_05=measures(matched_count),
                mean_abs_start_end_error=np.mean(boundary_errors, axis=0).tolist() if boundary_errors else None,
                normal_answer_any_alarm=float(np.mean(normal_alarms)) if normal_alarms else None,
                normal_answers=len(normal_alarms),
                post_span_8_normal_false_rate=float(np.mean(false_after)) if false_after else None,
                post_span_8_eligible=eligible_after, post_span_8_covered=covered_after)


def paired_gain(blocks, bootstrap):
    labels, primary, control, sources = [], [], [], []
    for block in blocks:
        first = block['scores']['offline_span']
        second = block['scores']['singleton']
        common = np.isfinite(first) & np.isfinite(second)
        labels.extend(block['views']['all_error'][0][common])
        primary.extend(first[common])
        control.extend(second[common])
        sources.extend([block['record']['source_id']] * int(common.sum()))
    real, single = Ranking(labels, primary), Ranking(labels, control)
    a, b = real.measure(), single.measure()
    delta = {key: a[key] - b[key] if a[key] is not None else None for key in ('auroc', 'ap')}
    unique, inverse = np.unique(sources, return_inverse=True)
    random = np.random.default_rng(17)
    draws = []
    for _ in range(bootstrap if len(unique) > 1 else 0):
        weights = np.bincount(random.integers(len(unique), size=len(unique)), minlength=len(unique))[inverse]
        first, second = real.measure(weights), single.measure(weights)
        if first['auroc'] is not None:
            draws.append([first['auroc'] - second['auroc'], first['ap'] - second['ap']])
    return dict(tokens=len(labels), delta=delta,
                delta_ci95=np.quantile(draws, [.025, .975], axis=0).tolist() if draws else None,
                order=['auroc', 'ap'], note='same-coverage paired source bootstrap; no refitting')


def evaluate_saved_predictions(prediction_dir, annotations, tokenizer=None, source_info=None, bootstrap=200):
    root = Path(prediction_dir)
    freeze = json.loads((root / 'prediction_freeze.json').read_text())
    if not freeze['complete'] or freeze['labels_used']:
        raise ValueError('complete frozen label-free predictions are required')
    records = freeze['records']
    wanted = {row['id'] for row in records}
    gold = {}
    with Path(annotations).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if str(row['id']) in wanted:
                gold[str(row['id'])] = row
    if set(gold) != wanted:
        raise ValueError('some saved response IDs are absent from response.jsonl')
    sources, _ = read_sources(annotations, source_info)
    # Every record stores its exact original input path; resolve_tokenizer can inspect its ancestors.
    binding = EvaluationBinding({'cache': '/'}, tokenizer)
    blocks = []
    for record in records:
        annotation = gold[record['id']]
        with np.load(root / record['file'], allow_pickle=False) as arrays:
            aligned, offsets = binding.bind(record, annotation, arrays, sources)
            views = label_views(offsets, annotation['labels'])
            gold_spans = []
            for span in annotation['labels']:
                hit = np.flatnonzero((offsets[:, 0] < span['end']) & (offsets[:, 1] > span['start'])
                                     & (offsets[:, 1] > offsets[:, 0]))
                gold_spans.append([int(hit[0]), int(hit[-1] + 1)])
            gold_spans = np.asarray(gold_spans, int).reshape(-1, 2)
            scores = {name: arrays[name].copy() for name in SCORES}
            if any(len(values) != len(offsets) for values in scores.values()):
                raise ValueError('saved scores and verified token offsets differ')
            blocks.append(dict(record=aligned, views=views, scores=scores, tokens=len(offsets),
                               predicted_spans=arrays['span_bounds'].copy(), gold_spans=gold_spans))
    groups = {'ALL': blocks}
    for block in blocks:
        name = block['record']['task'] + '|' + block['record']['generator']
        groups.setdefault(name, []).append(block)
    report = dict(mode='offline complete-answer', labels_used_for_training=False, groups={})
    for group, selected in groups.items():
        counts = {}
        for block in selected:
            source = block['record']['source_id']
            counts[source] = counts.get(source, 0) + block['tokens']
        metrics = {}
        for view in selected[0]['views']:
            metrics[view] = {}
            for name in SCORES:
                targets, scores, source_ids, weights = [], [], [], []
                for block in selected:
                    labels, mask = block['views'][view]
                    source = block['record']['source_id']
                    targets.extend(labels[mask])
                    scores.extend(block['scores'][name][mask])
                    source_ids.extend([source] * int(mask.sum()))
                    weights.extend([1 / counts[source]] * int(mask.sum()))
                metrics[view][name] = scoped_metrics(targets, scores, source_ids, weights, bootstrap)
        report['groups'][group] = dict(answers=len(selected), sources=len(counts), views=metrics,
                                       spans=span_metrics(selected), paired_vs_singleton=paired_gain(selected, bootstrap))
        print(json.dumps(dict(group=group, **metrics['all_error']['offline_span']), ensure_ascii=False), flush=True)
    write_json(root / 'evaluation.json', report)
    return report
