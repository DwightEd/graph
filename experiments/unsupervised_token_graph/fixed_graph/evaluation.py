"""预测文件冻结后才用标签；共同覆盖范围上比较图、属性与平滑对照。"""

from collections import defaultdict
import json

import numpy as np

from ..evaluate import Ranking, label_views, scoped_metrics
from ..evaluation_data import EvaluationBinding, read_sources
from ..offline_span.data import write_json
from ..offline_span.evaluation import span_metrics
from ..span_audit.inputs import default_observer_tokenizer
from .pipeline import SCORES, read_json


BASELINES = ('local_mass', 'attention_entropy', 'position')


def read_blocks(args):
    root = args.output / 'predictions'
    freeze = read_json(root / 'freeze.json')
    if not freeze['complete'] or freeze['labels_used']:
        raise ValueError('Freeze complete label-free scores before evaluating')
    annotations = args.dataset / 'response.jsonl'
    with annotations.open(encoding='utf-8') as stream:
        gold = {str(row['id']): row for row in map(json.loads, stream)}
    sources, _ = read_sources(annotations, args.source_info)
    fallback = default_observer_tokenizer(args.test_cache.resolve())
    binder = EvaluationBinding(dict(cache='/', tokenizer_fallback=fallback), args.tokenizer)
    blocks = []
    for row in freeze['records']:
        annotation = gold[row['id']]
        with np.load(root / row['file'], allow_pickle=False) as saved:
            identity, offsets = binder.bind(row, annotation, saved, sources)
            views = label_views(offsets, annotation['labels'])
            values = {name: saved[name].copy() for name in (*SCORES, *BASELINES)}
            spans = {name: saved[name + '_spans'].copy() for name in SCORES}
            blocks.append(dict(record=identity, views=views, scores=values,
                               spans=spans, gold=token_spans(offsets, annotation['labels']),
                               tokens=len(offsets)))
    return blocks


def token_spans(offsets, annotations):
    result = []
    for span in annotations:
        selected = (offsets[:, 0] < span['end']) & (offsets[:, 1] > span['start'])
        selected &= offsets[:, 1] > offsets[:, 0]
        hits = np.flatnonzero(selected)
        result.append((int(hits[0]), int(hits[-1] + 1)))
    return np.asarray(result, int).reshape(-1, 2)


def metric_arrays(blocks, view, method):
    source_sizes = defaultdict(int)
    for block in blocks:
        source_sizes[block['record']['source_id']] += block['tokens']
    labels, scores, sources, weights = [], [], [], []
    for block in blocks:
        target, mask = block['views'][view]
        source = block['record']['source_id']
        labels.extend(target[mask])
        scores.extend(block['scores'][method][mask])
        sources.extend([source] * int(mask.sum()))
        weights.extend([1. / source_sizes[source]] * int(mask.sum()))
    return np.asarray(labels), np.asarray(scores), np.asarray(sources), np.asarray(weights)


def compare_methods(blocks, view, control, draws):
    labels, graph, sources, _ = metric_arrays(blocks, view, 'graph')
    _, other, _, _ = metric_arrays(blocks, view, control)
    common = np.isfinite(graph) & np.isfinite(other)
    left = Ranking(labels[common], graph[common])
    right = Ranking(labels[common], other[common])
    a, b = left.measure(), right.measure()
    delta = {key: a[key] - b[key] if a[key] is not None else None for key in ('auroc', 'ap')}
    unique, inverse = np.unique(sources[common], return_inverse=True)
    random = np.random.default_rng(17)
    differences = []
    for _ in range(draws if len(unique) > 1 else 0):
        weights = np.bincount(random.integers(len(unique), size=len(unique)), minlength=len(unique))[inverse]
        a, b = left.measure(weights), right.measure(weights)
        if a['auroc'] is not None:
            differences.append([a['auroc'] - b['auroc'], a['ap'] - b['ap']])
    return dict(tokens=int(common.sum()), delta=delta,
                order=['auroc', 'ap'], bootstrap_valid=len(differences),
                ci95=np.quantile(differences, [.025, .975], axis=0).tolist() if differences else None)


def interval_report(blocks, name):
    """复用历史纯评价函数；offline_span键仅是旧函数接口，不调用退役模型。"""
    adapted = []
    for block in blocks:
        adapted.append(dict(views=block['views'], gold_spans=block['gold'],
                            predicted_spans=block['spans'][name],
                            scores={'offline_span': block['scores'][name]}))
    return span_metrics(adapted)


def evaluate_group(blocks, draws):
    views = {}
    for view in blocks[0]['views']:
        methods = {}
        for name in (*SCORES, *BASELINES):
            arrays = metric_arrays(blocks, view, name)
            methods[name] = scoped_metrics(*arrays, bootstrap=0)
        views[view] = methods
    paired = {}
    for view in ('all_error', 'first_error_until_first', 'continuation_vs_normal', 'strict_post_first'):
        paired[view] = {name: compare_methods(blocks, view, name, draws)
                        for name in SCORES if name != 'graph'}
    return dict(answers=len(blocks), views=views, graph_minus_control=paired,
                spans={name: interval_report(blocks, name) for name in SCORES})


def evaluate(args):
    blocks = read_blocks(args)
    if not blocks:
        raise ValueError('No saved test predictions')
    groups = defaultdict(list)
    groups['ALL'] = blocks
    for block in blocks:
        row = block['record']
        groups[row['task'] + '|' + row['generator']].append(block)
    report = dict(mode='offline unsupervised fixed graph', primary='graph',
                  labels_used_for_fit=False, score_direction='higher = more unusual', groups={})
    for name, selected in groups.items():
        result = evaluate_group(selected, args.bootstrap)
        report['groups'][name] = result
        for method in SCORES:
            metric = result['views']['all_error'][method]
            print(json.dumps(dict(group=name, method=method, tokens=metric['evaluated_tokens'],
                                  coverage=metric['coverage'], **metric['pooled'])), flush=True)
    write_json(args.output / 'predictions/evaluation.json', report)
    return report
