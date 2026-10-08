"""Freeze entropy/continuity ablations on all QA train answers; evaluate dev only.

The entropy CDF and alarm thresholds use fit sources, equally weighted. No
natural labels enter fitting, score construction, method choice or thresholds.
This tests a continuity assumption, not a causal propagation mechanism.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.native_support.evaluate import ranking
from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import source_bootstrap
from .data import PACKS, load_partition
from .renewal import GATE_SCALE, entropy_events, renewal_field
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .run_case_analysis import summarize_methods
from .run_unlabeled import answer_inputs, verify_freeze, weighted_reference_threshold
from .unlabeled import (equal_source_weights, fit_weighted_cdf, graph_fields,
                        local_weights, reference_rank)


DEFAULT_OUTPUT = Path('outputs/first_penalty_validation_20261008/renewal')
PRIMARY = 'entropy_unary_gated_huber'
BASELINE = 'source_route_native_huber'
METHODS = ('entropy_rank', 'source_unary', 'source_native_huber',
           'source_route_unary', BASELINE, 'source_route_gated_huber',
           'entropy_unary', 'entropy_unary_native_huber', PRIMARY)
SHARED_REPORT = Path('/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/first_penalty_validation_20261008/RENEWAL_VALIDATION.md')


def fit_entropy_reference(train):
    fit = ~train['development']
    values, cumulative = fit_weighted_cdf(train['observations'][fit, 6],
                                          equal_source_weights(train['source_index'][fit]))
    return dict(entropy_values=values, entropy_cumulative=cumulative)


def score_answer(reference, values, attention):
    source = .5 * sum(reference_rank(reference, name, values[name])
                       for name in ('source_local', 'source_full'))
    route = reference_rank(reference, 'raw_route', values['raw_route'])
    entropy = reference_rank(reference, 'entropy', values['entropy'])
    old_unary = .75 * source + .25 * route
    entropy_unary = .5 * source + .25 * route + .25 * entropy
    native = local_weights(attention)
    events = entropy_events(entropy)

    old_gated, old_diagnostics = renewal_field(old_unary, native, events)
    entropy_native, native_diagnostics = graph_fields(entropy_unary, dict(native=native))
    entropy_gated, gated_diagnostics = renewal_field(entropy_unary, native, events)
    scores = dict(entropy_rank=entropy, entropy_unary=entropy_unary,
        source_route_gated_huber=old_gated,
        entropy_unary_native_huber=entropy_native['native_huber'],
        entropy_unary_gated_huber=entropy_gated)
    diagnostics = dict(old_gate=old_diagnostics, entropy_native=native_diagnostics,
                       entropy_gate=gated_diagnostics)
    for name in ('old_gate', 'entropy_gate'):
        diagnostics[name].pop('solution')
    return scores, events, diagnostics


def score(args):
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / 'code_snapshot'
    snapshot.mkdir()
    for name in ('renewal.py', 'run_renewal_validation.py', 'smooth.py', 'unlabeled.py'):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    train, metadata = load_partition('train')
    old_directory = args.parent / 'unlabeled'
    old_freeze = verify_freeze(old_directory)
    reference = dict(np.load(old_directory / 'reference.npz'))
    reference.update(fit_entropy_reference(train))
    np.savez(args.output / 'reference.npz', **reference)
    previous = dict(np.load(old_directory / 'train_scores.npz'))
    scores = {name: (previous[name].copy() if name in previous else
                    np.full(len(train['token_id']), np.nan)) for name in METHODS}
    events = np.full(len(train['token_id']), np.nan)
    diagnostics = []
    for index, record in enumerate(metadata['records']):
        values, attention, targets, region = answer_inputs(args.cache, args.parent / 'capture', record, train)
        with np.load(args.cache / record['directory'] / 'observations.npz') as raw:
            values['entropy'] = raw['entropy'].copy()
        if not np.array_equal(values['entropy'][targets], train['observations'][region, 6]):
            raise ValueError(f"{record['id']}: original and fit-pack entropy differ")
        answer_scores, event_values, solver = score_answer(reference, values, attention)
        for name, value in answer_scores.items():
            scores[name][region] = value[targets]
        events[region] = event_values[targets]
        diagnostics.append(dict(id=record['id'], **solver))
        if (index + 1) % 300 == 0:
            print(f'RENEWAL train {index+1}/{len(metadata["records"])}', flush=True)
    if not all(np.isfinite(values).all() for values in (*scores.values(), events)):
        raise ValueError('Incomplete renewal scores')
    np.savez(args.output / 'train_scores.npz', **scores)
    np.savez(args.output / 'train_events.npz', entropy_event=events)
    write_json(args.output / 'SOLVER_DIAGNOSTICS.json', diagnostics)
    fit = ~train['development']
    weights = equal_source_weights(train['source_index'][fit])
    thresholds = {name: weighted_reference_threshold(values[fit], weights)
                  for name, values in scores.items()}
    write_json(args.output / 'THRESHOLDS.json', thresholds)
    write_json(args.output / 'FREEZE.json', dict(status='all_train_scores_frozen_before_dev_evaluation',
        primary=PRIMARY, baseline=BASELINE, entropy_unary_weights=[.5, .25, .25],
        event='max(0,2*fit_equal_source_mid_CDF_entropy-1)',
        edge_gate='exp(-4*max(E[j+1:t+1]))', gate_scale=GATE_SCALE,
        edge_budget='normalize native incident degree first; gate; no upward renormalization',
        penalty=.5, huber_delta=1., fit_sources=len(np.unique(train['source_index'][fit])),
        dev_sources=len(np.unique(train['source_index'][~fit])), train_answers=len(metadata['records']),
        fit_tokens=int(fit.sum()), dev_tokens=int((~fit).sum()),
        natural_labels_loaded_in_existing_train_pack=True, natural_labels_used_for_fit=False,
        natural_labels_used_for_selection=False, test_scored=False, new_llm_forwards=0,
        threshold_rule='strict_gt_equal_source_fit_mixture95; not guaranteed normal FPR',
        timing='offline; entropy prechoice; native graph posttoken',
        old_freeze_sha256=file_hash(old_directory / 'FREEZE.json'),
        old_primary=old_freeze['primary'],
        hashes={name: file_hash(args.output / name) for name in
                ('reference.npz', 'train_scores.npz', 'train_events.npz', 'THRESHOLDS.json')},
        code_hashes={path.name: file_hash(path) for path in snapshot.iterdir()},
        pack_hashes={name: file_hash(PACKS / name) for name in ('QA_train.npz', 'QA_train.json')},
        wall_seconds=time.time()-started))
    print(f'RENEWAL frozen {len(train["token_id"])} train tokens', flush=True)


def dev_partition(train, metadata):
    selected = train['development']
    dev = {name: values[selected] for name, values in train.items()}
    records, cursor = [], 0
    for record in metadata['records']:
        if record['partition'] != 'dev':
            continue
        count = record['packed_stop'] - record['packed_start']
        records.append(dict(record, packed_start=cursor, packed_stop=cursor+count))
        cursor += count
    return dev, dict(metadata, records=records), selected


def subgroup_metrics(dev, scores, thresholds, metadata, cache):
    first = dev['firsts']
    continuation = (dev['labels'] == 1) & ~dev['onsets']
    normal = dev['labels'] == 0
    opening = dev['target'] < 16
    punctuation = np.zeros(len(normal), bool)
    number = np.zeros(len(normal), bool)
    capitalized = np.zeros(len(normal), bool)
    for record in metadata['records']:
        region = slice(record['packed_start'], record['packed_stop'])
        response = json.loads((cache / record['directory'] / 'response.json').read_text())
        text = [response['token_text'][position].strip() for position in dev['target'][region]]
        punctuation[region] = [not any(character.isalnum() for character in token) for token in text]
        number[region] = [any(character.isdigit() for character in token) and
                          not any(character.isalpha() for character in token) for token in text]
        capitalized[region] = [next((character.isupper() for character in token
                                     if character.isalpha()), False) for token in text]
    summaries = {}
    for name, values in scores.items():
        alarm = values > thresholds[name]
        continuation_selected = continuation | normal
        summaries[name] = dict(continuation_vs_normal=ranking(
            continuation[continuation_selected].astype(int), values[continuation_selected]),
            continuation_tokens=int(continuation.sum()), continuation_detected=int(alarm[continuation].sum()),
            first_score_mean=float(values[first].mean()),
            normal_opening_tokens=int((normal & opening).sum()),
            normal_opening_false_alarms=int(alarm[normal & opening].sum()),
            normal_number_tokens=int((normal & number).sum()),
            normal_number_false_alarms=int(alarm[normal & number].sum()),
            normal_capitalized_tokens=int((normal & capitalized).sum()),
            normal_capitalized_false_alarms=int(alarm[normal & capitalized].sum()),
            normal_punctuation_tokens=int((normal & punctuation).sum()),
            normal_punctuation_false_alarms=int(alarm[normal & punctuation].sum()))
    return summaries


def report(results):
    lines = ['# 熵入口与连续性惩罚：固定开发集消融', '',
        '主方法预设为 entropy_unary_gated_huber。仅 fit 来源拟合等权熵 CDF 和混合总体95分位阈值，'
        '全 train 回答分数冻结后才评价 dev；没有 test 评分、自然标签拟合或阈值选型。'
        '熵门只是连续性先验，不是原生因果消息算子，也不是顶会方案有效性结论。', '',
        '| 方法 | 全词 AUROC | AP | 严格首错 AUROC | 首错检出 | 首错检出且此前无误报 | 正常回答报警 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, values in results['methods'].items():
        alarm = values['policies']['reference95']
        lines.append(f'| {name} | {values["all_tokens"]["auroc"]:.6f} | {values["all_tokens"]["ap"]:.6f} | '
            f'{values["strict_clean_prefix_first_error"]["auroc"]:.6f} | '
            f'{alarm["first_errors_detected"]}/{alarm["error_answers"]} | '
            f'{alarm["first_errors_detected_without_prior_false_alarm"]}/{alarm["error_answers"]} | '
            f'{alarm["normal_answers_with_alarm"]}/{alarm["normal_answers"]} |')
    lines += ['', '门值 g(j,t)=exp(-4 max(E[j+1:t+1]))；E=max(0,2CDF(H)-1)。'
        '跨过高熵入口的长边也被弱化，入口后的边不因该入口永久清零；每词保留自身 unary。'
        'native 边先做既有 incident-degree≤1 预算，再乘门，弱边不会被重新放大。', '',
        '固定加熵 unary=.5source+.25route+.25entropy；旧 unary=.75source+.25route。'
        '所有 Huber 仍 λ=.5、δ=1、同一优化器，未依标点或金标片段重置。'
        '完全回答优化为离线评分，不能把门只使用接收词及之前事件等同于整体在线检测。', '',
        f'来源成组 bootstrap：`{json.dumps(results["bootstrap"], ensure_ascii=False)}`。', '',
        '逐项误报、持续错误定位见 DEV_RESULTS.json/subgroups。原始分数、熵事件和冻结记录均保存。']
    return '\n'.join(lines) + '\n'


def evaluate(args):
    frozen = json.loads((args.output / 'FREEZE.json').read_text())
    for name, expected in frozen['hashes'].items():
        if file_hash(args.output / name) != expected:
            raise ValueError(f'Frozen artifact changed: {name}')
    for name, expected in frozen['code_hashes'].items():
        if file_hash(args.output / 'code_snapshot' / name) != expected:
            raise ValueError(f'Frozen execution snapshot changed: {name}')
        if file_hash(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Current code differs from frozen execution: {name}')
    for name, expected in frozen['pack_hashes'].items():
        if file_hash(PACKS / name) != expected:
            raise ValueError(f'Frozen pack changed: {name}')
    train, metadata = load_partition('train')
    dev, dev_metadata, selected = dev_partition(train, metadata)
    dev.update(evaluation_labels(args.cache, dev, dev_metadata))
    scores = {name: values[selected] for name, values in dict(np.load(args.output / 'train_scores.npz')).items()}
    thresholds = json.loads((args.output / 'THRESHOLDS.json').read_text())
    methods = {name: dict(scores=values, natural_label_fit=False,
        thresholds=dict(reference95=thresholds[name]),
        threshold_rules=dict(reference95='equal_source_fit_mixture95_strict_gt'))
        for name, values in scores.items()}
    summaries, counts = summarize_methods(dev, dev_metadata['records'], methods)
    pairs = {name + '_minus_' + control: source_bootstrap(dev, scores[name], scores[control], args.bootstrap)
        for name, control in ((PRIMARY, BASELINE),
                             ('source_route_gated_huber', BASELINE),
                             ('entropy_unary_native_huber', 'entropy_unary'),
                             (PRIMARY, 'entropy_unary'),
                             ('entropy_unary_gated_huber', 'entropy_unary_native_huber'))}
    results = dict(counts=counts, methods=summaries, primary=PRIMARY, bootstrap=pairs,
        subgroups=subgroup_metrics(dev, scores, thresholds, dev_metadata, args.cache),
        code_sha256=file_hash(Path(__file__)), no_method_selection=True)
    write_json(args.output / 'DEV_RESULTS.json', results)
    text = report(results)
    (args.output / 'DEV_RESULTS.md').write_text(text)
    args.shared_report.parent.mkdir(parents=True, exist_ok=True)
    args.shared_report.write_text(text)
    print(json.dumps({name: result['all_tokens'] for name, result in summaries.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('score', 'evaluate', 'all'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--parent', type=Path, default=OUTPUT)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--shared-report', type=Path, default=SHARED_REPORT)
    parser.add_argument('--bootstrap', type=int, default=300)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase in ('score', 'all'):
        score(args)
    if args.phase in ('evaluate', 'all'):
        evaluate(args)


if __name__ == '__main__':
    main()
