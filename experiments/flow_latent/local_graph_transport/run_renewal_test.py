"""Confirm all nine frozen entropy/penalty ablations on complete QA test.

No readout, entropy CDF, weight, threshold or primary method changes after dev.
All 900 answers are scored and frozen before official test annotations join.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import source_bootstrap
from .data import PACKS, load_partition
from .run_capture import CACHE, OUTPUT, file_hash, write_json
from .run_case_analysis import CASE_IDS, summarize_methods
from .run_renewal_validation import (BASELINE, METHODS, PRIMARY, score_answer,
                                     subgroup_metrics)
from .run_unlabeled import answer_inputs, verify_freeze


DEFAULT_DEV = Path('outputs/first_penalty_validation_20261008/renewal')
DEFAULT_OUTPUT = Path('outputs/first_penalty_validation_20261008/renewal_test')
SHARED_REPORT = Path('/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/first_penalty_validation_20261008/TEST_RESULTS.md')


def require_dev_freeze(directory):
    frozen = json.loads((directory / 'FREEZE.json').read_text())
    for name, expected in frozen['hashes'].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f'Dev-frozen artifact changed: {name}')
    for name, expected in frozen['code_hashes'].items():
        if file_hash(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Dev-frozen readout code changed: {name}')
    return frozen


def score(args):
    started = time.time()
    dev_freeze = require_dev_freeze(args.dev)
    verify_freeze(args.parent / 'unlabeled')
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / 'code_snapshot'
    snapshot.mkdir()
    for name in ('run_renewal_test.py', 'run_renewal_validation.py', 'renewal.py',
                 'smooth.py', 'unlabeled.py', 'data.py', 'run_unlabeled.py', 'run_case_analysis.py'):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    for name in ('reference.npz', 'THRESHOLDS.json'):
        shutil.copy2(args.dev / name, args.output / name)
    reference = dict(np.load(args.output / 'reference.npz'))
    pack, metadata = load_partition('test')
    previous = dict(np.load(args.parent / 'unlabeled/test_scores.npz'))
    scores = {name: (previous[name].copy() if name in previous else
                    np.full(len(pack['token_id']), np.nan)) for name in METHODS}
    events = np.full(len(pack['token_id']), np.nan)
    diagnostics = []
    for index, record in enumerate(metadata['records']):
        values, attention, targets, region = answer_inputs(args.cache, args.parent / 'capture', record, pack)
        with np.load(args.cache / record['directory'] / 'observations.npz') as raw:
            values['entropy'] = raw['entropy'].copy()
        if not np.array_equal(values['entropy'][targets], pack['observations'][region, 6]):
            raise ValueError(f"{record['id']}: original and test-pack entropy differ")
        answer_scores, event_values, solver = score_answer(reference, values, attention)
        for name, value in answer_scores.items():
            scores[name][region] = value[targets]
        events[region] = event_values[targets]
        diagnostics.append(dict(id=record['id'], **solver))
        if (index + 1) % 150 == 0:
            print(f'RENEWAL test {index+1}/{len(metadata["records"])}', flush=True)
    if not all(np.isfinite(values).all() for values in (*scores.values(), events)):
        raise ValueError('Incomplete renewal test scores')
    np.savez(args.output / 'test_scores.npz', **scores)
    np.savez(args.output / 'test_events.npz', entropy_event=events)
    write_json(args.output / 'SOLVER_DIAGNOSTICS.json', diagnostics)
    write_json(args.output / 'FREEZE.json', dict(status='all_test_scores_frozen_before_official_labels',
        primary=PRIMARY, methods=list(METHODS), thresholds_reused_from_dev=True,
        dev_freeze_sha256=file_hash(args.dev / 'FREEZE.json'),
        dev_protocol=dev_freeze, test_answers=len(metadata['records']),
        test_sources=len(metadata['sources']), test_tokens=len(pack['token_id']),
        natural_labels_used_for_fit=False, test_labels_loaded=False,
        test_labels_used_for_selection=False, new_llm_forwards=0, new_fits=0,
        historical_test_exposure=True, timing=dev_freeze['timing'],
        hashes={name: file_hash(args.output / name) for name in
                ('reference.npz', 'THRESHOLDS.json', 'test_scores.npz', 'test_events.npz')},
        code_hashes={path.name: file_hash(path) for path in snapshot.iterdir()},
        pack_hashes={name: file_hash(PACKS / name) for name in ('QA_test.npz', 'QA_test.json')},
        wall_seconds=time.time()-started))
    print(f'RENEWAL frozen {len(pack["token_id"])} test tokens', flush=True)


def export_cases(args, pack, metadata, scores, thresholds):
    counts = {}
    for record in metadata['records']:
        if record['id'] not in CASE_IDS:
            continue
        region = slice(record['packed_start'], record['packed_stop'])
        response = json.loads((args.cache / record['directory'] / 'response.json').read_text())
        labels = pack['labels'][region]
        counts[record['id']] = {}
        for name, values in scores.items():
            alarm = values[region] > thresholds[name]
            counts[record['id']][name] = dict(errors=int(labels.sum()),
                detected=int(alarm[labels == 1].sum()), false_alarms=int(alarm[labels == 0].sum()))
        rows = []
        for position in range(record['packed_start'], record['packed_stop']):
            token = int(pack['target'][position])
            row = dict(token_index=token, token_text=response['token_text'][token],
                       gold=int(pack['labels'][position]), first_error=int(pack['firsts'][position]))
            for name, values in scores.items():
                row[name] = float(values[position])
                row[name + '_flag'] = int(values[position] > thresholds[name])
            rows.append(row)
        with (args.output / f"case_{record['id']}.csv").open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    write_json(args.output / 'CASE_RESULTS.json', counts)


def report(results):
    lines = ['# 冻结熵/连续性消融：完整 QA test 实测', '',
        '全部9项方法及主方法 entropy_unary_gated_huber 维持开发前预定设置。'
        '复用同一 fit CDF、权重和 fit 混合分布95分位阈值，完整900答分数冻结后才加入官方标注。'
        '旧 test 已历史暴露，属于探索性复验；没有新的大模型前向、拟合或其他任务结果。', '',
        '| 方法 | 全词 AUROC | AP | 严格首错 AUROC | 首错检出 | 首错检出且此前无误报 | 正常回答报警 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, values in results['methods'].items():
        alarm = values['policies']['reference95']
        lines.append(f'| {name} | {values["all_tokens"]["auroc"]:.6f} | {values["all_tokens"]["ap"]:.6f} | '
            f'{values["strict_clean_prefix_first_error"]["auroc"]:.6f} | '
            f'{alarm["first_errors_detected"]}/{alarm["error_answers"]} | '
            f'{alarm["first_errors_detected_without_prior_false_alarm"]}/{alarm["error_answers"]} | '
            f'{alarm["normal_answers_with_alarm"]}/{alarm["normal_answers"]} |')
    lines += ['', '首错负例只含此前无错误的正常 token。阈值来自混合 fit 总体，不能解释为自然正常5% FPR。'
        '熵门仅减弱图平均，普通 Huber 在本轮 [0,1] unary 与 δ=1 下等价于 Laplacian 加权平均；'
        '不称其识别了真实语义片段、原生信息利用或在线首错预防。', '',
        f'固定来源 bootstrap：`{json.dumps(results["bootstrap"], ensure_ascii=False)}`。', '',
        'test 无方法或阈值选型；预定 entropy_unary_native_huber 对照即使更优，也不能被改名为本轮预定主方法。'
        '持续错误、正常开头/数字/大写/标点误报在 TEST_RESULTS.json/subgroups；7旧配对样例在 CASE_RESULTS.json'
        '及逐 token CSV，按官方标签原样评价。']
    return '\n'.join(lines) + '\n'


def evaluate(args):
    frozen = json.loads((args.output / 'FREEZE.json').read_text())
    for name, expected in frozen['hashes'].items():
        if file_hash(args.output / name) != expected:
            raise ValueError(f'Frozen test artifact changed: {name}')
    for name, expected in frozen['code_hashes'].items():
        if file_hash(args.output / 'code_snapshot' / name) != expected:
            raise ValueError(f'Frozen execution snapshot changed: {name}')
        if file_hash(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Current code differs from frozen execution: {name}')
    for name, expected in frozen['pack_hashes'].items():
        if file_hash(PACKS / name) != expected:
            raise ValueError(f'Frozen pack changed: {name}')
    pack, metadata = load_partition('test')
    pack.update(evaluation_labels(args.cache, pack, metadata))
    scores = dict(np.load(args.output / 'test_scores.npz'))
    thresholds = json.loads((args.output / 'THRESHOLDS.json').read_text())
    methods = {name: dict(scores=values, natural_label_fit=False,
        thresholds=dict(reference95=thresholds[name]),
        threshold_rules=dict(reference95='equal_source_fit_mixture95_strict_gt'))
        for name, values in scores.items()}
    summaries, counts = summarize_methods(pack, metadata['records'], methods)
    pairs = {name + '_minus_' + control: source_bootstrap(pack, scores[name], scores[control], args.bootstrap)
        for name, control in (('entropy_unary_native_huber', BASELINE),
                             (PRIMARY, 'entropy_unary_native_huber'),
                             (PRIMARY, BASELINE))}
    results = dict(counts=counts, methods=summaries, primary=PRIMARY, bootstrap=pairs,
        subgroups=subgroup_metrics(pack, scores, thresholds, metadata, args.cache),
        no_test_selection=True, code_sha256=file_hash(Path(__file__)))
    write_json(args.output / 'TEST_RESULTS.json', results)
    export_cases(args, pack, metadata, scores, thresholds)
    text = report(results)
    (args.output / 'TEST_RESULTS.md').write_text(text)
    args.shared_report.write_text(text)
    print(json.dumps({name: result['all_tokens'] for name, result in summaries.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('score', 'evaluate', 'all'))
    parser.add_argument('--dev', type=Path, default=DEFAULT_DEV)
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
