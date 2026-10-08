"""Audit frozen full-QA token scores without fitting or changing any detector.

The first-error diagnostic excludes every normal token after an earlier error.
It is still an offline observed-candidate observer analysis, not an online
prevention experiment with any of the six original answer generators.
"""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from experiments.native_support.evaluate import ranking
from experiments.probabilistic_detection.data import evaluation_labels, load_pack
from experiments.probabilistic_detection.evaluation import pairwise_within
from .data import PACKS
from .run_capture import CACHE, OUTPUT, file_hash, write_json


CASE_IDS = ('12219', '12297', '12471', '17199', '11907', '12015', '12045')
REQUIRED_FAMILIES = {'unlabeled', 'source_selfsup', 'source_choices_v2', 'supervised'}
RAW_SCALARS = ('source_local', 'source_full', 'raw_route', 'raw_attention', 'entropy')
MESSAGE_GROUPS = ('source', 'local', 'remote', 'self', 'other_prompt')
SHARED_REPORT = Path('/share/home/tm902089733300000/a903202310/lys/codex/research/'
                     'refine-logs/supervised_local_transport_20261008/CASE_ANALYSIS.md')


def score_families(root):
    """Select methods and existing threshold policies before reading annotations."""
    return (
        dict(name='unlabeled', directory=root / 'unlabeled', scores='test_scores.npz',
             freeze='FREEZE.json', hash_key='hashes', methods=('source_native_huber',
             'source_route_native_huber'), policies=('fixed95', 'reference95', 'dev_normal95'),
             natural_label_fit=False),
        dict(name='source_selfsup', directory=root / 'source_selfsup' / 'natural_test',
             scores='scores.npz', freeze='FREEZE.json', hash_key='prediction_sha256',
             methods=('real', 'native_only_real'), policies=('zero', 'program_correct95'),
             natural_label_fit=False),
        dict(name='source_choices_v2', directory=root / 'source_choices_v2' / 'natural_test',
             scores='scores.npz', freeze='FREEZE.json', hash_key='scores_sha256',
             methods=('real', 'native_only_real'), policies=('zero', 'program_correct95'),
             natural_label_fit=False),
        dict(name='supervised', directory=root / 'reader_fits', scores='test_scores.npz',
             freeze='TEST_FREEZE.json', hash_key='predictions_sha256',
             methods=('real_seed42', 'prechoice_real_seed42'), policies=('dev_normal95',),
             natural_label_fit=True),
        dict(name='source_address_v3', directory=root / 'source_address_v3' / 'natural_test',
             scores='scores.npz', freeze='FREEZE.json', hash_key='prediction_sha256',
             methods=('real', 'native_only_real'), policies=('zero', 'program_first_correct95'),
             natural_label_fit=False),
    )


def load_frozen_scores(root, token_count):
    """Require frozen score identity and completed existing evaluation policies."""
    methods, pending, bindings = {}, {}, {}
    for family in score_families(root):
        directory = family['directory']
        metric_files = {policy: directory / ('test_metrics.json' if family['name'] == 'supervised'
                        else f'test_metrics_{policy}.json') for policy in family['policies']}
        required = [directory / family['freeze'], directory / family['scores'], *metric_files.values()]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            pending[family['name']] = dict(status='pending', missing=missing)
            skipped = directory.parent / 'NATURAL_EVALUATION_SKIPPED.json'
            if family['name'] == 'source_address_v3' and skipped.exists():
                pending[family['name']] = dict(status='skipped_by_program_gate',
                    explanation=json.loads(skipped.read_text()), gate_record_sha256=file_hash(skipped))
            continue
        frozen = json.loads(required[0].read_text())
        expected = (frozen['hashes'][family['scores']] if family['hash_key'] == 'hashes'
                    else frozen[family['hash_key']])
        actual = file_hash(required[1])
        if actual != expected or frozen['test_tokens'] != token_count:
            raise ValueError(f"{family['name']}: frozen score identity/coverage differs")
        metrics = {policy: json.loads(path.read_text()) for policy, path in metric_files.items()}
        with np.load(required[1]) as arrays:
            for name in family['methods']:
                values = arrays[name].copy()
                if values.shape != (token_count,) or not np.isfinite(values).all():
                    raise ValueError(f"{family['name']}/{name}: incomplete/nonfinite scores")
                method_name = f"{family['name']}__{name}"
                thresholds = {policy: metrics[policy][name]['threshold'] for policy in family['policies']}
                methods[method_name] = dict(scores=values, thresholds=thresholds,
                    natural_label_fit=family['natural_label_fit'], family=family['name'], method=name,
                    threshold_rules={policy: metrics[policy][name]['threshold_rule']
                                     for policy in family['policies']})
        bindings[family['name']] = dict(scores_sha256=actual, freeze_sha256=file_hash(required[0]),
            directory=str(directory.resolve()), metrics_sha256={policy: file_hash(path)
                                                               for policy, path in metric_files.items()})
    return methods, pending, bindings


def clean_prefix_masks(pack, records):
    """Normal answers stay clean throughout; error answers stop at their first error."""
    clean = np.zeros(len(pack['labels']), dtype=bool)
    first = np.zeros(len(pack['labels']), dtype=bool)
    answer_rows = []
    for record in records:
        start, stop = record['packed_start'], record['packed_stop']
        errors = np.flatnonzero(pack['labels'][start:stop])
        first_position = start + int(errors[0]) if len(errors) else None
        clean[start:first_position if first_position is not None else stop] = True
        if first_position is not None:
            first[first_position] = True
        answer_rows.append(dict(id=record['id'], start=start, stop=stop,
                                first_position=first_position))
    if not np.array_equal(first, pack['firsts']):
        raise ValueError('Official first-error labels differ from first positive valid-token positions')
    return clean, first, answer_rows


def alarm_metrics(labels, scores, threshold, clean, first, answer_rows):
    """Count false alarms strictly before the first labeled error, never after it."""
    alarm = scores > threshold
    normal_any, error_prefix_any, first_without_prior, first_alarm = [], [], [], []
    for record in answer_rows:
        start, stop, onset = record['start'], record['stop'], record['first_position']
        if onset is None:
            normal_any.append(bool(alarm[start:stop].any()))
        else:
            prior = bool(alarm[start:onset].any())
            error_prefix_any.append(prior)
            first_without_prior.append(bool(alarm[onset] and not prior))
            first_alarm.append(bool(alarm[onset]))
    return dict(threshold=float(threshold), strict_gt=True,
        normal_tokens=int((labels == 0).sum()), error_tokens=int(labels.sum()),
        normal_token_false_alarms=int(alarm[labels == 0].sum()),
        error_tokens_detected=int(alarm[labels == 1].sum()),
        token_fpr=float(alarm[labels == 0].mean()), token_recall=float(alarm[labels == 1].mean()),
        clean_normal_tokens=int(clean.sum()), clean_normal_false_alarms=int(alarm[clean].sum()),
        clean_normal_fpr=float(alarm[clean].mean()),
        normal_answers=len(normal_any), normal_answers_with_alarm=int(sum(normal_any)),
        normal_answer_false_alarm=float(np.mean(normal_any)),
        error_answers=len(error_prefix_any), error_answers_with_false_alarm_before_first=int(sum(error_prefix_any)),
        first_errors_detected=int(sum(first_alarm)),
        first_errors_detected_without_prior_false_alarm=int(sum(first_without_prior)),
        first_error_recall=float(alarm[first].mean()))


def summarize_methods(pack, records, methods):
    clean, first, answer_rows = clean_prefix_masks(pack, records)
    selected = clean | first
    summaries = {}
    for name, method in methods.items():
        scores = method['scores']
        policies = {}
        for policy, threshold in method['thresholds'].items():
            policies[policy] = alarm_metrics(pack['labels'], scores, threshold, clean, first, answer_rows)
            policies[policy]['threshold_rule'] = method['threshold_rules'][policy]
            policies[policy]['natural_labels_used_for_threshold'] = policy == 'dev_normal95'
        summaries[name] = dict(natural_label_fit=method['natural_label_fit'],
            all_tokens=ranking(pack['labels'], scores),
            within_answer_auroc=pairwise_within(pack['labels'], scores, pack['answer_index']),
            strict_clean_prefix_first_error=ranking(first[selected].astype(int), scores[selected]),
            policies=policies)
    counts = dict(answers=len(records), sources=len({row['source_id'] for row in records}),
        valid_tokens=len(pack['labels']), error_tokens=int(pack['labels'].sum()),
        span_onsets=int(pack['onsets'].sum()), first_errors=int(first.sum()),
        clean_normal_tokens=int(clean.sum()), normal_answers=int(sum(row['first_position'] is None
                                                                   for row in answer_rows)))
    return summaries, counts


def case_rows(record, response, pack, methods, raw_signals):
    """Export each original valid token independently, without span aggregation."""
    start, stop = record['packed_start'], record['packed_stop']
    selected = np.arange(start, stop)
    targets = pack['target'][selected]
    if not np.array_equal(np.asarray(response['answer_ids'])[targets], pack['token_id'][selected]):
        raise ValueError(f"{record['id']}: saved token identities differ")
    rows = []
    for packed_position, target in zip(selected, targets):
        char_start, char_stop = response['offsets'][target]
        row = dict(answer_id=record['id'], source_id=record['source_id'], generator=record['generator'],
            token_index=int(target), token_id=int(pack['token_id'][packed_position]),
            token_text=response['token_text'][target], char_start=char_start, char_stop=char_stop,
            gold=int(pack['labels'][packed_position]), span_onset=int(pack['onsets'][packed_position]),
            first_error=int(pack['firsts'][packed_position]))
        for name, values in raw_signals.items():
            row[name] = float(values[target])
        for name, method in methods.items():
            risk = float(method['scores'][packed_position])
            row[name + '__risk'] = risk
            for policy, threshold in method['thresholds'].items():
                row[name + '__flag_' + policy] = int(risk > threshold)
        rows.append(row)
    return rows


def case_raw_signals(root, cache, record, response):
    """Keep historical scalars and every physical head's source/boundary mass."""
    with np.load(cache / record['directory'] / 'observations.npz') as saved:
        if saved['token_id'].tolist() != response['answer_ids']:
            raise ValueError(f"{record['id']}: original scalar token alignment differs")
        signals = {f'legacy__{name}': saved[name].copy() for name in RAW_SCALARS}
    with np.load(root / 'capture' / record['id'] / 'arrays.npz') as saved:
        if saved['answer_ids'].tolist() != response['answer_ids']:
            raise ValueError(f"{record['id']}: group-mass token alignment differs")
        masses = saved['group_mass']
        for timing, rows in (('prechoice', slice(None, -1)), ('posttoken', slice(1, None))):
            for world, world_name in enumerate(('native', 'source_blocked')):
                for group, group_name in enumerate(MESSAGE_GROUPS):
                    for head in range(masses.shape[-1]):
                        name = f'{timing}__{world_name}__{group_name}__head{head:02d}__attention_mass'
                        signals[name] = masses[world, rows, group, head].copy()
    return signals


def save_cases(destination, root, cache, records, case_ids, pack, methods):
    lookup = {row['id']: row for row in records}
    cases, bindings = {}, {}
    for identity in case_ids:
        record = lookup[identity]
        path = cache / record['directory'] / 'response.json'
        response = json.loads(path.read_text())
        raw_signals = case_raw_signals(root, cache, record, response)
        rows = case_rows(record, response, pack, methods, raw_signals)
        with (destination / f'{identity}_tokens.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        case = dict(source_id=record['source_id'], generator=record['generator'], valid_tokens=len(rows),
                    error_tokens=sum(row['gold'] for row in rows), onsets=sum(row['span_onset'] for row in rows))
        case['detections'] = {name: {policy: dict(
            error_tokens_detected=sum(row[name + '__flag_' + policy] * row['gold'] for row in rows),
            normal_tokens_flagged=sum(row[name + '__flag_' + policy] * (1 - row['gold']) for row in rows),
            onsets_detected=sum(row[name + '__flag_' + policy] * row['span_onset'] for row in rows))
            for policy in method['thresholds']} for name, method in methods.items()}
        cases[identity] = dict(summary=case, rows=rows)
        bindings[identity] = dict(response_sha256=file_hash(path),
            legacy_observations_sha256=file_hash(cache / record['directory'] / 'observations.npz'),
            capture_sha256=file_hash(root / 'capture' / record['id'] / 'arrays.npz'))
    return cases, bindings


def save_html(destination, cases, methods):
    """A standalone token table shows raw scores; shading represents annotations."""
    pieces = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>QA token traces</title>',
        '<style>body{font:14px sans-serif;margin:24px}table{border-collapse:collapse}',
        'td,th{border:1px solid #ddd;padding:4px}td.token{white-space:pre-wrap}',
        'tr.error{background:#ffedf0}tr.onset td{border-top:2px solid #aa2440}',
        'th{position:sticky;top:0;background:white}details{margin:20px 0}</style>',
        '<h1>Frozen QA per-token scores</h1><p>Raw risk units are method-specific, not factual probabilities. ',
        'Pink rows are official error annotations; a strong top border marks a span onset. ',
        'Each cell contains the raw score and the names of policies whose strict threshold was exceeded. ',
        'CSV files contain exact values and every independent token flag. This is offline observer evaluation.</p>']
    for identity, case in cases.items():
        summary = case['summary']
        pieces.append(f'<details><summary>{identity}: source {summary["source_id"]}, '
                      f'{summary["error_tokens"]}/{summary["valid_tokens"]} labeled error tokens</summary><table>')
        pieces.append('<tr><th>Token index</th><th>Token text</th><th>Gold</th>' +
                      ''.join(f'<th>legacy {name}</th>' for name in RAW_SCALARS) +
                      ''.join(f'<th>{html.escape(name)}</th>' for name in methods) + '</tr>')
        for row in case['rows']:
            style = ('error ' if row['gold'] else '') + ('onset' if row['span_onset'] else '')
            pieces.append(f'<tr class="{style}"><td>{row["token_index"]}</td>'
                          f'<td class="token">{html.escape(row["token_text"])}</td><td>{row["gold"]}</td>')
            for name in RAW_SCALARS:
                pieces.append(f'<td>{row["legacy__" + name]:.6g}</td>')
            for name, method in methods.items():
                flags = ', '.join(policy for policy in method['thresholds'] if row[name + '__flag_' + policy])
                pieces.append(f'<td>{row[name + "__risk"]:.6g}<br>{html.escape(flags)}</td>')
            pieces.append('</tr>')
        pieces.append('</table></details>')
    pieces.append('</html>')
    (destination / 'token_traces.html').write_text(''.join(pieces), encoding='utf-8')


def report_markdown(destination, summaries, counts, cases, pending):
    lines = ['# 冻结 QA 逐 token 样例与首错诊断', '',
        f'全量 {counts["sources"]} 来源 / {counts["answers"]} 回答 / {counts["valid_tokens"]} 有效 token，'
        f'{counts["error_tokens"]} 错误 token、{counts["span_onsets"]} 标注入口；'
        f'{counts["normal_answers"]} 条正常回答、{counts["first_errors"]} 条含错误回答。', '',
        '只读取已冻结分数，未训练或调整评分/阈值。严格 clean-prefix 首错 AUROC 的正例为每条错误回答的首个'
        '标错 token；负例仅含此前没有错误的正常 token（全部正常回答及错误回答的首错之前），排除首错后的正常词。'
        '它不是原评价器用全部正常词得到的 firsts AUROC。观察者显式读取实际候选 y，方法含离线分数；'
        '这些指标不能称六个原生成器的在线预防实验。官方 test 历史已暴露，属于探索性复验。', '',
        '| 方法 | 自然标签 fit | 全 token AUROC | AP | 答内 AUROC | 严格首错 AUROC |',
        '|---|---|---:|---:|---:|---:|']
    for name, values in summaries.items():
        lines.append(f'| {name} | {values["natural_label_fit"]} | {values["all_tokens"]["auroc"]:.6f} | '
            f'{values["all_tokens"]["ap"]:.6f} | {values["within_answer_auroc"]:.6f} | '
            f'{values["strict_clean_prefix_first_error"]["auroc"]:.6f} |')
    lines += ['', f'严格首错负例共 {counts["clean_normal_tokens"]} 个，正例 {counts["first_errors"]} 个。', '',
        '| 方法 / 既定阈值 | 阈值 | 正常回答报警 | 含错回答首错前误报 | 首错检出 | 首错检出且此前无误报 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, values in summaries.items():
        for policy, value in values['policies'].items():
            lines.append(f'| {name} / {policy} | {value["threshold"]:.6g} | '
                f'{value["normal_answers_with_alarm"]}/{value["normal_answers"]} | '
                f'{value["error_answers_with_false_alarm_before_first"]}/{value["error_answers"]} | '
                f'{value["first_errors_detected"]}/{value["error_answers"]} | '
                f'{value["first_errors_detected_without_prior_false_alarm"]}/{value["error_answers"]} |')
    lines += ['', 'dev_normal95 使用自然开发标签，仅作为标签辅助阈值单列；reference95 是混合 fit 分布的分位数，'
        'program_correct95 来自来源程序的兼容候选，均不保证自然正常 FPR。zero/fixed95 是固定阈值。', '',
        '样例是预先指定且历史已研究的回归例，不按本轮表现挑选。每个 CSV 包含各词独立 raw risk、所有既定阈值 flag、'
        '原 token ID/文本/字符区间、官方标签/入口/首错；未按 gold span 平均或广播。', '',
        '| 样例 | 来源 | 有效 token | 标错 token | 标注入口 |', '|---|---|---:|---:|---:|']
    for identity, case in cases.items():
        value = case['summary']
        lines.append(f'| {identity} | {value["source_id"]} | {value["valid_tokens"]} | '
                     f'{value["error_tokens"]} | {value["onsets"]} |')
    lines += ['', '12297 的历史核查显示正文内容由 Passage3 提供而引用写为 Passage2，整个引用段被标错；'
        '因此该段每个 gold=1 不等价于每个字面词都没有来源支持。12219 是错误否定来源提供步骤的旧例。'
        '本报告保留官方标注，不更改标签来改善任何方法。', '',
        f'原始诊断目录：`{destination.resolve()}`。可打开 `token_traces.html`，精确数据在各 `*_tokens.csv`、'
        '`summary.json` 与 `cases.json`。CSV 还保留原始来源/route/entropy 五个标量及 native/source-blocked '
        '世界 prechoice/posttoken 的全部 5×32 物理 head attention mass；不按 head 平均。读取质量不等于语义支持，'
        '最大质量也不命名为正确或主导来源。']
    if pending:
        descriptions = [f'{name} ({value["status"]})' for name, value in pending.items()]
        lines += ['', '未评价输出：' + ', '.join(descriptions) + '。本报告不补分数，也不宣称这些方法已评价。'
                  'v3 若未通过来源程序 gate，可以跳过自然评价；--require-complete 仅要求原四类方法完整。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUTPUT)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'case_analysis')
    parser.add_argument('--shared-report', type=Path, default=SHARED_REPORT)
    parser.add_argument('--case-ids', nargs='+', default=CASE_IDS)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    pack, metadata = load_pack(PACKS, 'QA', 'test')
    methods, pending, bindings = load_frozen_scores(args.root, len(pack['token_id']))
    required_pending = REQUIRED_FAMILIES & set(pending)
    if args.require_complete and required_pending:
        raise ValueError('Requested completed methods are pending: ' + ', '.join(sorted(required_pending)))
    if not methods:
        raise ValueError('No completed frozen scores are available')
    # All included prediction hashes and thresholds are fixed before this join.
    pack.update(evaluation_labels(args.cache, pack, metadata))
    summaries, counts = summarize_methods(pack, metadata['records'], methods)
    args.output.mkdir(parents=True, exist_ok=False)
    cases, responses = save_cases(args.output, args.root, args.cache, metadata['records'], args.case_ids, pack, methods)
    write_json(args.output / 'summary.json', dict(counts=counts, methods=summaries, pending=pending,
        timing='offline observed-candidate observer; no online prevention claim'))
    write_json(args.output / 'cases.json', {identity: case['summary'] for identity, case in cases.items()})
    write_json(args.output / 'INPUTS.json', dict(families=bindings, response_sha256=responses,
        code_sha256=file_hash(Path(__file__)), packs={name: file_hash(PACKS / name)
        for name in ('QA_test.npz', 'QA_test.json')},
        selected_case_ids=list(args.case_ids), case_selection='fixed historically exposed regression IDs',
        natural_labels_used_for_fit_or_selection=False, scores_modified=False))
    save_html(args.output, cases, methods)
    report = report_markdown(args.output, summaries, counts, cases, pending)
    (args.output / 'CASE_ANALYSIS.md').write_text(report)
    args.shared_report.write_text(report)
    print(json.dumps(dict(output=str(args.output), counts=counts, completed_methods=list(methods),
                          pending=list(pending)), ensure_ascii=False))


if __name__ == '__main__':
    main()
