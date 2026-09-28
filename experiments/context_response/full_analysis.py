"""Post-freeze full-test span coverage, generator groups and recurrence uncertainty."""
import argparse
import csv
from pathlib import Path
import numpy as np
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.unsupervised_graph.data import load_inputs
from experiments.probabilistic_detection.data import evaluation_labels
from experiments.probabilistic_detection.evaluation import source_bootstrap
from experiments.decision_risk_flow.verify import independent_metrics


def positive_runs(labels):
    changes = np.diff(np.r_[False, labels.astype(bool), False].astype(int))
    return list(zip(np.flatnonzero(changes==1), np.flatnonzero(changes==-1)))


def analyze(output, packs):
    read_json(output/'evaluation_complete.json')
    manifest = read_json(output/'manifest.json')
    thresholds = read_json(output/'thresholds.json')
    rows, groups, uncertainty = [], [], {}
    for task in ('QA', 'Summary', 'Data2txt'):
        pack, metadata = load_inputs(packs, task, 'test')
        pack.update(evaluation_labels(Path(manifest['source']), pack, metadata))
        with np.load(output/task/'test_scores.npz') as saved:
            scores = {name: saved[name] for name in saved.files}
        labels = pack['labels']
        generators = {}
        spans = []
        for row in metadata['records']:
            start, stop = row['packed_start'], row['packed_stop']
            generators.setdefault(row['generator'], []).extend(range(start, stop))
            spans.extend((start+a, start+b) for a,b in positive_runs(labels[start:stop]))
        for name, score in scores.items():
            alarm = score>thresholds[task][name]
            rows.append(dict(task=task, method=name, contiguous_error_runs=len(spans),
                any_detected=sum(alarm[a:b].any() for a,b in spans),
                fully_detected=sum(alarm[a:b].all() for a,b in spans),
                first_token_detected=sum(alarm[a] for a,b in spans),
                mean_run_token_recall=float(np.mean([alarm[a:b].mean() for a,b in spans]))))
            for generator, indices in generators.items():
                selected = np.asarray(indices)
                metric = independent_metrics(labels[selected], score[selected], thresholds[task][name])
                groups.append(dict(task=task, generator=generator, method=name, **metric))
        uncertainty[task] = {}
        for name in ('recurrence_offline', 'corroborated_only', 'original_recurrence', 'original_corroborated'):
            if name not in scores:
                continue
            uncertainty[task][name] = {baseline: source_bootstrap(pack, scores[name], scores[baseline])
                for baseline in ('strong_fused', 'original_full_reference_fixed')}
    for filename, records in (('error_run_metrics.csv', rows), ('generator_metrics.csv', groups)):
        with (output/filename).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    write_json(output/'recurrence_bootstrap.json', uncertainty)
    finish_report(output, rows, groups, uncertainty)


def finish_report(output, rows, groups, uncertainty):
    manifest = read_json(output/'manifest.json')
    thresholds = read_json(output/'thresholds.json')
    pilot = Path(manifest['pilot'])
    pilot_manifest = read_json(pilot/'manifest.json')
    pilot_selection = read_json(pilot/'scores_frozen.json')
    consistency, outside = [], []
    population = {r['id'] for r in manifest['records']}
    for row in pilot_manifest['records']:
        if row['role']!='regression':
            continue
        if row['id'] not in population:
            outside.append(row['id'])
            continue
        with np.load(pilot/row['key']/'scores.npz') as local, np.load(output/'responses'/row['id']/'scores.npz') as full:
            for name in pilot_selection['methods']:
                threshold = thresholds[row['task']][name]
                consistency.append(dict(id=row['id'], method=name,
                    max_abs_difference=float(np.max(np.abs(local[name]-full[name]))),
                    alarm_differences=int(np.sum((local[name]>threshold)!=(full[name]>threshold)))))
    write_json(output/'pilot_stream_consistency.json', dict(comparisons=consistency,
        regression_ids_outside_official_test=outside,
        max_abs_difference=max(r['max_abs_difference'] for r in consistency),
        alarm_differences=sum(r['alarm_differences'] for r in consistency)))
    write_json(output/'analysis_complete.json', dict(status='complete', task_methods=len(rows),
        generator_methods=len(groups), span_definition='contiguous official-positive token runs within each answer; not individual annotation objects',
        labels_used_for_scores=False, exploratory=True))
    lines = ['# 完整三任务：重复模式与风险起点对照', '',
        '2700个回答、450个来源、424408个token，所有1024物理头的新响应/JS已经采集。评分冻结后评价；同一测试总体历史已暴露，原起点追加对照还参考了本轮早期结果，属于探索复验。检测评分/校准无自然标签拟合，不声称盲测。', '',
        '|任务|方法|AUROC|答内AUROC|错误召回|正常token误报|正常回答任意报警|完整错误连续段|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    selected_methods = ('original_full_reference_fixed', 'strong_fused', 'recurrence_offline',
        'recurrence_shuffled', 'distance_offline', 'corroborated_only', 'barrier_corroborated',
        'original_recurrence', 'original_corroborated')
    for task in ('QA', 'Summary', 'Data2txt'):
        metrics = read_json(output/task/'metrics.json')
        for name in selected_methods:
            if name not in metrics:
                continue
            value = metrics[name]
            span = next(r for r in rows if r['task']==task and r['method']==name)
            fields = [value[k] for k in ('auroc', 'within_answer_auroc', 'token_recall',
                                          'token_fpr', 'normal_answer_false_alarm')]
            lines.append('|'+task+'|'+name+'|'+'|'.join(f'{x:.6f}' for x in fields)+
                         f"|{span['fully_detected']}/{span['contiguous_error_runs']}|")
    lines.extend(['', '完整错误连续段以每答官方token正标签的连续区间统计，和单个原始标注对象的数量可能不同。离线双向传播可用后续token；过去向消融单独保存在完整指标中。', '',
                  '来源簇bootstrap300次；以下是AUROC增量的名义95%区间，未对多候选作多重比较校正：', ''])
    for task, candidates in uncertainty.items():
        for name, baselines in candidates.items():
            for baseline, result in baselines.items():
                lines.append(f"- {task} / {name} − {baseline}: {result['auroc_delta_95ci']}")
    lines.extend(['', '原生响应直接影响当前原词相对竞争词，不是事实证据符号。重复图表示机制状态相似，不证明语义事实相同；max-min与双起点并非后验概率。', '',
        '四来源无标签dev95不是正常FPR=5%的保证；原固定基线使用原完整训练/dev参考与阈值。原评分数组未改写，已有对照逐元素核对后复用；full_verification.json为同agent另一实现的官方span和数值复核。', '',
        '所有14方法/任务的完整指标见RESULTS_ZH.md；错误连续段见error_run_metrics.csv；生成器分组见generator_metrics.csv；新采集与局部缓存的分数一致性见pilot_stream_consistency.json。'])
    (output/'COMPARISON_ZH.md').write_text('\n'.join(lines)+'\n')
    print('Completed span/generator analysis:', len(rows), len(groups), flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('all', 'report'), default='all')
    parser.add_argument('--packs', type=Path, default=Path('outputs/probabilistic_detection_20260928_full/packs'))
    args = parser.parse_args()
    if args.stage=='all':
        analyze(args.output, args.packs)
    else:
        with (args.output/'error_run_metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        with (args.output/'generator_metrics.csv').open() as stream:
            groups = list(csv.DictReader(stream))
        finish_report(args.output, rows, groups, read_json(args.output/'recurrence_bootstrap.json'))
