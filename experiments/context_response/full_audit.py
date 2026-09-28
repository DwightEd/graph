"""Recompute full-test official targets and ranking with a separate numeric implementation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.decision_risk_flow.verify import official_targets, independent_metrics


def verify(output):
    read_json(output/'evaluation_complete.json')
    manifest = read_json(output/'manifest.json')
    root = Path(manifest['source'])
    dataset = Path(read_json(root/'manifest.json')['dataset'])
    identities = {row['id'] for row in manifest['records']}
    official = {}
    for line in (dataset/'response.jsonl').open():
        row = json.loads(line)
        if str(row['id']) in identities:
            official[str(row['id'])] = row
    thresholds = read_json(output/'thresholds.json')
    methods = manifest['methods']
    labels = {task: [] for task in thresholds}
    scores = {task: {name: [] for name in methods} for task in thresholds}
    cases = []
    for row in manifest['records']:
        target, valid, response = official_targets(dict(row, root=str(root)), official)
        task = row['task']
        labels[task].append(target[valid])
        with np.load(output/'responses'/row['id']/'scores.npz') as saved:
            assert np.array_equal(saved['token_ids'], response['answer_ids'])
            for name in methods:
                selected = saved[name][valid]
                assert np.isfinite(selected).all()
                scores[task][name].append(selected)
                metric = independent_metrics(target[valid], selected, thresholds[task][name])
                cases.append(dict(id=row['id'], source_id=row['source_id'], task=task,
                    generator=row['generator'], method=name, **metric))
    checked = []
    for task in thresholds:
        target = np.concatenate(labels[task])
        expected = read_json(output/task/'metrics.json')
        for name in methods:
            metric = independent_metrics(target, np.concatenate(scores[task][name]), thresholds[task][name])
            for field in ('auroc', 'ap'):
                assert np.isclose(metric[field], expected[name][field], atol=1e-12, rtol=0), (task, name, field)
            assert np.isclose(metric['detected_errors']/metric['error_tokens'], expected[name]['token_recall'])
            assert np.isclose(metric['false_alarms']/metric['normal_tokens'], expected[name]['token_fpr'])
            checked.append(dict(task=task, method=name, **metric))
    with (output/'answer_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=cases[0])
        writer.writeheader()
        writer.writerows(cases)
    report = dict(status='passed', same_agent_independent_numeric_recomputation=True,
        answers=len(identities), case_method_groups=len(cases), task_method_groups=len(checked),
        valid_tokens=sum(len(np.concatenate(value)) for value in labels.values()), metrics=checked)
    write_json(output/'full_verification.json', report)
    return report


def report(output):
    manifest = read_json(output/'manifest.json')
    lines = ['# 完整三任务探索复验', '',
        '所有2700答的新评分已冻结后评价；已有测试总体此前暴露，不能称全新盲测。', '',
        '|任务|方法|AUROC|AP|答内AUROC（pair加权）|错误召回|正常token误报|正常回答任一报警|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    notes = []
    for task in ('QA', 'Summary', 'Data2txt'):
        metrics = read_json(output/task/'metrics.json')
        for name, value in metrics.items():
            fields = [value[key] for key in ('auroc', 'ap', 'within_answer_auroc', 'token_recall', 'token_fpr', 'normal_answer_false_alarm')]
            lines.append('|'+task+'|'+name+'|'+'|'.join('NA' if x is None else f'{x:.6f}' for x in fields)+'|')
        interval = read_json(output/task/'bootstrap.json')
        notes.extend(['', f"{task} 主方法为 `{manifest['main']}`。来源簇bootstrap差的95%区间：同参考基线 {interval['same_reference']['auroc_delta_95ci']}；此前完整训练参考基线 {interval['original_full_reference']['auroc_delta_95ci']}。", ''])
    lines.extend(notes)
    lines.extend(['', '评分没有自然标签拟合或测试阈值搜索。4 fit/4 dev来源只是小参考分布，混合95分位不保证正常FPR=5%。逐答数字见answer_metrics.csv；官方原span与另一个数值实现复算见full_verification.json。', '',
        '归因度量描述当前原词相对自动竞争词的改变，不能直接解释为事实反证；固定过去KV不是跨离散生成的完整因果轨迹。纯内部方法与融合方法分别列出，来源关联反向邻接不冒充原生因果流。'])
    (output/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.output)
    report(args.output)
    print({key: value for key, value in result.items() if key!='metrics'})


if __name__ == '__main__':
    main()
