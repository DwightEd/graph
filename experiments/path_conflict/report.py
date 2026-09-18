"""Within-question functional effects. Two pilot questions are not a detection benchmark."""

from pathlib import Path
import tarfile

import numpy as np
import pandas as pd


METRICS = ('next_margin', 'sequence_margin', 'mean_margin', 'correct_logp', 'wrong_logp', 'correct_first_logp', 'wrong_first_logp')


def compare(table):
    keys = ['case_id', 'side']
    baseline = table[table.variant == 'full'].set_index(keys)
    result = table.copy()
    for field in METRICS:
        reference = [baseline.loc[(row.case_id, row.side), field] for row in table.itertuples()]
        result['delta_' + field] = result[field] - np.asarray(reference)
    return result


def interactions(table):
    """Finite-difference interaction; effects are not additive causal percentages."""
    rows = []
    base = table[table.variant == 'full'].set_index(['case_id', 'side'])
    cuts = table[(table.layer >= 0) & table.variant.isin(['cut_evidence', 'cut_value_source', 'cut_history',
                                                       'cut_evidence_value_source', 'cut_evidence_history'])]
    for key, group in cuts.groupby(['case_id', 'side', 'layer', 'scope']):
        values = group.set_index('variant')
        full = base.loc[key[:2]]
        for rival in ('value_source', 'history'):
            joint_name = 'cut_evidence_' + rival
            row = dict(zip(['case_id', 'side', 'layer', 'scope'], key), rival=rival)
            for metric in METRICS[:3]:
                score = full[metric] - values.loc['cut_evidence', metric]
                row['evidence_support_' + metric] = score
                row['rival_removal_gain_' + metric] = values.loc['cut_' + rival, metric] - full[metric]
                row['interaction_' + metric] = (full[metric] - values.loc['cut_evidence', metric]
                    - values.loc['cut_' + rival, metric] + values.loc[joint_name, metric])
            rows.append(row)
    return pd.DataFrame(rows)


def paired_effects(table):
    fields = ['delta_' + metric for metric in METRICS]
    keys = ['case_id', 'layer', 'scope', 'variant']
    supported = table[table.side == 'supported'][keys + fields]
    unsupported = table[table.side == 'unsupported'][keys + fields]
    result = unsupported.merge(supported, on=keys, suffixes=('_unsupported', '_supported'), validate='one_to_one')
    for field in fields:
        result[field + '_difference'] = result[field + '_unsupported'] - result[field + '_supported']
    return result


def make_plots(table, trajectory, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = output / 'figures'
    directory.mkdir(exist_ok=True)
    for (case_id, scope), group in table[table.layer >= 0].groupby(['case_id', 'scope']):
        figure, axis = plt.subplots(figsize=(9, 5))
        for (side, variant), rows in group.groupby(['side', 'variant']):
            if variant not in ('cut_evidence', 'cut_value_source', 'cut_history', 'cut_mlp'):
                continue
            rows = rows.sort_values('layer')
            axis.plot(rows.layer, rows.delta_next_margin, marker='o', label=side + '/' + variant)
        axis.axhline(0, linestyle='--')
        axis.set(xlabel='Native LLM layer', ylabel='Change in correct-minus-wrong NEXT-token logit',
                 title=case_id + ' | ' + scope + ' | candidate-specific pilot')
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(directory / (case_id + '_' + scope + '.png'), dpi=150)
        figure.savefig(directory / (case_id + '_' + scope + '.svg'))
        plt.close(figure)
    baseline = trajectory[trajectory.variant == 'full']
    for case_id, group in baseline.groupby('case_id'):
        figure, axis = plt.subplots(figsize=(9, 5))
        for (side, site), rows in group.groupby(['side', 'site']):
            axis.plot(rows.layer, rows.margin, label=side + '/' + site)
        axis.axhline(0, linestyle='--')
        axis.set(xlabel='Native LLM layer', ylabel='Contextual logit-lens candidate margin',
                 title=case_id + ' | local readout, not causal evidence by itself')
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / 'figures' / (case_id + '_baseline_trajectory.png'), dpi=150)
        plt.close(figure)


def report(output):
    output = Path(output)
    records, trajectories, writes, changes = [], [], [], []
    for path in sorted((output / 'runs').glob('*.npz')):
        import json
        with np.load(path, allow_pickle=False) as saved:
            record = json.loads(str(saved['record']))
            records.append(record)
            identity = {name: record[name] for name in ('case_id', 'side', 'variant', 'scope')}
            identity['intervention_layer'] = record['layer']
            for key, target in [('trajectory', trajectories), ('writes', writes), ('changes', changes)]:
                for row in json.loads(str(saved[key])):
                    target.append({**identity, **row})
    table = compare(pd.DataFrame(records))
    table.to_csv(output / 'effects.csv', index=False)
    interactions(table).to_csv(output / 'path_interactions.csv', index=False)
    paired_effects(table).to_csv(output / 'same_question_effects.csv', index=False)
    trajectory = pd.DataFrame(trajectories)
    trajectory.to_csv(output / 'trajectories.csv.gz', index=False)
    pd.DataFrame(writes).to_csv(output / 'head_source_writes.csv.gz', index=False)
    pd.DataFrame(changes).to_csv(output / 'intervention_sizes.csv', index=False)
    make_plots(table, trajectory, output)
    plot_head_writes(pd.DataFrame(writes), output)
    notes = '''# 同题采样的原 LLM 路径干预

先看 inventory/sources.csv 和 decisions.csv，确认正在使用哪些原始轨迹、哪一项局部事实。
replay.json 记录原始采样 logits 与此次无缓存回放的差异；缺少原始 trace 不会伪造回放。

- effects.csv：正的 delta_next_margin 表示当前干预使正确候选更有优势；分别保留两候选自身概率。
- path_interactions.csv：原 M - 删证据 M 为正才支持“证据原本有效”；删竞争来源后 M 增加才支持其推动错误。
- same_question_effects.csv：错误侧效应减同题有依据侧效应；前缀词面不同仍有混杂。
- head_source_writes.csv.gz：每层每头的来源质量、value能量、实际W_O写入及局部候选支持，不先平均头。
- trajectories.csv.gz：各次原生运行的残差读出变化；logit lens是观察，不独自证明因果。

序列总logp、平均logp和首token margin分别报告。候选长度不同；不能把序列均分当概率。
支持/不支持仅针对两项核验过的陈述，并非整篇标签。样本抽中错误token不等于原生分布偏好错误。
模型输入始终是保存的token ID，不重套chat模板。只运行原Llama，不调用CHARM/LDA。

query只改变最终输入位置的写入；prefix改变从起点到该位置的所有写入。候选后续位置不施加新的删除。
source cut按A*V*W_O删除消息，不修改softmax分母。随机对照匹配残差写入L2大小，未保证语义或在分布上。
restore将下游头恢复到同一前缀未干预时的值，测试上游效应的依赖，不把恢复直接叫自然中介比例。

本批只有两个探索性case；不作自然总体AUROC、p值或“幻觉普遍有冲突”的结论。
要支持证据落败，需同时看到证据必要、竞争路径切断有利、正确候选概率改善、正常对照不被普遍破坏。
不满足时保留“未读入证据”“候选/词面混杂”“采样选错”“未定位”的解释，不自动命名self lock-in。
'''
    (output / 'REPORT_zh.md').write_text(notes, encoding='utf-8')
    archive_path = output / 'path_conflict_review.tar.gz'
    with tarfile.open(archive_path.with_suffix('.partial'), 'w:gz') as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name not in (archive_path.name, archive_path.with_suffix('.partial').name):
                archive.add(path, arcname=path.relative_to(output))
    archive_path.with_suffix('.partial').replace(archive_path)
    print(table[['case_id', 'side', 'layer', 'scope', 'variant', 'delta_next_margin', 'delta_sequence_margin']].to_string(index=False))


def plot_head_writes(writes, output):
    import matplotlib.pyplot as plt

    for (case_id, side, source), frame in writes.groupby(['case_id', 'side', 'source_group']):
        if source not in ('evidence', 'value_source', 'history'):
            continue
        matrix = frame.pivot(index='layer', columns='head', values='local_lens_support')
        figure, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(matrix.to_numpy(), aspect='auto')
        figure.colorbar(image, ax=axis, label='Contextual lens margin lost after removing one write')
        axis.set(xlabel='Native head index', ylabel='Native layer index', title=f'{case_id} | {side} | {source}')
        figure.tight_layout()
        figure.savefig(output / 'figures' / f'{case_id}_{side}_{source}_heads.png', dpi=150)
        plt.close(figure)
