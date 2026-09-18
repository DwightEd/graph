"""Candidate-conditional support, interactions and restoration; no invented semantic labels."""

from pathlib import Path
import json
import tarfile
import html

import numpy as np
import pandas as pd


KEYS = ['case_id', 'side', 'panel']
MEASURES = ['next_margin', 'correct_first_logp', 'wrong_first_logp', 'sequence_margin', 'tail_margin']


def load_captures(output):
    records, trajectories, writes, changes = [], [], [], []
    for path in sorted((output / 'runs').glob('*.npz')):
        with np.load(path, allow_pickle=False) as saved:
            record = json.loads(str(saved['record']))
            records.append(record)
            identity = {key: record[key] for key in KEYS + ['variant', 'kind']}
            for field, target in [('trajectory', trajectories), ('writes', writes), ('changes', changes)]:
                target.extend(dict(identity, **row) for row in json.loads(str(saved[field])))
    return pd.DataFrame(records), pd.DataFrame(trajectories), pd.DataFrame(writes), pd.DataFrame(changes)


def effects_from_full(table):
    baseline = table[table.variant == 'full'][KEYS + MEASURES]
    result = table.merge(baseline, on=KEYS, suffixes=('', '_full'), validate='many_to_one')
    for field in MEASURES:
        result['delta_' + field] = result[field] - result[field + '_full']
    result['candidate_support_lost'] = -result.delta_next_margin
    result['correct_first_probability'] = np.exp(result.correct_first_logp)
    result['wrong_first_probability'] = np.exp(result.wrong_first_logp)
    result['candidate_order_flipped'] = (result.next_margin_full < 0) & (result.next_margin > 0)
    return result


def derived_contrasts(table):
    rows = []
    for key, group in table.groupby(KEYS, sort=False):
        lookup = group.set_index('variant')
        baseline = lookup.loc['full']
        for trial in group.itertuples():
            if trial.kind not in ('joint', 'restore', 'random', 'dose'):
                continue
            if any(parent not in lookup.index for parent in trial.parents):
                continue
            row = dict(zip(KEYS, key), variant=trial.variant, kind=trial.kind)
            for field in MEASURES:
                value = getattr(trial, field)
                if trial.kind == 'joint':
                    row['interaction_' + field] = value - sum(lookup.loc[p, field] for p in trial.parents) + baseline[field]
                else:
                    row['minus_parent_' + field] = value - lookup.loc[trial.parents[0], field]
            rows.append(row)
    return pd.DataFrame(rows)


def same_question_effects(table):
    columns = ['case_id', 'panel', 'variant']
    fields = ['delta_' + name for name in MEASURES]
    left = table[table.side == 'unsupported'][columns + fields]
    right = table[table.side == 'supported'][columns + fields]
    merged = left.merge(right, on=columns, suffixes=('_unsupported', '_supported'), validate='one_to_one')
    for field in fields:
        merged[field + '_difference'] = merged[field + '_unsupported'] - merged[field + '_supported']
    return merged


def plot_effects(table, trajectories, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = output / 'figures'
    folder.mkdir(exist_ok=True)
    selected = ['early_22', 'early_23', 'early_joint', 'late_history', 'early_and_history', 'history_hold_mlp31']
    for (case_id, panel), group in table.groupby(['case_id', 'panel']):
        figure, axis = plt.subplots(figsize=(10, 5))
        for offset, (side, values) in zip((-.18, .18), group.groupby('side')):
            values = values.set_index('variant').reindex(selected)
            axis.bar(np.arange(len(selected)) + offset, values.delta_next_margin, .34, label=side)
        axis.set_xticks(np.arange(len(selected)), selected, rotation=20, ha='right')
        axis.set_ylabel('Intervention minus full candidate log-odds')
        axis.set_title(case_id + ' / ' + panel)
        axis.axhline(0, linestyle='--', linewidth=.8)
        axis.legend()
        figure.tight_layout()
        figure.savefig(folder / f'{case_id}_{panel}_effects.png', dpi=160)
        plt.close(figure)
    for key, group in trajectories.groupby(KEYS):
        figure, axis = plt.subplots(figsize=(8, 4))
        for name in ('full', 'early_joint', 'early_and_history'):
            rows = group[(group.variant == name) & (group.site == 'after_mlp')].sort_values('layer')
            axis.plot(rows.layer, rows.margin, label=name)
        axis.axhline(0, linestyle='--', linewidth=.8)
        axis.set(xlabel='Original LLM layer', ylabel='Local lens candidate margin', title='/'.join(key))
        axis.legend()
        figure.tight_layout()
        figure.savefig(folder / ('_'.join(key) + '_trajectory.png'), dpi=160)
        plt.close(figure)


def route_gallery(output):
    panels = json.loads((output / 'panels.json').read_text())
    source = pd.read_csv(output / 'source_tokens.csv.gz', keep_default_na=False)
    page = ['<!doctype html><meta charset="utf-8"><h1>固定头的逐来源读取轨迹</h1>',
            '<p>每行仅展示前三个来源；NPZ保留全部来源。这里观察地址持续性，不等同于语义锁定。</p>']
    for panel in panels:
        mask = np.logical_and.reduce([source[key].eq(panel[key]) for key in KEYS])
        tokens = source[mask].set_index('source')
        stem = '_'.join(panel[key] for key in KEYS)
        with np.load(output / 'routes' / (stem + '.npz'), allow_pickle=False) as saved:
            for field in sorted(saved.files):
                if not field.endswith('_attention'):
                    continue
                key = field.removesuffix('_attention')
                rows = []
                for query, weights in zip(saved[key + '_queries'], saved[field]):
                    for index in np.argsort(-weights)[:3]:
                        rows.append(dict(query=int(query), source=int(index), self_key=bool(query == index),
                            attention=float(weights[index]), text=tokens.loc[index, 'token_text']))
                page.append('<details><summary>' + html.escape(stem + '/' + key) + '</summary>')
                page.append(pd.DataFrame(rows).to_html(index=False, escape=True) + '</details>')
    (output / 'source_routes.html').write_text('\n'.join(page), encoding='utf-8')


def report_focused(output):
    output = Path(output)
    table, trajectories, writes, changes = load_captures(output)
    effects = effects_from_full(table)
    effects.to_csv(output / 'effects.csv', index=False)
    derived_contrasts(table).to_csv(output / 'joint_and_restoration.csv', index=False)
    same_question_effects(effects).to_csv(output / 'same_question_effects.csv', index=False)
    trajectories.to_csv(output / 'trajectories.csv.gz', index=False)
    writes.to_csv(output / 'head_source_writes.csv.gz', index=False)
    changes.to_csv(output / 'intervention_sizes.csv', index=False)
    config = json.loads((output / 'focused_config.json').read_text())
    # Two headwear panels and one onion panel, each with supported/unsupported prefixes.
    expected_panels = sum(4 if case['case_id'] == '14315_headwear_scope' else 2 for case in config['cases'])
    expected = expected_panels * len(config['schedule'])
    status = dict(completed_runs=len(table), expected_runs=expected, complete=len(table) == expected,
        max_first_logp_identity_error=float(abs(table.next_margin - table.correct_first_logp + table.wrong_first_logp).max()),
        first_metric='same-prefix readout', natural_cases=len(config['cases']), semantic_causation_proven=False)
    (output / 'status.json').write_text(json.dumps(status, indent=2))
    plot_effects(effects, trajectories, output)
    route_gallery(output)
    write_notes(output)
    archive_path = output / 'path_conflict_review.tar.gz'
    with tarfile.open(archive_path.with_suffix('.partial'), 'w:gz') as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name not in (archive_path.name, archive_path.with_suffix('.partial').name):
                archive.add(path, arcname=path.relative_to(output))
    archive_path.with_suffix('.partial').replace(archive_path)
    print(status, flush=True)
    print(effects[effects.variant.isin(['full', 'early_joint', 'late_history', 'history_hold_mlp31'])]
          [KEYS + ['variant', 'delta_next_margin', 'delta_correct_first_logp', 'delta_tail_margin']].to_string(index=False))


def write_notes(output):
    text = '''# 定点证据路径实验

每个next_margin与两个first_logp来自同一前缀、同一次前向；输出层用FP32乘法，内部仍为记录的dtype。
先看status和replay，确认完成范围和数值误差。每种候选面板单独比较，不能合并成多条独立自然样本。

- candidate_support_lost = full margin - cut margin：正值是被删写入支持候选C胜过W。
- local_lens_support只读中间状态；两者不相加，不把局部镜头当最终因果效果。
- scope与supported_value分别切断，测试限定语所在位置与直接候选描述所在位置。
- query_self / recent_history / remote_history互斥，历史不会再混入最终query的self写入。
- early_joint与两个单头切断构成2×2对照；interaction是联合剩余项，不是机制百分比。
- restore把同一前缀的下游head/MLP恢复为未切断值；没有在自然正误前缀间偷换状态。
- history_hold_mlp31固定最后MLP输出，区分历史直接写入与MLP对改变输入的反应。
- parallel_singular面板共同强制输入a，测cap/headdress的分叉；它是词面控制，不是新自然采样。
- tail_margin只统计首个分叉词之后的续写影响，防止首词效应冒充整段恢复。
- 原词表KL、两候选绝对概率和随机同范数扰动一并保留，不只看相对差改善。

这些是来源位置上的路径干预：V本身已有上下文，删scope位置的边不代表抹除了所有约束语义。
两自然前缀不同，候选词与长度不同；局部标签不保证整答正确。相邻head控制未匹配语义。
L22H28/L23H6/L31H14/L31H21来自此前同批探索，不能称独立确认或无监督检测。
'''
    (output / 'REPORT_zh.md').write_text(text, encoding='utf-8')
