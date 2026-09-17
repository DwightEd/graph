"""Report numerical validity, signed contributions, and fixed-pair interventions."""

from pathlib import Path
import html
import tarfile

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from .data import read_json, write_json
from .evaluate import source_interval
from .lockin_report import regions


def token_table(capture, pair):
    count = pair['length']
    difference = capture['error_x'] - capture['normal_x']
    logits = capture['logits']
    score = capture['baseline_score']
    threshold = float(capture['threshold'])
    low, high = capture['tail_cutoffs']
    data = dict(id=str(pair['id']), source_id=str(pair['source_id']), pair_start=pair['error_start'],
        normal_start=pair['normal_start'], length=count, offset=np.arange(count),
        error_token=capture['tokens'][0], normal_token=capture['tokens'][1],
        error_text=capture['text'][0], normal_text=capture['text'][1],
        error_score=score[0], normal_score=score[1], threshold=threshold,
        error_hit=score[0] > threshold, normal_false_alarm=score[1] > threshold,
        error_high=score[0] > high, error_low=score[0] < low,
        normal_high=score[1] > high, normal_low=score[1] < low,
        logit_gap=logits[0] - logits[1], attribution_sum=capture['attribution'].sum(axis=-1),
        converged=capture['converged'], quadrature_points=capture['points'],
        completeness_residual=capture['completeness_residual'], allocation_change=capture['allocation_change'],
        error_local_prediction=(capture['beta'][0] * difference).sum(axis=-1),
        normal_local_prediction=(capture['beta'][1] * difference).sum(axis=-1),
        error_local_intercept=capture['intercept'][0], normal_local_intercept=capture['intercept'][1],
        gate_flip_fraction=(capture['gates'][0] != capture['gates'][1]).mean(axis=-1),
        readout_gap_sum=(capture['contributions'][0] - capture['contributions'][1]).sum(axis=-1))
    return pd.DataFrame(data)


def intervention_rows(capture, pair):
    """Per-token two-sided effects; never confuse score margin with probability."""
    base_error, base_normal = capture['logits']
    gap = base_error - base_normal
    change = capture['error_x'] - capture['normal_x']
    rows = []
    families = [('layer', capture['layer_removed'], capture['layer_added']),
                ('selected', capture['selected_removed'], capture['selected_added'])]
    for family, removed, added in families:
        for index in range(len(removed)):
            name = f'LLM_layer_{index}' if family == 'layer' else ('IG_top' if index == 0 else f'random_{index-1}')
            mask = None
            if family == 'selected':
                mask = capture['selected_mask'] if index == 0 else capture['random_masks'][index-1]
            for offset in range(pair['length']):
                row = dict(id=str(pair['id']), source_id=str(pair['source_id']), pair_start=pair['error_start'],
                    normal_start=pair['normal_start'], length=pair['length'], offset=offset,
                    family=family, experiment=name, llm_layer=index if family == 'layer' else -1,
                    base_error=base_error[offset], base_normal=base_normal[offset],
                    removed_error=removed[index, offset], added_normal=added[index, offset],
                    removal_effect=base_error[offset]-removed[index, offset],
                    insertion_effect=added[index, offset]-base_normal[offset],
                    interaction=base_error[offset]-removed[index, offset]-added[index, offset]+base_normal[offset],
                    baseline_gap=gap[offset], converged=bool(capture['converged'][offset]),
                    threshold=float(capture['threshold']))
                if mask is not None:
                    row.update(selected_count=int(mask[offset].sum()),
                        overlap_with_IG=int((mask[offset] & capture['selected_mask'][offset]).sum()),
                        exchangeable_channels=int(capture['exchangeable'][offset]),
                        changed_l1=float(abs(change[offset, mask[offset]]).sum()),
                        predicted_contribution=float(capture['attribution'][offset, mask[offset]].sum()))
                rows.append(row)
    return pd.DataFrame(rows)


def pair_metrics(rows):
    """Contrast-only diagnostics. 'Added' uses normal background on both sides."""
    records = []
    keys = ['id', 'source_id', 'pair_start', 'family', 'experiment', 'llm_layer']
    for identity, group in rows.groupby(keys, sort=False):
        group = group.sort_values('offset')
        masks = regions(int(group.length.iloc[0]))
        for region, mask in masks.items():
            subset = group[mask[group.offset.to_numpy()]]
            if identity[3] == 'selected':
                subset = subset[subset.converged]
            if subset.empty:
                continue
            labels = np.r_[np.ones(len(subset)), np.zeros(len(subset))]
            base = roc_auc_score(labels, np.r_[subset.base_error, subset.base_normal])
            removed = roc_auc_score(labels, np.r_[subset.removed_error, subset.base_normal])
            added = roc_auc_score(labels, np.r_[subset.added_normal, subset.base_normal])
            row = dict(zip(keys, identity), region=region, tokens=len(subset), base_auc=base,
                removed_auc=removed, added_auc=added, removed_auc_delta=removed-base,
                baseline_gap=subset.baseline_gap.mean(), removal_effect=subset.removal_effect.mean(),
                insertion_effect=subset.insertion_effect.mean(), interaction=subset.interaction.mean())
            records.append(row)
    return pd.DataFrame(records)


def summarize_interventions(frame, bootstrap):
    records = []
    if frame.empty:
        return pd.DataFrame()
    fields = ['base_auc', 'removed_auc_delta', 'added_auc', 'baseline_gap',
              'removal_effect', 'insertion_effect', 'interaction']
    for (family, name, layer, region), group in frame.groupby(['family', 'experiment', 'llm_layer', 'region']):
        row = dict(family=family, experiment=name, llm_layer=layer, region=region, pairs=len(group))
        for field in fields:
            interval = source_interval(group, field, bootstrap)
            row.update({field + '_' + key: value for key, value in interval.items()})
        records.append(row)
    return pd.DataFrame(records)


def source_contrast(values, table):
    """Equal-token means inside pair, equal-pair inside source, equal-source outside."""
    pair_values, pair_sources = [], []
    for (source, _, _), index in table.groupby(['source_id', 'id', 'pair_start']).indices.items():
        pair_values.append(values[index].mean(axis=0))
        pair_sources.append(source)
    pair_values, pair_sources = np.asarray(pair_values), np.asarray(pair_sources)
    return np.stack([pair_values[pair_sources == source].mean(axis=0) for source in np.unique(pair_sources)])


def channel_summary(values, table, heads, bootstrap):
    """Signed head contributions; all heads retained. Selected tails are descriptive."""
    output = []
    fraction = (table.offset + .5) / table.length
    cohorts = dict(all=np.ones(len(table), bool), front=fraction < .5, back=fraction >= .5,
        high_error=table.error_high, low_error=table.error_low,
        high_normal=table.normal_high, low_normal=table.normal_low,
        error_TP=table.error_hit, error_FN=~table.error_hit)
    for name, mask in cohorts.items():
        mask = np.asarray(mask) & table.converged.to_numpy()
        if not mask.any():
            continue
        subset = table.loc[mask].reset_index(drop=True)
        source_values = source_contrast(values[mask], subset)
        mean = source_values.mean(axis=0)
        low, high = np.full_like(mean, np.nan), np.full_like(mean, np.nan)
        if bootstrap and len(source_values) > 1:
            rng = np.random.default_rng(42)
            draws = [source_values[rng.integers(len(source_values), size=len(source_values))].mean(axis=0) for _ in range(bootstrap)]
            low, high = np.quantile(draws, [.025, .975], axis=0)
        for channel in range(values.shape[1]):
            output.append(dict(cohort=name, llm_layer=channel//heads, llm_head=channel%heads,
                tokens=int(mask.sum()), sources=len(source_values), contribution=mean[channel], low=low[channel], high=high[channel]))
    return pd.DataFrame(output)


def random_comparison(effects, bootstrap):
    selected = effects[(effects.family == 'selected') & effects.converged]
    keys = ['id', 'source_id', 'pair_start', 'offset']
    selected = selected.copy()
    selected['abs_removal'] = abs(selected.removal_effect)
    selected['abs_insertion'] = abs(selected.insertion_effect)
    fields = ['removal_effect', 'insertion_effect', 'changed_l1', 'overlap_with_IG', 'abs_removal', 'abs_insertion']
    target = selected[selected.experiment == 'IG_top']
    random = selected[selected.experiment.str.startswith('random_')].groupby(keys)[fields].mean().reset_index()
    joined = target.merge(random, on=keys, suffixes=('_IG', '_random'), validate='one_to_one')
    joined['removal_advantage'] = joined.removal_effect_IG - joined.removal_effect_random
    joined['insertion_advantage'] = joined.insertion_effect_IG - joined.insertion_effect_random
    joined['abs_removal_advantage'] = joined.abs_removal_IG - joined.abs_removal_random
    joined['abs_insertion_advantage'] = joined.abs_insertion_IG - joined.abs_insertion_random
    rows = []
    comparison_fields = ['removal_advantage', 'insertion_advantage', 'abs_removal_advantage', 'abs_insertion_advantage']
    for scope, mask in [('all', np.ones(len(joined), bool)), ('exchangeable', joined.exchangeable_channels > 0)]:
        group = joined[mask]
        pair = group.groupby(['source_id', 'id', 'pair_start'])[comparison_fields].mean().reset_index()
        row = dict(scope=scope, tokens=len(group), pairs=len(pair))
        for field in comparison_fields:
            row.update({field+'_'+key: value for key, value in source_interval(pair, field, bootstrap).items()})
        rows.append(row)
    return joined, pd.DataFrame(rows)


def plots(channels, effects, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = output / 'figures'
    directory.mkdir(exist_ok=True)
    if not channels.empty:
        for cohort in ('all', 'back', 'high_error', 'high_normal'):
            rows = channels[channels.cohort == cohort]
            if rows.empty:
                continue
            matrix = rows.pivot(index='llm_layer', columns='llm_head', values='contribution')
            figure, axis = plt.subplots(figsize=(8, 5))
            scale = max(float(abs(matrix.to_numpy()).max()), 1e-8)
            image = axis.imshow(matrix, aspect='auto', vmin=-scale, vmax=scale)
            figure.colorbar(image, ax=axis, label='Error - normal logit contribution')
            axis.set(xlabel='Original LLM head', ylabel='Original LLM layer',
                     title=cohort + ': paired input contributions (numerically accepted only)')
            figure.tight_layout()
            figure.savefig(directory / f'channels_{cohort}.png', dpi=160)
            plt.close(figure)
    if effects.empty:
        return
    rows = effects[(effects.family == 'layer') & (effects.region == 'all')].sort_values('llm_layer')
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(rows.llm_layer, rows.removal_effect_mean, marker='.', label='Remove layer difference from error')
    axis.plot(rows.llm_layer, rows.insertion_effect_mean, marker='.', label='Insert layer difference into normal')
    axis.axhline(0, linestyle='--', linewidth=.8)
    axis.set(xlabel='Original LLM layer', ylabel='Signed logit effect',
             title='Frozen paired contrast: NOT a trained single-layer detector')
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / 'layer_contrast.png', dpi=160)
    plt.close(figure)


def write_explanation(table, channels, status, output):
    text = ['# 节点模型白盒验证', '', f"已完成 {status['completed_pairs']}/{status['expected_pairs']} 对。",
        f"数值积分通过 {status['converged_tokens']}/{status['tokens']} 个配对位置。未通过的不进入通道汇总；原分数仍完整保留。", '',
        '1. tokens.csv.gz：查看真实logit差、积分误差、局部线性预测误差、ReLU开关差异。',
        '2. channel_contributions.csv.gz：全部L/H的有符号贡献；高低分子组仅为描述，不能当独立检测验证。',
        '3. intervention_summary.csv：删除某层差异、只引入该层差异，以及两者的交互。',
        '4. random_comparison.csv：归因选中通道比相同层/输入变化幅度组的随机通道多影响多少。',
        '5. readout_contributions.csv.gz：最后读出单元的精确贡献差；其总和等于logit差。', '',
        '**单层对照只隔离配对输入差异；背景取自已知正常对照。它不能证明可部署的单层检测器足够。**',
        '没有Q/K/V或证据角色干预，不能给通道赋予事实语义，也不能判定原LLM发生self lock-in。',
        '下一阶段应以这里的全通道结果生成独立来源的QK/OV对照；不在本test上挑头后冒称独立确认。',
        '完整路径数值可靠仍不保证语义因果解释：直线路径/混合向量可能不在自然输入分布上。']
    (output / 'REPORT_zh.md').write_text('\n'.join(text), encoding='utf-8')
    page = ['<!doctype html><meta charset="utf-8"><h1>全部配对 token 的模型内解释</h1>',
            '<p>按身份/位置排序，不按高分挑例子。局部系数和完整路径贡献在captures中逐头保存。</p>']
    columns = ['offset', 'error_text', 'normal_text', 'error_score', 'normal_score', 'logit_gap',
               'gate_flip_fraction', 'completeness_residual', 'converged']
    for (identity, start), group in table.groupby(['id', 'pair_start'], sort=True):
        page.append(f'<details><summary>{html.escape(str(identity))} / {start}</summary>')
        page.append(group[columns].to_html(index=False, escape=True))
        page.append('</details>')
    (output / 'gallery.html').write_text('\n'.join(page), encoding='utf-8')


def report(output, bootstrap):
    """Re-report from completed per-pair captures without loading a checkpoint."""
    manifest = read_json(output / 'manifest.json')
    tables, interventions, attributions, units = [], [], [], []
    heads = read_json(output / 'geometry.json')['heads']
    completed = 0
    for pair in manifest['pairs']:
        path = output / 'captures' / pair['file']
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as saved:
            capture = {key: saved[key] for key in saved.files}
        table = token_table(capture, pair)
        tables.append(table)
        interventions.append(intervention_rows(capture, pair))
        attributions.append(capture['attribution'])
        delta = capture['contributions'][0] - capture['contributions'][1]
        for unit in range(delta.shape[1]):
            units.extend(dict(id=str(pair['id']), source_id=str(pair['source_id']), pair_start=pair['error_start'],
                offset=offset, unit=unit, contribution=delta[offset, unit]) for offset in range(pair['length']))
        completed += 1
    if not tables:
        raise ValueError('No completed whitebox captures')
    table = pd.concat(tables, ignore_index=True)
    effects = pd.concat(interventions, ignore_index=True)
    channels = channel_summary(np.concatenate(attributions), table, heads, bootstrap)
    pair = pair_metrics(effects)
    summary = summarize_interventions(pair, bootstrap)
    random_tokens, random = random_comparison(effects, bootstrap)
    outputs = dict(tokens=table, token_interventions=effects, pair_interventions=pair,
        channel_contributions=channels, intervention_summary=summary, random_tokens=random_tokens,
        random_comparison=random, readout_contributions=pd.DataFrame(units))
    for name, frame in outputs.items():
        suffix = '.csv.gz' if name in ('tokens', 'token_interventions', 'channel_contributions', 'readout_contributions') else '.csv'
        frame.to_csv(output / (name + suffix), index=False)
    status = dict(completed_pairs=completed, expected_pairs=len(manifest['pairs']), tokens=len(table),
        converged_tokens=int(table.converged.sum()),
        max_abs_completeness_error=float(table.completeness_residual.abs().max()),
        complete=completed == len(manifest['pairs']), native_llm_intervention=False, model='node_only')
    write_json(output / 'status.json', status)
    plots(channels, summary, output)
    write_explanation(table, channels, status, output)
    with tarfile.open(output / 'whitebox_review.tar.gz', 'w:gz') as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and 'captures' not in path.parts and path != output / 'whitebox_review.tar.gz':
                archive.add(path, arcname=path.relative_to(output))
    print(read_json(output / 'status.json'), flush=True)
