"""Exact head rules, matched normal controls, previous-state controls and compact reports."""

from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from .data import read_json, write_json
from .evaluate import metrics, source_interval


def save_channel_rules(stats, full, terms, test_data, pairs, output):
    values, table, _, geometry = test_data
    table = table.copy()
    table['row'] = np.arange(len(table))
    heads = geometry[1]
    channels = np.arange(len(full['weight']))
    difference = stats['means'][1] - stats['means'][0]
    rules = pd.DataFrame(dict(channel=channels, layer=channels // heads, head=channels % heads,
        normal_mean=stats['means'][0], error_mean=stats['means'][1], mean_gap=difference,
        variance=np.diag(stats['covariance']), weight=full['weight'],
        fit_gap_contribution=difference * full['weight'], **terms))
    rules.to_csv(output / 'head_rules.csv', index=False)
    links = covariance_links(stats, full['weight'], heads)
    links.to_csv(output / 'covariance_links.csv.gz', index=False)
    captures = []
    for pair in pairs:
        error, normal = pair_indices(table, pair)
        length = len(error)
        regions = dict(all=np.ones(length, bool), front=(np.arange(length) + .5) / length < .5,
                       back=(np.arange(length) + .5) / length >= .5)
        for region, mask in regions.items():
            if not mask.any():
                continue
            error_mean = values[error[mask]].mean(axis=0)
            normal_mean = values[normal[mask]].mean(axis=0)
            delta = error_mean - normal_mean
            captures.append(pd.DataFrame(dict(id=str(pair['id']), source_id=str(pair['source_id']),
                start=pair['error_start'], region=region, channel=channels,
                error_mean=error_mean, normal_mean=normal_mean,
                input_gap=delta, score_contribution=delta * full['weight'])))
    columns = ['id', 'source_id', 'start', 'region', 'channel', 'error_mean', 'normal_mean', 'input_gap', 'score_contribution']
    result = pd.concat(captures, ignore_index=True) if captures else pd.DataFrame(columns=columns)
    result.to_csv(output / 'pair_head_contributions.csv.gz', index=False)


def covariance_links(stats, weight, heads, count=3):
    """FIT-selected strongest terms in the exact coefficient equation; not causal edges."""
    covariance = stats['covariance']
    rows = []
    for target in range(len(weight)):
        influence = -covariance[target] * weight / covariance[target, target]
        influence[target] = 0
        partners = np.argsort(-abs(influence), kind='stable')[:min(count, len(weight) - 1)]
        for source in partners:
            if source == target:
                continue
            rows.append(dict(target_layer=target // heads, target_head=target % heads,
                source_layer=int(source // heads), source_head=int(source % heads),
                covariance=covariance[target, source], coefficient_adjustment=influence[source]))
    return pd.DataFrame(rows)


def pair_indices(table, pair):
    group = table[table.id == str(pair['id'])].set_index('token')
    offsets = np.arange(pair['length'])
    # These are frozen gold diagnostics, never model inputs or detected boundaries.
    error = group.loc[pair['error_start'] + offsets]
    normal = group.loc[pair['normal_start'] + offsets]
    assert error.gold.eq(1).all() and normal.gold.eq(0).all(), 'Locked pair labels changed'
    assert error.source_id.eq(str(pair['source_id'])).all(), 'Locked pair source changed'
    return error.row.to_numpy(), normal.row.to_numpy()


def score_populations(table):
    normal = table.gold == 0
    return dict(all=np.ones(len(table), bool),
        answer_first=normal | (table.role == 'answer_first'),
        continuation=normal | (table.role == 'continuation'),
        previous_normal=table.previous_gold == 0,
        previous_error=table.previous_gold == 1)


def evaluate_scores(table, calibration, names):
    pooled, within, thresholds = [], [], {}
    for name in names:
        negatives = calibration.loc[calibration.gold.eq(0) & calibration.text_valid, name].dropna()
        thresholds[name] = float(np.quantile(negatives, .95, method='higher'))
        for population, mask in score_populations(table).items():
            group = table.loc[mask & np.isfinite(table[name])]
            pooled.append(dict(model=name, population=population, **metrics(group.gold, group[name], thresholds[name])))
            for identity, answer in group.groupby('id', sort=False):
                measured = metrics(answer.gold, answer[name], thresholds[name])
                within.append(dict(model=name, population=population, id=identity,
                                   source_id=str(answer.source_id.iloc[0]), **measured))
    return pd.DataFrame(pooled), pd.DataFrame(within), thresholds


def matched_scores(table, pairs, names, thresholds):
    rows = []
    for pair in pairs:
        error_index, normal_index = pair_indices(table, pair)
        length = len(error_index)
        fractions = (np.arange(length) + .5) / length
        for region, mask in dict(all=np.ones(length, bool), front=fractions < .5, back=fractions >= .5).items():
            for name in names:
                error = table.loc[error_index[mask], name].to_numpy()
                normal = table.loc[normal_index[mask], name].to_numpy()
                valid = np.isfinite(error) & np.isfinite(normal)
                if not valid.any():
                    continue
                error, normal = error[valid], normal[valid]
                measured = metrics(np.r_[np.ones(len(error)), np.zeros(len(normal))],
                                   np.r_[error, normal], thresholds[name])
                rows.append(dict(model=name, region=region, id=str(pair['id']),
                    source_id=str(pair['source_id']), start=pair['error_start'], tokens_each=len(error),
                    error_mean=error.mean(), normal_mean=normal.mean(), gap=(error - normal).mean(),
                    **measured))
    return pd.DataFrame(rows)


def summarize_groups(frame, keys, bootstrap):
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for key, group in frame.groupby(keys, sort=False):
        valid = group.dropna(subset=['auroc'])
        interval = source_interval(valid, 'auroc', bootstrap)
        rows.append(dict(zip(keys, key), units=len(valid), **interval))
    return pd.DataFrame(rows)


def conditional_ranking(table, names, bootstrap):
    """Within-answer comparisons with identical observed nuisance strata; disclose support."""
    rows = []
    conditions = dict(previous=['previous_gold'],
                      context=['past_run_bin', 'position_bin', 'surface'])
    for name in names:
        for condition, columns in conditions.items():
            cells = []
            for _, group in table.dropna(subset=[name]).groupby(['id', *columns], sort=False):
                result = metrics(group.gold, group[name], np.inf)
                if result['auroc'] is not None:
                    cells.append(dict(source_id=str(group.source_id.iloc[0]), id=group.id.iloc[0], **result))
            cells = pd.DataFrame(cells)
            if cells.empty:
                rows.append(dict(model=name, condition=condition, mixed_cells=0, tokens=0))
                continue
            weights = cells.positives * cells.negatives
            cells['weighted_auc'] = weights * cells.auroc
            cells['pair_count'] = weights
            answers = cells.groupby(['source_id', 'id']).agg(numerator=('weighted_auc', 'sum'),
                denominator=('pair_count', 'sum'))
            answers['auroc'] = answers.numerator / answers.denominator
            interval = source_interval(answers.reset_index(), 'auroc', bootstrap)
            rows.append(dict(model=name, condition=condition, mixed_cells=len(cells),
                tokens=int(cells.tokens.sum()), positives=int(cells.positives.sum()),
                negatives=int(cells.negatives.sum()), **interval))
    return pd.DataFrame(rows)


def paired_differences(frame, bootstrap):
    """Paired source intervals, including full-minus-covariance controls on identical pairs."""
    if frame.empty:
        return pd.DataFrame()
    keys = ['id', 'source_id', 'start', 'region', 'tokens_each']
    base = frame[frame.model == 'full'][keys + ['auroc', 'gap']]
    rows = []
    for name, group in frame.groupby('model', sort=False):
        common = group.merge(base, on=keys, suffixes=('', '_full'), validate='one_to_one')
        common['auc_delta'] = common.auroc - common.auroc_full
        common['gap_delta'] = common.gap - common.gap_full
        for region, selected in common.groupby('region'):
            row = dict(model=name, region=region, pairs=len(selected))
            for field in ('auc_delta', 'gap_delta'):
                row.update({field + '_' + k: v for k, v in source_interval(selected, field, bootstrap).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def history_table(table):
    """All four transition cells, including correct tokens immediately after errors."""
    rows = []
    for (previous, current), group in table[table.previous_gold >= 0].groupby(['previous_gold', 'gold']):
        for field in ('full', 'history_mean', 'current_increment'):
            source = group.groupby('source_id')[field].mean()
            rows.append(dict(previous_gold=int(previous), current_gold=int(current),
                             field=field, tokens=len(group), sources=len(source), mean=source.mean()))
    return pd.DataFrame(rows)


def draw_figures(pooled, paired, rules, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = output / 'figures'
    directory.mkdir(exist_ok=True)
    names = ['raw_mean', 'diagonal', 'layer_block', 'full', 'layer_mean', 'head_contrast', 'unordered_heads']
    values = pooled[pooled.population == 'all'].set_index('model').loc[names]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(names, values.auroc)
    axis.set(ylim=(0, 1), ylabel='Full-stream AUROC', title='Which head information does LDA use?')
    axis.tick_params(axis='x', rotation=20)
    figure.tight_layout()
    figure.savefig(directory / 'covariance_and_identity.png', dpi=160)
    plt.close(figure)
    for field in ('weight', 'direct', 'same_layer', 'cross_layer'):
        image = rules.pivot(index='layer', columns='head', values=field)
        figure, axis = plt.subplots(figsize=(8, 5))
        plot = axis.imshow(image, aspect='auto')
        axis.set(xlabel='LLM head', ylabel='LLM layer', title=field + ' (raw input units)')
        figure.colorbar(plot, ax=axis)
        figure.tight_layout()
        figure.savefig(directory / (field + '.png'), dpi=160)
        plt.close(figure)


def report(output, bootstrap=2000):
    output = Path(output)
    protocol = read_json(output / 'protocol.json')
    types = dict(id=str, source_id=str)
    table = pd.read_csv(output / 'test_scores.csv.gz', dtype=types, keep_default_na=False)
    calibration = pd.read_csv(output / 'calibration_scores.csv.gz', dtype=types, keep_default_na=False)
    names = protocol['score_columns']
    for frame in (table, calibration):
        frame[names] = frame[names].replace('', np.nan).astype(float)
    table['row'] = np.arange(len(table))
    pairs = read_json(output / 'pairs.json')
    pooled, within, thresholds = evaluate_scores(table, calibration, names)
    paired = matched_scores(table, pairs, names, thresholds)
    outputs = dict(metrics=pooled, within_answer=within, matched_pairs=paired,
        within_summary=summarize_groups(within, ['model', 'population'], bootstrap),
        matched_summary=summarize_groups(paired, ['model', 'region'], bootstrap),
        paired_deltas=paired_differences(paired, bootstrap),
        conditional_ranking=conditional_ranking(table, names, bootstrap), history_decomposition=history_table(table))
    for name, frame in outputs.items():
        frame.to_csv(output / (name + '.csv'), index=False)
    write_json(output / 'thresholds.json', thresholds)
    rules = pd.read_csv(output / 'head_rules.csv')
    source_channel_summary(output, rules, bootstrap)
    draw_figures(pooled, paired, rules, output)
    time_summary = outputs['matched_summary']
    if not time_summary.empty:
        history_plot(time_summary[time_summary.model.isin(['full', 'history_mean', 'current_increment'])], output)
    write_findings(pooled, rules, protocol, output)
    with tarfile.open(output / 'lda_review.tar.gz', 'w:gz') as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and 'parameters' not in path.relative_to(output).parts and path.suffix != '.npz':
                if path != output / 'lda_review.tar.gz':
                    archive.add(path, arcname=path.relative_to(output))
    print(pooled[pooled.population == 'all'][['model', 'auroc', 'ap', 'recall', 'fpr']].to_string(index=False))
    print('LDA results:', output, flush=True)


def write_findings(pooled, rules, protocol, output):
    lines = ['# LDA具体判别规则审计', '',
        '先核对baseline_replay.json；旧模型和本轮full使用相同ridge时应接近数值误差。',
        'covariance_and_identity.png：diagonal→layer_block→full分别加入同层及跨层协方差。',
        'layer_mean只看每层整体自关注；head_contrast只看层内相对差；unordered_heads保留数值列表但去掉head编号。',
        'matched_support与balanced_context使用完全相同训练token；后者在位置/词面/过去错误长度组中平衡类别。',
        '历史长度使用金标，只作诊断分层/训练重权，不是检测输入。分组缺少一类的训练词会排除，覆盖见fit_context_balance.csv。',
        'history_mean仅用过去10词（或指定窗口），current_increment=当前分数−过去均值，不能把两项AUROC相加。',
        'head_rules中的direct+same_layer+cross_layer精确等于weight。协方差项反映统计补偿，不是Transformer的因果边。',
        'pair_head_contributions是w*(错误x−正常x)，求和精确重建同一配对的logit差；不需要近似归因。',
        'conditional_ranking只比较同回答、同条件且两类都存在的单元，不能把低覆盖当作没有信号。',
        '每个评分在calibration正常词上单独取5%FPR阈值；TEST实际FPR单列。各分量重新校准，非原模型概率。',
        '全流程是监督诊断；仍不能用自然标签找到的方向直接宣称无监督方法。', '',
        '## 训练中贡献最大的通道（由FIT确定，全通道另存）', '']
    selected = rules.loc[rules.fit_gap_contribution.abs().sort_values(ascending=False).index[:10]]
    lines.append(selected[['layer', 'head', 'mean_gap', 'weight', 'direct', 'same_layer', 'cross_layer']].to_string(index=False))
    if protocol['prompt']:
        lines.extend(['', 'prompt_only使用保留prompt权重；self_after_prompt_regression只减去FIT线性可预测部分。',
            'prompt_predicted_full与full_unexplained_by_prompt精确相加为full；残差有效只排除线性prompt解释。',
            'self_plus_prompt测试联合预测增量。prompt总量不是适用证据，截断丢失和非线性混杂未被消除。'])
    else:
        lines.extend(['', '本轮未读取prompt边；不能从self向量推断减少的质量流向何处。使用--lda-prompt单独运行。'])
    (output / 'REPORT_zh.md').write_text('\n'.join(lines), encoding='utf-8')


def report_saved_scores(table, pairs, output, bootstrap):
    """Reproduce history-vs-current rankings on identical observable test positions."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = ['full', 'history_mean', 'current_increment']
    common = table.dropna(subset=names)
    rows = []
    groups = dict(common_all=common, previous_normal=common[common.previous_gold == 0],
                  previous_error=common[common.previous_gold == 1])
    for population, group in groups.items():
        for name in names:
            result = metrics(group.gold, group[name], np.inf)
            rows.append(dict(model=name, population=population,
                **{key: result[key] for key in ('tokens', 'positives', 'negatives', 'auroc', 'ap')}))
    pooled = pd.DataFrame(rows)
    paired = matched_scores(table, pairs, names, dict.fromkeys(names, np.inf))
    if not paired.empty:
        paired = paired.drop(columns=['tp', 'fp', 'fn', 'tn', 'recall', 'fpr'])
    conditional = conditional_ranking(common, names, bootstrap)
    summary = summarize_groups(paired, ['model', 'region'], bootstrap)
    for name, frame in [('metrics', pooled), ('matched_pairs', paired),
                        ('matched_summary', summary), ('conditional_ranking', conditional)]:
        frame.to_csv(output / (name + '.csv'), index=False)
    table.to_csv(output / 'tokens.csv.gz', index=False)
    write_json(output / 'protocol.json', dict(scope='Existing supervised LDA scores; no new fitting',
        common_tokens=len(common), excluded_first_tokens=len(table) - len(common),
        past_is_computed_from='past detector scores, never past labels',
        oracle='past gold only defines diagnostic strata, never the score',
        limitations='Predictable temporal scores do not prove the pointwise LDA computes temporal aggregation.'))
    history_plot(summary, output)
    with tarfile.open(output / 'lda_scores_review.tar.gz', 'w:gz') as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path != output / 'lda_scores_review.tar.gz':
                archive.add(path, arcname=path.relative_to(output))
    print(pooled.to_string(index=False), flush=True)


def history_plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name, group in summary.groupby('model', sort=False):
        group = group.set_index('region').reindex(['front', 'back'])
        axis.plot(['front', 'back'], group['mean'], marker='o', label=name)
    axis.set(ylim=(0, 1), ylabel='Within-pair AUROC, equal-source mean',
             title='Current score vs past-only score vs current increment')
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / 'history_vs_current.png', dpi=160)
    figure.savefig(output / 'history_vs_current.svg')
    plt.close(figure)


def source_channel_summary(output, rules, bootstrap):
    """All channels, fixed pairs and regions; source bootstrap of actual score contributions."""
    paired = pd.read_csv(output / 'pair_head_contributions.csv.gz', dtype={'id': str, 'source_id': str})
    rows = []
    for region, group in paired.groupby('region', sort=False):
        source = group.groupby(['source_id', 'channel']).score_contribution.mean().unstack()
        values = source.to_numpy()
        mean = values.mean(axis=0)
        low, high = np.full_like(mean, np.nan), np.full_like(mean, np.nan)
        if bootstrap and len(values) > 1:
            generator = np.random.default_rng(42)
            counts = generator.multinomial(len(values), np.ones(len(values)) / len(values), size=bootstrap)
            low, high = np.quantile(counts @ values / len(values), [.025, .975], axis=0)
        averages = group.groupby(['source_id', 'channel'])[['error_mean', 'normal_mean', 'input_gap']].mean()
        averages = averages.groupby('channel').mean().reindex(source.columns)
        for index, channel in enumerate(source.columns):
            rows.append(dict(region=region, channel=int(channel), sources=len(values),
                error_mean=averages.error_mean.iloc[index], normal_mean=averages.normal_mean.iloc[index],
                input_gap=averages.input_gap.iloc[index], contribution=mean[index], low=low[index], high=high[index]))
    if rows:
        frame = pd.DataFrame(rows).merge(rules[['channel', 'layer', 'head', 'weight']], on='channel')
    else:
        frame = pd.DataFrame(columns=['region', 'channel', 'sources', 'contribution'])
    frame.to_csv(output / 'channel_summary.csv', index=False)
