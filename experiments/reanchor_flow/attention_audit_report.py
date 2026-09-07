"""Stream sources through a finite audit matrix; save every head and head pair."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .attention_audit_stats import (
    HORIZONS, STATES, Moments, carrier_entry, incoming_reads, joint_tables,
    matched_difference, matched_joint_gap, mean, ratio, read_metrics, sample_contrasts, target_chain_tables,
)
from .attention_rhythm_report import save_json


def analyze_sample(path, horizons=HORIZONS, match_window=32, onset_radius=8):
    with np.load(path, allow_pickle=False) as stored:
        trace = dict(stored)
    with np.load(path.with_suffix('.labels.npz'), allow_pickle=False) as stored:
        labels = stored['labels']
    metrics = read_metrics(trace)
    reads = []
    with np.load(path.with_suffix('.history.npz'), allow_pickle=False) as history:
        for l in range(trace['ordinary_mass'].shape[0]):
            reads.append(incoming_reads(history[f'L{l}'], trace['ordinary_mass'][l], trace, horizons))
        fai = {k: np.stack([r[k] for r in reads]) for k in ('fai', 'enrichment', 'count')}
        full = reads[0]['full']
        # These metrics use carrier labels; their names explicitly distinguish
        # them from the predictor-coordinate reading metrics above.
        for w, (lo, hi) in enumerate(horizons):
            suffix = f'{lo}_{hi or "end"}'
            for name in ('fai', 'enrichment'):
                value = fai[name][..., w].copy()
                if hi:
                    value[..., ~full[:, w]] = np.nan
                metrics[f'carrier_{name}_{suffix}'] = value
        contrast = sample_contrasts(trace, labels, metrics, match_window, onset_radius)
        chain = target_chain_tables(trace, history, labels, contrast['pairs'], contrast['valid'])
    writes = np.stack([trace[k][..., :-1] for k in ('attention_margin', 'mlp_margin', 'rounding_margin')])
    contrast['writes_raw'] = np.stack([mean(writes[..., contrast['valid'] & (labels == y)], -1) for y in (0, 1)])
    pairs = contrast['pairs']
    contrast['writes_matched'] = mean(matched_difference(writes,pairs), -1)
    controls = np.stack((trace['predictor_entropy'][:-1], trace['predictor_logprob'][:-1],
                         np.arange(len(labels), dtype=float)))
    contrast['controls_raw'] = np.stack([mean(controls[..., contrast['valid'] & (labels == y)], -1) for y in (0, 1)])
    contrast['controls_matched'] = mean(matched_difference(controls,pairs), -1)
    from .attention_audit_plot import write_sample_text
    write_sample_text(path, trace, labels)
    entry, gain = carrier_entry(trace)
    # First finite horizon is the registered primary four-state comparison.
    reuse = np.where((fai['count'][..., 0] > 0) & full[:, 0],
                     fai['enrichment'][..., 0] >= 2, np.nan)
    joint = joint_tables(entry, reuse, labels, contrast['valid'], entry.shape[1])
    gap = matched_joint_gap(entry, reuse, contrast['pairs'], entry.shape[1])
    # Persist compact per-sample arrays. Large pair matrices are aggregated
    # immediately and do not create another sample x head x head archive.
    audit = {**fai, 'full_horizon': full, 'carrier_entry': entry.astype(np.float32),
             'carrier_evidence_gain': gain.astype(np.float32),
             'metric_names': np.array(list(metrics)), 'raw': contrast['raw'],
             'matched': contrast['matched'], 'onset': contrast['onset'], 'onset_raw': contrast['onset_raw'],
             'pairs': contrast['pairs'], 'onset_pairs': contrast['onset_pairs'],
             'horizons': np.array(horizons), 'labels': labels,
             'valid_target': contrast['valid']}
    np.savez_compressed(path.with_suffix('.audit.npz'), **audit)
    return contrast, {'joint_raw': joint, 'joint_matched': gap,
                      'chain_raw': chain['raw'], 'chain_matched': chain['matched'],
                      'chain_excess_matched': chain['excess_matched']}, list(metrics)


def save_moments(output, group, moments, names, layers, heads, horizons, onset_radius):
    folder = output / 'cohorts'
    folder.mkdir(exist_ok=True)
    values = {'metric_names': np.array(names), 'layers': np.array(layers), 'heads': np.array(heads),
              'horizons': np.array(horizons), 'states': np.array(STATES),
              'onset_offsets': np.arange(-onset_radius, onset_radius+1)}
    inference = {'matched', 'onset', 'joint_matched', 'chain_matched', 'chain_excess_matched',
                 'writes_matched', 'controls_matched'}
    for name, accumulator in moments.items():
        if accumulator.total is not None:
            for statistic, value in accumulator.finish(name in inference).items():
                values[name + '_' + statistic] = value
    joint=values['joint_raw_mean']
    values['joint_reuse_given_entry']=ratio(joint[:,3],joint[:,1]+joint[:,3]).astype(np.float32)
    values['joint_reuse_given_no_entry']=ratio(joint[:,2],joint[:,0]+joint[:,2]).astype(np.float32)
    path = folder / (group.replace('/', '_') + '.npz')
    np.savez_compressed(path, **values)
    return path


def summarize(output, manifest, *, horizons=HORIZONS, match_window=32, onset_radius=8,
              plots_per_class=2, cpu_threads=4):
    from threadpoolctl import threadpool_limits
    from .attention_audit_plot import plot_cohort, plot_sample, write_gallery

    output = Path(output)
    grouped = defaultdict(lambda: defaultdict(list))
    for e in manifest['samples']:
        grouped[e['split'] + '/' + e['task_type']][e['source_id']].append(e)
    report = {'schema': 3, 'labels_used_for_capture': False, 'labels_used_for_comparison': True,
              'detection_metrics_run': False, 'groups': {}, 'horizons': horizons,
              'matching': {'same_response': True, 'max_position_gap': match_window,
                           'same_token_class': True, 'primary': 'two-sided normal interpolation at the exact H position',
                           'secondary': 'nearest normal without replacement'},
              'inference': 'source-level Student intervals; BY FDR within each cohort/contrast across every reported metric/head or pair',
              'not_established': ['claim-specific evidence relevance', 'causal factual mediation',
                                  'a validated hallucination detector']}
    all_moments, all_coverage, all_sources = {}, {}, {}
    rows_for_gallery, names = [], None
    with threadpool_limits(limits=cpu_threads):
        for group, sources in grouped.items():
            moments = defaultdict(Moments)
            coverage = dict(samples=0, sources=len(sources), normal_tokens=0, hallucinated_tokens=0,
                            unknown_tokens=0, special_targets=0, matched_pairs=0, matched_onsets=0,
                            negative_answers=0, positive_answers=0, unknown_answers=0)
            examples = {'negative': [], 'positive': []}
            progress = tqdm(sources.items(), desc=f'audit {group}', unit='source')
            for source, entries in progress:
                source_stats = defaultdict(Moments)
                for e in entries:
                    path = output / e['path']
                    progress.set_postfix_str(str(e['sample_id']))
                    contrast, matrices, current_names = analyze_sample(path, horizons, match_window, onset_radius)
                    if names is None:
                        names = current_names
                    if names != current_names:
                        raise ValueError('metric axes changed between samples')
                    layers, heads = contrast['matched'].shape[-2:]
                    for key in ('raw', 'matched', 'nearest_control_matched', 'onset', 'onset_raw', 'slope', 'writes_raw', 'writes_matched', 'controls_raw', 'controls_matched'):
                        source_stats[key].add(contrast[key])
                    source_stats['answer_' + contrast['answer_class']].add(contrast['answer'])
                    if contrast['answer_class'] == 'negative':
                        source_stats['negative_answer_slope'].add(contrast['slope'][0])
                    for key, value in matrices.items():
                        source_stats[key].add(value)
                    coverage['samples'] += 1
                    for key in ('normal_tokens', 'hallucinated_tokens', 'unknown_tokens', 'special_targets'):
                        coverage[key] += contrast[key]
                    coverage['matched_pairs'] += len(contrast['pairs'])
                    coverage['matched_onsets'] += len(contrast['onset_pairs'])
                    coverage[contrast['answer_class'] + '_answers'] += 1
                    display = dict(e, answer_class=contrast['answer_class'],
                                   hallucinated_tokens=contrast['hallucinated_tokens'],
                                   normal_tokens=contrast['normal_tokens'])
                    display['text'] = str(Path(e['path']).with_suffix('.review.html'))
                    rows_for_gallery.append(display)
                    if contrast['answer_class'] in examples:
                        options = examples[contrast['answer_class']]
                        if len(options) < plots_per_class:
                            options.append(display)
                    del contrast, matrices
                for key, acc in source_stats.items():
                    moments[key].add(acc.finish()['mean'])
            file = save_moments(output, group, moments, names, layers, heads, horizons, onset_radius)
            coverage['statistics'] = str(file.relative_to(output))
            report['groups'][group] = coverage
            plot_cohort(file)
            for entries in examples.values():
                for e in entries:
                    # Explicit post-hoc label-balanced examples. Their labels
                    # do not change capture, the head axes or cohort statistics.
                    plot_sample(output / e['path'], title=f"{group}/{e['sample_id']}")
                    e['figure'] = str(Path(e['path']).with_suffix('.review.png'))
                    e['text'] = str(Path(e['path']).with_suffix('.review.html'))
            split = group.split('/')[0]
            seen = all_sources.setdefault(split, set())
            if seen & sources.keys():
                raise ValueError('a source occurs in multiple task groups; ALL needs a shared-source aggregation')
            seen.update(sources)
            combined = all_moments.setdefault(split, defaultdict(Moments))
            for key, acc in moments.items():
                combined[key].merge(acc)
            totals = all_coverage.setdefault(split, defaultdict(int))
            for key, value in coverage.items():
                if isinstance(value, int):
                    totals[key] += value
            save_json(output / 'summary.json', report)
        for split, moments in all_moments.items():
            group = split + '/ALL'
            path = save_moments(output, group, moments, names, layers, heads, horizons, onset_radius)
            report['groups'][group] = {**all_coverage[split], 'statistics': str(path.relative_to(output))}
            plot_cohort(path)
    report['replication'] = replication(output, report)
    save_json(output / 'summary.json', report)
    save_json(output / 'review_index.json', rows_for_gallery)
    write_summary(output, report)
    import shutil
    shutil.copyfile(Path(__file__).with_name('view_attention_audit.ipynb'), output / 'view_attention_audit.ipynb')
    write_gallery(output, rows_for_gallery, report)
    return report


def replication(output, report):
    """Freeze rankings on train; inspect the same physical head/pair on test."""
    result = {}
    for task in ('QA', 'Summary', 'Data2txt', 'ALL'):
        if any(split+'/'+task not in report['groups'] for split in ('train', 'test')):
            continue
        with np.load(output / report['groups']['train/'+task]['statistics']) as a, np.load(
                output / report['groups']['test/'+task]['statistics']) as b:
            heads = int(a['heads'])
            rows = []
            for k, name in enumerate(a['metric_names']):
                train, test = a['matched_mean'][k].ravel(), b['matched_mean'][k].ravel()
                valid = np.isfinite(train) & np.isfinite(test)
                if not np.isfinite(train).any():
                    continue
                candidates=np.flatnonzero(np.isfinite(train))
                i = candidates[np.argmax(np.abs(train[candidates]))]
                correlation = float(np.corrcoef(train[valid], test[valid])[0, 1]) if valid.sum() > 1 and np.std(train[valid]) and np.std(test[valid]) else None
                rows.append({'metric': str(name), 'train_selected_head': [int(i//heads), int(i%heads)],
                             'train_effect': float(train[i]), 'test_effect': float(test[i]),
                             'test_q_by': float(b['matched_q_by'][k].ravel()[i]),
                             'profile_correlation_descriptive': correlation})
            pairs=[]
            for name in ('chain_matched','chain_excess_matched'):
                train,test=a[name+'_mean'],b[name+'_mean']
                if not np.isfinite(train).any():
                    continue
                wi,ri=np.unravel_index(np.nanargmax(np.abs(train)),train.shape)
                pairs.append({'comparison':name,'writer':[int(wi//heads),int(wi%heads)],
                              'reader':[int(ri//heads),int(ri%heads)],
                              'train_effect':float(train[wi,ri]),'test_effect':float(test[wi,ri]),
                              'test_sources':int(b[name+'_sources'][wi,ri]),
                              'test_ci95':b[name+'_ci95'][:,wi,ri].tolist(),
                              'test_q_by':float(b[name+'_q_by'][wi,ri])})
            result[task]={'heads':rows,'pairs':pairs}
    return result


def write_summary(output, report):
    lines = ['# 正常／幻觉全量结构审计', '',
             '主要路由统计与路径排除特殊来源、载体和目标；特殊质量对照和原生写入保留真实计算。N=正常，H=幻觉。',
             '总体图保留每个真实 layer/head。两跳矩阵保留每对严格跨层 writer/reader。', '',
             '| 分组 | 样本 | N token | H token | 排除特殊目标 | 未知标签 | 配对 token | 配对起点 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group, c in report['groups'].items():
        lines.append(f"| {group} | {c['samples']} | {c['normal_tokens']} | {c['hallucinated_tokens']} | {c['special_targets']} | {c['unknown_tokens']} | {c['matched_pairs']} | {c['matched_onsets']} |")
    lines += ['', '## 查看结果', '',
              '- `gallery.html`：按任务和正负回答查看样本、完整标红文本、总体图。',
              '- `cohorts/*.reads.png`：总览指标显示 N、H、同回答位置及 token 类型配对的 H−N；notebook 可查看全部指标，不平均 head。',
              '- `cohorts/*.chains.png`：完整四种进入／复用状态、载体标签差异、以目标标签对齐的两跳差异。',
              '- `cohorts/*.npz`：全部均值、贡献 source 数、95% Student 区间、p 值、BY 校正 q 值；可离线筛选每个 head。',
              '- 每样本 `.history.npz`：所有 head 的完整回答内部 attention；`.audit.npz`：各复用窗口、标签及配对坐标。',
              '- 每样本 `.qk.npz`：实际 Q/K、精度、缩放；CPU 重建任意 head 的完整 prompt+response 来源，不加载模型权重。',
              '- `view_attention_audit.ipynb`：切换样本和任意 layer/head 的离线查看入口。', '',
              '## 解释边界', '',
              '读取指标按 q→q+1 对齐标签；以 carrier_ 开头的指标及四状态表按载体 b 的标签对齐。',
              'chain_matched 使用最终目标 q+1 的标签，计算严格跨层的两跳 attention 系数组合。',
              'chain_excess_matched 减去载体证据份额在 16 个位置块内循环移动后的对照。',
              '这些是结构关联；材料来源标签并不自动确认它与当前 claim 相关，系数组合也不证明事实内容经过该路径。',
              'head_margin 和 MLP/残差读出支持的是已有 token 相对 runner，已有 token 自身可能错误。',
              '主配对用同类型、前后两侧正常位置插值，抵消线性位置趋势；最近单个正常位置作为次要对照。',
              '该配对仍未控制全部非线性位置、语义、生成模型与历史需求。',
              '置信区间以 source 为统计单位；BY 在每个 cohort/contrast 的所有指标/head 或所有路径上控制多重比较。',
              '本轮不拟合新检测器，不把发现方向更好的测试集指标改成检测分数。', '']
    (output / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
