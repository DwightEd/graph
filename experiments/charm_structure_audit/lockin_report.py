"""Same-pair causal sensitivity, exact readout accounting, and explicit denominators."""

from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from .data import read_json, write_json
from .evaluate import source_interval


def regions(length):
    offset = np.arange(length)
    fraction = (offset + .5) / length
    return dict(all=np.ones(length, bool), first=offset == 0,
                front_half=fraction < .5, back_half=fraction >= .5,
                early=fraction < 1/3, middle=(fraction >= 1/3) & (fraction < 2/3),
                late=fraction >= 2/3)


def components(logits, names):
    index = {name: i for i, name in enumerate(names)}
    full = logits[:, index['full']]
    no_inner = logits[:, index['cut_internal']]
    no_outer = logits[:, index['cut_external']]
    own = logits[:, index['cut_all']]
    inner = .5 * ((full - no_inner) + (no_outer - own))
    outer = .5 * ((full - no_outer) + (no_inner - own))
    interaction = full - no_inner - no_outer + own
    np.testing.assert_allclose(own + inner + outer, full, atol=2e-5, rtol=1e-5)
    return dict(own=own, internal=inner, external=outer, interaction=interaction)


def frozen_cohorts(capture):
    score = capture['baseline_score'][0]
    low, high = capture['tail_cutoffs']
    return dict(all=np.ones(len(score), bool), high_error=score > high,
                low_error=score < low, error_TP=score > capture['threshold'],
                error_FN=score <= capture['threshold'])


def one_pair_row(capture, experiment, selected):
    labels = np.r_[np.ones(selected.sum()), np.zeros(selected.sum())]
    base = capture['logits'][:, 0, selected]
    changed = capture['logits'][:, experiment, selected]
    base_score = capture['baseline_score'][:, selected]
    score = base_score if experiment == 0 else expit(changed)
    threshold = float(capture['threshold'])
    effect = (base - changed).mean(axis=1)
    return dict(tokens_each_side=int(selected.sum()),
                base_auc=float(roc_auc_score(labels, base_score.reshape(-1))),
                changed_auc=float(roc_auc_score(labels, score.reshape(-1))),
                error_effect=float(effect[0]), normal_effect=float(effect[1]),
                specific_effect=float(effect[0] - effect[1]),
                base_logit_margin=float((base[0] - base[1]).mean()),
                changed_logit_margin=float((changed[0] - changed[1]).mean()),
                error_TP=int((score[0] > threshold).sum()),
                normal_FP=int((score[1] > threshold).sum()),
                base_TP=int((base_score[0] > threshold).sum()),
                base_FP=int((base_score[1] > threshold).sum()),
                error_gate_flip=float(capture['gate_flips'][0, experiment, selected].mean()),
                normal_gate_flip=float(capture['gate_flips'][1, experiment, selected].mean()))


def pair_rows(capture, identity):
    rows = []
    masks = regions(capture['logits'].shape[2])
    cohorts = frozen_cohorts(capture)
    for number, experiment in enumerate(capture['names']):
        for region, inside in masks.items():
            for cohort, chosen in cohorts.items():
                selected = inside & chosen
                if not selected.any():
                    continue
                row = dict(identity, experiment=str(experiment), region=region, cohort=cohort)
                row.update(one_pair_row(capture, number, selected))
                row['auc_delta'] = row['changed_auc'] - row['base_auc']
                rows.append(row)
    return rows


def token_rows(capture, identity):
    rows = []
    parts = components(capture['logits'], capture['names'])
    score = capture['baseline_score']
    threshold = float(capture['threshold'])
    low, high = capture['tail_cutoffs']
    names = list(capture['names'])
    for role, side in enumerate(('error', 'normal')):
        for offset in range(score.shape[1]):
            row = dict(identity, side=side, offset=offset, token=int(capture['tokens'][role, offset]),
                       text=str(capture['text'][role, offset]), score=float(score[role, offset]),
                       alarm=bool(score[role, offset] > threshold),
                       tail='high' if score[role, offset] > high else 'low' if score[role, offset] < low else 'middle')
            row.update({key: float(value[role, offset]) for key, value in parts.items()})
            row['outcome'] = ('TP' if row['alarm'] else 'FN') if role == 0 else ('FP' if row['alarm'] else 'TN')
            row['reuse_effect'] = float(capture['logits'][role, 0, offset] - capture['logits'][role, names.index('no_reuse'), offset])
            row['exchangeable_edges'] = int(capture['exchangeable'][role, offset])
            rows.append(row)
    return rows


def component_rows(capture, identity):
    rows = []
    parts = components(capture['logits'], capture['names'])
    for region, selected in regions(capture['logits'].shape[2]).items():
        if not selected.any():
            continue
        for component, values in parts.items():
            means = values[:, selected].mean(axis=1)
            rows.append(dict(identity, region=region, component=component,
                             error=float(means[0]), normal=float(means[1]),
                             difference=float(means[0] - means[1])))
    return rows


def random_adjusted_rows(capture, identity):
    """Only jointly exchangeable positions; no fake zero when contrast is unidentified."""
    names = list(capture['names'])
    random = [i for i, name in enumerate(names) if name.startswith('random_')]
    if not random:
        return []
    eligible = (capture['exchangeable'] > 0).all(axis=0)
    # (full-cutI) - mean(full-random) = mean(random) - cutI.
    excess = capture['logits'][:, random].mean(axis=1) - capture['logits'][:, names.index('cut_internal')]
    rows = []
    for region, selected in regions(len(eligible)).items():
        selected = selected & eligible
        values = excess[:, selected].mean(axis=1) if selected.any() else [np.nan, np.nan]
        rows.append(dict(identity, region=region, eligible_tokens=int(selected.sum()),
                         error=float(values[0]), normal=float(values[1]),
                         difference=float(values[0] - values[1])))
    return rows


def readout_rows(capture, identity):
    """Every unit; no test-selected shortlist. Contributions sum to the paired margin."""
    values = capture['contributions']
    rebuilt = values.sum(axis=-1) + float(capture['readout_bias'][0])
    np.testing.assert_allclose(rebuilt, capture['logits'], atol=2e-5, rtol=1e-5)
    rows = []
    for index, experiment in enumerate(capture['names']):
        for region in ('all', 'front_half', 'back_half'):
            selected = regions(values.shape[2])[region]
            if not selected.any():
                continue
            mean = values[:, index, selected].mean(axis=1)
            for unit in range(mean.shape[-1]):
                rows.append(dict(identity, experiment=str(experiment), region=region, unit=unit,
                                 error=float(mean[0, unit]), normal=float(mean[1, unit]),
                                 difference=float(mean[0, unit] - mean[1, unit])))
    return rows


def summarize(frame, keys, fields, bootstrap):
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for values, group in frame.groupby(keys, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values))
        row['pairs'] = len(group)
        for field in fields:
            valid = group[np.isfinite(group[field])]
            interval = source_interval(valid, field, bootstrap)
            row.update({field + '_' + key: value for key, value in interval.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def detection_counts(frame):
    """Any hit is partial, not point-adjusted token success."""
    rows = []
    for (experiment, region, cohort), group in frame.groupby(['experiment', 'region', 'cohort']):
        total = group.tokens_each_side.sum()
        rows.append(dict(experiment=experiment, region=region, cohort=cohort,
            pairs=len(group), tokens_each_side=int(total), error_TP=int(group.error_TP.sum()),
            normal_FP=int(group.normal_FP.sum()), recall=float(group.error_TP.sum()/total),
            fpr=float(group.normal_FP.sum()/total), error_any=int((group.error_TP > 0).sum()),
            normal_clear=int((group.normal_FP == 0).sum()),
            both_any_clear=int(((group.error_TP > 0) & (group.normal_FP == 0)).sum()),
            both80_clear=int(((group.error_TP/group.tokens_each_side >= .8) & (group.normal_FP == 0)).sum())))
    return pd.DataFrame(rows)


def save_plots(summary, parts, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    directory = Path(output) / 'figures'
    directory.mkdir(exist_ok=True)
    order = ['early', 'middle', 'late']
    frame = summary[(summary.cohort == 'all') & summary.region.isin(order)]
    for field in ('specific_effect', 'auc_delta'):
        figure, axis = plt.subplots(figsize=(8, 5))
        for name in ('cut_internal', 'no_reuse', 'cut_external', 'last_swap_internal', 'last_swap_external'):
            rows = frame[frame.experiment == name].set_index('region').reindex(order)
            axis.plot(order, rows[field + '_mean'], marker='o', label=name)
        axis.axhline(0, linestyle='--', linewidth=.8)
        axis.set_ylabel('Error-minus-normal logit effect' if field == 'specific_effect' else 'Paired AUROC: intervened - full')
        axis.set_title('Same fixed pairs; GNN intervention, not native LLM lock-in')
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(directory / (field + '.png'), dpi=160)
        figure.savefig(directory / (field + '.svg'))
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 5))
    for component in ('own', 'internal', 'external', 'interaction'):
        rows = parts[parts.component == component].set_index('region').reindex(order)
        axis.plot(order, rows.difference_mean, marker='o', label=component)
    axis.axhline(0, linestyle='--', linewidth=.8)
    axis.set_title('Exact two-source logit decomposition (interaction is separate)')
    axis.set_ylabel('Error-minus-normal contribution; signed values')
    axis.legend()
    figure.tight_layout()
    figure.savefig(directory / 'components.png', dpi=160)
    figure.savefig(directory / 'components.svg')
    plt.close(figure)



def pulse_plot(summary, output):
    import matplotlib.pyplot as plt

    selected = summary[(summary.cohort == 'all') & summary.experiment.str.startswith('pulse_')].copy()
    selected['step'] = selected.experiment.str.replace('pulse_', '', regex=False).astype(int)
    matrix = selected.pivot(index='step', columns='region', values='specific_effect_mean')
    matrix = matrix.reindex(columns=['early', 'middle', 'late'])
    values = matrix.to_numpy()
    finite = np.isfinite(values)
    scale = max(float(np.abs(values[finite]).max()) if finite.any() else 0., 1e-8)
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(np.ma.masked_invalid(values), aspect='auto', vmin=-scale, vmax=scale)
    axis.set_xticks(range(3), ['Early', 'Middle', 'Late'])
    axis.set_yticks(range(len(matrix)), matrix.index)
    axis.set_ylabel('GNN round where internal messages were cut (NOT LLM layer)')
    axis.set_title('Final-logit effect of a one-round cut: error minus normal')
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    for suffix in ('png', 'svg'):
        figure.savefig(Path(output) / 'figures' / ('pulse_rounds.' + suffix), dpi=160)
    plt.close(figure)


def readout_plot(frame, output):
    import matplotlib.pyplot as plt

    full = frame[frame.experiment == 'full']
    per_source = full.groupby(['source_id', 'region', 'unit']).difference.mean().reset_index()
    means = per_source.groupby(['region', 'unit']).difference.mean().unstack('unit')
    means = means.reindex(['all', 'front_half', 'back_half'])
    values = means.to_numpy()
    finite = np.isfinite(values)
    scale = max(float(np.abs(values[finite]).max()) if finite.any() else 0., 1e-8)
    figure, axis = plt.subplots(figsize=(10, 4))
    image = axis.imshow(np.ma.masked_invalid(values), aspect='auto', vmin=-scale, vmax=scale)
    axis.set_yticks(range(3), ['All', 'Front half', 'Back half'])
    axis.set_xlabel('Final readout unit, original index (all units shown)')
    axis.set_title('Exact signed logit contribution: error minus normal')
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    for suffix in ('png', 'svg'):
        figure.savefig(Path(output) / 'figures' / ('readout_units.' + suffix), dpi=160)
    plt.close(figure)

def report(output, bootstrap):
    """Completed-pair resume and CPU-only re-report; incomplete runs are disclosed."""
    output = Path(output)
    manifest = read_json(output / 'manifest.json')
    rows, tokens, parts, random, units = [], [], [], [], []
    available = []
    for pair in manifest['pairs']:
        path = output / 'captures' / pair['file']
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as saved:
            capture = {key: saved[key] for key in saved.files}
        identity = {key: pair[key] for key in ('id', 'source_id', 'error_start', 'normal_start', 'length')}
        rows.extend(pair_rows(capture, identity))
        tokens.extend(token_rows(capture, identity))
        parts.extend(component_rows(capture, identity))
        random.extend(random_adjusted_rows(capture, identity))
        units.extend(readout_rows(capture, identity))
        available.append(pair)
    if not available:
        raise ValueError('No completed lockin pair captures; run --lockin-stage all first')
    frame = pd.DataFrame(rows)
    summary = summarize(frame, ['experiment', 'region', 'cohort'],
                        ['specific_effect', 'error_effect', 'normal_effect', 'auc_delta'], bootstrap)
    part_summary = summarize(pd.DataFrame(parts), ['region', 'component'], ['error', 'normal', 'difference'], bootstrap)
    tables = dict(pair_effects=frame, tokens=pd.DataFrame(tokens), pair_components=pd.DataFrame(parts),
                  component_summary=part_summary, summary=summary, random_adjusted=pd.DataFrame(random),
                  detection_counts=detection_counts(frame), readout_units=pd.DataFrame(units))
    if random:
        tables['random_summary'] = summarize(pd.DataFrame(random), ['region'], ['difference'], bootstrap)
    for name, table in tables.items():
        suffix = '.csv.gz' if name in ('pair_effects', 'tokens', 'readout_units') else '.csv'
        table.to_csv(output / (name + suffix), index=False)
    save_plots(summary, part_summary, output)
    pulse_plot(summary, output)
    readout_plot(pd.DataFrame(units), output)
    status = dict(completed_pairs=len(available), expected_pairs=len(manifest['pairs']),
                  complete=len(available) == len(manifest['pairs']),
                  interpretation='Label-assisted frozen-detector sensitivity, not deployable detection or native LLM causality')
    write_json(output / 'report_status.json', status)
    text = make_report(summary, tables['detection_counts'], status)
    (output / 'REPORT_zh.md').write_text(text, encoding='utf-8')
    with tarfile.open(output / 'lockin_review.tar.gz', 'w:gz') as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file() and (path.suffix != '.gz' or path.name.endswith('.csv.gz')):
                if 'captures' not in path.parts and not path.name.endswith('.partial'):
                    archive.add(path, arcname=path.relative_to(output))
    print(text, flush=True)


def make_report(summary, counts, status):
    lines = ['# CHARM片段内部复用审计', '',
             f"已完成 {status['completed_pairs']}/{status['expected_pairs']} 对。不是新的检测成绩。", '',
             '|实验|全段AUROC变化|错误侧减正常侧的logit效应|两侧同时：错误有命中且正常无误报|',
             '|---|---:|---:|---:|']
    selected = summary[(summary.region == 'all') & (summary.cohort == 'all')]
    for row in selected.itertuples():
        count = counts[(counts.experiment == row.experiment) & (counts.region == 'all') & (counts.cohort == 'all')].iloc[0]
        lines.append(f'|{row.experiment}|{row.auc_delta_mean:.5f}|{row.specific_effect_mean:.5f}|{int(count.both_any_clear)}/{int(count.pairs)}|')
    lines += ['', '先看 cut_internal / no_reuse 是否在错误侧作用更强，而非两侧同时掉分。',
              '再看前后半、TP/FN和正常高低分逐词表；高低分分组只描述，不能作为独立验证。',
              'random_summary只统计两侧都有可交换来源的词；零覆盖不是零效应。',
              'own+internal+external精确相加为full；interaction另列，不要再相加一次。',
              'readout_units保存所有单元的实际贡献；不能把通道编号直接命名为语义机制。',
              '本实验没有验证自回归LLM的闭环、吸引子或证据语义；prompt+早前历史不是正确证据。']
    return '\n'.join(lines)
