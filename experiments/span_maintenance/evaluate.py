"""Read labels only after score freezing; state diagnostics and exact token audits."""
import argparse
import csv
import html
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from experiments.constraint_uptake.evaluate import label_tables, rag_units, step_units, metrics, first_error
from .run import METHODS, read_json, write_json
from .state import FIELDS


def summarize(rows, name, thresholds):
    selected = [row for row in rows if row['gold'] >= 0]
    result = metrics(rows, name, 0)
    gold = np.array([row['gold'] for row in selected])
    values = np.array([row[name] for row in selected])
    alarm = np.array([row[name] > thresholds[row['task']] for row in selected])
    result.update(tp=int((alarm & (gold == 1)).sum()), fp=int((alarm & (gold == 0)).sum()),
        fn=int((~alarm & (gold == 1)).sum()), threshold='original per-task baseline threshold')
    if len(set(gold)) == 2:
        positive = gold.sum()
        auc = (rankdata(values)[gold == 1].sum() - positive * (positive + 1) / 2) / (positive * (len(gold) - positive))
        np.testing.assert_allclose(result['auroc'], auc, atol=1e-12)
    return result


def measured_columns(directory):
    with np.load(directory / 'states.npz') as saved:
        state = saved['state'].astype(float).reshape(1024, -1, len(FIELDS))
        shuffled = saved['shuffled'].astype(float).reshape(state.shape)
        signed = saved['signed'].astype(float).reshape(1024, state.shape[1], 4)
        stability = saved['signed_stability'].mean((0, 1))
    columns = {name: state[..., index].mean(0) for index, name in enumerate(FIELDS)}
    columns['continuation_p90'] = np.quantile(state[..., FIELDS.index('continuation')], .9, axis=0)
    columns['shuffle_age'] = shuffled[..., FIELDS.index('age')].mean(0)
    columns['shuffle_excess'] = shuffled[..., FIELDS.index('distance_excess')].mean(0)
    columns['signed_stability'] = stability
    for index, name in enumerate(('prompt_positive', 'prompt_negative', 'history_positive', 'history_negative')):
        columns[name] = signed[..., index].sum(0)
    return columns


def make_rows(record, output, truth, local, gsm, methods):
    count = len(record['response']['answer_ids'])
    row = dict(record, answer=record['response']['answer_ids'], positions=list(range(count)),
        text=record['response']['token_text'])
    directory = output / record['key']
    observed = measured_columns(directory)
    with np.load(directory / 'scores.npz') as saved:
        scores = {name: saved[name].copy() for name in methods}
    if row['dataset'] == 'ragtruth':
        units = rag_units(row, scores, methods, truth, local)
    else:
        units = step_units(row, scores, methods, gsm)
    if (directory / 'boundary_states.npz').exists():
        with np.load(directory / 'boundary_states.npz') as saved:
            observed['punctuation_hint'] = saved['boundaries'].astype(float)
            for mode in ('hard', 'soft'):
                head_mean = saved[mode].mean((0, 1))
                for index, field in enumerate(FIELDS):
                    observed[f'{mode}_{field}'] = head_mean[:, index]
    with np.load(directory / 'route_scores.npz') as saved:
        route_scores = {f'route_{name}': saved[name].copy() for name in saved.files}
    for unit in units:
        target = unit['position']
        selected = target if row['dataset'] == 'ragtruth' else slice(*row['step_ranges'][target])
        unit.update({name: float(np.mean(values[selected])) for name, values in observed.items()})
        unit.update({name: float(np.mean(values[selected])) for name, values in route_scores.items()})
        unit['task'] = row['task']
        unit['state_scope'] = 'token' if row['dataset'] == 'ragtruth' else 'step aggregate, not token gold'
    return units


def position_groups(rows):
    """Gold is used only to describe errors; contiguous gold is not semantic truth."""
    for key in dict.fromkeys(row['key'] for row in rows):
        answer = [row for row in rows if row['key'] == key]
        labels = {row['position']: row['gold'] for row in answer}
        for row in answer:
            position, gold = row['position'], row['gold']
            if row['cohort'] == 'gsm_step':
                group = 'first_error_step' if gold == 1 else 'normal_step' if gold == 0 else 'unknown'
            elif gold == 1:
                group = 'error_inside' if labels.get(position - 1) == 1 else 'error_start'
            elif gold == 0:
                nearby = any(labels.get(position + delta) == 1 for delta in (-2, -1, 1, 2))
                group = 'normal_near_error' if nearby else 'normal_other'
            else:
                group = 'unknown'
            row['position_group'] = group


def state_summary(rows):
    fields = (*FIELDS, 'continuation_p90', 'shuffle_age', 'shuffle_excess', 'signed_stability')
    result = {}
    for group in dict.fromkeys(row['position_group'] for row in rows):
        selected = [row for row in rows if row['position_group'] == group]
        result[group] = dict(count=len(selected),
            means={name: float(np.mean([row[name] for row in selected])) for name in fields},
            medians={name: float(np.median([row[name] for row in selected])) for name in fields})
    return result


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def changes(rows, thresholds, methods):
    result = []
    for row in rows:
        if row['gold'] < 0:
            continue
        old = row['base'] > thresholds[row['task']]
        for name in methods:
            alarm = row[name] > thresholds[row['task']]
            if alarm != bool(row['gold']) or alarm != old:
                status = ('TP' if row['gold'] else 'FP') if alarm else ('FN' if row['gold'] else 'TN')
                result.append(dict(key=row['key'], position=row['position'], text=row['text'],
                    gold=row['gold'], method=name, score=row[name], status=status,
                    changed_from_base=bool(alarm != old), position_group=row['position_group']))
    return result


def render(rows, output, thresholds, primary):
    parts = ['<!doctype html><meta charset="utf-8"><title>片段维持状态逐位置审计</title>',
        '<style>body{font:15px system-ui;margin:28px}td,th{padding:5px;border-bottom:1px solid #ddd}',
        'table{border-collapse:collapse}td.text{white-space:pre-wrap;max-width:500px}.bad{background:#ffe1dd}',
        '.unknown{color:#777}th{position:sticky;top:0;background:white}</style>',
        '<h1>片段维持状态：18答开发回归</h1><p>状态稳定不等于正确；GSM每行是步骤，不是token标签。',
        f'红色表示{primary}误报或漏检；权重是统计汇聚权重，不是已验证的因果影响。</p>']
    for key in dict.fromkeys(row['key'] for row in rows):
        selected = [row for row in rows if row['key'] == key]
        parts.append(f'<h2>{html.escape(key)}</h2><table><tr><th>位置</th><th>原文</th><th>标签</th>')
        parts.append(''.join(f'<th>{name}</th>' for name in ('base', 'raw', primary, 'history_mass',
            'prompt_refresh', 'age', 'distance_excess', 'signed_stability')) + '<th>状态权重最高的历史位置</th></tr>')
        kernel = np.load(output / key / 'weights.npz')['state']
        if primary == 'soft_both':
            kernel = np.load(output / key / 'boundary_weights.npz')['soft']
        for row in selected:
            alarm = row[primary] > thresholds[row['task']]
            style = 'unknown' if row['gold'] < 0 else 'bad' if alarm != bool(row['gold']) else ''
            parts.append(f'<tr class="{style}"><td>{row["position"]}</td><td class="text">{html.escape(row["text"])}</td><td>{row["gold"]}</td>')
            parts.append(''.join(f'<td>{row[name]:.4f}</td>' for name in ('base', 'raw', primary,
                'history_mass', 'prompt_refresh', 'age', 'distance_excess', 'signed_stability')))
            target = row['position']
            parents = np.argsort(kernel[target, :target])[-4:][::-1]
            detail = ', '.join(f'{int(index)}:{kernel[target,index]:.3f}' for index in parents)
            parts.append('<td>' + (detail if row['cohort'] != 'gsm_step' else '见原token weights.npz') + '</td></tr>')
        parts.append('</table>')
    (output / 'TOKEN_AUDIT.html').write_text('\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frozen = read_json(args.output / 'scores_frozen.json')
    assert frozen['status'] == 'complete'
    records = read_json(args.output / 'manifest.json')['records']
    thresholds = read_json(args.output / 'thresholds.json')
    truth, local, gsm = label_tables(records)
    methods, primary = frozen['methods'], frozen['primary']
    rows = [unit for record in records for unit in make_rows(record, args.output, truth, local, gsm, methods)]
    position_groups(rows)
    groups = {name: [row for row in rows if row['cohort'] == name] for name in ('rag_official', 'rag_local', 'gsm_step')}
    metrics_by_group = {group: {name: summarize(items, name, thresholds) for name in methods} for group, items in groups.items()}
    route_names = [name for name in rows[0] if name.startswith('route_')]
    routes = {group: {name: metrics(items, name, 0)['auroc'] for name in route_names} for group, items in groups.items()}
    result = dict(status='complete', primary=primary, cohorts=metrics_by_group, route_auc=routes,
        states={name: state_summary(items) for name, items in groups.items()},
        cases={key: {name: summarize([row for row in rows if row['key'] == key], name, thresholds) for name in methods}
               for key in dict.fromkeys(row['key'] for row in rows)},
        first_error={name: first_error(groups['gsm_step'], name, thresholds['GSM8K']) for name in methods},
        no_semantic_phase_ground_truth=True, inference='descriptive, exposed development cases',
        verification='all combined AUCs independently rank-recomputed; all 1024 physical heads retained')
    write_json(args.output / 'evaluation.json', result)
    write_csv(args.output / 'positions.csv', rows)
    write_csv(args.output / 'errors_and_changes.csv', changes(rows, thresholds, methods))
    render(rows, args.output, thresholds, primary)
    for group, values in metrics_by_group.items():
        print(group, {name: (value['auroc'], value['tp'], value['fp']) for name, value in values.items()})


if __name__ == '__main__':
    main()
