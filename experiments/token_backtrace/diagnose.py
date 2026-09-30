"""Inspect every exposed token, with gold span membership kept evaluation-only."""

import argparse
import csv
import html
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoTokenizer

from experiments.decision_risk_flow.data import labels, read_json, write_json
from experiments.span_source_control.measure import MODEL
from .readout import METHODS


TRACES = (Path('outputs/token_backtrace_20260930_pilot'),
          Path('outputs/token_backtrace_20260930_controls'))


def trace_features(row, trace):
    prompt_length = len(row['prompt'])
    source = np.array(row['source']['source_mask'], bool)
    effect = trace['root_effect']
    source_effect = effect[:, :prompt_length][:, source]
    source_mass = np.abs(source_effect).sum(-1)
    total = np.abs(effect).sum(-1)
    history_mass = np.abs(effect[:, prompt_length:]).sum(-1)
    safe_total = np.maximum(total, np.finfo(np.float32).tiny)
    return dict(source_abs=source_mass, history_abs=history_mass,
                weak_source=1 - source_mass / safe_total,
                history_fraction=history_mass / safe_total,
                source_negative=np.maximum(-source_effect, 0).sum(-1) / safe_total,
                nll=-trace['logp'], negative_margin=-trace['margin'])


def metrics(rows, name):
    known = [row for row in rows if row['valid']]
    gold = np.array([row['gold'] for row in known])
    scores = np.array([row[name] for row in known])
    result = dict(tokens=len(known), positive=int(gold.sum()))
    if len(np.unique(gold)) == 2:
        result.update(auroc=float(roc_auc_score(gold, scores)),
                      ap=float(average_precision_score(gold, scores)))
    if known and f'{name}_alarm' in known[0]:
        alarm = np.array([row[f'{name}_alarm'] for row in known])
        result.update(tp=int(np.sum(alarm & (gold == 1))),
                      fp=int(np.sum(alarm & (gold == 0))),
                      fn=int(np.sum(~alarm & (gold == 1))))
    return result


def token_rows(output, records, methods):
    truth = labels([row['original'] for row in records])
    thresholds = read_json(output / 'thresholds.json')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    result = []
    for row in records:
        directory = next(root / row['key'] for root in TRACES if (root / row['key']).exists())
        with np.load(directory / 'trace.npz') as trace, np.load(output / 'pilot' / row['key'] / 'scores.npz') as saved:
            np.testing.assert_array_equal(trace['token_ids'], saved['token_id'])
            features = trace_features(row, trace)
            root_ids = row['prompt'] + row['response']['answer_ids'][:-1]
            for target, ((start, stop), token) in enumerate(zip(row['response']['offsets'], saved['token_id'])):
                values = dict(key=row['key'], task=row['task'], target=target,
                    text=row['response']['token_text'][target], start=start, stop=stop,
                    gold=int(truth[row['key']][target]),
                    valid=bool(stop > start and token not in row['response']['special_ids']))
                for name in methods:
                    values[name] = float(saved[name][target])
                    values[f'{name}_alarm'] = bool(saved[name][target] > thresholds[row['task']][name])
                values.update({name: float(value[target]) for name, value in features.items()})
                effect = trace['root_effect'][target]
                roots = np.argsort(-np.abs(effect))[:5]
                values['top_roots'] = repr([(int(index), tokenizer.decode([root_ids[index]]),
                                             round(float(effect[index]), 6)) for index in roots])
                result.append(values)
    return result


def write_html(output, rows, primary):
    body = ['<!doctype html><meta charset="utf-8"><title>逐 token 对照</title>',
            '<style>body{max-width:1200px;margin:2em auto;font:16px sans-serif}',
            'span{white-space:pre-wrap;border:1px solid transparent;line-height:2.3}',
            '.gold{border-bottom:3px solid #d0354e}.alarm{background:#ffddb3}</style>',
            f'<h1>逐 token 开发诊断</h1><p>下划线：官方字符 span 成员；橙色：冻结 {primary} 报警。',
            '悬停查看位置、分数、原生梯度根。标注成员不等于每个词独立错误。</p>']
    for key in dict.fromkeys(row['key'] for row in rows):
        body.append(f'<h2>{html.escape(key)}</h2><p>')
        for row in (row for row in rows if row['key'] == key):
            classes = ('gold ' if row['gold'] else '') + ('alarm' if row[f'{primary}_alarm'] else '')
            title = f"t={row['target']} risk={row[primary]:.4f} nll={row['nll']:.3f}; {row['top_roots']}"
            body.append(f'<span class="{classes}" title="{html.escape(title, quote=True)}">{html.escape(row["text"])}</span>')
        body.append('</p>')
    (output / 'pilot.html').write_text('\n'.join(body))


def graph_token_rows(output, records, methods, prefix=''):
    from experiments.context_response.restore import original_threshold

    truth = labels([row['original'] for row in records])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    result = []
    for row in records:
        with np.load(output / row['key'] / (prefix + 'scores.npz')) as scores, np.load(
                output / row['key'] / 'graph.npz') as graph:
            np.testing.assert_array_equal(scores['token_ids'], row['response']['answer_ids'])
            for target, ((start, stop), text) in enumerate(zip(
                    row['response']['offsets'], row['response']['token_text'])):
                item = dict(key=row['key'], source_id=row['original']['source_id'], task=row['task'],
                    target=target, text=text, start=start, stop=stop,
                    valid=bool(scores['valid'][target]), gold=int(truth[row['key']][target]))
                for method in methods:
                    threshold = original_threshold(row['task']) if method == 'base' else .95
                    item[method] = float(scores[method][target])
                    item[method + '_alarm'] = bool(scores[method][target] > threshold)
                    if method != 'base':
                        item['raw_' + method] = float(scores['raw_' + method][target])
                history = graph['history_effect'][target]
                parents = np.argsort(-np.abs(history))[:5]
                item['parents'] = [{'target': int(parent), 'text': row['response']['token_text'][parent],
                    'effect': float(history[parent])} for parent in parents if history[parent] != 0]
                prompt = graph['prompt_effect'][target]
                roots = np.argsort(-np.abs(prompt))[:5]
                item['roots'] = [{'position': int(root), 'text': tokenizer.decode([row['prompt'][root]]),
                    'effect': float(prompt[root])} for root in roots]
                result.append(item)
    return result


def graph_comparisons(rows, methods, primary_method='graph_ridge'):
    """Cluster resampling describes this exposed eight-source pilot only."""
    keys = list(dict.fromkeys(row['source_id'] for row in rows))
    groups = [np.array([i for i, row in enumerate(rows) if row['source_id'] == key]) for key in keys]
    gold = np.array([row['gold'] for row in rows])
    scores = {name: np.array([row[name] for row in rows]) for name in methods}
    rng = np.random.default_rng(42)
    deltas = {name: [] for name in methods if name != primary_method}
    for _ in range(1000):
        selected = np.concatenate([groups[index] for index in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(gold[selected])) < 2:
            continue
        primary = roc_auc_score(gold[selected], scores[primary_method][selected])
        for name in deltas:
            deltas[name].append(primary - roc_auc_score(gold[selected], scores[name][selected]))
    return {name: dict(auroc_delta=float(roc_auc_score(gold, scores[primary_method])
                         - roc_auc_score(gold, scores[name])),
        ci95=np.quantile(values, [.025, .975]).tolist(), bootstrap_draws=len(values))
        for name, values in deltas.items()}


def graph_structure_diagnostics(output, records):
    """Descriptive topology check; the tiny-mass cutoff never enters scoring."""
    from scipy.stats import spearmanr

    result = {}
    for record in records:
        with np.load(output / record['key'] / 'graph.npz') as graph, np.load(
                output / record['key'] / 'scores.npz') as scores:
            weights = graph['history_effect']
            tokens = scores['token_ids']
            exchangeable = 0.
            for target in range(1, len(tokens)):
                groups = 2 * np.floor(np.log2(target - np.arange(target))).astype(int)
                groups += tokens[:target] == tokens[target]
                for group in np.unique(groups):
                    indices = np.flatnonzero(groups == group)
                    if len(indices) > 1:
                        exchangeable += np.abs(weights[target, indices]).sum()
            result[record['key']] = dict(
                exchangeable_absolute_mass_fraction=float(exchangeable / np.abs(weights).sum()),
                graph_vs_node_rank_correlation=float(spearmanr(scores['graph_ridge'], scores['node_ridge']).statistic),
                graph_vs_temporal_rank_correlation=float(spearmanr(scores['graph_ridge'], scores['temporal_ridge']).statistic),
                negligible_history_rows=int((np.abs(weights).sum(-1)[1:] < 1e-5).sum()),
                tokens=len(tokens))
    return result


def write_graph_html(output, rows, methods, prefix='', primary='graph_ridge'):
    import json

    data = json.dumps(rows, ensure_ascii=False).replace('<', '\\u003c')
    options = ''.join(f'<option>{name}</option>' for name in methods)
    page = '''<!doctype html><meta charset="utf-8"><title>全回答逐token图异常</title>
<style>body{max-width:1250px;margin:24px auto;font:16px sans-serif}span.token{white-space:pre-wrap;
line-height:2.4;cursor:pointer;border-bottom:3px solid transparent}.gold{border-bottom-color:#b52149!important}
.alarm{background:#ffe0a8}pre{white-space:pre-wrap;background:#f3f5f7;padding:16px;max-height:380px;overflow:auto}
#detail{position:sticky;top:0;background:white;border:1px solid #ccc;padding:8px}select{font:inherit}</style>
<h1>8个完整回答：逐token图异常对照</h1><p>旧暴露样本；下划线是官方span成员，橙色是无标签冻结阈值报警。
边是原词logp的根梯度总作用，不是事实支持证明；hidden为末层归一化状态的固定投影。点击任何token看真实父地址与全部方法分数。</p>
<div id="detail"><select id="method">OPTIONS</select><pre id="info">请选择token</pre></div><div id="answers"></div>
<script>const rows=DATA;const methods=METHODS;const picker=document.getElementById('method');
picker.value='graph_ridge';function draw(){const out=document.getElementById('answers');out.replaceChildren();
for(const key of [...new Set(rows.map(r=>r.key))]){const title=document.createElement('h2');title.textContent=key;out.append(title);
const para=document.createElement('p');out.append(para);for(const r of rows.filter(r=>r.key===key)){
const el=document.createElement('span');el.textContent=r.text;el.className='token'+(r.gold?' gold':'')+(r[picker.value+'_alarm']?' alarm':'');
el.title='t='+r.target+' score='+r[picker.value].toFixed(4);el.onclick=()=>{const selected={key:r.key,target:r.target,
text:r.text,chars:[r.start,r.stop],gold:r.gold,valid:r.valid,scores:Object.fromEntries(methods.map(m=>[m,r[m]])),
raw:Object.fromEntries(methods.filter(m=>m!=='base').map(m=>[m,r['raw_'+m]])),history_parents:r.parents,prompt_roots:r.roots};
document.getElementById('info').textContent=JSON.stringify(selected,null,2)};para.append(el)}}}picker.onchange=draw;draw();</script>'''
    page = page.replace('OPTIONS', options).replace('METHODS', json.dumps(methods)).replace('DATA', data)
    page = page.replace("picker.value='graph_ridge'", "picker.value='" + primary + "'")
    (output / (prefix + 'TOKEN_GRAPH.html')).write_text(page)


def evaluate_graph(output, preserve_mass=False):
    from .span_metrics import character_counts
    from experiments.native_support.ragtruth_benchmark.data import annotations

    prefix = 'mass_' if preserve_mass else ''
    primary = 'graph_mass' if preserve_mass else 'graph_ridge'
    frozen = read_json(output / (prefix + 'scores_frozen.json'))
    assert frozen['labels_accessed'] is False
    records = read_json(output / 'manifest.json')['records']
    diagnostics = graph_structure_diagnostics(output, records)
    diagnostic_path = output / 'structure_diagnostics.json'
    if diagnostic_path.exists():
        prior = read_json(diagnostic_path)
        for key, values in diagnostics.items():
            for name, value in values.items():
                np.testing.assert_allclose(value, prior[key][name], rtol=1e-6)
    else:
        write_json(diagnostic_path, diagnostics)
    methods = frozen['methods']
    rows = graph_token_rows(output, records, methods, prefix)
    known = [row for row in rows if row['valid']]
    report = dict(pooled={name: metrics(rows, name) for name in methods},
        answers={record['key']: {name: metrics([row for row in rows if row['key'] == record['key']], name)
                  for name in methods} for record in records},
        primary_vs=graph_comparisons(known, methods, primary), common_budget={})
    gold = np.array([row['gold'] for row in known])
    for name in methods:
        score = np.array([row[name] for row in known])
        alarm = np.zeros(len(known), dtype=bool)
        alarm[np.argsort(-score, kind='stable')[:79]] = True
        report['common_budget'][name] = dict(budget=79, tp=int(gold[alarm].sum()), fp=int((1-gold[alarm]).sum()))
    root = Path(records[0]['original']['root'])
    truth = annotations(root, read_json(root / 'manifest.json'), [row['original'] for row in records])
    for name in methods:
        totals = np.zeros(6, dtype=int)
        onsets, correct_alarms, correct_answers = [], 0, 0
        for record in records:
            tokens = [row for row in rows if row['key'] == record['key']]
            annotation = truth[record['key']]
            response = read_json(root / record['original']['directory'] / 'response.json')
            alarms = np.array([row[name + '_alarm'] and row['valid'] for row in tokens])
            totals += character_counts(response['text'], annotation['character_spans'],
                                       record['response']['offsets'], alarms)
            for span in annotation['character_spans']:
                members = [i for i, row in enumerate(tokens) if row['valid']
                           and row['start'] < span['end'] and row['stop'] > span['start']]
                if members:
                    onsets.append(bool(alarms[members[0]]))
            if not any(row['gold'] for row in tokens):
                correct_answers += 1
                correct_alarms += int(alarms.any())
        report['pooled'][name].update(character_counts=totals.tolist(), span_onsets_hit=sum(onsets),
            span_onsets=len(onsets), correct_answer_alarms=correct_alarms, correct_answers=correct_answers)
    with (output / (prefix + 'tokens.csv')).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_graph_html(output, rows, methods, prefix, primary)
    write_json(output / (prefix + 'evaluation.json'), report)
    print({name: (round(values['auroc'], 6), values['tp'], values['fp'])
           for name, values in report['pooled'].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--logic', action='store_true')
    args = parser.parse_args()
    records = read_json(args.output / 'pilot_frozen.json')['records']
    methods = (*METHODS, 'base', 'logic_full', 'logic_only') if args.logic else (*METHODS, 'base')
    rows = token_rows(args.output, records, methods)
    with (args.output / 'pilot_tokens.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = (*methods, 'weak_source', 'history_fraction', 'source_negative', 'nll', 'negative_margin')
    report = dict(pooled={name: metrics(rows, name) for name in names},
        answers={key: {name: metrics([row for row in rows if row['key'] == key], name) for name in names}
                 for key in dict.fromkeys(row['key'] for row in rows)})
    write_json(args.output / 'pilot_results.json', report)
    write_html(args.output, rows, 'logic_full' if args.logic else 'odds_full')
    print(report['pooled'], flush=True)


if __name__ == '__main__':
    main()
