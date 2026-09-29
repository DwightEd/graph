"""Post-measurement paired diagnostics; no fitted detector or token gold for GSM."""
import argparse
import csv
import html
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from experiments.constraint_uptake.evaluate import label_tables, rag_units, step_units
from experiments.decision_risk_flow.data import read_json, write_json
from experiments.span_maintenance.boundary import punctuation_boundaries
from .measure import MODEL

FEATURES = ('continuation', 'prompt_attention', 'prompt_effect_share', 'prompt_opposition',
            'persistent_history', 'effective_heads', 'address_agreement', 'nll',
            'address_reuse', 'signed_reuse', 'continued_source_control')


def read_features(directory):
    with np.load(directory / 'heads.npz') as saved:
        measured = saved['measured'].astype(float)
        continuation = saved['continuation'].astype(float)
        agreement = (saved['effect_keys'] == saved['attention_keys']).mean((0, 1))
    positive, negative, history_positive, history_negative = np.moveaxis(measured[..., :4], -1, 0)
    prompt = positive + negative
    history = history_positive + history_negative
    total = (prompt + history).sum((0, 1)) + 1e-12
    result = dict(continuation=continuation.mean((0, 1)), prompt_attention=measured[..., 4].mean((0, 1)),
        prompt_effect_share=prompt.sum((0, 1)) / total,
        prompt_opposition=negative.sum((0, 1)) / (prompt.sum((0, 1)) + 1e-12),
        persistent_history=(continuation * history).sum((0, 1)) / total,
        effective_heads=prompt.sum((0, 1)) ** 2 / ((prompt ** 2).sum((0, 1)) + 1e-12),
        address_agreement=agreement)
    with np.load(directory / 'baseline.npz') as saved:
        result['nll'] = -saved['logp']
    with np.load(directory / 'memory.npz') as saved:
        for index, name in enumerate(saved['fields']):
            result[str(name)] = saved['measured'][..., index].mean((0, 1))
    return result


def annotate_tokens(row, directory, features, units, tokenizer):
    count = len(row['response']['answer_ids'])
    with np.load(directory / 'heads.npz') as saved:
        effects = saved['measured'][..., 5]
        keys = saved['effect_keys']
    with np.load(directory / 'baseline.npz') as saved:
        margin = saved['margin']
        alternatives = saved['alternatives']
    flat_heads = np.argmax(np.abs(effects).reshape(1024, count), axis=0)
    segments = np.cumsum(punctuation_boundaries(row['response']['token_text']))
    result = []
    for target in range(count):
        layer, head = divmod(int(flat_heads[target]), 32)
        source = int(keys[layer, head, target])
        gold = units[target]['gold'] if row['dataset'] == 'ragtruth' else -1
        step = next((index for index, (start, end) in enumerate(row.get('step_ranges', [])) if start <= target < end), -1)
        result.append(dict(key=row['key'], target=target, text=row['response']['token_text'][target],
            token_gold=gold, step=step, step_gold=units[step]['gold'] if step >= 0 else -1,
            segment=int(segments[target]), layer=layer, head=head, source_key=source,
            source_text=tokenizer.decode([row['prompt'][source]]), effect=float(effects[layer, head, target]),
            rival=tokenizer.decode([int(alternatives[target])]), margin=float(margin[target]),
            **{name: float(values[target]) for name, values in features.items()}))
    return result


def propagation_rows(row, directory):
    segments = np.cumsum(punctuation_boundaries(row['response']['token_text']))
    result = []
    for index in range(2):
        metadata = read_json(directory / f'propagation_{index}.json')
        probe = metadata['probe']
        start = probe['target']
        with np.load(directory / f'propagation_{index}.npz') as saved:
            for treatment in metadata['audits']:
                name = treatment['kind']
                for target in range(start, len(segments)):
                    result.append(dict(key=row['key'], probe=index, kind=name, receiver=start,
                        target=target, lag=target - start, same_segment=bool(segments[target] == segments[start]),
                        source=treatment['text'], source_keys=str(treatment['keys']),
                        slope=float(saved[f'{name}_margin_slope'][target]),
                        logp_slope=float(saved[f'{name}_logp_slope'][target]),
                        hidden_slope=float(saved[f'{name}_hidden_slope'][target]),
                        current_slope=float(saved[f'{name}_margin_slope'][start])))
    return result


def matched_lags(rows):
    """Equal lag means; different events remain a semantic/length confound."""
    comparisons = []
    for lag in sorted({row['lag'] for row in rows if row['lag'] > 0}):
        selected = [row for row in rows if row['lag'] == lag and row['kind'] == 'effect']
        within = [row for row in selected if row['same_segment']]
        across = [row for row in selected if not row['same_segment']]
        if within and across:
            comparisons.append(dict(lag=lag, within_count=len(within), across_count=len(across),
                within=float(np.mean([abs(r['slope']) for r in within])),
                across=float(np.mean([abs(r['slope']) for r in across])),
                within_hidden=float(np.mean([r['hidden_slope'] for r in within])),
                across_hidden=float(np.mean([r['hidden_slope'] for r in across]))))
    return comparisons


def group_summary(units):
    result = {}
    for cohort in sorted({row['cohort'] for row in units}):
        selected = [row for row in units if row['cohort'] == cohort and row['gold'] >= 0]
        labels = np.array([row['gold'] for row in selected])
        result[cohort] = dict(known=len(selected), positive=int(labels.sum()), features={})
        for name in FEATURES:
            values = np.array([row[name] for row in selected])
            auc = float(roc_auc_score(labels, values)) if len(set(labels)) == 2 else None
            result[cohort]['features'][name] = dict(auroc_high_is_error=auc,
                normal_mean=float(values[labels == 0].mean()), error_mean=float(values[labels == 1].mean()))
    return result


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(path, tokens):
    columns = ('target', 'text', 'token_gold', 'step_gold', 'continuation', 'prompt_attention',
               'prompt_effect_share', 'prompt_opposition', 'layer', 'head', 'source_text', 'effect', 'rival', 'margin')
    parts = ['<!doctype html><meta charset="utf-8"><title>片段与来源控制逐token审计</title>',
        '<style>body{font:14px system-ui;margin:25px}td,th{padding:5px;border:1px solid #ddd}',
        'table{border-collapse:collapse}th{position:sticky;top:0;background:white}.error{background:#ffd9d0}</style>',
        '<h1>片段维持与来源输出作用</h1><p>GSM step_gold仅表示所在步骤，不是token标签。',
        'effect是当前margin对来源attention logit的导数，不是事实支持。层/头从0编号。</p>']
    for key in dict.fromkeys(row['key'] for row in tokens):
        parts.append(f'<h2>{key}</h2><table><tr>' + ''.join(f'<th>{name}</th>' for name in columns) + '</tr>')
        for row in tokens:
            if row['key'] != key:
                continue
            style = ' class="error"' if row['token_gold'] == 1 else ''
            values = [f'{row[name]:.4f}' if isinstance(row[name], float) else str(row[name]) for name in columns]
            parts.append(f'<tr{style}>' + ''.join(f'<td>{html.escape(value)}</td>' for value in values) + '</tr>')
        parts.append('</table>')
    path.write_text('\n'.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.output / 'finite_complete.json')
    read_json(args.output / 'memory_complete.json')
    records = read_json(args.output / 'manifest.json')['records']
    features = {row['key']: read_features(args.output / row['key']) for row in records}
    write_json(args.output / 'readout_frozen.json', dict(columns=FEATURES, labels_used=False,
        scoring='descriptive raw features, no fitted detector, thresholds, or selected orientation'))
    truth, local, gsm = label_tables(records)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    units, tokens, propagation = [], [], []
    for record in records:
        row = dict(record, answer=record['response']['answer_ids'],
            positions=list(range(len(record['response']['answer_ids']))), text=record['response']['token_text'])
        scores = features[row['key']]
        current = rag_units(row, scores, FEATURES, truth, local) if row['dataset'] == 'ragtruth' else step_units(row, scores, FEATURES, gsm)
        units.extend(current)
        tokens.extend(annotate_tokens(row, args.output / row['key'], scores, current, tokenizer))
        propagation.extend(propagation_rows(row, args.output / row['key']))
    write_csv(args.output / 'tokens.csv', tokens)
    write_csv(args.output / 'units.csv', units)
    write_csv(args.output / 'propagation.csv', propagation)
    write_json(args.output / 'summary.json', group_summary(units))
    write_json(args.output / 'matched_lags.json', matched_lags(propagation))
    render(args.output / 'TOKEN_AUDIT.html', tokens)
    print('evaluated', len(tokens), 'tokens;', len(units), 'label units', flush=True)


if __name__ == '__main__':
    main()
