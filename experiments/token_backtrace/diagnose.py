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
    if name in (*METHODS, 'base', 'logic_full', 'logic_only'):
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
