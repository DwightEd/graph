"""Post-freeze token locations and error analysis; never changes predictions."""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import inputs, read_json, write_json


def contiguous(selected):
    groups = []
    for token in selected:
        if not groups or token['position'] != groups[-1][-1]['position'] + 1:
            groups.append([])
        groups[-1].append(token)
    return [dict(start=group[0]['position'], stop=group[-1]['position']+1,
                 text=''.join(token['text'] for token in group)) for group in groups]


def audit_rows(output, records, methods, thresholds):
    with (output / 'tokens.csv').open() as stream:
        scored = list(csv.DictReader(stream))
    observations, spans, display = [], [], []
    for record in records:
        if record['role'] != 'regression':
            continue
        key = record['key']
        rows = [row for row in scored if row['key'] == key]
        _, response = inputs(record)
        context = np.load(output / key / 'context.npz')['values']
        tokens = []
        for row in rows:
            position = int(row['token'])
            label = int(row['label'])
            scores = {method: float(row[method]) for method in methods}
            status = {method: ('TP' if label else 'FP') if scores[method] > thresholds[record['task']][method]
                      else ('FN' if label else 'TN') for method in methods}
            token = dict(position=position, text=row['text'], label=label, scores=scores, status=status)
            tokens.append(token)
            for method in methods:
                observations.append(dict(key=key, task=record['task'], method=method,
                    position=position, fraction=position/max(len(response['token_text'])-1, 1),
                    text=row['text'], label=label, score=scores[method], status=status[method],
                    entropy_nats=float(context[position, 0]), surprisal=float(context[position, 1]),
                    prompt_copy=int(context[position, 6]), surface=int(context[position, 7:].argmax()),
                    context=''.join(response['token_text'][max(0, position-6):position+7])))
        for method in methods:
            for category in ('TP', 'FP', 'FN'):
                spans.append(dict(key=key, method=method, category=category,
                    groups=contiguous([token for token in tokens if token['status'][method] == category])))
        display.append(dict(key=key, task=record['task'], tokens=tokens))
    return observations, spans, display


def summaries(observations):
    result = []
    methods = list(dict.fromkeys(row['method'] for row in observations))
    for method in methods:
        for category in ('TP', 'FP', 'FN', 'TN'):
            selected = [row for row in observations if row['method'] == method and row['status'] == category]
            result.append(dict(method=method, category=category, count=len(selected),
                position_quartiles=[sum(min(int(row['fraction']*4), 3) == i for row in selected) for i in range(4)],
                mean_entropy_nats=float(np.mean([row['entropy_nats'] for row in selected])) if selected else None,
                prompt_copy_fraction=float(np.mean([row['prompt_copy'] for row in selected])) if selected else None,
                surface_counts=[sum(row['surface'] == i for row in selected) for i in range(5)]))
    return result


def browser(output, rows, methods):
    payload = json.dumps(dict(rows=rows, methods=methods), ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><meta charset="utf-8"><title>逐token误报与漏检审计</title>
<style>body{font:17px system-ui;max-width:1250px;margin:auto;padding:25px}select{font:inherit;margin:10px}.tokens{white-space:pre-wrap;line-height:2.1}.TP{background:#c6ead3}.FP{background:#ffbdb8}.FN{background:#e5d1f4}.TN{background:#f4f5f6}span{cursor:pointer}#detail{position:sticky;top:0;padding:12px;background:#e8eef5}h2{margin-top:35px}</style>
<h1>原始token位置：检出、误报、漏检</h1><p>绿=TP，红=FP，紫=FN，灰=TN。位置为0起始回答token。官方标签仅用于冻结后显示；“正常”表示官方未标错。点击查看原始分数。</p>
<select id="method"></select><div id="detail">点击token查看</div><main id="body"></main>
<script>const data=PAYLOAD;const select=document.getElementById('method');for(const name of data.methods){const o=document.createElement('option');o.value=name;o.textContent=name;select.appendChild(o)}function render(){const root=document.getElementById('body');root.replaceChildren();for(const row of data.rows){const title=document.createElement('h2');title.textContent=row.key+' / '+row.task;root.appendChild(title);const block=document.createElement('div');block.className='tokens';for(const token of row.tokens){const span=document.createElement('span');span.className=token.status[select.value];span.textContent=token.text;span.title='token '+token.position+' '+span.className;span.onclick=()=>{document.getElementById('detail').textContent=row.key+' token '+token.position+' '+JSON.stringify(token.scores)};block.appendChild(span)}root.appendChild(block)}}select.onchange=render;render();</script>'''.replace('PAYLOAD', payload)
    (output / 'TOKEN_AUDIT.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read_json(args.output / 'evaluation.json')
    records = read_json(args.output / 'manifest.json')['records']
    methods = read_json(args.output / 'scores_frozen.json')['methods']
    thresholds = read_json(args.output / 'thresholds.json')
    observations, spans, display = audit_rows(args.output, records, methods, thresholds)
    with (args.output / 'token_audit.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=observations[0])
        writer.writeheader()
        writer.writerows(observations)
    write_json(args.output / 'alarm_spans.json', spans)
    write_json(args.output / 'position_summary.json', summaries(observations))
    browser(args.output, display, methods)
    print('audited', len(observations), 'token-method rows', flush=True)


if __name__ == '__main__':
    main()
