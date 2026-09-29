"""Interactive answer-token to source-span inspection; no rescoring."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json


def report(output, state=None):
    records = read_json(output / 'manifest.json')['records']
    destination = output if state is None else state
    score_name = 'localized_odds' if state is None else 'path_localized_odds'
    source_name = 'source_index' if state is None else 'path_source'
    with (destination / 'tokens.csv').open() as stream:
        rows = {(row['key'], int(row['target'])): row for row in csv.DictReader(stream)}
    data, answers = {}, []
    for row in records:
        key = row['key']
        groups = read_json(output / key / 'sources.json')
        boundaries = {span['start'] for span in read_json(destination / key / 'output_spans.json')}
        pieces = []
        with np.load(output / key / 'effects.npz') as measured, np.load(output / key / 'prior.npz') as prior:
            for target, text in enumerate(row['response']['token_text']):
                item = rows[(key, target)]
                identity = key+'_'+str(target)
                entries = []
                for source in np.argsort(-measured['output_js'][target]):
                    entries.append(dict(source=int(source), text=groups[source]['text'],
                        keys=[groups[source]['keys'][0], groups[source]['keys'][-1]+1],
                        js=float(measured['output_js'][target, source]),
                        delta_logp=float(measured['masked_logp'][target, source]-measured['original_logp'][target]),
                        prior=float(prior['prior'][target, source])))
                data[identity] = dict(key=key, target=target, text=text, gold=item['gold'],
                    score=item[score_name], selected=item[source_name], entries=entries)
                style = ('gold ' if item['gold']=='1' else '') + ('alarm ' if item[score_name+'_alarm']=='True' else '')
                style += 'boundary' if target in boundaries else ''
                pieces.append(f'<span class="token {style}" onclick="show(\'{identity}\')" title="t={target}">{html.escape(text)}</span>')
        answers.append('<h2>回答 '+html.escape(key)+'</h2><div>'+''.join(pieces)+'</div>')
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    document = '''<!doctype html><meta charset="utf-8"><title>自动证据span追踪</title>
<style>body{font:16px system-ui;margin:24px;display:grid;grid-template-columns:1fr 1fr;gap:24px}
.token{white-space:pre-wrap;line-height:2;cursor:pointer}.gold{border-bottom:3px solid #b62444}
.alarm{background:#ffe1a1}.boundary{border-left:2px solid #387bad}#detail{position:sticky;top:20px;max-height:93vh;overflow:auto}
pre{white-space:pre-wrap}article{border-bottom:1px solid #ddd;padding:8px}button{margin:5px}</style>
<main><h1>逐token自动来源归因</h1><p>点击任一token，右侧列出全部来源span的实际消融作用。红线=官方span成员；黄色=主候选报警；蓝线=自动状态分段。作用不等于事实支持。</p>ANSWERS</main>
<aside id="detail">点击左侧token。位置从0起，来源key范围为[start,stop)。</aside>
<script>const data=PAYLOAD;
function show(id){const x=data[id],p=document.getElementById('detail');p.replaceChildren();
const title=document.createElement('h2');title.textContent=`${x.key} t${x.target}: ${x.text}`;p.append(title);
const note=document.createElement('p');note.textContent=`标注成员=${x.gold}；主分数=${Number(x.score).toFixed(4)}；内部联合排序选择源${x.selected}。以下按完整词表JS排列。Δlogp>0表示屏蔽后原词更容易出现，不直接等于事实矛盾。`;p.append(note);
for(const e of x.entries){const a=document.createElement('article'),h=document.createElement('strong'),q=document.createElement('pre');
h.textContent=`源${e.source} keys[${e.keys}] JS=${e.js.toFixed(5)} Δlogp=${e.delta_logp.toFixed(4)} prior=${e.prior.toFixed(4)}`;
q.textContent=e.text;a.append(h,q);p.append(a);}}
</script>'''.replace('ANSWERS',''.join(answers)).replace('PAYLOAD',payload)
    (destination / 'TOKEN_EVIDENCE.html').write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state', type=Path)
    args = parser.parse_args()
    report(args.output, args.state)


if __name__ == '__main__':
    main()
