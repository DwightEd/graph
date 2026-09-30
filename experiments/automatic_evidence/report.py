"""Interactive answer-token to source-span inspection; no rescoring."""
import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from experiments.decision_risk_flow.data import read_json


def relation_entries(candidates, effects, target, original_logp, source_prior):
    entries = []
    for index in np.argsort(-effects['flip_js'][target]):
        candidate = candidates['edits'][index]
        entries.append(dict(source=candidate['source'], keys=candidate['keys'],
            js=float(effects['flip_js'][target, index]),
            delta_logp=float(effects['flip_logp'][target, index]-original_logp), prior=float(source_prior[candidate['source']]),
            text=candidate['original']+'\n改变关系: '+candidate['flip_text']+'\n等价对照: '+candidate['equivalent_text'],
            extra=f"类型={candidate['family']}；对照JS={effects['equivalent_js'][target,index]:.5f}；对照Δlogp={effects['equivalent_logp'][target,index]-original_logp:.4f}"))
    for index, group in enumerate(candidates['repeated_sets']):
        entries.append(dict(source=group['sources'], keys=[], text='重复值集合: '+group['value'],
            js=float(effects['repeat_js'][target, index]),
            delta_logp=float(effects['repeat_logp'][target,index]-original_logp), prior=float(source_prior[group['sources']].sum()),
            extra=f"相同key数量随机对照JS={effects['repeat_control_js'][target,index]:.5f}"))
    return entries


def report(output, state=None, prefix=''):
    records = read_json(output / 'manifest.json')['records']
    protocol = read_json(output / 'protocol.json')
    relation = protocol.get('mode') == 'relation'
    source_root = Path(protocol['input']) if relation else output
    destination = output if state is None else state
    score_name = protocol['primary'] if state is None else 'path_localized_odds'
    if prefix:
        score_name = read_json(output/(prefix+'scores_frozen.json'))['primary']
    source_name = 'source_index' if state is None else 'path_source'
    with (destination / (prefix+'tokens.csv')).open() as stream:
        rows = {(row['key'], int(row['target'])): row for row in csv.DictReader(stream)}
    data, answers = {}, []
    for row in records:
        key = row['key']
        groups = read_json(source_root / key / 'sources.json')
        boundaries = {span['start'] for span in read_json(destination / key / (prefix+'output_spans.json'))}
        pieces = []
        effects_file = 'relation_effects.npz' if relation else 'effects.npz'
        candidates = read_json(output / key / 'candidates.json') if relation else None
        with np.load(output / key / effects_file) as measured, np.load(source_root / key / 'prior.npz') as prior:
            for target, text in enumerate(row['response']['token_text']):
                item = rows[(key, target)]
                identity = key+'_'+str(target)
                entries = []
                for source in ([] if relation else np.argsort(-measured['output_js'][target])):
                    entries.append(dict(source=int(source), text=groups[source]['text'],
                        keys=[groups[source]['keys'][0], groups[source]['keys'][-1]+1],
                        js=float(measured['output_js'][target, source]),
                        delta_logp=float(measured['masked_logp'][target, source]-measured['original_logp'][target]),
                        prior=float(prior['prior'][target, source])))
                if relation:
                    entries = relation_entries(candidates, measured, target, measured['original_logp'][target], prior['prior'][target])
                data[identity] = dict(key=key, target=target, text=text, gold=item['gold'],
                    score=item[score_name], selected=item[source_name], entries=entries,
                    covered=item.get('relation_covered', 'True'))
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
if(x.covered==='False'){const c=document.createElement('p');c.textContent='本token所选来源没有可解析关系：主分数0表示弃权，不表示正确。';p.append(c);}
for(const e of x.entries){const a=document.createElement('article'),h=document.createElement('strong'),q=document.createElement('pre');
h.textContent=`源${e.source} keys[${e.keys}] JS=${e.js.toFixed(5)} Δlogp=${e.delta_logp.toFixed(4)} prior=${e.prior.toFixed(4)}`;
q.textContent=e.text+(e.extra?String.fromCharCode(10)+e.extra:'');a.append(h,q);p.append(a);}}
</script>'''.replace('ANSWERS',''.join(answers)).replace('PAYLOAD',payload)
    if relation:
        document = document.replace('全部来源span的实际消融作用', '全部自动关系候选的改写作用和重复来源集合屏蔽作用')
        document = document.replace('自动状态分段', '逐词所选来源改变')
        document = document.replace('Δlogp>0表示屏蔽后原词更容易出现', '关系候选Δlogp>0表示翻转后原词更容易出现；prior为缓存来源权重，集合项为成员权重之和')
    if prefix:
        document = document.replace('内部联合排序选择源', '关系响应选择源')
        document = document.replace('本token所选来源没有可解析关系', '本回答没有自动关系候选')
    (destination / (prefix+'TOKEN_EVIDENCE.html')).write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state', type=Path)
    parser.add_argument('--readout', choices=('selected', 'all'), default='selected')
    args = parser.parse_args()
    report(args.output, args.state, 'all_' if args.readout == 'all' else '')


if __name__ == '__main__':
    main()
