"""A portable browser for every token and every frozen detector."""

import argparse
import csv
import json
from pathlib import Path

from experiments.decision_risk_flow.data import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    evaluation = read_json(args.output / 'evaluation.json')
    rows = list(csv.DictReader((args.output / 'tokens.csv').open()))
    cases = {}
    for row in rows:
        cases.setdefault(row['key'], []).append(row)
    main_method = read_json(args.output / 'scores_frozen.json')['main']
    payload = json.dumps(dict(evaluation=evaluation, cases=cases, main=main_method), ensure_ascii=False).replace('<', '\\u003c')
    html = '''<!doctype html><meta charset="utf-8"><title>JS 与消息传播实测</title>
<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 20px;color:#182431}
table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}
select{font:inherit;padding:8px} .token{display:inline-block;padding:3px;margin:2px;border:2px solid transparent;white-space:pre-wrap}
.error{background:#ffe3a8}.alarm{border-color:#bb3150}.both{background:#efbdc8}article{margin:24px 0;line-height:2}
.note{background:#eef2f7;padding:15px}small{color:#58616b}</style>
<h1>JS 与逐头消息传播：全部回归位置</h1>
<p class="note">无标签拟合与校准。8 个历史已暴露回答；不是未见测试。黄色为官方错误，红框为报警，粉红为检出错误。鼠标停留查看 token 位置和分数。自然采样仅有四处局部声明标签，结果另列。</p>
<label>检测方法 <select id="method"></select></label><div id="summary"></div><div id="cases"></div>
<h2>自然采样局部对照</h2><div id="natural"></div>
<script>const data=PAYLOAD;
const selector=document.querySelector('#method');
for(const name of Object.keys(data.evaluation.aggregate)){const option=document.createElement('option');option.value=name;option.textContent=name;selector.appendChild(option)}
selector.value=data.main;
function render(){const name=selector.value;const a=data.evaluation.aggregate[name];
document.querySelector('#summary').textContent=`错误检出 ${a.detected_errors}/${a.error_tokens}；正常误报 ${a.false_alarms}/${a.normal_tokens}；段覆盖 ${a.any_span}/${a.total_spans}；整段覆盖 ${a.full_span}/${a.total_spans}；平均答内 AUROC ${a.mean_within_answer_auroc.toFixed(4)}`;
const container=document.querySelector('#cases');container.replaceChildren();
for(const [key,tokens] of Object.entries(data.cases)){const metric=data.evaluation.cases.find(x=>x.key===key&&x.method===name);const article=document.createElement('article');const title=document.createElement('h2');title.textContent=`${key}：检出 ${metric.detected_errors}/${metric.error_tokens}，误报 ${metric.false_alarms}/${metric.normal_tokens}`;article.appendChild(title);
for(const row of tokens){const token=document.createElement('span');const score=Number(row[name]);const error=Number(row.label)===1;const alarm=score>metric.threshold;token.className='token'+(error?' error':'')+(alarm?' alarm':'')+(error&&alarm?' both':'');token.textContent=row.text;token.title=`位置 ${row.token}；分数 ${row[name]}；阈值 ${metric.threshold}`;article.appendChild(token)}container.appendChild(article)}
const natural=document.querySelector('#natural');natural.replaceChildren();for(const row of data.evaluation.natural_local.filter(x=>x.method===name)){const p=document.createElement('p');p.textContent=`${row.case} ${row.side}：错误检出 ${row.detected_errors}/${row.error_tokens}，正常误报 ${row.false_alarms}/${row.normal_tokens}，可测 ${row.scored_tokens}/${row.eligible_tokens}`;natural.appendChild(p)}}
selector.onchange=render;render();</script>'''.replace('PAYLOAD', payload)
    (args.output / 'CASE_BROWSER.html').write_text(html)


if __name__ == '__main__':
    main()
