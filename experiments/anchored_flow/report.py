"""Render exact paired before/after alarms without labelling unknown text as normal."""
import argparse
import csv
import html
from pathlib import Path
from experiments.decision_risk_flow.data import read_json


def status(row,name,threshold):
    if int(row['gold'])<0:
        return 'unknown'
    alarm = float(row[name])>threshold
    if int(row['gold']):
        return 'tp' if alarm else 'fn'
    return 'fp' if alarm else 'tn'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.output/'units.csv').open()))
    limits = read_json(args.output/'thresholds.json')
    content = ['<!doctype html><meta charset="utf-8"><title>基线与消息约束修正</title>',
        '<style>body{max-width:1200px;margin:32px auto;font:16px/1.8 sans-serif}pre{white-space:pre-wrap}',
        '.tp{background:#ffc6c6}.fn{background:#ffe0a6}.fp{background:#d5c9ff}',
        '.unknown{color:#888}section{border-top:1px solid #ccc;padding:20px 0}',
        'span{white-space:pre-wrap}small{color:#555}</style>',
        '<h1>旧基线与新修正逐位置对照</h1>',
        '<p>红色=检出错误；橙色=漏检；紫色=误报；灰色=真值未知。每答上方旧基线、下方新方法。',
        '悬停查看位置和分数。自然回答仅局部有真值；GSM每个块为步骤，后续未知步骤不判真假。</p>']
    for key in dict.fromkeys(r['key'] for r in rows):
        group = [r for r in rows if r['key']==key]
        threshold = limits[group[0]['task']]
        content.append(f'<section><h2>{html.escape(key)}</h2><small>阈值 {threshold:.6f}</small>')
        for name in ('base','anchored_flow'):
            content.append(f'<h3>{name}</h3><div>')
            for row in group:
                title = f"position={row['position']}; gold={row['gold']}; base={float(row['base']):.6f}; new={float(row['anchored_flow']):.6f}"
                content.append(f'<span class="{status(row,name,threshold)}" title="{html.escape(title)}">{html.escape(row["text"])}</span>')
                if row['cohort']=='gsm_step':
                    content.append('<hr>')
            content.append('</div>')
        content.append('</section>')
    (args.output/'TOKEN_AUDIT.html').write_text('\n'.join(content))
    print('rendered',args.output/'TOKEN_AUDIT.html')


if __name__=='__main__':
    main()
