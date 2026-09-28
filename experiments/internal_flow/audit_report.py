"""Tabulate every measured query and keep evidence roles separate from truth claims."""
import argparse
from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from experiments.entropy_detection.run import read_json,write_json
from .audit_capture import GROUPS,FIELDS


def summarize(root,scope,tokenizer):
    manifest=read_json(root/'manifest.json')
    rows,heads,keys,interventions,layers,coverage=[],[],[],[],[],[]
    for item in manifest['items']:
        directory=root/item['key']
        if not (directory/'audit.npz').exists():
            continue
        with np.load(directory/'audit.npz') as saved:
            values={name:saved[name] for name in saved.files}
        groups=GROUPS[:values['stats'].shape[-2]]
        positions=values['positions'].tolist()
        all_ids=item['prompt_ids']+item['answer_ids'][:-1]
        pieces=[tokenizer.decode([token]) for token in all_ids]
        prefix=len(item['prompt_ids'])
        info=dict(scope=scope,key=item['key'],kind=item['kind'],response_id=item['response_id'],
            source_id=item['source_id'],task=item['task'],evidence_status=item['evidence_status'])
        coverage.append(dict(info,queries=len(positions),physical_heads=int(np.prod(values['stats'].shape[:2])),
            reconstruction_max=float(values['reconstruction'].max())))
        for qi,position in enumerate(positions):
            basic=dict(info,position=position,token=tokenizer.decode([item['answer_ids'][position]]),
                annotated_start=item['spans'][0][0] if item['spans'] else None,
                decision=item.get('decisions',[None])[0] if item.get('decisions') else None)
            for gi,group in enumerate(groups):
                for block,left in [('all',0),('late16',16)]:
                    array=values['stats'][left:,:,qi,gi]
                    signed=array[...,2]
                    rows.append(dict(basic,group=group,block=block,
                        attention_mass=float(array[...,0].mean()),write_norm=float(array[...,1].mean()),
                        local_projection_net=float(signed.sum(1).mean()),
                        local_projection_positive=float(np.maximum(signed,0).sum(1).mean()),
                        local_projection_negative=float(np.minimum(signed,0).sum(1).mean()),
                        residual_cosine=float(array[...,3].mean())))
                array=values['stats'][:,:,qi,gi]
                for layer,head in zip(*np.unravel_index(np.argsort(-np.abs(array[...,2]),axis=None)[:5],array.shape[:2])):
                    heads.append(dict(basic,group=group,layer=int(layer),head=int(head),
                        **{name:float(array[layer,head,j]) for j,name in enumerate(FIELDS)}))
            # Aggregate only for display. Complete physical-head/key arrays remain in the NPZ.
            attention=values['attention'][:,:,qi].astype(float).mean((0,1))
            projection=values['key_projection'][:,:,qi].astype(float).sum(1).mean(0)
            source=np.array(item['groups']['source'],dtype=int)
            for criterion,score in [('source_attention',attention),('source_positive_projection',projection),('source_negative_projection',-projection)]:
                selected=source[np.argsort(-score[source])[:8]]
                for key in selected:
                    keys.append(dict(basic,criterion=criterion,key_position=int(key),key_text=pieces[key],
                        context=''.join(pieces[max(0,key-3):min(len(pieces),key+4)]),
                        evidence=bool(key in item['groups']['evidence']),constraint=bool(key in item['groups']['constraint']),
                        attention_mass=float(attention[key]),local_projection=float(projection[key])))
            if 'site_readouts' in values:
                for layer in range(len(values['site_readouts'])):
                    site=values['site_readouts'][layer,qi]
                    mlp=values['mlp_stats'][layer,qi]
                    layers.append(dict(basic,layer=layer+1,pre_entropy_bits=float(site[0,0]/np.log(2)),
                        attention_entropy_bits=float(site[1,0]/np.log(2)),post_entropy_bits=float(site[2,0]/np.log(2)),
                        pre_surprisal=float(site[0,1]),attention_surprisal=float(site[1,1]),post_surprisal=float(site[2,1]),
                        attention_logp_gain=float(site[0,1]-site[1,1]),mlp_logp_gain=float(site[1,1]-site[2,1]),
                        mlp_norm=float(mlp[0]),mlp_local_projection=float(mlp[1]),mlp_residual_cosine=float(mlp[2])))
        if (directory/'interventions.json').exists():
            for row in read_json(directory/'interventions.json'):
                row=dict(row)
                intervention_scope=row.pop('scope','single_head')
                interventions.append(dict(info,intervention_scope=intervention_scope,**row))
    return dict(measurements=rows,strongest_heads=heads,prompt_keys=keys,interventions=interventions,
        layer_readouts=layers,coverage=coverage)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--decisions',type=Path,required=True)
    parser.add_argument('--reanchors',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,use_fast=True)
    combined={}
    for scope,root in [('span',args.audit),('decision',args.decisions),('reanchor',args.reanchors)]:
        for name,rows in summarize(root,scope,tokenizer).items():
            combined.setdefault(name,[]).extend(rows)
    for name,rows in combined.items():
        pd.DataFrame(rows).to_csv(args.output/f'{name}.csv',index=False)
    table=pd.DataFrame(combined['measurements'])
    decisive=table[(table.scope=='decision')&(table.block=='late16')&(table.position==table.decision)]
    pivot=decisive.pivot(index=['key','kind','token'],columns='group',values='attention_mass').reset_index()
    pivot.to_csv(args.output/'decision_attention.csv',index=False)
    reanchor=[]
    for item in read_json(args.reanchors/'manifest.json')['items']:
        selection=read_json(args.reanchors/item['key']/'selection.json')
        reanchor.append(dict(key=item['key'],kind=item['kind'],decisions=selection['decisions'],
            total_nodes=len(selection['all_nodes']),measured_nodes=selection['measured_nodes']))
    write_json(args.output/'reanchor_coverage.json',reanchor)
    html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>漏检内部信号审计</title>',
        '<style>body{font:15px system-ui;margin:24px;color:#20242a}table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #ddd;padding:6px}details{margin:20px 0}input{padding:8px;width:350px}</style>',
        '<h1>漏检内部信号审计</h1><p>标签辅助机制诊断；正值只表示支持实际输出词。原回答使用观察模型重放，人工正确对照的措辞/历史不同。原始全头数组保留在各实验目录。</p>',
        '<input id="filter" placeholder="输入样本号，如 15604 或 9022">']
    for key in table.key.unique():
        rows=table[(table.key==key)&(table.block=='late16')]
        detail=rows[['scope','position','token','group','attention_mass','write_norm','local_projection_net']]
        effects=pd.DataFrame([r for r in combined['interventions'] if r['key']==key and r['dose']>0])
        html.append('<section data-key="'+escape(key)+'"><h2>'+escape(key)+'</h2><details><summary>所有测量位置：后16层汇总</summary>'+detail.to_html(index=False,float_format=lambda x:f'{x:.5f}')+'</details><details><summary>实际内部干预</summary>'+effects.to_html(index=False,float_format=lambda x:f'{x:.5f}')+'</details></section>')
    html.append('<script>document.getElementById("filter").oninput=e=>document.querySelectorAll("section").forEach(s=>s.hidden=!s.dataset.key.includes(e.target.value));</script></html>')
    (args.output/'index.html').write_text(''.join(html),encoding='utf-8')
    write_json(args.output/'verification.json',dict(measured_items=len(combined['coverage']),
        max_reconstruction_error=max(row['reconstruction_max'] for row in combined['coverage']),
        zero_dose_max=max(abs(row['support']) for row in combined['interventions'] if row['dose']==0),
        source_text_deleted=False,new_detector=False,reviewer='same-agent numerical checks'))
    print('REPORT_COMPLETE',args.output,flush=True)


if __name__=='__main__':
    main()
