"""Check finite-effect signs with FP32 unembedding and a dose sweep."""
import argparse
from pathlib import Path
from time import perf_counter
import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from experiments.entropy_detection.run import read_json,write_json
from experiments.path_conflict.native import NativeRun,Intervention
from experiments.path_conflict.operators import vocabulary_logits


def readout(model,item,position,patch=None):
    prefix=item['prompt_ids']+item['answer_ids'][:position]
    target=item['answer_ids'][position]
    ids=torch.tensor([prefix],device=model.device)
    with torch.inference_mode(),NativeRun(model,len(prefix),{},[target,target],
            interventions=() if patch is None else patch if isinstance(patch,tuple) else (patch,),record_writes=False):
        state=model.model(ids,use_cache=False).last_hidden_state[0,-1]
        return float(vocabulary_logits(model,state).log_softmax(-1)[target])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','audit','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    items=read_json(args.audit/'manifest.json')['items']
    keys=['15604_0_original','219_0_original','9022_1_original','9022_2_original']
    selected=[item for item in items if item['key'] in keys]
    write_json(args.output/'manifest.json',dict(keys=keys,readout='FP32 unembedding; native transformer remains bfloat16',
        selection='predeclared four primary remaining failures; head choices frozen from original audit',doses=[0.,.05,.1,.25]))
    torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True).to('cuda:0').eval()
    started=perf_counter()
    rows=[]
    calls=0
    for item in selected:
        old=read_json(args.audit/item['key']/'interventions.json')
        old=[row for row in old if row['scope']=='single_head' and row['operation']=='cut' and row['dose']==.25]
        baselines={position:readout(model,item,position) for position in sorted({row['position'] for row in old})}
        calls+=len(baselines)
        for row in old:
            position,group=row['position'],row['group']
            prompt=len(item['prompt_ids'])
            if group=='local_history':
                sources=list(range(max(prompt,prompt+position-16),prompt+position))
            elif group=='claim_prefix':
                sources=list(range(prompt+item['spans'][0][0],prompt+position))
            else:
                sources=item['groups'][group]
            for dose in (0.,.05,.1,.25):
                patch=Intervention(layer=row['layer'],heads=(row['head'],),groups=(group,),sources=tuple(sources),dose=dose)
                changed=readout(model,item,position,patch)
                calls+=1
                rows.append(dict(key=item['key'],position=position,group=group,layer=row['layer'],head=row['head'],dose=dose,
                    local_projection=row['local_projection'],bf16_support_at_025=row['support'],fp32_support=baselines[position]-changed))
        print(item['key'],round(perf_counter()-started,2),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(args.output/'effects.csv',index=False)
    zero=float(frame.loc[frame.dose==0,'fp32_support'].abs().max())
    if zero>1e-6:
        raise ValueError('FP32 zero-dose control failed')
    write_json(args.output/'completed.json',dict(forwards=calls,seconds=perf_counter()-started,zero_dose_error=zero))


if __name__=='__main__':
    main()
