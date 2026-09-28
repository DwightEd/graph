"""Compare causal dependence on the reviewed choice prefix and earlier text."""
import argparse
from pathlib import Path
from time import perf_counter
import pandas as pd
import torch
from transformers import AutoModelForCausalLM
from experiments.entropy_detection.run import read_json,write_json
from experiments.path_conflict.native import Intervention
from .audit_precision import readout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','audit','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    items=read_json(args.audit/'manifest.json')['items']
    write_json(args.output/'manifest.json',dict(keys=[item['key'] for item in items],
        prefix='first up to2 tokens at reviewed choice, not the official span start',
        receiver='min(choice+3, claim_end-1); skip when no later claim token exists',
        control='2 preceding answer tokens; distance/wording are not exactly matched',
        readout='FP32 observed next-token logp; not semantic correctness'))
    torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True).to('cuda:0').eval()
    rows,skipped=[],[]
    calls=0
    started=perf_counter()
    for item in items:
        decision=item['decisions'][0]
        position=min(decision+3,item['spans'][0][1]-1)
        if position<=decision:
            skipped.append(dict(key=item['key'],reason='No subsequent token inside reviewed claim'))
            continue
        prompt=len(item['prompt_ids'])
        baseline=readout(model,item,position)
        calls+=1
        groups=dict(choice_prefix=list(range(prompt+decision,prompt+min(decision+2,position))),
                    earlier_prefix=list(range(prompt+max(0,decision-2),prompt+decision)))
        for group,sources in groups.items():
            for dose in (0.,.1,.25):
                patches=tuple(Intervention(layer=layer,groups=(group,),sources=tuple(sources),dose=dose) for layer in range(32))
                changed=readout(model,item,position,patches)
                calls+=1
                rows.append(dict(key=item['key'],kind=item['kind'],decision=decision,position=position,
                    group=group,tokens=len(sources),dose=dose,support=baseline-changed,baseline_logp=baseline))
        print(item['key'],round(perf_counter()-started,2),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(args.output/'effects.csv',index=False)
    zero=float(frame.loc[frame.dose==0,'support'].abs().max())
    if zero>1e-6:
        raise ValueError('Choice-prefix sham failed')
    write_json(args.output/'completed.json',dict(forwards=calls,seconds=perf_counter()-started,
        zero_dose_error=zero,skipped=skipped))


if __name__=='__main__':
    main()
