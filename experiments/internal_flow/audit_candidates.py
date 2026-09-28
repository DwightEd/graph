"""Same-prefix correct/wrong readouts and exhaustive layer-wise write interventions."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.entropy_detection.run import read_json,write_json
from experiments.path_conflict.native import NativeRun,Intervention
from .inputs import CASES


def compile_cases(audit,tokenizer):
    items=read_json(audit/'manifest.json')['items']
    result=[]
    for case in CASES:
        number=[row['name'] for row in CASES if row['response_id']==case['response_id']].index(case['name'])
        item=next(row for row in items if row['key']==f"{case['response_id']}_{number}_original")
        begin=item['answer'].index(case['anchor']+case['wrong'])+len(case['anchor'])
        prefix=item['answer'][:begin]
        candidates={side:tokenizer(prefix+case[side],add_special_tokens=False)['input_ids'] for side in ('correct','wrong')}
        common=next(i for i,(left,right) in enumerate(zip(candidates['correct'],candidates['wrong'])) if left!=right)
        np.testing.assert_array_equal(item['answer_ids'][:common+1],candidates['wrong'][:common+1])
        prompt=len(item['prompt_ids'])
        groups=dict(item['groups'],local_history=list(range(max(prompt,prompt+common-16),prompt+common)))
        result.append(dict(item,case=case['name'],decision=common,groups=groups,
            prefix_ids=item['prompt_ids']+item['answer_ids'][:common],
            candidates={side:tokens[common:] for side,tokens in candidates.items()}))
    return result


def score(model,case,patch=None,side=None):
    prefix=case['prefix_ids']
    candidates=case['candidates']
    target=[candidates[name][0] for name in ('correct','wrong')]
    continuation=[] if side is None else candidates[side][:-1]
    ids=torch.tensor([prefix+continuation],device=model.device)
    with torch.inference_mode(),NativeRun(model,len(prefix),case['groups'],target,
            interventions=() if patch is None else (patch,),record_writes=False):
        hidden=model.model(ids,use_cache=False).last_hidden_state[0,len(prefix)-1:]
        lp=model.lm_head(hidden).float().log_softmax(-1)
    if side is None:
        return dict(margin=float(lp[0,target[0]]-lp[0,target[1]]),correct_logp=float(lp[0,target[0]]),wrong_logp=float(lp[0,target[1]]))
    selected=lp[torch.arange(len(candidates[side]),device=model.device),torch.tensor(candidates[side],device=model.device)]
    return dict(sum=float(selected.sum()),mean=float(selected.mean()),tokens=len(selected))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','audit','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,use_fast=True)
    cases=compile_cases(args.audit,tokenizer)
    write_json(args.output/'manifest.json',dict(cases=cases,discovery_only=True,
        candidate_warning='reviewed continuations; first-token margin may test a connector, so full-candidate sum and mean are also measured',
        complete_prompt=True,dose=.25,layer_selection='all32; no cherry-picked layer in first-token scan'))
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True).to('cuda:0').eval()
    started=perf_counter()
    calls=0
    for case in cases:
        directory=args.output/case['case']
        directory.mkdir()
        baseline=score(model,case)
        sham=score(model,case,Intervention(layer=0,groups=('source',),sources=tuple(case['groups']['source']),dose=0.))
        calls+=2
        if abs(sham['margin']-baseline['margin'])>1e-6:
            raise ValueError('Zero-dose candidate margin changed')
        rows=[]
        for layer in range(32):
            for group in ('source','evidence','constraint','local_history','mlp'):
                if group!='mlp' and not case['groups'][group]:
                    continue
                patch=Intervention(layer=layer,groups=(group,),dose=.25,
                    sources=() if group=='mlp' else tuple(case['groups'][group]))
                changed=score(model,case,patch)
                calls+=1
                rows.append(dict(layer=layer,group=group,**changed,
                    margin_change=changed['margin']-baseline['margin'],
                    native_support_for_correct=baseline['margin']-changed['margin']))
        pd.DataFrame(rows).to_csv(directory/'layer_interventions.csv',index=False)
        write_json(directory/'baseline.json',dict(**baseline,correct_first=tokenizer.decode([case['candidates']['correct'][0]]),
            wrong_first=tokenizer.decode([case['candidates']['wrong'][0]]),zero_dose_error=sham['margin']-baseline['margin']))
        # Keep all groups and both signs; extrema are explicitly discovery-selected.
        sequence_baseline={side:score(model,case,side=side) for side in ('correct','wrong')}
        calls+=2
        sequences=[]
        for group in ('source','constraint','local_history','mlp'):
            selected=[row for row in rows if row['group']==group]
            if not selected:
                continue
            for criterion,row in [('largest_correct_support',max(selected,key=lambda r:r['native_support_for_correct'])),
                                  ('largest_wrong_support',min(selected,key=lambda r:r['native_support_for_correct']))]:
                patch=Intervention(layer=row['layer'],groups=(group,),dose=.25,
                    sources=() if group=='mlp' else tuple(case['groups'][group]))
                changed={side:score(model,case,patch,side) for side in ('correct','wrong')}
                calls+=2
                sequences.append(dict(group=group,layer=row['layer'],selection=criterion,
                    first_token_margin_change=row['margin_change'],scores=changed,
                    sum_margin_change=(changed['correct']['sum']-changed['wrong']['sum'])-(sequence_baseline['correct']['sum']-sequence_baseline['wrong']['sum']),
                    mean_margin_change=(changed['correct']['mean']-changed['wrong']['mean'])-(sequence_baseline['correct']['mean']-sequence_baseline['wrong']['mean'])))
        write_json(directory/'sequence.json',dict(baseline=sequence_baseline,interventions=sequences,
            limitations='forced candidate likelihood, length/wording sensitivity; not free generation repair'))
        print(case['case'],baseline,'seconds',round(perf_counter()-started,2),flush=True)
    write_json(args.output/'completed.json',dict(cases=len(cases),forwards=calls,seconds=perf_counter()-started))


if __name__=='__main__':
    main()
