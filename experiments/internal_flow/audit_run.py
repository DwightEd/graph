"""Audit exposed failures and matched controls, without changing any source text."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.entropy_detection.run import write_json
from experiments.path_conflict.native import Intervention, NativeRun
from .audit_capture import capture_audit, GROUPS, FIELDS, group_masks
from .audit_inputs import original_items, natural_items


def select_queries(item, trajectory):
    source = trajectory['mass'][:,:,:,0].mean((0,1))
    gain = np.array([source[t]-source[max(0,t-8):t].mean() if t else 0 for t in range(len(source))])
    automatic = np.argsort(-gain)[:3]
    automatic = [int(t) for t in automatic if gain[t] > 0]
    positions = set(automatic)
    if not item['spans']:
        positions.update(np.linspace(0,len(source)-1,5,dtype=int).tolist())
    for start, stop in item['spans']:
        positions.update(t for t in [start-1,start,start+1,start+3,start+7,(start+stop)//2,stop-1,stop,stop+3] if 0<=t<len(source))
    return sorted(positions), dict(automatic_positions=automatic, source_mass_gain=gain.tolist(),
        rule='top3 positive mean source-attention rises above previous8 positions; proxy, not validated reanchor',
        gold_positions_for_audit_only=item['spans'])


def token_logp(model, item, position, intervention=None):
    ids = item['prompt_ids']+item['answer_ids'][:position]
    target = item['answer_ids'][position]
    tensor = torch.tensor([ids],device=model.device)
    with torch.inference_mode(), NativeRun(model,len(ids),{},[target,target],
            interventions=() if intervention is None else intervention if isinstance(intervention,tuple) else (intervention,),record_writes=False):
        hidden = model.model(tensor,use_cache=False).last_hidden_state[0,-1]
        logp = model.lm_head(hidden).float().log_softmax(-1)
    return float(logp[target])


def intervene(model, item, captured, directory, probes=None, broad=False):
    positions = captured['positions'].tolist()
    if probes is None:
        probes = set()
        for start,stop in item['spans']:
            probes.update([start,(start+stop)//2])
    rows=[]
    for position in sorted(probes):
        qi=positions.index(position)
        masks=group_masks(item,[position],len(item['prompt_ids'])+position,model.device)[0]
        baseline=token_logp(model,item,position)
        for group in ('source','evidence','constraint','local_history','claim_prefix'):
            gi=GROUPS.index(group)
            sources=torch.nonzero(masks[gi]).flatten().tolist()
            if not sources:
                continue
            signal=captured['stats'][:,:,qi,gi,2]
            layer,head=np.unravel_index(np.abs(signal).argmax(),signal.shape)
            for operation,dose in [('cut',0.),('cut',.25),('random',.25)]:
                patch=Intervention(layer=int(layer),groups=(group,),heads=(int(head),),
                    sources=tuple(sources),operation=operation,dose=dose)
                changed=token_logp(model,item,position,patch)
                rows.append(dict(position=position,group=group,layer=int(layer),head=int(head),
                    operation=operation,dose=dose,scope='single_head',baseline_logp=baseline,changed_logp=changed,
                    support=baseline-changed,local_projection=float(signal[layer,head]),
                    interpretation='positive means native write supports observed token, not factual correctness'))
            if broad:
                for scope,layers in [('selected_layer_all_heads',[int(layer)]),('all_layers',range(len(model.model.layers)))]:
                    patches=tuple(Intervention(layer=li,groups=(group,),sources=tuple(sources),dose=.25) for li in layers)
                    changed=token_logp(model,item,position,patches)
                    rows.append(dict(position=position,group=group,layer=int(layer),head=-1,
                        operation='cut',dose=.25,scope=scope,baseline_logp=baseline,changed_logp=changed,
                        support=baseline-changed,local_projection=float(signal.sum()),
                        interpretation='multi-head/layer perturbation includes interactions, not additive path attribution'))
    if any(abs(row['support'])>1e-6 for row in rows if row['dose']==0):
        raise ValueError('Zero-dose internal intervention changed observed log probability')
    write_json(directory/'interventions.json',rows)
    return len(rows)+len(probes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','packs','fixtures','samples','states','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--keys',nargs='*')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,use_fast=True)
    items=original_items(args.packs,args.fixtures,tokenizer)+natural_items(args.samples,args.states,tokenizer)
    if args.keys:
        items=[item for item in items if item['key'] in args.keys]
    write_json(args.output/'manifest.json',dict(items=items,groups=GROUPS,fields=FIELDS,
        source_deletion=False,labels_for_discovery=True,new_detector=False,
        local_projection='gradient of local RMSNorm+unembedding observed logp; no later layers',
        native_intervention='25% removal or equal-norm random internal head write, complete prompt retained'))
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True).to('cuda:0').eval()
    started=perf_counter()
    calls=0
    for number,item in enumerate(items):
        directory=args.output/item['key']
        directory.mkdir()
        trajectory=capture_audit(model,item)
        calls+=1
        positions,selection=select_queries(item,trajectory)
        write_json(directory/'selection.json',selection)
        np.savez_compressed(directory/'trajectory.npz',**trajectory)
        captured=capture_audit(model,item,positions)
        calls+=1
        if float(captured['reconstruction'].max())>.02:
            raise ValueError('Head writes failed to reconstruct native attention output')
        np.savez_compressed(directory/'audit.npz',**captured)
        calls+=intervene(model,item,captured,directory)
        write_json(directory/'tokens.json',dict(prompt_tokens=tokenizer.convert_ids_to_tokens(item['prompt_ids']),
            answer_tokens=tokenizer.convert_ids_to_tokens(item['answer_ids'])))
        print(number+1,len(items),item['key'],'seconds',round(perf_counter()-started,2),flush=True)
    write_json(args.output/'completed.json',dict(items=len(items),forwards=calls,
        seconds=perf_counter()-started,max_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    main()
