"""Locate original-rule reanchor nodes, then inspect their signed information writes."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.entropy_detection.run import read_json,write_json
from .audit_capture import capture_audit
from .audit_inputs import attach_decisions
from .audit_reanchors import states
from .audit_run import intervene


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','output','model'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,use_fast=True)
    items=read_json(args.audit/'manifest.json')['items']
    attach_decisions(items,tokenizer)
    for item in items:
        item['special_ids']=tokenizer.all_special_ids
    write_json(args.output/'manifest.json',dict(items=items,
        rule='original label-free-local-to-old-switch-v2: current cutoff, window10, baseline3, shift .1, episode starts',
        node_selection_labels=False, audit_subset='nearest preceding and following nodes within8 of reviewed decision; normal controls first3 nodes'))
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
        heads,nodes=states(trajectory['switch_measures'])
        node_positions=np.flatnonzero(nodes==1)
        positions=[]
        for decision in item['decisions']:
            before=node_positions[(node_positions<=decision)&(node_positions>=decision-8)]
            after=node_positions[(node_positions>decision)&(node_positions<=decision+8)]
            if len(before):
                positions.append(int(before[-1]))
            if len(after):
                positions.append(int(after[0]))
        if not item['decisions']:
            positions=node_positions[:3].tolist()
        positions=sorted(set(positions))
        np.savez_compressed(directory/'trajectory.npz',**trajectory,head_states=heads,node_states=nodes)
        write_json(directory/'selection.json',dict(all_nodes=node_positions.tolist(),measured_nodes=positions,
            decisions=item['decisions'],window=8,all_nodes_label_free=True,measured_subset_label_assisted=True))
        if positions:
            captured=capture_audit(model,item,positions)
            calls+=1
            if captured['reconstruction'].max()>.02:
                raise ValueError('Native head-write reconstruction failed')
            np.savez_compressed(directory/'audit.npz',**captured)
            calls+=intervene(model,item,captured,directory,positions,broad=False)
        print(number+1,len(items),item['key'],'nodes',len(node_positions),'measured',positions,round(perf_counter()-started,2),flush=True)
    write_json(args.output/'completed.json',dict(items=len(items),forwards=calls,seconds=perf_counter()-started,
        max_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    main()
