"""Supplement the complete audit at reviewed semantic choices with broader write cuts."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.entropy_detection.run import read_json,write_json
from .audit_inputs import attach_decisions
from .audit_capture import capture_audit
from .audit_run import intervene


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,use_fast=True)
    manifest=read_json(args.audit/'manifest.json')
    items=manifest['items']
    attach_decisions(items,tokenizer)
    items=[item for item in items if item['decisions']]
    write_json(args.output/'manifest.json',dict(items=items,base_audit=str(args.audit),
        decision_positions='reviewed correct/wrong continuation divergence; discovery labels, not automatic detector',
        complete_prompt=True, perturbation_scopes=['single_head','selected_layer_all_heads','all_layers']))
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True).to('cuda:0').eval()
    started=perf_counter()
    calls=0
    for number,item in enumerate(items):
        directory=args.output/item['key']
        directory.mkdir()
        positions=sorted({t for decision in item['decisions'] for t in (decision,min(decision+3,len(item['answer_ids'])-1))})
        captured=capture_audit(model,item,positions)
        if float(captured['reconstruction'].max())>.02:
            raise ValueError('Head-write reconstruction mismatch')
        np.savez_compressed(directory/'audit.npz',**captured)
        calls+=1+intervene(model,item,captured,directory,positions,broad=True)
        print(number+1,len(items),item['key'],round(perf_counter()-started,2),flush=True)
    write_json(args.output/'completed.json',dict(items=len(items),forwards=calls,seconds=perf_counter()-started,
        max_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':
    main()
