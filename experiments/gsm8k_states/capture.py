"""Capture full-layer step states and fixed-candidate output readouts."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from experiments.gsm8k_recurrence.measure import MODEL, write_json


@torch.no_grad()
def output_readouts(model, states, logits, answer):
    final = logits.float()
    log_probability = final.log_softmax(-1)
    token_nll = -log_probability.gather(1,answer[:,None])[:,0]
    entropy = -(log_probability.exp()*log_probability).sum(-1)
    others = final.clone()
    others.scatter_(1,answer[:,None],-torch.inf)
    competitors = others.topk(31,dim=-1).indices
    candidates = torch.cat((answer[:,None],competitors),-1)
    weights = model.lm_head.weight[candidates]
    margins, conditional = [], []
    for state in states:
        normalized = model.model.norm(state)
        readout = torch.einsum('td,tcd->tc',normalized,weights).float()
        margins.append(readout[:,0]-readout[:,1])
        conditional.append(readout.log_softmax(-1))
    reference = final.gather(1,answer[:,None])[:,0]-others.max(-1).values
    error = float((margins[-1]-reference).abs().max())
    # BF16 matmul kernels can differ between full-vocabulary and selected readouts.
    return dict(margin=torch.stack(margins).cpu().numpy(),
        conditional_logp=torch.stack(conditional).cpu().numpy(),
        nll=token_nll.cpu().numpy(),entropy=entropy.cpu().numpy(),
        final_margin=reference.cpu().numpy(),candidate_ids=candidates.cpu().numpy()),error


def step_states(states, ranges, first):
    mean,entry,end = [],[],[]
    for state in states:
        mean.append(torch.stack([state[first+a:first+b].float().mean(0) for a,b in ranges]))
        entry.append(torch.stack([state[first+a-1] for a,b in ranges]))
        end.append(torch.stack([state[first+b-1] for a,b in ranges]))
    return {name:torch.stack(value).to(torch.float16).cpu().numpy()
            for name,value in [('mean',mean),('entry',entry),('end',end)]}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,default=Path('outputs/gsm8k_recurrence_20260929_v1'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = json.loads((args.previous/'manifest.json').read_text())
    manifest['previous'] = str(args.previous.resolve())
    write_json(args.output/'manifest.json',manifest)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval().requires_grad_(False)
    captured = {}
    hook = model.model.norm.register_forward_pre_hook(lambda module,inputs:captured.update(raw=inputs[0]))
    started = perf_counter()
    errors = []
    for index,row in enumerate(manifest['records']):
        with np.load(row['cache']) as saved:
            ids = torch.tensor(saved['token_ids'][None,:],device='cuda')
            first = int(saved['response_idx'])
            ranges = saved['step_ranges']-first
        result = model(ids,output_hidden_states=True,use_cache=False)
        states = [value[0] for value in result.hidden_states[1:-1]]+[captured['raw'][0]]
        pooled = step_states(states,ranges,first)
        readouts,error = output_readouts(model,[value[first-1:-1] for value in states],
            result.logits[0,first-1:-1],ids[0,first:])
        errors.append(error)
        np.savez_compressed(args.output/(row['id']+'.npz'),**pooled,**readouts,step_ranges=ranges,
            response_ids=ids[0,first:].cpu().numpy())
        del result,states,pooled,readouts
        captured.clear()
        if (index+1)%10==0:
            write_json(args.output/'progress.json',dict(completed=index+1,total=400,seconds=perf_counter()-started))
            print('captured',index+1,round(perf_counter()-started,1),flush=True)
    hook.remove()
    write_json(args.output/'capture_complete.json',dict(status='complete',answers=400,layers=32,hidden=4096,
        seconds=perf_counter()-started,max_selected_readout_margin_difference=max(errors),
        scope='new BF16 SDPA observer replay; full-layer step states; logit lens not native Jacobian'))


if __name__=='__main__':
    main()
