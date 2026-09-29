"""Validate signed transport and automatic source-block interactions by replay."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from experiments.decision_risk_flow.data import read_json,write_json
from experiments.decision_risk_flow.native import prefill,replay
from experiments.decision_risk_flow.run import load_model
from .score import key_effects


@torch.no_grad()
def margin(model,cache,tokens,position,target,competitor,gate=None):
    final,_,_ = replay(model,cache,tokens,position,checkpoints=(32,),gate=gate)
    contrast = model.lm_head.weight[target].float()-model.lm_head.weight[competitor].float()
    return float((final[0]*contrast).sum())


def choose_blocks(block_effect,blocks):
    positive = int(block_effect.argmax())
    negative = min((i for i in range(len(block_effect)) if i!=positive),key=lambda i:block_effect[i])
    length = np.diff(np.array(blocks),axis=-1)[:,0]
    eligible = [i for i in range(len(length)) if i not in (positive,negative)]
    selected = dict(positive=[positive],negative=[negative],joint=sorted({positive,negative}))
    # Two-block prompts support the joint test but have no third matched block.
    if eligible:
        control = min(eligible,key=lambda i:(abs(length[i]-length[negative]),i))
        selected['control'] = [control]
    return selected


def select_probe(row,directory,query):
    features = np.load(directory/'features.npz')
    activity = features['features'][:,:,query,:,0]*features['energy'][:,:,query]
    layer,head,candidate = np.unravel_index(activity.argmax(),activity.shape)
    names = ('native','values','self_values','attention')
    files = {name:np.load(directory/(name+'.npy'),mmap_mode='r') for name in names}
    response = key_effects(files['native'][layer,:,query:query+1],files['values'][layer],
        files['self_values'][layer,:,query:query+1],files['attention'][layer,:,query:query+1])
    effect = response[head,0,candidate,:-1]
    block_effect = np.array([effect[start:end].sum() for start,end in row['blocks']])
    selected = choose_blocks(block_effect,row['blocks'])
    attention = np.asarray(files['attention'][layer,head,query,:-1])
    return int(layer),int(head),int(candidate),effect,attention,selected


@torch.no_grad()
def probe_query(model,cache,row,directory,query):
    layer,head,candidate,effect,attention,selected = select_probe(row,directory,query)
    saved = np.load(directory/'readout.npz')
    target = int(saved['target'][query])
    competitor = int(saved['candidates'][query,candidate])
    token = row['positions'][query]
    position = torch.tensor([len(row['prompt'])-1+token],device=model.device)
    tokens = row['prompt']+row['answer'][:-1]
    baseline = margin(model,cache,tokens,position,target,competitor)
    assert abs(baseline-float(saved['margin'][query,candidate]))<.005
    rows = []
    for treatment,blocks in selected.items():
        mask = np.zeros(len(tokens),dtype=bool)
        for index in blocks:
            start,end = row['blocks'][index]
            mask[start:end] = True
        delta = torch.tensor((attention*mask)[None],device=model.device)
        slope = float(effect[mask].sum())
        for dose in (0.,-.05,.05,-.25,.25):
            gate = dict(layer=layer,head=head,attention_delta=delta,dose=dose)
            observed = margin(model,cache,tokens,position,target,competitor,gate)-baseline
            rows.append(dict(key=row['key'],position=token,query=query,layer=layer,head=head,
                candidate=candidate,competitor=competitor,target=target,treatment=treatment,
                blocks=blocks,dose=dose,predicted=dose*slope,observed=observed,
                prompt_attention=float(attention[mask].sum())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    read_json(args.output/'scores_frozen.json')
    manifest = read_json(args.output/'manifest.json')
    model = load_model(manifest['model'])
    started = perf_counter()
    results = read_json(args.output/'finite_progress.json')['rows'] if args.resume else []
    completed = {r['key'] for r in results}
    for row in manifest['records']:
        if row['key'] in completed:
            continue
        directory = args.output/row['key']
        score = np.load(directory/'scores.npz')['binding_conflict']
        eligible = np.flatnonzero(np.array(row['positions'])>0)
        chosen = sorted({int(eligible[len(eligible)//2]),int(eligible[score[eligible].argmax()])})
        cache,_,_ = prefill(model,row['prompt'],row['answer'],checkpoints=(32,))
        for query in chosen:
            results.extend(probe_query(model,cache,row,directory,query))
        write_json(args.output/'finite_progress.json',dict(rows=results,last=row['key'],seconds=perf_counter()-started))
        print('finite',row['key'],len(results),round(perf_counter()-started,1),flush=True)
    write_json(args.output/'finite.json',dict(status='complete',rows=results,seconds=perf_counter()-started,
        selection='middle and strongest unsigned-label-free conflict query; head selected by conflict times activity',
        intervention='single-head block message amplitude, no renormalization, fixed original history',
        control='different source block matched by token count; absent on two-block prompts, never imputed',
        missing_control_keys=[r['key'] for r in manifest['records'] if len(r['blocks'])<3]))


if __name__=='__main__':
    main()
