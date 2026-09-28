"""Validate context-completion derivatives by native finite reallocation."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from experiments.decision_risk_flow.data import read_json,write_json
from experiments.decision_risk_flow.native import prefill,replay
from experiments.decision_risk_flow.run import load_model
from experiments.route_complement.capture import get_inputs
from .measure import relation_matrices,completion_direction,response


def selected_queries(features):
    count = features.shape[2]
    uniform = np.linspace(0,count-1,3).astype(int)
    extreme = np.argsort(np.abs(features[...,:2]).max((0,1,3)))[-3:]
    return np.unique(np.r_[uniform,extreme])


def prepare_delta(row,manifest,token,layer,head,direction,device):
    source = Path(manifest['relation_base'])/'sources'/str(row['source_id'])
    saved = np.load(source/'input.npz')
    indices = np.flatnonzero(saved['source_mask'])
    query = np.load(source/'query.npy',mmap_mode='r')[layer,head:head+1][:,indices]
    key = np.load(source/'key.npy',mmap_mode='r')[layer,head//4:head//4+1][:,indices]
    relations = relation_matrices(torch.tensor(query,device=device),torch.tensor(key,device=device),
                                  torch.tensor(indices,device=device))
    base = Path(manifest['base'])/row['key']
    attention = np.load(base/'attention.npy',mmap_mode='r')[layer,head,token]
    gradient = np.load(base/'derivative.npy',mmap_mode='r')[layer,head,token]
    reading = torch.tensor(attention[indices],device=device)[None,None]
    effect = torch.tensor(gradient[indices],device=device)[None,None]
    change = completion_direction(reading,relations[direction])
    slope = float(response(reading,effect,change).item())
    delta = torch.zeros(1,len(attention),device=device)
    delta[:,indices] = change[0]
    return delta,slope


@torch.no_grad()
def native_margin(model,cache,tokens,position,target,alternative,gate=None):
    final,_,_ = replay(model,cache,tokens,position,checkpoints=(32,),gate=gate)
    logits = model.lm_head(final).float()[0]
    return float((logits[target]-logits[alternative]).item())


def check_record(model,row,manifest,output,inputs_tuple):
    prompt,answer,_ = inputs_tuple
    cache,_,_ = prefill(model,prompt,answer,checkpoints=(32,))
    readout = np.load(Path(manifest['base'])/row['key']/'readouts.npz')
    features = np.load(output/row['key']/'responses.npz')['values']
    tokens = prompt+answer[:-1]
    rows = []
    for token in selected_queries(features):
        position = torch.tensor([len(prompt)-1+int(token)],device=model.device)
        target,alternative = answer[token],int(readout['alternative'][token])
        baseline = native_margin(model,cache,tokens,position,target,alternative)
        assert abs(baseline-float(readout['confidence'][token,2]))<.005
        for direction in (0,1):
            layer,head = np.unravel_index(np.abs(features[:,:,token,direction]).argmax(),(32,32))
            delta,slope = prepare_delta(row,manifest,token,int(layer),int(head),direction,model.device)
            assert abs(float(delta.sum()))<1e-5
            for dose in (0.,.01,.05,.25):
                gate = dict(layer=int(layer),head=int(head),attention_delta=delta,dose=dose)
                measured = native_margin(model,cache,tokens,position,target,alternative,gate)-baseline
                rows.append(dict(key=row['key'],token=int(token),direction=direction,layer=int(layer),
                    head=int(head),dose=dose,predicted=dose*slope,measured=measured))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    read_json(args.output/'scores_frozen.json')
    manifest = read_json(args.output/'manifest.json')
    root = Path(next(row['root'] for row in manifest['records'] if row['kind']=='observer'))
    lookup = {str(r['source_id']):r['source_file'] for r in read_json(root/'manifest.json')['records']}
    selected = [r for r in manifest['records'] if r['role']=='regression' or r['key'] in ('00005','00006','00012','00013')]
    model = load_model(manifest['model'])
    rows,started = [],perf_counter()
    for row in selected:
        values = get_inputs(row,Path(manifest['samples']),root,lookup)
        rows.extend(check_record(model,row,manifest,args.output,values))
        write_json(args.output/'finite_progress.json',dict(status='running',last_key=row['key'],rows=rows))
        print(dict(key=row['key'],seconds=perf_counter()-started),flush=True)
    write_json(args.output/'finite_effects.json',dict(status='complete',rows=rows,seconds=perf_counter()-started,
        selection='three uniform and three largest absolute first-order response positions per answer; strongest head per direction',
        labels_used=False,scope='single-head source reallocation with original past KV; downstream native replay'))


if __name__=='__main__':
    main()
