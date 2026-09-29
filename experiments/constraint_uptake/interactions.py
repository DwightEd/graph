"""Label-selected mechanism audit, with anonymous source blocks and joint controls."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from transformers import AutoTokenizer
from experiments.decision_risk_flow.data import read_json,write_json
from experiments.decision_risk_flow.native import prefill,replay
from experiments.decision_risk_flow.run import load_model
from experiments.role_free_flow.diagnostics import read_local_spans
from .score import key_effects
from .sink_center import exchange_effects


@torch.no_grad()
def trajectory(model,cache,tokens,position,contrast,gate=None):
    checkpoints = tuple(range(1,model.config.num_hidden_layers+1))
    final,states,_ = replay(model,cache,tokens,position,checkpoints=checkpoints,gate=gate)
    states = states[0]
    lens = (model.model.norm(states)*contrast).sum(-1)
    return float((final[0]*contrast).sum()),states,lens


def diagnostic_points(manifest):
    gsm_manifest = read_json('outputs/gsm8k_states_20260929_v1/manifest.json')
    originals = {r['id']:r for r in read_json(gsm_manifest['data'])}
    first = {row['pair']:originals[row['key']]['label'] for row in manifest['records']
             if row['dataset']=='gsm8k' and originals[row['key']]['label']>=0}
    local = read_local_spans(Path('outputs/role_free_flow_20260928'),Path('experiments/path_conflict/paired_cases.json'))
    result = []
    for row in manifest['records']:
        if row['dataset']=='gsm8k':
            step = min(first[row['pair']],len(row['step_ranges'])-1)
            start,end = row['step_ranges'][step]
            token = (start+end-1)//2
            scope = 'middle of annotated first-error step or same ordinal step in correct paired answer'
        elif row['original']['kind']=='natural_replay':
            positions = local[local.trace==row['original']['trace']].position.to_numpy()
            token = int(positions[len(positions)//2])
            scope = 'middle token of previously reviewed local paired claim'
        else:
            continue
        result.append(dict(key=row['key'],token=token,query=row['positions'].index(token),scope=scope))
    return result


def source_treatments(row,directory,query,scored=None):
    saved = np.load((scored or directory)/'features.npz')
    activity = saved['features'][:,:,query,:,0]*saved['energy'][:,:,query]
    layer,head,candidate = np.unravel_index(activity.argmax(),activity.shape)
    files = {name:np.load(directory/(name+'.npy'),mmap_mode='r')
             for name in ('native','values','self_values','attention')}
    reference,reference_mass = None,1.
    if scored is None:
        effects = key_effects(files['native'][layer,:,query:query+1],files['values'][layer],
            files['self_values'][layer,:,query:query+1],files['attention'][layer,:,query:query+1])
    else:
        references = saved['references'][layer]
        effects,mass = exchange_effects(files['native'][layer,:,query:query+1],files['values'][layer],
            files['self_values'][layer,:,query:query+1],files['attention'][layer,:,query:query+1],
            references,len(row['prompt']),np.array([row['positions'][query]]))
        reference,reference_mass = int(references[head]),float(mass[head,0])
    effect = effects[head,0,candidate,:-1]
    attention = np.asarray(files['attention'][layer,head,query,:-1])
    prompt = len(row['prompt'])
    blocks = [(start,min(start+16,prompt)) for start in range(0,prompt,16)]
    impacts = np.array([effect[a:b].sum() for a,b in blocks])
    masses = np.array([attention[a:b].sum() for a,b in blocks])
    positive = int(impacts.argmax())
    negative = min((i for i in range(len(blocks)) if i!=positive),key=lambda i:impacts[i])
    eligible = [i for i in range(len(blocks)) if i not in (positive,negative)]
    control = min(eligible,key=lambda i:(abs((blocks[i][1]-blocks[i][0])-(blocks[negative][1]-blocks[negative][0])),
                                          abs(masses[i]-masses[negative])))
    treatments = dict(positive=[positive],negative=[negative],joint=[positive,negative],
        control=[control],control_joint=[positive,control])
    return int(layer),int(head),int(candidate),blocks,treatments,effect,attention,reference,reference_mass


@torch.no_grad()
def measure_point(model,tokenizer,row,point,directory,output,scored=None):
    query = point['query']
    layer,head,candidate,blocks,treatments,effect,attention,reference,reference_mass = source_treatments(row,directory,query,scored)
    saved = np.load(directory/'readout.npz')
    target,competitor = int(saved['target'][query]),int(saved['candidates'][query,candidate])
    tokens = row['prompt']+row['answer'][:-1]
    position = torch.tensor([len(row['prompt'])-1+point['token']],device=model.device)
    cache,_,_ = prefill(model,row['prompt'],row['answer'],checkpoints=(32,))
    contrast = model.lm_head.weight[target].float()-model.lm_head.weight[competitor].float()
    baseline,base_states,base_lens = trajectory(model,cache,tokens,position,contrast)
    assert abs(baseline-float(saved['margin'][query,candidate]))<.005
    results = []
    for treatment,indices in treatments.items():
        mask = np.zeros(len(tokens),dtype=bool)
        for index in indices:
            start,end = blocks[index]
            mask[start:end] = True
        change = attention*mask*reference_mass
        if reference is not None:
            change[reference] -= change.sum()
            assert abs(float(change.sum()))<1e-7
            assert np.min(attention-.25*change)>-1e-7
            assert np.min(attention+.25*change)>-1e-7
        delta = torch.tensor(change[None],device=model.device)
        slope = float(effect[mask].sum())
        for dose in (0.,-.05,.05,-.25,.25):
            gate = dict(layer=layer,head=head,attention_delta=delta,dose=dose)
            changed,states,lens = trajectory(model,cache,tokens,position,contrast,gate)
            observed = changed-baseline
            difference = states-base_states
            state_file = f"paths/{row['key']}_{treatment}_{dose}.npz"
            np.savez_compressed(output/state_file,delta_state=difference.cpu().numpy(),
                baseline_state=base_states.cpu().numpy(),delta_lens=(lens-base_lens).cpu().numpy())
            results.append(dict(**point,layer=layer,head=head,candidate=candidate,
                target_text=tokenizer.decode([target]),competitor_text=tokenizer.decode([competitor]),
                treatment=treatment,blocks=[blocks[i] for i in indices],
                reference=reference,reference_mass=reference_mass,
                block_text=[tokenizer.decode(row['prompt'][blocks[i][0]:blocks[i][1]]) for i in indices],
                source_attention=float(attention[mask].sum()),dose=dose,predicted=dose*slope,observed=observed,
                state_file=state_file,path_norm=torch.linalg.vector_norm(difference,dim=-1).cpu().tolist(),
                path_lens=(lens-base_lens).cpu().tolist()))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dense',type=Path,required=True)
    parser.add_argument('--rag-factors',type=Path,required=True)
    parser.add_argument('--scored',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    read_json(args.dense/'capture_complete.json')
    manifest = read_json(args.dense/'manifest.json')
    output = args.output or args.dense
    if args.output is not None:
        output.mkdir(exist_ok=False)
    (output/'paths').mkdir()
    points = diagnostic_points(manifest)
    model = load_model(manifest['model'])
    tokenizer = AutoTokenizer.from_pretrained(manifest['model'],local_files_only=True)
    rows = []
    started = perf_counter()
    for point in points:
        row = next(r for r in manifest['records'] if r['key']==point['key'])
        root = args.dense if row['dataset']=='gsm8k' else args.rag_factors
        scored = args.scored/row['key'] if args.scored is not None else None
        rows.extend(measure_point(model,tokenizer,row,point,root/row['key'],output,scored))
        write_json(output/'interaction_progress.json',dict(rows=rows,last=row['key']))
        print('interaction',row['key'],len(rows),round(perf_counter()-started,1),flush=True)
    write_json(output/'interactions.json',dict(status='complete',rows=rows,seconds=perf_counter()-started,
        operator='mass-preserving exchange' if args.scored is not None else 'message amplitude',
        scope='label-selected diagnostic positions; automatic 16-token prompt blocks, no evidence-role labels',
        caution='same ordinal steps and different natural prefixes are not identical semantic decisions',
        detector_input=False))


if __name__=='__main__':
    main()
