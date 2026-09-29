"""Fixed-head, fixed-candidate diagnostic of one value used in different roles."""
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments.decision_risk_flow.data import read_json,write_json
from experiments.decision_risk_flow.native import prefill
from experiments.decision_risk_flow.run import load_model
from .interactions import trajectory
from .sink_center import exchange_effects


def selected_edges(row,files,references,query,candidate):
    layer,head = 30,29
    effect,mass = exchange_effects(files['native'][layer,:,query:query+1],files['values'][layer],
        files['self_values'][layer,:,query:query+1],files['attention'][layer,:,query:query+1],
        references[layer],len(row['prompt']),np.array([query]))
    effect = effect[head,0,candidate,:-1]
    prompt = len(row['prompt'])
    source = int(effect[:prompt].argmin())
    history = prompt+int(effect[prompt:prompt+query-1].argmax())
    eligible = [prompt+i for i in range(query-1) if row['answer'][i]!=row['answer'][query]]
    control = max(eligible,key=lambda i:effect[i])
    treatments = dict(positive=[history],negative=[source],joint=[history,source],
        control=[control],control_joint=[control,source])
    return effect,float(mass[head,0]),treatments


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dense',type=Path,required=True)
    parser.add_argument('--centered',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    (args.output/'paths').mkdir()
    manifest = read_json(args.dense/'manifest.json')
    row = next(r for r in manifest['records'] if r['key']=='gsm8k-49')
    directory = args.dense/row['key']
    files = {name:np.load(directory/(name+'.npy'),mmap_mode='r')
             for name in ('native','values','self_values','attention')}
    references = np.load(args.centered/row['key']/'features.npz')['references']
    saved = np.load(directory/'readout.npz')
    model = load_model(manifest['model'])
    cache,_,_ = prefill(model,row['prompt'],row['answer'],checkpoints=(32,))
    tokens = row['prompt']+row['answer'][:-1]
    results = []
    for query,candidate in ((62,1),(93,0),(101,0),(112,1)):
        target,competitor = int(saved['target'][query]),int(saved['candidates'][query,candidate])
        assert target==1187 and competitor==508
        effect,reference_mass,treatments = selected_edges(row,files,references,query,candidate)
        attention = np.asarray(files['attention'][30,29,query,:-1])
        reference = int(references[30,29])
        contrast = model.lm_head.weight[target].float()-model.lm_head.weight[competitor].float()
        position = torch.tensor([len(row['prompt'])-1+query],device=model.device)
        baseline,base_states,base_lens = trajectory(model,cache,tokens,position,contrast)
        for treatment,keys in treatments.items():
            change = np.zeros(len(tokens),dtype=np.float32)
            change[keys] = attention[keys]*reference_mass
            change[reference] -= change.sum()
            assert abs(float(change.sum()))<1e-7
            for dose in (0.,-.05,.05,-.25,.25):
                gate = dict(layer=30,head=29,attention_delta=torch.tensor(change[None],device=model.device),dose=dose)
                changed,states,lens = trajectory(model,cache,tokens,position,contrast,gate)
                state_file = f'paths/{query}_{treatment}_{dose}.npz'
                np.savez_compressed(args.output/state_file,delta_state=(states-base_states).cpu().numpy(),
                    baseline_state=base_states.cpu().numpy(),delta_lens=(lens-base_lens).cpu().numpy())
                results.append(dict(key=f'gsm8k-49_q{query}',token=query,layer=30,head=29,
                    target_text='24',competitor_text='20',treatment=treatment,dose=dose,
                    keys=keys,history_positions=[k-len(row['prompt']) for k in keys if k>=len(row['prompt'])],
                    block_text=[''.join(row['text'][max(0,k-len(row['prompt'])-4):k-len(row['prompt'])+5])
                                for k in keys if k>=len(row['prompt'])],
                    predicted=dose*float(effect[keys].sum()),observed=changed-baseline,state_file=state_file))
        print('same-value',query,len(results),flush=True)
    write_json(args.output/'interactions.json',dict(status='complete',rows=results,
        scope='posthoc one-answer fixed-head fixed-24-vs-20 diagnostic; not a fitted detector',
        labels='first-error step membership does not label every token; token112 can denote the correct previous price',
        controls='strongest noncopy past message, excluding current self; generator and prompt identical'))


if __name__=='__main__':
    main()
