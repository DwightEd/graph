"""Second-round effect-weighted readout; frozen factor observations stay intact."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import softmax
from experiments.decision_risk_flow.data import read_json,write_json
from .score import key_effects


def weighted_heads(values,energy):
    return (values*energy).sum((0,1))/np.maximum(energy.sum((0,1)),1e-12)


def net_conflict(row,directory,energy):
    files = {name:np.load(directory/(name+'.npy'),mmap_mode='r')
             for name in ('native','values','self_values','attention')}
    result = []
    prompt = len(row['prompt'])
    positions = np.array(row['positions'])
    for layer in range(len(files['native'])):
        effect = key_effects(files['native'][layer],files['values'][layer],files['self_values'][layer],files['attention'][layer])
        source = effect[...,:prompt].sum(-1)
        source += effect[...,-1]*(positions==0)[None,:,None]
        history = effect.sum(-1)-source
        conflict = 2*np.sqrt(np.maximum(-source,0)*np.maximum(history,0))
        result.append(conflict/np.maximum(energy[layer],1e-12))
    return weighted_heads(np.stack(result),energy)


def record(row,source,output):
    directory = source/row['key']
    target = output/row['key']
    target.mkdir()
    features = np.load(directory/'features.npz')
    values,energy = features['features'],features['energy']
    margins = np.load(directory/'readout.npz')['margin']
    # z_competitor = z_actual - margin; the unknown common actual logit cancels.
    weights = softmax(-margins,axis=-1)
    candidates = dict(energy_conflict=weighted_heads(values[...,0],energy),
        energy_opposition=weighted_heads(values[...,1],energy),
        energy_reversal=weighted_heads(values[...,2],energy),
        energy_cancellation=weighted_heads(values[...,3],energy),
        net_conflict=net_conflict(row,directory,energy))
    scores = {name:(value*weights).sum(-1) for name,value in candidates.items()}
    scores['energy_conflict_max'] = candidates['energy_conflict'].max(-1)
    with np.load(directory/'scores.npz') as old:
        for name in old.files:
            scores[name] = old[name]
    np.savez_compressed(target/'scores.npz',**scores)
    np.savez_compressed(target/'candidate_scores.npz',**candidates,weights=weights)
    return list(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    read_json(args.source/'scores_frozen.json')
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.source/'manifest.json')
    manifest['factor_source'] = str(args.source.resolve())
    write_json(args.output/'manifest.json',manifest)
    for row in manifest['records']:
        methods = record(row,args.source,args.output)
        print('reweighted',row['key'],flush=True)
    write_json(args.output/'scores_frozen.json',dict(status='complete',methods=methods,
        primary='energy_conflict',labels_used=False,threshold=.5,
        scope='exposed-feedback iteration, all-head influence and final competitor weighting; no parameter search'))


if __name__=='__main__':
    main()
