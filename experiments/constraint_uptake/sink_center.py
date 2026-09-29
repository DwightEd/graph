"""Content-specific message responses under exactly mass-preserving exchanges."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import softmax
from experiments.decision_risk_flow.data import read_json,write_json
from .score import key_effects,head_features
from .reweight import weighted_heads


def exchange_effects(gradient,values,own,attention,references,prompt,positions):
    repeated = np.repeat(values,gradient.shape[0]//values.shape[0],axis=0)
    reference_values = repeated[np.arange(len(references)),references]
    reference_mass = np.take_along_axis(attention[:,:,:-1],references[:,None,None],axis=-1)[:,:,0]
    reference_mass += attention[:,:,-1]*((references[:,None]==prompt-1)&(positions[None]==0))
    centered = repeated-reference_values[:,None]
    own_centered = own-reference_values[:,None]
    effect = key_effects(gradient,centered,own_centered,attention*reference_mass[:,:,None])
    return effect,reference_mass


def read_features(row,directory):
    files = {name:np.load(directory/(name+'.npy'),mmap_mode='r')
             for name in ('native','local','values','self_values','attention')}
    positions = np.array(row['positions'])
    prompt = len(row['prompt'])
    layers,heads,queries = files['native'].shape[:3]
    features = np.empty((layers,heads,queries,3,5),dtype=np.float32)
    energy = np.empty((layers,heads,queries,3),dtype=np.float32)
    net = np.empty_like(energy)
    references = []
    for layer in range(layers):
        mean = files['attention'][layer,:,:,:prompt].mean(1)
        if positions[0]==0:
            mean[:,-1] += files['attention'][layer,:,0,-1]/queries
        reference = mean.argmax(-1)
        references.append(reference)
        for start in range(0,queries,32):
            end = min(start+32,queries)
            attention = files['attention'][layer,:,start:end]
            effects = {name:exchange_effects(files[name][layer,:,start:end],files['values'][layer],
                files['self_values'][layer,:,start:end],attention,reference,prompt,positions[start:end])[0]
                for name in ('native','local')}
            values,absolute = head_features(effects['native'],effects['local'],attention,prompt,positions[start:end])
            features[layer,:,start:end] = values
            energy[layer,:,start:end] = absolute
            source = effects['native'][...,:prompt].sum(-1)
            source += effects['native'][...,-1]*(positions[start:end]==0)[None,:,None]
            history = effects['native'].sum(-1)-source
            net[layer,:,start:end] = 2*np.sqrt(np.maximum(-source,0)*np.maximum(history,0))/np.maximum(absolute,1e-12)
    return features,energy,net,np.stack(references)


def score_record(row,directory,output):
    target = output/row['key']
    target.mkdir()
    features,energy,net,references = read_features(row,directory)
    readout = np.load(directory/'readout.npz')
    weights = softmax(-readout['margin'],axis=-1)
    names = ('energy_conflict','energy_opposition','energy_reversal','energy_cancellation')
    candidates = {name:weighted_heads(features[...,index],energy) for index,name in enumerate(names)}
    candidates['net_conflict'] = weighted_heads(net,energy)
    scores = {name:(values*weights).sum(-1) for name,values in candidates.items()}
    scores['energy_conflict_max'] = candidates['energy_conflict'].max(-1)
    scores['negative_margin'] = -readout['margin'][:,0]
    scores['nll'] = readout['nll']
    scores['prompt_deficit'] = 1-features[...,4].mean((0,1,3))
    np.savez_compressed(target/'scores.npz',**scores)
    np.savez_compressed(target/'features.npz',features=features,energy=energy,references=references,net=net)
    return list(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dense',type=Path,required=True)
    parser.add_argument('--rag-factors',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.dense/'manifest.json')
    manifest['operator'] = 'mass-preserving exchange against per-head resident prompt reference'
    write_json(args.output/'manifest.json',manifest)
    for row in manifest['records']:
        root = args.dense if row['dataset']=='gsm8k' else args.rag_factors
        methods = score_record(row,root/row['key'],args.output)
        print('sink-centered',row['key'],flush=True)
    write_json(args.output/'scores_frozen.json',dict(status='complete',methods=methods,primary='energy_conflict',
        labels_used=False,threshold=.5,scope='third exposed pilot iteration; formulas fixed before evaluation'))


if __name__=='__main__':
    main()
