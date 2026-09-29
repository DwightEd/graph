"""Reconstruct signed key responses; score current answers without truth labels."""
import argparse
from pathlib import Path
import numpy as np
from experiments.decision_risk_flow.data import read_json,write_json

FIELDS = ('binding_conflict','prompt_opposition','adoption_reversal','cancellation','prompt_mass')


def key_effects(gradient,values,self_values,attention):
    """Native head factorization is exact, with a separate recomputed self-key."""
    repeated = np.repeat(values,gradient.shape[0]//values.shape[0],axis=0)
    heads,queries,candidates,dimension = gradient.shape
    products = gradient.reshape(heads,queries*candidates,dimension)@repeated.transpose(0,2,1)
    past = products.reshape(heads,queries,candidates,-1)*attention[:,:,None,:-1]
    own = np.einsum('hqcd,hqd->hqc',gradient,self_values)*attention[:,:,-1,None]
    return np.concatenate((past,own[:,:,:,None]),axis=-1)


def head_features(native,local,attention,prompt,positions):
    keys = native.shape[-1]-1
    source = np.broadcast_to(np.arange(keys)<prompt,(len(positions),keys))
    source = np.concatenate((source,(prompt-1+positions<prompt)[:,None]),axis=-1)
    positive = np.maximum(native,0)
    negative = np.maximum(-native,0)
    prompt_positive = (positive*source[None,:,None]).sum(-1)
    prompt_negative = (negative*source[None,:,None]).sum(-1)
    history_positive = (positive*(~source)[None,:,None]).sum(-1)
    absolute = np.abs(native).sum(-1)
    denominator = np.maximum(absolute,1e-12)
    conflict = 2*np.sqrt(prompt_negative*history_positive)/denominator
    opposition = prompt_negative/np.maximum(prompt_positive+prompt_negative,1e-12)
    reversal = (negative*(local>0)*source[None,:,None]).sum(-1)/denominator
    cancellation = 1-np.abs(native.sum(-1))/denominator
    cancellation = np.where(absolute>1e-12,cancellation,0)
    mass = (attention*source[None]).sum(-1)
    mass = np.broadcast_to(mass[:,:,None],conflict.shape)
    return np.stack((conflict,opposition,reversal,cancellation,mass),-1),absolute


def record_scores(row,output):
    directory = output/row['key']
    files = {name:np.load(directory/(name+'.npy'),mmap_mode='r')
             for name in ('native','local','attention','values','self_values')}
    features,energy = [],[]
    for layer in range(files['native'].shape[0]):
        effects = {name:key_effects(files[name][layer],files['values'][layer],files['self_values'][layer],files['attention'][layer])
                   for name in ('native','local')}
        measured,absolute = head_features(effects['native'],effects['local'],files['attention'][layer],
            len(row['prompt']),np.array(row['positions']))
        features.append(measured)
        energy.append(absolute)
    features = np.stack(features)
    energy = np.stack(energy)
    flat = features.reshape(-1,len(row['positions']),3,len(FIELDS))
    scores = {}
    for index,name in enumerate(FIELDS[:-1]):
        scores[name] = np.sort(flat[:,:,:,index],axis=0)[-8:].mean(0).max(-1)
    scores['prompt_deficit'] = 1-features[...,4].mean((0,1,3))
    readout = np.load(directory/'readout.npz')
    scores['negative_margin'] = -readout['margin'][:,0]
    scores['nll'] = readout['nll']
    np.savez_compressed(directory/'features.npz',features=features,energy=energy,fields=FIELDS)
    np.savez_compressed(directory/'scores.npz',**scores)
    return list(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    read_json(args.output/'capture_complete.json')
    manifest = read_json(args.output/'manifest.json')
    for row in manifest['records']:
        methods = record_scores(row,args.output)
        print('scored',row['key'],flush=True)
    write_json(args.output/'scores_frozen.json',dict(status='complete',methods=methods,
        primary='binding_conflict',labels_used=False,threshold=.5,
        threshold_scope='fixed exploratory ratio threshold, not calibrated false-positive control',
        step_pool='mean of at most eight uniformly sampled query scores within each step'))


if __name__=='__main__':
    main()
