"""Read full native message edges; no risk seeds or truth labels are accessed."""
import argparse
from pathlib import Path
import numpy as np
from experiments.decision_risk_flow.data import read_json,write_json
from experiments.constraint_uptake.score import key_effects
from .model import overlap_parts,combine_overlap

RAG = Path('outputs/message_js_20260928_v1')
GSM = Path('outputs/constraint_uptake_20260929_dense')


def scatter_self(array, prompt):
    """Map separately recomputed self key back to its true absolute address."""
    result = array[...,:-1].copy()
    positions = prompt-1+np.arange(result.shape[1])
    result[:,np.arange(len(positions)),positions] += array[...,-1]
    return result


def layer_arrays(row,layer):
    key = row['key']
    if row['dataset']=='ragtruth':
        attention = np.load(RAG/key/'attention.npy',mmap_mode='r')[layer]
        effect = np.load(RAG/key/'derivative.npy',mmap_mode='r')[layer]
        return np.asarray(attention),np.asarray(effect)
    directory = GSM/key
    attention = np.load(directory/'attention.npy',mmap_mode='r')[layer]
    gradient = np.load(directory/'native.npy',mmap_mode='r')[layer,:,:,:1]
    values = np.load(directory/'values.npy',mmap_mode='r')[layer]
    own = np.load(directory/'self_values.npy',mmap_mode='r')[layer]
    # Query chunks bound temporary reconstructed signed effects, retaining every key.
    effects = []
    for start in range(0,len(row['response']['answer_ids']),32):
        stop = start+32
        effects.append(key_effects(gradient[:,start:stop],values,own[:,start:stop],attention[:,start:stop])[:,:,0])
    effect = np.concatenate(effects,axis=1)
    return scatter_self(attention,len(row['prompt'])),scatter_self(effect,len(row['prompt']))


def measure_record(row):
    energy,parts,shuffled_parts = [],[],[]
    rng = np.random.default_rng(713)
    for layer in range(32):
        attention,effect = layer_arrays(row,layer)
        head_energy,overlap = overlap_parts(attention,effect)
        # Independently relabel the physical heads at every query within each layer.
        shuffled_attention = np.empty_like(attention)
        shuffled_effect = np.empty_like(effect)
        for query in range(attention.shape[1]):
            permutation = rng.permutation(32)
            shuffled_attention[:,query] = attention[permutation,query]
            shuffled_effect[:,query] = effect[permutation,query]
        _,null_overlap = overlap_parts(shuffled_attention,shuffled_effect)
        energy.append(head_energy)
        parts.append(overlap)
        shuffled_parts.append(null_overlap)
    energy = np.concatenate(energy)
    parts = np.concatenate(parts)
    shuffled_parts = np.concatenate(shuffled_parts)
    return dict(energy=energy,head_overlap=parts,head_shuffled_overlap=shuffled_parts,
        edges=combine_overlap(energy,parts),shuffled=combine_overlap(energy,shuffled_parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    for row in read_json(args.output/'manifest.json')['records']:
        if row['role']!='case':
            continue
        measured = measure_record(row)
        np.savez_compressed(args.output/row['key']/'edges.npz',**measured)
        print('message edges',row['key'],flush=True)
    write_json(args.output/'edges_complete.json',dict(status='complete',heads=1024,
        effect='native current-query margin to strongest nonactual candidate; original past KV fixed',
        control='independent physical-head permutation per query within layer',labels_used=False))


if __name__=='__main__':
    main()
