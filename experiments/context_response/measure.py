"""Exact first-order margin response to source-context attention completion."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.decision_risk_flow.data import read_json, write_json
from experiments.source_relation.measure import normalize, source_relation, js

FIELDS = ('backward_response','forward_response','backward_sham','forward_sham',
          'backward_relative','forward_relative','backward_specific','forward_specific',
          'backward_js','forward_js','source_mass','source_absolute_effect')


def completion_direction(attention, relation):
    """Keep total source mass, replace only source-address proportions."""
    destination = normalize(normalize(attention) @ relation)
    return attention.sum(-1,keepdim=True)*destination-attention


def response(attention, gate_gradient, direction):
    if not torch.all(attention > 0):
        raise ValueError('Zero attention prevents gradient recovery; capture native head gradient instead.')
    sensitivity = gate_gradient.double()/attention.double()
    return (sensitivity*direction.double()).sum(-1)


def relation_matrices(query,key,positions):
    backward = source_relation(query,key,positions)
    forward = normalize(backward.transpose(-1,-2))
    permutation = torch.tensor(np.random.default_rng(42).permutation(len(positions)),device=query.device)
    earlier = positions[None,:] < positions[:,None]
    backward_sham = normalize(backward[...,permutation]*earlier)
    forward_sham = normalize(forward[...,permutation]*earlier.T)
    return backward,forward,backward_sham,forward_sham


def layer_features(attention,gradient,relations):
    directions = [completion_direction(attention,relation) for relation in relations]
    effects = [response(attention,gradient,direction) for direction in directions]
    absolute = gradient.abs().sum(-1).double()
    relative = [effect/absolute.clamp_min(1e-20) for effect in effects[:2]]
    specific = [effects[2]-effects[0],effects[3]-effects[1]]
    divergences = [js(attention,attention+direction) for direction in directions[:2]]
    return torch.stack(effects+relative+specific+divergences+
                       [attention.sum(-1),absolute],-1).float()


def measure_record(row,manifest,output,device):
    directory = output/row['key']
    directory.mkdir()
    source = Path(manifest['relation_base'])/'sources'/str(row['source_id'])
    saved = np.load(source/'input.npz')
    indices = np.flatnonzero(saved['source_mask'])
    query = np.load(source/'query.npy',mmap_mode='r')
    key = np.load(source/'key.npy',mmap_mode='r')
    attention = np.load(Path(manifest['base'])/row['key']/'attention.npy',mmap_mode='r')
    gradient = np.load(Path(manifest['base'])/row['key']/'derivative.npy',mmap_mode='r')
    values = np.empty((*attention.shape[:3],len(FIELDS)),dtype=np.float32)
    positions = torch.tensor(indices,device=device)
    for layer in range(attention.shape[0]):
        query_layer = torch.tensor(np.asarray(query[layer])[:,indices],device=device)
        key_layer = torch.tensor(np.asarray(key[layer])[:,indices],device=device)
        relations = relation_matrices(query_layer,key_layer,positions)
        reading = torch.tensor(np.asarray(attention[layer])[...,indices],device=device)
        effect = torch.tensor(np.asarray(gradient[layer])[...,indices],device=device)
        values[layer] = layer_features(reading,effect,relations).cpu().numpy()
    np.savez_compressed(directory/'responses.npz',values=values,fields=FIELDS)
    previous = Path(manifest['previous'])/row['key']
    (directory/'context.npz').symlink_to((previous/'context.npz').resolve())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--relation-base',type=Path,default=Path('outputs/source_relation_20260928_v1'))
    parser.add_argument('--previous',type=Path,default=Path('outputs/source_relation_20260928_v5'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.relation_base/'manifest.json')
    manifest.update(relation_base=str(args.relation_base.resolve()),previous=str(args.previous.resolve()))
    write_json(args.output/'manifest.json',manifest)
    started = perf_counter()
    for index,row in enumerate(manifest['records']):
        measure_record(row,manifest,args.output,'cuda')
        print(dict(key=row['key'],done=index+1,seconds=perf_counter()-started),flush=True)
    write_json(args.output/'features_complete.json',dict(status='complete',fields=FIELDS,
        answers=len(manifest['records']),labels_used=False,seconds=perf_counter()-started))


if __name__ == '__main__':
    main()
