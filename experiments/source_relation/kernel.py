"""Full native message geometry kernel: no rank reduction or random probes."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from safetensors import safe_open

from experiments.decision_risk_flow.data import read_json, write_json
from .measure import normalize, source_relation, FIELDS

KERNEL_FIELDS = tuple(name.replace('_js','_mmd2') for name in FIELDS)


def message_kernel(values, gram):
    """Exact output-space squared distances from native head factors."""
    inner = (values @ gram) @ values.transpose(-1,-2)
    squared_norm = inner.diagonal(dim1=-2,dim2=-1)
    distance = (squared_norm[:,:,None]+squared_norm[:,None,:]-2*inner).clamp_min(0)
    offdiag = ~torch.eye(values.shape[1],device=values.device,dtype=torch.bool)
    bandwidth = distance[:,offdiag].median(-1).values.clamp_min(1e-12)
    return torch.exp(-distance/(2*bandwidth[:,None,None])),bandwidth


def mmd(left,right,kernel):
    valid = (left.sum(-1)>1e-20)&(right.sum(-1)>1e-20)
    delta = normalize(left)-normalize(right)
    distance = ((delta@kernel)*delta).sum(-1).clamp_min(0)
    return torch.where(valid,distance,torch.nan)


def projection_grams(model,device):
    index = read_json(model/'model.safetensors.index.json')['weight_map']
    result = []
    for layer in range(32):
        name = f'model.layers.{layer}.self_attn.o_proj.weight'
        with safe_open(model/index[name],framework='pt',device='cpu') as checkpoint:
            weight = checkpoint.get_tensor(name).to(device=device,dtype=torch.float32)
        heads = weight.reshape(4096,32,128).permute(1,0,2)
        result.append((heads.transpose(-1,-2)@heads).cpu().numpy())
    return np.stack(result)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = read_json(args.first/'manifest.json')
    manifest.update(measurement='native_message_kernel_mmd2',relation_base=str(args.first.resolve()))
    write_json(args.output/'manifest.json',manifest)
    for name in ('sources','capture_complete.json'):
        (args.output/name).symlink_to((args.first/name).resolve())
    grams = projection_grams(Path(manifest['model']),args.device)
    np.save(args.output/'projection_grams.npy',grams)
    base,route = Path(manifest['base']),Path(manifest['route_base'])
    started = perf_counter()
    for row in manifest['records']:
        directory = args.output/row['key']
        directory.mkdir()
        source = args.output/'sources'/str(row['source_id'])
        inp = np.load(source/'input.npz')
        positions = np.flatnonzero(inp['source_mask'])
        length = len(inp['token_ids'])
        q = np.load(source/'query.npy',mmap_mode='r')
        k = np.load(source/'key.npy',mmap_mode='r')
        v = np.load(route/row['key']/'values.npy',mmap_mode='r')
        attn = np.load(base/row['key']/'attention.npy',mmap_mode='r')
        grad = np.load(base/row['key']/'derivative.npy',mmap_mode='r')
        count = attn.shape[2]
        results = np.empty((32,32,count,len(FIELDS)),dtype=np.float32)
        bandwidths = []
        pos = torch.tensor(positions,device=args.device)
        perm = torch.tensor(np.random.default_rng(42).permutation(len(pos)),device=args.device)
        visible = pos[None,:]<pos[:,None]
        for layer in range(32):
            query = torch.tensor(np.asarray(q[layer])[:,positions],device=args.device)
            key = torch.tensor(np.asarray(k[layer])[:,positions],device=args.device)
            relation = source_relation(query,key,pos)
            shuffled = normalize(relation[...,perm]*visible)
            value = torch.tensor(np.asarray(v[layer])[:,positions],device=args.device).repeat_interleave(4,dim=0)
            gram = torch.tensor(grams[layer],device=args.device)
            kernel,bandwidth = message_kernel(value,gram)
            bandwidths.append(bandwidth.cpu().numpy())
            fields = []
            for raw in (attn,grad):
                weights = torch.tensor(np.array(raw[layer]),device=args.device).abs()
                direct = normalize(weights[...,positions])
                history = normalize(weights[...,length:])
                relay = history@direct[:,:-1]
                state = history@direct[:,1:]
                context = direct@relation
                fields.extend((mmd(direct,relay,kernel),mmd(context,relay,kernel),
                    mmd(context,state,kernel),mmd(direct@shuffled,relay,kernel)))
            a = torch.tensor(np.array(attn[layer]),device=args.device)
            fields.extend((fields[1]-fields[0],fields[5]-fields[4],
                           a[...,positions].sum(-1),a[...,length:].sum(-1)))
            results[layer] = torch.stack(fields,-1).cpu().numpy()
        np.savez_compressed(directory/'relations.npz',values=results,fields=KERNEL_FIELDS,
                            source_positions=positions,prompt_length=length,bandwidths=bandwidths)
        print(dict(key=row['key'],seconds=perf_counter()-started),flush=True)
    write_json(args.output/'features_complete.json',dict(status='complete',fields=KERNEL_FIELDS,
        axes=['layer','physical_head','prediction_token','field'],labels_used=False,
        answers=len(manifest['records']),seconds=perf_counter()-started,
        metric='squared Gaussian kernel MMD in exact native output message geometry'))


if __name__ == '__main__':
    main()
