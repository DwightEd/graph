"""Event-seeded differential DAG with exact 0/1/2+ position-hop bookkeeping."""
from time import perf_counter

import numpy as np
import torch

from .differential import DifferentialLayer, final_directions

VARIANTS = ('full','fixed_qk','no_mlp_paths')


def route_hops(state, same, cross):
    post = state+same
    post[1] = post[1]+cross[0]
    post[2] = post[2]+cross[1]+cross[2]
    return post


def _step(op, state, rows, sites, seeds):
    """Advance independent GPU events through ONE shared native layer."""
    count,r,d = state.shape[2:]
    post = torch.empty_like(state)
    for variant in range(3):
        same,cross,codes = op.attention_jvp(state[variant].flatten(0,1),routing=variant!=1)
        post[variant] = route_hops(state[variant],same.reshape(3,count,r,d),cross.reshape(3,count,r,d))
        if variant==0:
            code = codes.reshape(3,count,op.h,r,op.hd).sum(0)
            energy = torch.einsum('bhrd,hde,bhre->bh',code,op.output_gram,code).clamp_min(0)
    injected = torch.zeros((count,op.h,d),device=op.device)
    roots = np.zeros((count,len(op.cache.trace['token_ids'])),np.float32)
    if len(sites):
        owners = np.searchsorted(rows,sites[:,2])
        pairs = [seeds[int(h),int(row)] for _,h,row in sites]
        injected[torch.as_tensor(owners,device=op.device),torch.as_tensor(sites[:,1],device=op.device)] = torch.stack([p[0] for p in pairs])
        np.add.at(roots,owners,np.stack([p[1] for p in pairs]))
    ids = torch.arange(count,device=op.device)
    seed_rows = torch.as_tensor(rows,device=op.device)
    existing = torch.einsum('bhc,hdc->bhd',code[ids,:,seed_rows],op.output_blocks)
    energy = (energy+injected.square().sum(-1)+2*(existing*injected).sum(-1)).clamp_min(0)
    combined = injected.sum(1)
    norm,gross = combined.norm(dim=-1),injected.norm(dim=-1).sum(1)
    post[:,0,ids,seed_rows] += combined
    full_post = post[0].sum(0)
    # Keep the original diagnostic arithmetic, including exact cancellation.
    full_mlp = op.mlp_jvp(full_post)
    state[0] = post[0]+op.mlp_jvp(post[0].flatten(0,1)).reshape(3,count,r,d)
    state[1] = post[1]+op.mlp_jvp(post[1].flatten(0,1)).reshape(3,count,r,d)
    state[2] = post[2]
    denominator = full_post.norm(dim=-1)*full_mlp.norm(dim=-1)
    alignment = (full_post*full_mlp).sum(-1)/denominator.clamp_min(1e-30)
    stats = dict(injection=norm,seed_square=norm.square(),
                 cancellation=torch.where(gross>0,(1-norm/gross.clamp_min(1e-30)).clamp_min(0),torch.nan),
                 energy=energy,mlp_norm=full_mlp.norm(dim=-1),
                 alignment=torch.where(denominator>0,alignment,torch.nan),
                 state_norm=state[0].sum(0).norm(dim=-1))
    return {name:value.cpu().numpy() for name,value in stats.items()},roots


def _record_cut(op, state, reader, recorders, rows):
    from .transport import last_crossing_edges
    for begin,end,a,effects in last_crossing_edges(op,state[0].sum(0),reader):
        for i,row in enumerate(rows):
            recorders[int(row)].write(op,begin,end,a,effects[i])


@torch.inference_mode()
def trace_events(cache, coordinates, *, window=10, query_chunk=8, contrasts=None, progress=None,
                 cut_readout=None, cut_recorders=None, event_batch=None, profile=None):
    """Layer-major propagation; event_batch bounds GPU work, not coverage.

    All events in this group share each layer's loaded weights, capture and
    attention. If a CUDA group exceeds event_batch, its persistent states stay
    on CPU; only the working slice moves to GPU. The caller bounds group RAM.
    Head identities, all future targets and the three variants are retained.
    """
    coordinates = np.asarray(coordinates,int).reshape(-1,3)
    if not len(coordinates): return []
    if len(np.unique(coordinates,axis=0))!=len(coordinates): raise ValueError('duplicate read sites')
    event_rows = np.unique(coordinates[:,2])
    groups = [coordinates[coordinates[:,2]==row] for row in event_rows]
    count = len(groups)
    cfg = cache.weights.config
    r,d,l,h = cache.rows,cfg['hidden_size'],cache.layers,cache.heads
    if np.any(coordinates<0) or np.any(coordinates>=np.array([l,h,r])):
        raise ValueError('event coordinate outside native layer/head/row bounds')
    batch_size = count if event_batch is None else event_batch
    if batch_size<1: raise ValueError('event_batch must be positive')
    device = torch.device(cache.weights.device)
    def clock():
        if profile is not None and device.type=='cuda': torch.cuda.synchronize(device)
        return perf_counter()
    def record(name,started):
        if profile is not None: profile[name] = profile.get(name,0.)+clock()-started
    started = clock()
    storage = 'cpu' if count>batch_size else device
    state = torch.zeros((3,3,count,r,d),device=storage)
    direction,positive,negative,semantic,baseline = final_directions(cache,contrasts)
    record('state_and_readout_seconds',started)
    seed_norm = np.zeros(count,np.float32)
    injection_norm = np.zeros((count,l),np.float32)
    injection_cancellation = np.full((count,l),np.nan,np.float32)
    root_weights = np.zeros((count,len(cache.trace['token_ids'])),np.float32)
    norms = np.zeros((count,l+1,r),np.float32)
    mlp_norm = np.zeros((count,l,r),np.float32)
    mlp_alignment = np.full_like(mlp_norm,np.nan)
    heads = np.zeros((count,l,h),np.float32)
    for layer in range(int(coordinates[:,0].min()),l):
        if progress: progress(f'differential DAG L{layer+1}/{l}, shared events={count}')
        started = clock()
        op = DifferentialLayer(cache,layer,query_chunk)
        op.cache_attention()
        reader = op.tensor(cut_readout[f'L{layer}']) if cut_readout is not None else None
        record('operator_and_attention_seconds',started)
        started = clock()
        layer_sites = coordinates[coordinates[:,0]==layer]
        seeds = op.remote_seeds(layer_sites[:,1:],window)
        record('native_seed_seconds',started)
        started = clock()
        for begin in range(0,count,batch_size):
            end = min(begin+batch_size,count)
            rows = event_rows[begin:end]
            current = state[:,:,begin:end].to(device,memory_format=torch.contiguous_format)
            if reader is not None: _record_cut(op,current,reader,cut_recorders,rows)
            stats,roots = _step(op,current,rows,layer_sites[np.isin(layer_sites[:,2],rows)],seeds)
            state[:,:,begin:end].copy_(current)
            injection_norm[begin:end,layer] = stats['injection']
            injection_cancellation[begin:end,layer] = stats['cancellation']
            seed_norm[begin:end] += stats['seed_square']
            root_weights[begin:end] += roots
            norms[begin:end,layer+1] = stats['state_norm']
            mlp_norm[begin:end,layer] = stats['mlp_norm']
            mlp_alignment[begin:end,layer] = stats['alignment']
            heads[begin:end,layer] = stats['energy']
            if progress: progress(f'differential DAG L{layer+1}/{l}, events {end}/{count}, batch={end-begin}')
            del current
        record('propagation_and_edges_seconds',started)
        if profile is not None:
            profile['layer_builds'] = profile.get('layer_builds',0)+1
        del op,seeds,reader
    started = clock()
    response = np.empty((count,3,3,r-1),np.float32)
    for begin in range(0,count,batch_size):
        end = min(begin+batch_size,count)
        current = state[:,:,begin:end].to(device,memory_format=torch.contiguous_format)
        response[begin:end] = torch.einsum('vkbrd,rd->bvkr',current,direction).cpu().numpy()[...,:-1]
        del current
    result = []
    for i,row in enumerate(event_rows):
        result.append(dict(event_sites=groups[i],event_row=np.array(row),
                           event_position=np.array(cache.trace['row_position'][row]),
                           target_position=cache.trace['row_position'][:-1]+1,
                           variants=np.array(VARIANTS),hop_names=np.array(['0','1','2+']),
                           margin_response=response[i],seed_norm=np.sqrt(seed_norm[i]),
                           root_attention_sum=root_weights[i],state_response_norm=norms[i],
                           injection_norm=injection_norm[i],injection_cancellation=injection_cancellation[i],
                           mlp_response_norm=mlp_norm[i],mlp_alignment=mlp_alignment[i],
                           attention_head_energy=heads[i],positive_id=positive,negative_id=negative,
                           explicit_contrast=semantic,baseline_margin=baseline,labels_used=np.array(False)))
        if cut_readout is not None: result[-1].update(cut_recorders[int(row)].finish(result[-1]))
    record('readout_and_finalize_seconds',started)
    return result
