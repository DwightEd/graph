"""Event-seeded differential DAG with exact 0/1/2+ position-hop bookkeeping."""
import numpy as np
import torch

from .differential import DifferentialLayer, final_directions

VARIANTS = ('full','fixed_qk','no_mlp_paths')


def route_hops(state, same, cross):
    post = state+same
    post[1] = post[1]+cross[0]
    post[2] = post[2]+cross[1]+cross[2]
    return post


@torch.inference_mode()
def trace_events(cache, coordinates, *, window=10, query_chunk=8, contrasts=None, progress=None):
    """All later output positions per event; caller bounds the event batch.

    Sites are (writer layer, writer head, response-row index). Sites at the
    same response position form one event, adding their actual messages at
    their original layer. Physical head IDs are retained, never averaged. Native
    forwards are never modified. Labels are not inputs to this function.
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
    state = torch.zeros((3,3,count,r,d),device=cache.weights.device)
    direction,positive,negative,semantic,baseline = final_directions(cache,contrasts)
    seed_norm = np.zeros(count,np.float32)
    injection_norm = np.zeros((count,l),np.float32)
    injection_cancellation = np.full((count,l),np.nan,np.float32)
    root_weights = np.zeros((count,len(cache.trace['token_ids'])),np.float32)
    norms = np.zeros((count,l+1,r),np.float32)
    mlp_norm = np.zeros((count,l,r),np.float32)
    mlp_alignment = np.full_like(mlp_norm,np.nan)
    heads = np.zeros((count,l,h),np.float32)
    for layer in range(int(coordinates[:,0].min()),l):
        if progress: progress(f'differential DAG L{layer+1}/{l}, events={count}')
        op = DifferentialLayer(cache,layer,query_chunk)
        post = torch.empty_like(state)
        for variant in range(3):
            same,cross,codes = op.attention_jvp(state[variant].flatten(0,1),routing=variant!=1)
            post[variant] = route_hops(state[variant],same.reshape(3,count,r,d),cross.reshape(3,count,r,d))
            if variant==0:
                code = codes.reshape(3,count,h,r,op.hd).sum(0)
                block = op.w['output'].reshape(d,h,op.hd).permute(1,0,2)
                gram = block.transpose(-1,-2)@block
                heads[:,layer] = torch.einsum('bhrd,hde,bhre->bh',code,gram,code).clamp_min(0).cpu().numpy()
        seeds = op.remote_seeds(coordinates[coordinates[:,0]==layer,1:],window)
        for i,sites in enumerate(groups):
            combined = torch.zeros(d,device=cache.weights.device)
            gross = 0.
            for _,head,row in sites[sites[:,0]==layer]:
                seed,weights = seeds[int(head),int(row)]
                root_weights[i] += weights
                combined += seed
                gross += float(seed.norm())
                existing = block[head]@code[i,head,row]
                heads[i,layer,head] += float(seed.square().sum()+2*(existing*seed).sum())
                heads[i,layer,head] = max(0.,heads[i,layer,head])
            post[:,0,i,event_rows[i]] += combined
            norm = float(combined.norm())
            injection_norm[i,layer] = norm
            seed_norm[i] += norm**2
            if gross: injection_cancellation[i,layer] = max(0.,1-norm/gross)
        full_post = post[0].sum(0)
        full_mlp = op.mlp_jvp(full_post)
        mlp_norm[:,layer] = full_mlp.norm(dim=-1).cpu().numpy()
        denominator = full_post.norm(dim=-1)*full_mlp.norm(dim=-1)
        alignment = (full_post*full_mlp).sum(-1)/denominator.clamp_min(1e-30)
        mlp_alignment[:,layer] = torch.where(denominator>0,alignment,torch.nan).cpu().numpy()
        for variant in range(3):
            state[variant] = post[variant] if variant==2 else post[variant]+op.mlp_jvp(post[variant].flatten(0,1)).reshape(3,count,r,d)
        norms[:,layer+1] = state[0].sum(0).norm(dim=-1).cpu().numpy()
        del op
    response = torch.einsum('vkbrd,rd->bvkr',state,direction).cpu().numpy()[...,:-1]
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
    return result
