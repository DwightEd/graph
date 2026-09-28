"""Reuse the original local-to-old switch rule on complete native attention."""
import numpy as np
import torch
from experiments.unsupervised_token_graph.span_audit.onset_changes import classify,event_starts,aggregate_heads


def switch_measures(attention,item):
    """Exact old rule: window10, previous3 at current cutoff, shift>=.1, specials removed."""
    weights=attention[0].float()
    prompt=len(item['prompt_ids'])
    length=weights.shape[-1]
    queries=torch.arange(prompt-1,length,device=weights.device)
    keys=torch.arange(length,device=weights.device)
    ids=torch.tensor(item['prompt_ids']+item['answer_ids'][:-1],device=weights.device)
    special=torch.isin(ids,torch.tensor(item['special_ids'],device=weights.device))
    old=(keys[None]<torch.maximum(torch.full_like(queries,prompt),queries-9)[:,None]) & ~special[None]
    def parts(offset):
        selected=weights[:,queries-offset]
        total=selected.sum(-1)
        selected=selected/total.clamp_min(1.)[:,:,None]
        old_mass=(selected*old).sum(-1)
        local_mass=(selected*(~old & ~special[None])).sum(-1)
        missing=(1-total).clamp_min(0)
        return torch.stack((old_mass,local_mass,missing),-1)
    previous=torch.stack([parts(i) for i in (1,2,3)]).mean(0)
    now=parts(0)
    ob,lb,mb=previous.unbind(-1)
    on,ln,mn=now.unbind(-1)
    gain,drop=on-ob,lb-ln
    before,after=lb-ob,on-ln
    measures=torch.stack((ob,lb,on,ln,mb,mn,torch.minimum(gain-mb,drop-mn),
        torch.minimum(gain+mn,drop+mb),before-mb,before+mb,after-mn,after+mn),-1)
    invalid=torch.stack([special[queries-i] for i in (0,1,2,3)]).any(0)
    measures[:,invalid]=float('nan')
    return measures.cpu().numpy()


def states(measures):
    layer_states=np.array([[event_starts(classify(head,.1)) for head in layer] for layer in measures])
    return layer_states,aggregate_heads(layer_states.reshape(-1,layer_states.shape[-1]))
