"""Signed last-crossing paths through native V/K edges.

Each cross-position path has exactly one LAST crossing. Summing this cut
recovers the 1/2+ hop response, without counting a path once at every layer.
All physical edges are streamed to NPZ. No attention threshold or top-k is
used in computation/storage; pruning belongs only to the HTML display.
"""
import torch
import torch.nn.functional as F

from .differential import rms_jvp, rotate


def last_crossing_edges(op, delta, reader):
    """Yield [event,head,query,source, V/K] signed output effects.

    delta is the complete prefix response, summed over its previous hop
    counts. The reader is the same-position suffix adjoint, not the full
    backward adjoint. K edges denote routing control, not content provenance.
    """
    batch,r,_ = delta.shape
    z = rms_jvp(delta,op.x,op.w['input_norm'],op.eps)
    dv = F.linear(z,op.w['value']).reshape(batch,r,op.kv,op.hd).transpose(1,2)
    dk = F.linear(z,op.wk).reshape(batch,r,op.kv,op.hd).transpose(1,2)
    dv = dv.repeat_interleave(op.h//op.kv,1)
    dk = rotate(dk,op.cos,op.sin).repeat_interleave(op.h//op.kv,1)
    g = F.linear(reader,op.w['output'].T).reshape(r,op.h,op.hd).transpose(0,1)
    for begin,end,a in op.rows_attention(r-1):
        ar = a[...,op.rows]
        projected = g[:,begin:end]
        value = ar[None]*torch.einsum('hqd,bhsd->bhqs',projected,dv)
        ds = torch.einsum('hqd,bhsd->bhqs',op.q[:,begin:end],dk)*op.scale
        av = torch.einsum('hqs,hsd->hqd',a,op.v)
        contrast = torch.einsum('hqd,hsd->hqs',projected,op.v[:,op.rows])
        contrast -= (projected*av).sum(-1,keepdim=True)
        key = ar[None]*ds*contrast[None]
        cross = op.rows[None,:] < op.rows[begin:end,None]
        effects = torch.stack((value,key),-1)*cross[None,None,:,:,None]
        yield begin,end,ar,effects
