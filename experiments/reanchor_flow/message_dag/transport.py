"""Signed last-crossing paths: complete prefix, native V/K edge, local suffix.

Each cross-position path has exactly one LAST crossing. Summing this cut
recovers the 1/2+ hop response, without counting a path once at every layer.
All physical edges are streamed to NPZ. No attention threshold or top-k is
used in computation/storage; pruning belongs only to the HTML display.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
import torch.nn.functional as F

from ..attention_audit import write_array
from .differential import DifferentialLayer, final_directions, rms_jvp, rotate


@torch.inference_mode()
def prepare_local_readout(cache, path, *, query_chunk=8, contrasts=None, progress=None):
    """One reverse sweep per sample, shared by ALL events and future targets.

    Only same-position suffixes remain after a last crossing. Consequently
    every row can hold its own target's reader without an extra target axis.
    A temporary NPZ holds L*R*D floats on disk, not on the GPU.
    """
    reader = final_directions(cache,contrasts)[0].clone()
    reader[-1] = 0  # the final input row has no captured next-token target
    with ZipFile(path,'w',ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for layer in reversed(range(cache.layers)):
            if progress: progress(f'local output readers L{layer+1}/{cache.layers}')
            op = DifferentialLayer(cache,layer,query_chunk)
            post = reader+op.mlp_vjp(reader)
            write_array(archive,f'L{layer}',post.cpu().numpy())
            reader = post+op.attention_same_vjp(post)
            del op


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


class CutRecorder:
    """Stream one event's edges and retain exact, small marginal summaries."""
    def __init__(self, path, cache, event_row):
        self.path,self.row = Path(path),int(event_row)
        self.temporary = self.path.with_suffix('.tmp.npz')
        self.archive = ZipFile(self.temporary,'w',ZIP_DEFLATED,compresslevel=1,allowZip64=True)
        l,h,t = cache.layers,cache.heads,cache.rows-1
        self.signed = np.zeros((l,h,t,2),np.float32)
        self.positive = np.zeros_like(self.signed)
        self.negative = np.zeros_like(self.signed)
        self.hop_positive = np.zeros((2,t,2),np.float32)
        self.hop_negative = np.zeros_like(self.hop_positive)
        self.carriers = np.zeros((t,cache.rows,2),np.float32)
        # Display cache only. ALL edges are still written above without top-k.
        self.preview_index = np.full((t,24,4),-1,np.int32)
        self.preview_effect = np.zeros((t,24),np.float32)
        self.preview_attention = np.zeros_like(self.preview_effect)
        self.rows = cache.trace['row_position']
        self.saved_layers = set()
        write_array(self.archive,'row_position',self.rows)
        write_array(self.archive,'event_row',np.array(self.row))
        write_array(self.archive,'branch_names',np.array(['V_content','K_routing']))

    def write(self, op, begin, end, attention, effects):
        begin_event = max(begin,self.row+1)
        if begin_event>=end: return
        if op.layer not in self.saved_layers:
            if not hasattr(op,'cut_energies'):
                native = op.v[:,op.rows]
                energy = torch.einsum('hsd,hde,hse->hs',native,op.output_gram,native).clamp_min(0)
                op.cut_energies = native.square().sum(-1).cpu().numpy(),energy.cpu().numpy()
            write_array(self.archive,f'value_energy_L{op.layer}',op.cut_energies[0])
            write_array(self.archive,f'write_energy_L{op.layer}',op.cut_energies[1])
            self.saved_layers.add(op.layer)
        # Source rows before the seed cannot be affected; future/self sources
        # are excluded by the native causal cut, not by an attention threshold.
        sl = slice(begin_event-begin,end-begin)
        values = effects[:,sl,self.row:end-1].cpu().numpy()
        a = attention[:,sl,self.row:end-1].cpu().numpy()
        name = f'L{op.layer}Q{begin_event}'
        write_array(self.archive,name,values)
        write_array(self.archive,'A'+name,a)
        positive,negative = np.maximum(values,0),np.maximum(-values,0)
        self.signed[op.layer,:,begin_event:end] += values.sum(2)
        self.positive[op.layer,:,begin_event:end] += positive.sum(2)
        self.negative[op.layer,:,begin_event:end] += negative.sum(2)
        self.carriers[begin_event:end,self.row:end-1] += values.sum(0)
        for total,part in ((self.hop_positive,positive),(self.hop_negative,negative)):
            total[0,begin_event:end] += part[:,:,0].sum(0)
            total[1,begin_event:end] += part[:,:,1:].sum((0,2))
        for j,q in enumerate(range(begin_event,end)):
            flat = values[:,j].reshape(-1)
            count = min(24,len(flat))
            take = np.argpartition(np.abs(flat),len(flat)-count)[-count:]
            take = take[flat[take]!=0]
            head,source,kind = np.unravel_index(take,values[:,j].shape)
            index = np.column_stack((np.full(len(take),op.layer),head,source+self.row,kind))
            effects = np.r_[self.preview_effect[q],flat[take]]
            order = np.argsort(-np.abs(effects),kind='stable')[:24]
            self.preview_effect[q] = effects[order]
            self.preview_attention[q] = np.r_[self.preview_attention[q],a[head,j,source]][order]
            self.preview_index[q] = np.concatenate((self.preview_index[q],index))[order]

    def finish(self, event):
        net = (self.hop_positive-self.hop_negative).sum(-1)
        expected = event['margin_response'][0,1:]
        error = net-expected
        scale = np.maximum(np.abs(expected),(self.hop_positive+self.hop_negative).sum(-1))
        if not (np.isfinite(error).all() and np.isfinite(scale).all()):
            raise ValueError('nonfinite last-crossing response; cannot certify path conservation')
        if np.any(np.abs(error)>2e-7+2e-4*scale):
            raise ValueError('last-crossing cut does not reconstruct the native 1/2+ hop response')
        result = dict(cut_schema=np.array(1),cut_signed=self.signed,cut_positive=self.positive,cut_negative=self.negative,
                      cut_hop_positive=self.hop_positive,cut_hop_negative=self.hop_negative,
                      cut_carrier_effect=self.carriers,cut_closure_error=error,
                      cut_preview_index=self.preview_index,cut_preview_effect=self.preview_effect,
                      cut_preview_attention=self.preview_attention)
        for name,value in result.items(): write_array(self.archive,name,value)
        write_array(self.archive,'event_sites',event['event_sites'])
        self.archive.close()
        self.temporary.replace(self.path)
        return result

    def close(self):
        self.archive.close()
        if self.temporary.exists(): self.temporary.unlink()
