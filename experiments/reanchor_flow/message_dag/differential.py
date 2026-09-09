"""Analytic response-state Jacobians on the captured Llama computation DAG.

No fitted graph, parameter gradients, or arbitrary SwiGLU allocations. For
rounded captures this is a smooth local approximation at the stored states.
Prompt states are fixed because the seed is an internal response message.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .native_layer import NativeLayer


def rms_jvp(delta, reference, weight, epsilon):
    variance = reference.square().mean(-1, keepdim=True) + epsilon
    return weight * torch.rsqrt(variance) * (
        delta - reference * (delta*reference).mean(-1, keepdim=True)/variance)


def rms_vjp(reader, reference, weight, epsilon):
    weighted = reader*weight
    variance = reference.square().mean(-1, keepdim=True)+epsilon
    return torch.rsqrt(variance)*(weighted-reference*(weighted*reference).mean(-1, keepdim=True)/variance)


def rotate(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return x*cos + torch.cat((-b, a), dim=-1)*sin


class DifferentialLayer(NativeLayer):
    """Native local Jacobian; deliberately independent of source allocation."""

    def __init__(self, cache, layer, chunk=8):
        super().__init__(cache, layer, chunk)
        prefix = f'model.layers.{layer}.self_attn.'
        self.wq = cache.weights.get(prefix+'q_proj.weight').float()
        self.wk = cache.weights.get(prefix+'k_proj.weight').float()
        # Keep native Q/K on the device across all attention consumers.
        for name in ('query','key'):
            key = f'{name}_{layer}'
            self.qk[key] = torch.as_tensor(self.qk[key],device=self.device)
        self.q = self.qk[f'query_{layer}'].float()
        self.k = self.qk[f'key_{layer}'].float().repeat_interleave(self.h//self.kv, 0)
        self.v = self.value.repeat_interleave(self.h//self.kv, 0)
        self.scale = float(self.qk[f'scale_{layer}'])
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        rotary = LlamaRotaryEmbedding(config=LlamaConfig(**cache.weights.config), device=self.device)
        self.cos, self.sin = (v[0].float() for v in rotary(self.x[None], self.rows[None]))
        z = self.post*self.post_scale
        self.gate = F.linear(z, self.w['gate'])
        self.up = F.linear(z, self.w['up'])
        sigmoid = self.gate.sigmoid()
        self.silu_prime = sigmoid * (1 + self.gate*(1-sigmoid))
        self.activation = F.silu(self.gate)
        self.output_blocks = self.w['output'].reshape(self.d,self.h,self.hd).permute(1,0,2)
        self.output_gram = self.output_blocks.transpose(-1,-2)@self.output_blocks
        self.cached_attention = None

    def cache_attention(self, max_bytes=128*2**20):
        """Reuse this layer's exact rows if they fit a bounded device buffer."""
        shape = (self.h,len(self.rows),self.k.shape[1])
        if np.prod(shape)*4>max_bytes: return
        # No threshold, head averaging, or reduced precision in this cache.
        value = torch.empty(shape,device=self.device)
        for begin,end,a in super().rows_attention(): value[:,begin:end] = a
        self.cached_attention = value

    def rows_attention(self, stop=None):
        if self.cached_attention is None:
            yield from super().rows_attention(stop)
        else:
            end_rows = len(self.rows) if stop is None else min(stop,len(self.rows))
            for begin in range(0,end_rows,self.chunk):
                end = min(begin+self.chunk,end_rows)
                yield begin,end,self.cached_attention[:,begin:end]

    def mlp_jvp(self, delta):
        z = rms_jvp(delta, self.post, self.w['post_norm'], self.eps)
        hidden = self.activation*F.linear(z, self.w['up'])
        hidden += self.up*self.silu_prime*F.linear(z, self.w['gate'])
        return F.linear(hidden, self.w['down'])

    def mlp_vjp(self, reader):
        """Adjoint of the native SwiGLU derivative, including RMS denominator."""
        hidden = F.linear(reader, self.w['down'].T)
        z = F.linear(hidden*self.activation, self.w['up'].T)
        z += F.linear(hidden*self.up*self.silu_prime, self.w['gate'].T)
        return rms_vjp(z, self.post, self.w['post_norm'], self.eps)

    def attention_same_vjp(self, reader):
        """Diagonal position blocks only: each row keeps its own output reader.

        Includes Q control over the WHOLE attention row, self K competition,
        self V, RoPE and RMSNorm. This is not a frozen-attention backward pass.
        """
        projected = F.linear(reader, self.w['output'].T).reshape(-1,self.h,self.hd).transpose(0,1)
        dq,dk,dv = (torch.zeros_like(projected) for _ in range(3))
        for begin,end,a in self.rows_attention():
            g = projected[:,begin:end]
            av = torch.einsum('hqs,hsd->hqd',a,self.v)
            score = torch.einsum('hqd,hsd->hqs',g,self.v)-(g*av).sum(-1,keepdim=True)
            ds = a*score
            local = torch.arange(end-begin,device=self.device)
            absolute = self.rows[begin:end]
            dq[:,begin:end] = torch.einsum('hqs,hsd->hqd',ds,self.k)*self.scale
            dk[:,begin:end] = ds[:,local,absolute,None]*self.q[:,begin:end]*self.scale
            dv[:,begin:end] = a[:,local,absolute,None]*g
        dq = rotate(dq,self.cos,-self.sin).transpose(0,1).flatten(-2)
        dk = rotate(dk,self.cos,-self.sin)
        def shared_heads(x):
            return x.reshape(self.kv,self.h//self.kv,len(self.rows),self.hd).sum(1).transpose(0,1).flatten(-2)
        z = F.linear(dq,self.wq.T)+F.linear(shared_heads(dk),self.wk.T)
        z += F.linear(shared_heads(dv),self.w['value'].T)
        return rms_vjp(z,self.x,self.w['input_norm'],self.eps)

    def attention_jvp(self, delta, *, routing=True):
        """Return same-position and strictly earlier-position derivative branches.

        Query changes are same-position control. Key/value changes at s<q are
        cross-position transmission. This counts real position hops, not layers.
        """
        batch, r, _ = delta.shape
        z = rms_jvp(delta, self.x, self.w['input_norm'], self.eps)
        dv = F.linear(z, self.w['value']).reshape(batch,r,self.kv,self.hd).transpose(1,2)
        dv = dv.repeat_interleave(self.h//self.kv,1)
        if routing:
            dq = F.linear(z,self.wq).reshape(batch,r,self.h,self.hd).transpose(1,2)
            dk = F.linear(z,self.wk).reshape(batch,r,self.kv,self.hd).transpose(1,2)
            dq = rotate(dq,self.cos,self.sin)
            dk = rotate(dk,self.cos,self.sin).repeat_interleave(self.h//self.kv,1)
        same = torch.zeros((batch,r,self.d), device=self.device)
        total = torch.zeros_like(same)
        codes = torch.zeros((batch,self.h,r,self.hd), device=self.device)
        for begin,end,a in self.rows_attention():
            qrows = self.rows[begin:end]
            ar = a[...,self.rows]
            local = torch.arange(end-begin,device=self.device)
            a_self = a[:,local,qrows]
            all_code = torch.einsum('hqs,bhsd->bhqd',ar,dv)
            self_code = a_self[None,...,None]*dv[:,:,begin:end]
            if routing:
                ds_q = torch.einsum('bhqd,hsd->bhqs',dq[:,:,begin:end],self.k)*self.scale
                ds_k = torch.einsum('hqd,bhsd->bhqs',self.q[:,begin:end],dk)*self.scale
                ds = ds_q.clone()
                ds[...,self.rows] += ds_k
                da = a[None]*(ds-(ds*a[None]).sum(-1,keepdim=True))
                all_code += torch.einsum('bhqs,hsd->bhqd',da,self.v)
                da_query = a[None]*(ds_q-(ds_q*a[None]).sum(-1,keepdim=True))
                self_code += torch.einsum('bhqs,hsd->bhqd',da_query,self.v)
                # A change to K_q changes the whole softmax row, not only A_qq.
                key_self = ds_k[:,:,local,torch.arange(begin,end,device=self.device)]
                av = torch.einsum('hqs,hsd->hqd',a,self.v)
                self_code += (a_self[None]*key_self)[...,None]*(self.v[:,qrows]-av)[None]
            codes[:,:,begin:end] = all_code
            total[:,begin:end] = F.linear(all_code.transpose(1,2).flatten(-2),self.w['output'])
            same[:,begin:end] = F.linear(self_code.transpose(1,2).flatten(-2),self.w['output'])
        return same, total-same, codes

    def remote_seeds(self, sites, window):
        """Batch native head writes; one CPU weight transfer per query chunk."""
        sites = np.asarray(sites,int).reshape(-1,2)  # head, row
        result = {}
        if not len(sites): return result
        source = torch.arange(len(self.cache.trace['token_ids']),device=self.device)
        ordinary = ~torch.as_tensor(self.cache.trace['special_mask'],device=self.device)
        for begin,end,a in self.rows_attention(int(sites[:,1].max())+1):
            take = sites[(sites[:,1]>=begin)&(sites[:,1]<end)]
            if not len(take): continue
            mask = ((self.rows[begin:end,None]-source)>window) & ordinary
            weights = a*mask[None]
            code = torch.einsum('hqs,hsd->hqd',weights,self.v)
            messages = torch.einsum('hqc,hdc->hqd',code,self.output_blocks)
            heads = torch.as_tensor(take[:,0],device=self.device)
            rows = torch.as_tensor(take[:,1]-begin,device=self.device)
            selected = messages[heads,rows]
            mass = weights[heads,rows].cpu().numpy()
            for i,(head,row) in enumerate(take):
                result[int(head),int(row)] = selected[i],mass[i]
        if len(result)!=len(sites): raise ValueError('duplicate/absent read site')
        return result

    def remote_seed(self, head, row, window):
        return self.remote_seeds([(head,row)],window)[int(head),int(row)]


def final_directions(cache, contrasts=None):
    """True final-RMSNorm gradient, with fixed observed/runner candidates."""
    key = tuple(sorted((int(c['target']),int(c['positive_id']),int(c['negative_id'])) for c in (contrasts or [])))
    if key in cache.event_readouts: return cache.event_readouts[key]
    from ..message_lineage import readout_directions
    frozen, runners = readout_directions(cache.trace,cache.states,cache.weights)
    raw = torch.as_tensor(cache.states['final_residual'],device=cache.weights.device).float()
    rows = cache.trace['row_position']
    positive = cache.trace['token_ids'][rows[:-1]+1].copy()
    semantic = np.zeros(len(positive),bool)
    if contrasts:
        norm = cache.weights.get('model.norm.weight').float()
        unembed = cache.weights.get('lm_head.weight').float()
        if len({int(c['target']) for c in contrasts})!=len(contrasts):
            raise ValueError('duplicate contrast targets')
        for c in contrasts:
            index = np.flatnonzero(rows[:-1]+1==int(c['target']))
            if not len(index): raise ValueError(f"contrast target {c['target']} is not a predicted position")
            i = int(index[0]);pos,neg = int(c['positive_id']),int(c['negative_id'])
            if not (0<=pos<len(unembed) and 0<=neg<len(unembed)) or pos==neg:
                raise ValueError('contrast IDs must be distinct valid vocabulary IDs')
            variance = raw[i].square().mean()+cache.weights.config['rms_norm_eps']
            frozen[i] = (unembed[pos]-unembed[neg])*norm*torch.rsqrt(variance)
            positive[i],runners[i],semantic[i] = pos,neg,True
    variance = raw.square().mean(-1,keepdim=True)+cache.weights.config['rms_norm_eps']
    baseline = (frozen*raw).sum(-1)[:-1].cpu().numpy()
    direction = frozen-raw*(frozen*raw).mean(-1,keepdim=True)/variance
    result = direction, positive, runners, semantic, baseline
    cache.event_readouts[key] = result
    return result
