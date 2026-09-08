"""Native-reference linear allocations and their exact algebraic adjoints.

Attention and RMS denominators are fixed. The SwiGLU product is allocated
symmetrically (or gate-frozen for a sensitivity control). No trained graph
weights, autograd, ablation, dense D x D operators, or averaged heads.
"""
import torch
import torch.nn.functional as F

from ..message_lineage import attention_chunks


class LayerOperator:
    def __init__(self, cache, layer, rule="symmetric", chunk=8):
        self.cache, self.layer, self.chunk = cache, layer, chunk
        trace, states, weights = cache.trace, cache.states, cache.weights
        self.device = weights.device
        self.source_values = None  # shared across targets in the same layer block
        prefix = f"model.layers.{layer}."
        names = {"input_norm": "input_layernorm.weight", "post_norm": "post_attention_layernorm.weight",
                 "value": "self_attn.v_proj.weight", "output": "self_attn.o_proj.weight",
                 "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight", "down": "mlp.down_proj.weight"}
        native = {k: weights.get(prefix + name) for k, name in names.items()}
        self.w = {k: value.float() for k, value in native.items()}
        self.rows = torch.as_tensor(trace["row_position"], device=self.device)
        self.carrier = self.rows >= int(trace["response_start"])
        self.h = weights.config["num_attention_heads"]
        self.kv = weights.config["num_key_value_heads"]
        self.d = weights.config["hidden_size"]
        self.hd = self.d // self.h
        self.x = self.tensor(states[f"residual_{layer}"])
        self.value = self.tensor(states[f"value_{layer}"])
        self.qk = {f"{name}_{layer}": cache.qk[f"{name}_{layer}"] for name in ("query", "key", "dtype", "scale")}
        self.history = {f"L{layer}": self.tensor(cache.history[f"L{layer}"])}
        dtype = getattr(torch, str(cache.qk[f"dtype_{layer}"]))
        self.post = (self.x.to(dtype) + self.tensor(states[f"attention_{layer}"]).to(dtype)).float()
        key = f"residual_{layer+1}" if layer+1 < cache.layers else "final_residual"
        self.next = self.tensor(states[key])
        eps = weights.config["rms_norm_eps"]
        scale = lambda x, w: w * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        self.input_scale = scale(self.x, self.w["input_norm"])
        self.post_scale = scale(self.post, self.w["post_norm"])
        z = (self.post * self.post_scale).to(dtype)
        gate, up = F.linear(z, native["gate"]).float(), F.linear(z, native["up"]).float()
        activation = F.silu(gate.to(dtype)).float()
        if rule == "symmetric":
            secant = torch.where(gate != 0, activation / torch.where(gate != 0, gate, 1), .5)
            self.up_factor, self.gate_factor = .5 * activation, .5 * up * secant
        elif rule == "up":
            self.up_factor, self.gate_factor = activation, None
        else:
            raise ValueError("MLP allocation must be symmetric or up")

    def tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    def rows_attention(self, stop=None):
        c = self.cache
        for begin,end,a in attention_chunks(c.trace,self.qk,self.history,self.layer,self.device,self.chunk):
            if stop is not None:
                if begin>=stop: return
                end=min(end,stop)
                a=a[:,:end-begin]
            yield begin,end,a
            if stop is not None and end==stop: return

    def values(self, part):
        """[group,R,D] -> [group,Hkv,R,d]; prompt predictor is a boundary."""
        shape = part.shape[:-2]
        code = F.linear(part * self.input_scale, self.w["value"])
        code = code.reshape(*shape, len(self.rows), self.kv, self.hd).transpose(-3, -2)
        return code * self.carrier[None, :, None]

    def output_direction(self, adjoint):
        """[target,R,D] -> [target,H,R,d]."""
        return F.linear(adjoint, self.w["output"].T).reshape(-1, len(self.rows), self.h, self.hd).transpose(1, 2)

    def attention(self, part, boundary_mask):
        """All response messages plus disjoint prompt-V boundary injections."""
        value = self.values(part).repeat_interleave(self.h // self.kv, dim=1)
        native = self.value.repeat_interleave(self.h // self.kv, dim=0)
        result = torch.empty_like(part)
        for begin, end, a in self.rows_attention():
            code = torch.einsum("hqs,ghsd->gqhd", a[..., self.rows], value)
            code += torch.einsum("gs,hqs,hsd->gqhd", boundary_mask, a, native)
            result[:, begin:end] = F.linear(code.flatten(-2), self.w["output"])
        return result

    def attention_adjoint(self, adjoint, stop=None):
        projected = self.output_direction(adjoint)
        code = torch.zeros((len(adjoint), self.h, len(self.rows), self.hd), device=self.device)
        for begin, end, a in self.rows_attention(stop):
            code += torch.einsum("hqs,bhqd->bhsd", a[..., self.rows], projected[:, :, begin:end])
        code *= self.carrier[None, None, :, None]
        code = code.reshape(len(adjoint), self.kv, self.h // self.kv, len(self.rows), self.hd).sum(2)
        code = code.transpose(1, 2).flatten(-2)
        return F.linear(code, self.w["value"].T) * self.input_scale

    def mlp(self, part):
        z = part * self.post_scale
        hidden = F.linear(z, self.w["up"]) * self.up_factor
        if self.gate_factor is not None:
            hidden += F.linear(z, self.w["gate"]) * self.gate_factor
        return F.linear(hidden, self.w["down"])

    def mlp_adjoint(self, adjoint):
        hidden = F.linear(adjoint, self.w["down"].T)
        result = F.linear(hidden * self.up_factor, self.w["up"].T)
        if self.gate_factor is not None:
            result += F.linear(hidden * self.gate_factor, self.w["gate"].T)
        return result * self.post_scale
