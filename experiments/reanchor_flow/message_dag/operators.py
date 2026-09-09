"""Native-reference linear allocations and their exact algebraic adjoints.

Attention and RMS denominators are fixed. The SwiGLU product is allocated
symmetrically (or gate-frozen for a sensitivity control). No trained graph
weights, autograd, ablation, dense D x D operators, or averaged heads.
"""
import torch
import torch.nn.functional as F

from .native_layer import NativeLayer


class LayerOperator(NativeLayer):
    """Conditional source-allocation operator at a native reference layer."""

    def __init__(self, cache, layer, rule="symmetric", chunk=8):
        super().__init__(cache, layer, chunk)
        self.source_values = None  # shared across targets in the same layer block
        key = f"residual_{layer+1}" if layer+1 < cache.layers else "final_residual"
        self.next = self.tensor(cache.states[key])
        dtype = getattr(torch, str(cache.qk[f"dtype_{layer}"]))
        z = (self.post * self.post_scale).to(dtype)
        gate = F.linear(z, self.native_weights["gate"]).float()
        up = F.linear(z, self.native_weights["up"]).float()
        activation = F.silu(gate.to(dtype)).float()
        if rule == "symmetric":
            secant = torch.where(gate != 0, activation / torch.where(gate != 0, gate, 1), .5)
            self.up_factor, self.gate_factor = .5 * activation, .5 * up * secant
        elif rule == "up":
            self.up_factor, self.gate_factor = activation, None
        else:
            raise ValueError("MLP allocation must be symmetric or up")

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
