"""Shared native Llama operands at one captured reference layer.

This module owns only checkpoint/capture reconstruction. Source allocation and
local differentiation extend it independently because they compute different
mathematical objects.
"""

import torch

from ..message_lineage import attention_chunks


class NativeLayer:
    """Load the captured operands shared by allocation and tangent engines."""

    def __init__(self, cache, layer, chunk=8):
        self.cache, self.layer, self.chunk = cache, layer, chunk
        trace, states, weights = cache.trace, cache.states, cache.weights
        self.device = weights.device
        prefix = f"model.layers.{layer}."
        names = {
            "input_norm": "input_layernorm.weight",
            "post_norm": "post_attention_layernorm.weight",
            "value": "self_attn.v_proj.weight",
            "output": "self_attn.o_proj.weight",
            "gate": "mlp.gate_proj.weight",
            "up": "mlp.up_proj.weight",
            "down": "mlp.down_proj.weight",
        }
        self.native_weights = {key: weights.get(prefix + name) for key, name in names.items()}
        self.w = {key: value.float() for key, value in self.native_weights.items()}
        self.rows = torch.as_tensor(trace["row_position"], device=self.device)
        self.carrier = self.rows >= int(trace["response_start"])
        self.h = weights.config["num_attention_heads"]
        self.kv = weights.config["num_key_value_heads"]
        self.d = weights.config["hidden_size"]
        self.hd = self.d // self.h
        self.x = self.tensor(states[f"residual_{layer}"])
        self.value = self.tensor(states[f"value_{layer}"])
        self.qk = {
            f"{name}_{layer}": cache.qk[f"{name}_{layer}"]
            for name in ("query", "key", "dtype", "scale")
        }
        self.history = {f"L{layer}": self.tensor(cache.history[f"L{layer}"])}
        dtype = getattr(torch, str(cache.qk[f"dtype_{layer}"]))
        self.post = (
            self.x.to(dtype) + self.tensor(states[f"attention_{layer}"]).to(dtype)
        ).float()
        self.eps = weights.config["rms_norm_eps"]
        self.input_scale = self.w["input_norm"] * torch.rsqrt(
            self.x.square().mean(-1, keepdim=True) + self.eps
        )
        self.post_scale = self.w["post_norm"] * torch.rsqrt(
            self.post.square().mean(-1, keepdim=True) + self.eps
        )

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def rows_attention(self, stop=None):
        for begin, end, attention in attention_chunks(
            self.cache.trace,
            self.qk,
            self.history,
            self.layer,
            self.device,
            self.chunk,
        ):
            if stop is not None:
                if begin >= stop:
                    return
                end = min(end, stop)
                attention = attention[:, : end - begin]
            yield begin, end, attention
            if stop is not None and end == stop:
                return
