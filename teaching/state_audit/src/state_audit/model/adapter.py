"""One explicit adapter for the shared Llama/Mistral/Qwen2 decoder layout."""

from contextlib import contextmanager
from importlib import import_module
from types import MethodType
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..operations.messages import replace_source_readouts

SUPPORTED_MODELS = ("llama", "mistral", "qwen2")


class ModelAdapter:
    """Maps native modules to public representations; no dataset or audit logic."""

    def __init__(self, native):
        family = native.config.model_type
        if family not in SUPPORTED_MODELS:
            raise ValueError(f"Supported layouts: {SUPPORTED_MODELS}; add an explicit adapter")
        scaling = native.config.to_dict().get("rope_scaling") or {}
        if scaling.get("rope_type") in {"dynamic", "longrope"}:
            raise ValueError("Length-dependent RoPE needs stepwise capture, not full-prefix replay")
        native.set_attn_implementation("eager")
        self.native = native.eval()
        self.layers = native.model.layers
        self.implementation = import_module(f"transformers.models.{family}.modeling_{family}")

    def input_ids(self, token_ids):
        return torch.tensor([token_ids], device=self.native.device, dtype=torch.long)

    def forward(self, token_ids):
        """Full-prefix replay, retaining native masks; no KV cache or labels."""
        output = self.native.model(
            input_ids=self.input_ids(token_ids), use_cache=False, return_dict=True
        )
        return output.last_hidden_state[0]

    def score(self, hidden, target_ids):
        """Native log probabilities and entropy, independent of sampling temperature."""
        logits = self.native.lm_head(hidden).float()
        log_probs = logits.log_softmax(-1)
        actual = log_probs.gather(-1, target_ids[..., None]).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        return actual, entropy

    def head_layout(self, layer):
        """Native query-head count, KV-head count, and head width, including GQA."""
        attention = self.layers[layer].self_attn
        width = attention.head_dim
        return attention.q_proj.out_features // width, attention.k_proj.out_features // width, width

    def project_heads(self, layer, readout, heads):
        """[position, head, width] -> individual W_O writes; exclude shared output bias."""
        projection = self.layers[layer].self_attn.o_proj.weight
        count, _, width = self.head_layout(layer)
        matrices = projection.reshape(projection.shape[0], count, width).permute(1, 2, 0)
        selected = matrices[list(heads)].float()
        values = torch.as_tensor(readout, device=projection.device, dtype=torch.float32)
        return (values.transpose(0, 1) @ selected).transpose(0, 1)

    def module_at(self, name, layer):
        """Return (module, input_or_output) for a public representation."""
        if name == "embedding":
            return self.native.model.embed_tokens, "output"
        if name == "final_hidden":
            return self.native.model.norm, "output"
        block = self.layers[layer]
        sites = {
            "residual_before": (block, "input"),
            "attention_input": (block.input_layernorm, "output"),
            "query": (block.self_attn.q_proj, "output"),
            "key": (block.self_attn.k_proj, "output"),
            "value": (block.self_attn.v_proj, "output"),
            "head_readout": (block.self_attn.o_proj, "input"),
            "attention_write": (block.self_attn.o_proj, "output"),
            "residual_mid": (block.post_attention_layernorm, "input"),
            "mlp_input": (block.post_attention_layernorm, "output"),
            "mlp_activation": (block.mlp.down_proj, "input"),
            "mlp_write": (block.mlp, "output"),
            "residual_after": (block, "output"),
        }
        return sites[name]

    def transform_native(self, name, layer, tensor, transform):
        value = tensor[0]
        if name in ("query", "key", "value", "head_readout"):
            width = self.layers[layer].self_attn.head_dim
            value = value.reshape(value.shape[0], -1, width)
        if name in ("query", "key", "value"):
            value = value.transpose(0, 1)
        changed = transform(value)
        if name in ("query", "key", "value"):
            changed = changed.transpose(0, 1)
        return changed.reshape(tensor.shape)

    @contextmanager
    def bind(self, name, layer, transform, *, edit=False):
        """Scope an observer/transform to one site and remove it even on failure."""
        if name == "attention":
            with self.bind_attention(layer, transform, edit):
                yield
            return
        module, location = self.module_at(name, layer)

        def before(module, inputs):
            value = self.transform_native(name, layer, inputs[0], transform)
            return (value, *inputs[1:])

        def after(module, inputs, output):
            return self.transform_native(name, layer, output, transform)

        if location == "input":
            handle = module.register_forward_pre_hook(before)
        else:
            handle = module.register_forward_hook(after)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def bind_attention(self, layer, transform, edit, replacements=()):
        attention = self.layers[layer].self_attn
        if edit:
            forward = self.attention_forward(transform, replacements)
            with patch.object(attention, "forward", MethodType(forward, attention)):
                yield
            return

        def observe(module, inputs, output):
            transform(output[1][0])

        handle = attention.register_forward_hook(observe)
        try:
            yield
        finally:
            handle.remove()

    def attention_forward(self, transform, replacements=()):
        """Native Q/K/RoPE/mask; edit probabilities before A @ V and o_proj."""
        implementation = self.implementation

        def forward(module, hidden_states, position_embeddings, attention_mask, **kwargs):
            if kwargs.get("past_key_values") is not None:
                raise ValueError("Attention edits require full-prefix replay: use_cache=False")
            shape = (*hidden_states.shape[:-1], -1, module.head_dim)
            query = module.q_proj(hidden_states).view(shape).transpose(1, 2)
            key = module.k_proj(hidden_states).view(shape).transpose(1, 2)
            value = module.v_proj(hidden_states).view(shape).transpose(1, 2)
            cosine, sine = position_embeddings
            query, key = implementation.apply_rotary_pos_emb(query, key, cosine, sine)
            _, weights = implementation.eager_attention_forward(
                module, query, key, value, attention_mask, scaling=module.scaling, dropout=0.0
            )
            changed = transform(weights[0]).unsqueeze(0)
            validate_attention(changed, attention_mask)
            expanded_value = implementation.repeat_kv(value, module.num_key_value_groups)
            output = replace_source_readouts(changed, expanded_value, replacements)
            output = output.transpose(1, 2)
            output = output.reshape(*hidden_states.shape[:-1], -1).contiguous()
            return module.o_proj(output), changed

        return forward


def validate_attention(weights, mask):
    """Edits must not introduce negative weights or open forbidden edges."""
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Attention weights must remain finite and nonnegative")
    if mask is None:
        length = weights.shape[-1]
        forbidden = torch.ones((length, length), device=weights.device, dtype=torch.bool).triu(1)
    else:
        forbidden = mask[..., : weights.shape[-1]] < 0
    if torch.any(weights.masked_select(forbidden) != 0):
        raise ValueError("Attention operation opened a causal or sliding-window masked edge")


def load_model(name: str, revision: str, device: str, dtype: str):
    native = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=getattr(torch, dtype), attn_implementation="eager"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Verified character offsets require a fast tokenizer")
    return ModelAdapter(native), tokenizer
