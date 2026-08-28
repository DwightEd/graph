"""Deterministic synthetic fixtures for operator-graph tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from ..basis import extract_operator_basis, q_to_kv_mapping
from ..schema import ExactSampleCapture, LayerCapture, OperatorBasis


@dataclass
class SyntheticBundle:
    capture: ExactSampleCapture
    basis: OperatorBasis


def synthetic_bundle(
    *,
    layer_count: int = 2,
    head_count: int = 2,
    kv_head_count: int = 1,
    head_dim: int = 2,
    token_count: int = 5,
    response_start: int = 2,
    seed: int = 7,
) -> SyntheticBundle:
    generator = torch.Generator().manual_seed(seed)
    hidden = head_count * head_dim
    response = token_count - response_start
    mapping = q_to_kv_mapping(head_count, kv_head_count)
    captures = []
    output_factors = []
    output_biases = []
    normalized_factors = []
    for _layer in range(layer_count):
        output_weight = torch.randn(hidden, hidden, generator=generator) * 0.1
        output_factor = output_weight.reshape(
            hidden, head_count, head_dim
        ).permute(1, 0, 2).contiguous()
        output_bias = torch.randn(hidden, generator=generator) * 0.01
        value = torch.randn(
            token_count, kv_head_count, head_dim, generator=generator
        )
        attention = torch.zeros(head_count, response, token_count)
        for query in range(response):
            target = response_start + query
            row = torch.rand(head_count, target + 1, generator=generator)
            row = row / row.sum(dim=-1, keepdim=True)
            attention[:, query, : target + 1] = row
        value_by_head = value[:, mapping]
        context = torch.einsum("hrn,nhd->rhd", attention, value_by_head)
        o_proj_input = torch.randn(
            token_count, head_count, head_dim, generator=generator
        )
        o_proj_input[response_start:] = context
        attention_output = torch.randn(
            token_count, hidden, generator=generator
        ) * 0.01
        attention_output[response_start:] = torch.nn.functional.linear(
            context.reshape(response, hidden),
            output_weight,
            output_bias,
        )
        residual = torch.randn(token_count, hidden, generator=generator)
        post_attention = residual + attention_output
        mlp = torch.randn(token_count, hidden, generator=generator) * 0.1
        layer_output = post_attention + mlp
        captures.append(
            LayerCapture(
                attention=attention,
                value_states=value,
                o_proj_input=o_proj_input,
                residual_input=residual,
                pre_attention_hidden=torch.randn(
                    token_count, hidden, generator=generator
                ),
                attention_output=attention_output,
                post_attention_residual=post_attention,
                pre_mlp_hidden=torch.randn(
                    token_count, hidden, generator=generator
                ),
                mlp_output=mlp,
                layer_output=layer_output,
            )
        )
        output_factors.append(output_factor)
        output_biases.append(output_bias)
        normalized_factors.append(torch.eye(head_count))
    capture = ExactSampleCapture(
        checkpoint="synthetic-checkpoint",
        token_ids=torch.arange(token_count, dtype=torch.long),
        response_start=response_start,
        final_hidden=torch.randn(token_count, hidden, generator=generator),
        layers=tuple(captures),
        q_to_kv=mapping,
        head_count=head_count,
        kv_head_count=kv_head_count,
        head_dim=head_dim,
        hidden_size=hidden,
        attention_cache_binding=None,
    )
    basis = OperatorBasis(
        checkpoint="synthetic-checkpoint",
        output_factor=tuple(output_factors),
        output_bias=tuple(output_biases),
        has_output_bias=torch.ones(layer_count, dtype=torch.bool),
        normalized_operator_factor=torch.stack(normalized_factors),
        q_to_kv=mapping,
    )
    return SyntheticBundle(capture=capture, basis=basis)


class TinyAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.q_to_kv = q_to_kv_mapping(heads, kv_heads)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, _ = hidden_states.shape
        value = self.v_proj(hidden_states).reshape(
            batch, tokens, self.kv_heads, self.head_dim
        )
        value = value[:, :, self.q_to_kv]
        source = torch.arange(tokens, device=hidden_states.device)
        target = torch.arange(tokens, device=hidden_states.device)
        allowed = source[None] <= target[:, None]
        weights = allowed.to(dtype=hidden_states.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights[None, None].expand(batch, self.heads, -1, -1)
        context = torch.einsum("bhqs,bshd->bqhd", weights, value)
        output = self.o_proj(context.reshape(batch, tokens, -1))
        return output, weights


class TinyLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.self_attn = TinyAttention(hidden, heads, kv_heads, head_dim)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any):
        residual = hidden_states
        attention, weights = self.self_attn(
            self.input_layernorm(hidden_states), **kwargs
        )
        hidden_states = residual + attention
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, weights


class TinyBackbone(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        layers: int,
        hidden: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.embed_tokens = embedding
        self.layers = nn.ModuleList(
            TinyLayer(hidden, heads, kv_heads, head_dim) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any):
        hidden = self.embed_tokens(input_ids)
        attentions = []
        for layer in self.layers:
            output = layer(hidden, **kwargs)
            hidden = output[0]
            attentions.append(output[1])
        return SimpleNamespace(
            last_hidden_state=self.norm(hidden),
            attentions=tuple(attentions),
        )


class TinyCausalLM(nn.Module):
    def __init__(
        self,
        *,
        layers: int = 2,
        hidden: int = 4,
        heads: int = 2,
        kv_heads: int = 1,
        vocabulary: int = 32,
    ) -> None:
        super().__init__()
        head_dim = hidden // heads
        self.config = SimpleNamespace(
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            hidden_size=hidden,
            head_dim=head_dim,
        )
        self.embedding = nn.Embedding(vocabulary, hidden)
        self.model = TinyBackbone(
            self.embedding, layers, hidden, heads, kv_heads, head_dim
        )
        self.lm_head = nn.Linear(hidden, vocabulary, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


def tiny_model_bundle() -> tuple[TinyCausalLM, OperatorBasis]:
    torch.manual_seed(17)
    model = TinyCausalLM()
    basis = extract_operator_basis(
        model,
        checkpoint="tiny-checkpoint",
        compute_device="cpu",
        compute_dtype=torch.float32,
    )
    return model, basis


class FakeAttentionCache:
    """Canonical-cache-shaped object derived from an exact synthetic capture."""

    def __init__(self, capture: ExactSampleCapture, *, floor: float = 0.08) -> None:
        self.num_layers = capture.layer_count
        self.num_heads = capture.head_count
        self.num_tokens = capture.token_count
        self.num_response_tokens = capture.response_count
        self.response_idx = capture.response_start
        self.attention_floor = float(floor)
        self.token_ids = capture.token_ids.clone()
        self.attention_diagonal = torch.zeros(
            self.num_layers, self.num_heads, self.num_tokens
        )
        row_ptr = [0]
        columns = []
        values = []
        for layer, layer_capture in enumerate(capture.layers):
            for head in range(self.num_heads):
                for query in range(self.num_response_tokens):
                    target = self.response_idx + query
                    row = layer_capture.attention[head, query]
                    self.attention_diagonal[layer, head, target] = row[target]
                    retained = torch.nonzero(
                        (torch.arange(self.num_tokens) < target)
                        & (row >= self.attention_floor),
                        as_tuple=False,
                    ).flatten()
                    columns.extend(retained.tolist())
                    values.extend(row[retained].tolist())
                    row_ptr.append(len(columns))
        self.response_row_ptr = torch.tensor(row_ptr, dtype=torch.long)
        self.response_column_indices = torch.tensor(columns, dtype=torch.long)
        self.response_values = torch.tensor(values, dtype=torch.float32)
