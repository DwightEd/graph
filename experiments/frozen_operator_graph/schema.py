"""Typed in-memory and persisted schemas for frozen operator graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


PROMPT = 0
HISTORY = 1
SELF = 2
ROLE_COUNT = 3
ROLE_NAMES = ("prompt", "history", "self")

GRAPH_SCHEMA = "frozen-hypernetwork-operator-graph"
GRAPH_VERSION = 1


def _finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def source_roles(
    sources: torch.Tensor,
    *,
    target: int,
    response_start: int,
) -> torch.Tensor:
    """Classify sources; future positions receive sentinel role ``-1``.

    The sentinel keeps full absolute source coordinates available to vectorized
    callers while making future-token positions impossible to confuse with any
    causal role.  Graph construction must still intersect roles with the causal
    support before exposing edges.
    """

    sources = torch.as_tensor(sources, dtype=torch.long)
    if sources.ndim != 1:
        raise ValueError("sources must be one-dimensional")
    target = int(target)
    response_start = int(response_start)
    if bool((sources < 0).any()):
        raise ValueError("sources must be non-negative")
    role = torch.full_like(sources, -1)
    causal = sources <= target
    role[causal] = HISTORY
    role[causal & (sources < response_start)] = PROMPT
    role[sources == target] = SELF
    return role


@dataclass(frozen=True)
class LayerCapture:
    """Exact signals for one frozen Llama-like decoder layer.

    ``attention`` contains only response-query rows and has shape ``[H,R,N]``.
    The other tensors contain all token positions so source values and residual
    identities are never reconstructed from an approximation.
    """

    attention: torch.Tensor
    value_states: torch.Tensor
    o_proj_input: torch.Tensor
    residual_input: torch.Tensor
    pre_attention_hidden: torch.Tensor
    attention_output: torch.Tensor
    post_attention_residual: torch.Tensor
    pre_mlp_hidden: torch.Tensor
    mlp_output: torch.Tensor
    layer_output: torch.Tensor

    def validate(
        self,
        *,
        response_start: int,
        head_count: int,
        kv_head_count: int,
        head_dim: int,
        hidden_size: int,
        atol: float,
        rtol: float,
    ) -> "LayerCapture":
        tokens = int(self.residual_input.shape[0])
        response = tokens - int(response_start)
        expected_hidden = (tokens, hidden_size)
        hidden_tensors = {
            "residual_input": self.residual_input,
            "pre_attention_hidden": self.pre_attention_hidden,
            "attention_output": self.attention_output,
            "post_attention_residual": self.post_attention_residual,
            "pre_mlp_hidden": self.pre_mlp_hidden,
            "mlp_output": self.mlp_output,
            "layer_output": self.layer_output,
        }
        for name, tensor in hidden_tensors.items():
            if tensor.shape != expected_hidden:
                raise ValueError(f"{name} must have shape {expected_hidden}")
            _finite(name, tensor)
        if self.attention.shape != (head_count, response, tokens):
            raise ValueError(
                "attention must have shape [query_head,response_query,source]"
            )
        if self.value_states.shape != (tokens, kv_head_count, head_dim):
            raise ValueError("value_states has incompatible Llama/GQA geometry")
        if self.o_proj_input.shape != (tokens, head_count, head_dim):
            raise ValueError("o_proj_input must be [token,query_head,head_dim]")
        _finite("attention", self.attention)
        _finite("value_states", self.value_states)
        _finite("o_proj_input", self.o_proj_input)
        if bool((self.attention < -atol).any()):
            raise ValueError("attention probabilities must be non-negative")
        row_sum = self.attention.float().sum(dim=-1)
        if not torch.allclose(
            row_sum,
            torch.ones_like(row_sum),
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError("attention response rows must sum to one")
        response_query = torch.arange(response, device=self.attention.device)
        absolute_target = int(response_start) + response_query
        source = torch.arange(tokens, device=self.attention.device)
        future = source[None, :] > absolute_target[:, None]
        if bool((self.attention * future[None]).abs().max() > atol):
            raise ValueError("attention capture contains a future-token edge")
        if not torch.allclose(
            self.post_attention_residual.float(),
            self.residual_input.float() + self.attention_output.float(),
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError("post-attention residual does not equal residual + attention")
        if not torch.allclose(
            self.layer_output.float(),
            self.post_attention_residual.float() + self.mlp_output.float(),
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError("layer output does not equal post-attention residual + MLP")
        return self


@dataclass(frozen=True)
class ExactSampleCapture:
    checkpoint: str
    token_ids: torch.Tensor
    response_start: int
    final_hidden: torch.Tensor
    layers: Sequence[LayerCapture]
    q_to_kv: torch.Tensor
    head_count: int
    kv_head_count: int
    head_dim: int
    hidden_size: int
    attention_cache_binding: Mapping[str, object] | None = None

    @property
    def token_count(self) -> int:
        return int(self.token_ids.numel())

    @property
    def response_count(self) -> int:
        return self.token_count - int(self.response_start)

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def validate(self, *, atol: float, rtol: float) -> "ExactSampleCapture":
        if self.token_ids.ndim != 1 or self.token_ids.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise ValueError("token_ids must be an integer vector")
        if not 0 < int(self.response_start) < self.token_count:
            raise ValueError("response_start must split prompt and response")
        if self.layer_count < 1:
            raise ValueError("at least one decoder layer is required")
        if self.final_hidden.shape != (self.token_count, self.hidden_size):
            raise ValueError("final_hidden has the wrong shape")
        _finite("final_hidden", self.final_hidden)
        if self.q_to_kv.shape != (self.head_count,):
            raise ValueError("q_to_kv must contain one KV index per query head")
        if self.q_to_kv.dtype not in {torch.int32, torch.int64}:
            raise ValueError("q_to_kv must be integer")
        if bool(((self.q_to_kv < 0) | (self.q_to_kv >= self.kv_head_count)).any()):
            raise ValueError("q_to_kv contains an invalid KV-head index")
        if self.head_count * self.head_dim != self.hidden_size:
            raise ValueError("query-head geometry does not span hidden_size")
        for layer in self.layers:
            layer.validate(
                response_start=self.response_start,
                head_count=self.head_count,
                kv_head_count=self.kv_head_count,
                head_dim=self.head_dim,
                hidden_size=self.hidden_size,
                atol=atol,
                rtol=rtol,
            )
        return self


@dataclass(frozen=True)
class OperatorBasis:
    """Frozen output-projection views and operator-code geometry.

    ``output_factor`` stores one exact CPU tensor per layer without stacking
    them into another monolithic allocation.  The dtype is preserved from the
    checkpoint, and graph building upcasts only the current layer for stable
    feature arithmetic.
    """

    checkpoint: str
    output_factor: Sequence[torch.Tensor]
    output_bias: Sequence[torch.Tensor]
    has_output_bias: torch.Tensor
    normalized_operator_factor: torch.Tensor
    q_to_kv: torch.Tensor

    @property
    def layer_count(self) -> int:
        return len(self.output_factor)

    @property
    def head_count(self) -> int:
        return int(self.output_factor[0].shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.output_factor[0].shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.output_factor[0].shape[2])

    def validate(self) -> "OperatorBasis":
        if not self.output_factor:
            raise ValueError("operator basis must contain at least one layer")
        if len(self.output_bias) != self.layer_count:
            raise ValueError("output_bias must contain one vector per layer")
        heads, hidden, head_dim = self.output_factor[0].shape
        if heads * head_dim != hidden:
            raise ValueError("output factors do not span hidden size")
        for layer, factor in enumerate(self.output_factor):
            if factor.shape != (heads, hidden, head_dim):
                raise ValueError(f"layer {layer} output factor has inconsistent shape")
        for layer, bias in enumerate(self.output_bias):
            if bias.shape != (hidden,):
                raise ValueError(f"layer {layer} output bias has inconsistent shape")
            _finite(f"output_bias[{layer}]", bias)
        if self.has_output_bias.shape != (self.layer_count,):
            raise ValueError("has_output_bias must be [layer]")
        if self.has_output_bias.dtype != torch.bool:
            raise ValueError("has_output_bias must be boolean")
        if self.normalized_operator_factor.shape != (
            self.layer_count, heads, heads
        ):
            raise ValueError("normalized_operator_factor must be [layer,head,head]")
        if self.q_to_kv.shape != (heads,):
            raise ValueError("operator q_to_kv must be [head]")
        _finite("normalized_operator_factor", self.normalized_operator_factor)
        return self


@dataclass(frozen=True)
class OperatorGraphArtifact:
    """Persisted graph and deterministic response-token encodings."""

    sample_id: str
    source_id: str
    metadata: Mapping[str, object]
    token_ids: torch.Tensor
    response_start: int
    edge_index: torch.Tensor
    edge_layer: torch.Tensor
    edge_role: torch.Tensor
    edge_attention_code: torch.Tensor
    edge_features: torch.Tensor
    edge_feature_names: Sequence[str]
    remainder_features: torch.Tensor
    remainder_feature_names: Sequence[str]
    route_features: torch.Tensor
    route_feature_names: Sequence[str]
    layer_features: torch.Tensor
    layer_feature_names: Sequence[str]
    temporal_features: torch.Tensor
    temporal_feature_names: Sequence[str]
    final_hidden: torch.Tensor
    node_embedding: torch.Tensor
    node_feature_names: Sequence[str]
    audit: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @property
    def response_count(self) -> int:
        return int(self.token_ids.numel()) - int(self.response_start)

    def validate(self) -> "OperatorGraphArtifact":
        response = self.response_count
        if self.token_ids.ndim != 1:
            raise ValueError("artifact token_ids must be one-dimensional")
        if not 0 < int(self.response_start) < int(self.token_ids.numel()):
            raise ValueError("artifact response_start is invalid")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must be [2,E]")
        edges = int(self.edge_index.shape[1])
        for name, tensor in (
            ("edge_layer", self.edge_layer),
            ("edge_role", self.edge_role),
        ):
            if tensor.shape != (edges,):
                raise ValueError(f"{name} must have one value per edge")
        if self.edge_attention_code.shape[0] != edges:
            raise ValueError("edge_attention_code must have one row per edge")
        if self.edge_features.shape != (edges, len(self.edge_feature_names)):
            raise ValueError("edge feature names do not match edge_features")
        if self.remainder_features.shape[-1] != len(self.remainder_feature_names):
            raise ValueError("remainder feature names do not match tensor")
        if self.route_features.shape[0] != response:
            raise ValueError("route_features must be response aligned")
        if self.route_features.shape[-1] != len(self.route_feature_names):
            raise ValueError("route feature names do not match tensor")
        if self.layer_features.shape[0] != response:
            raise ValueError("layer_features must be response aligned")
        if self.layer_features.shape[-1] != len(self.layer_feature_names):
            raise ValueError("layer feature names do not match tensor")
        if self.temporal_features.shape != (
            response,
            len(self.temporal_feature_names),
        ):
            raise ValueError("temporal_features has incompatible geometry")
        if self.final_hidden.shape[0] != response:
            raise ValueError("final_hidden must be response aligned")
        if self.node_embedding.shape != (
            response,
            len(self.node_feature_names),
        ):
            raise ValueError("node_embedding names do not match its width")
        for name, tensor in (
            ("edge_attention_code", self.edge_attention_code),
            ("edge_features", self.edge_features),
            ("remainder_features", self.remainder_features),
            ("route_features", self.route_features),
            ("layer_features", self.layer_features),
            ("temporal_features", self.temporal_features),
            ("final_hidden", self.final_hidden),
            ("node_embedding", self.node_embedding),
        ):
            _finite(name, tensor)
        return self
