"""Sparse layer/head edge encoding and label-free sensitivity refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .data import PROMPT, RESPONSE, SourceReuseGraph
from .grounding_config import GroundingGraphConfig


@dataclass(frozen=True)
class PairEncoding:
    sources: torch.Tensor
    embedding: torch.Tensor
    mass: torch.Tensor
    origin: torch.Tensor
    sensitivity: torch.Tensor
    gate: torch.Tensor


class SparseLayerHeadEncoder(nn.Module):
    """Encode token-source incidences while preserving transformer depth order."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        config: GroundingGraphConfig,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.config = config
        self.layer = nn.Embedding(num_layers, config.layer_embedding_dim)
        self.head = nn.Embedding(num_heads, config.head_embedding_dim)
        self.relation = nn.Embedding(2, config.relation_embedding_dim)
        self.lag = nn.Embedding(config.response_lag_bins, config.lag_embedding_dim)
        event_dim = (
            config.layer_embedding_dim
            + config.head_embedding_dim
            + config.relation_embedding_dim
            + config.lag_embedding_dim
            + 4
        )
        self.event = nn.Sequential(
            nn.Linear(event_dim, config.hidden_dim),
            nn.PReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.layer_projection = nn.Sequential(
            nn.Linear(config.hidden_dim + 2, config.hidden_dim),
            nn.PReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.depth = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_dim + 4, config.hidden_dim),
            nn.PReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def _lag_bin(
        self,
        graph: SourceReuseGraph,
        token: int,
        source: torch.Tensor,
    ) -> torch.Tensor:
        prompt = source < graph.response_idx
        prompt_position = source.float() / float(max(graph.response_idx - 1, 1))
        prompt_bin = torch.floor(
            prompt_position * self.config.response_lag_bins
        ).long().clamp_max(self.config.response_lag_bins - 1)
        response_lag = token - (source - graph.response_idx)
        response_bin = torch.floor(
            torch.log2(response_lag.clamp_min(1).float())
        ).long().clamp_max(self.config.response_lag_bins - 1)
        return torch.where(prompt, prompt_bin, response_bin)

    def forward(
        self,
        graph: SourceReuseGraph,
        *,
        token: int,
        observed_weight: torch.Tensor,
        base_weight: torch.Tensor,
        origin: torch.Tensor,
        sensitivity: torch.Tensor | None = None,
        refine: bool,
    ) -> PairEncoding:
        current = graph.token_slice(token)
        source = graph.source[current]
        if source.numel() == 0:
            empty = base_weight.new_empty((0, self.config.hidden_dim))
            scalar = base_weight.new_empty(0)
            return PairEncoding(source, empty, scalar, scalar, scalar, scalar)

        layer = graph.layer[current]
        head = graph.head[current]
        relation = torch.where(
            source < graph.response_idx,
            torch.full_like(source, PROMPT),
            torch.full_like(source, RESPONSE),
        )
        if sensitivity is None:
            sensitivity = observed_weight.new_zeros(observed_weight.shape)
        numeric = torch.stack(
            (
                observed_weight,
                torch.log1p(
                    observed_weight / max(graph.attention_floor, 1e-8)
                ),
                origin,
                sensitivity,
            ),
            dim=-1,
        )
        event = self.event(
            torch.cat(
                (
                    self.layer(layer),
                    self.head(head),
                    self.relation(relation),
                    self.lag(self._lag_bin(graph, token, source)),
                    numeric,
                ),
                dim=-1,
            )
        )

        unique_source, inverse = torch.unique(
            source,
            sorted=True,
            return_inverse=True,
        )
        pair_count = unique_source.shape[0]
        flat = inverse * self.num_layers + layer
        layer_mass = observed_weight.new_zeros(pair_count * self.num_layers)
        layer_count = observed_weight.new_zeros(pair_count * self.num_layers)
        layer_sum = event.new_zeros(
            (pair_count * self.num_layers, event.shape[-1])
        )
        layer_mass.index_add_(0, flat, observed_weight)
        layer_count.index_add_(0, flat, (observed_weight > 0).float())
        layer_sum.index_add_(0, flat, event * observed_weight.unsqueeze(-1))
        layer_mean = layer_sum / layer_mass.clamp_min(1e-8).unsqueeze(-1)
        layer_feature = torch.cat(
            (
                layer_mean,
                torch.log1p(
                    layer_mass / max(graph.attention_floor, 1e-8)
                ).unsqueeze(-1),
                (layer_count / float(self.num_heads)).unsqueeze(-1),
            ),
            dim=-1,
        )
        layer_feature = self.layer_projection(layer_feature).reshape(
            pair_count,
            self.num_layers,
            -1,
        )
        _, hidden = self.depth(layer_feature)
        pair_embedding = hidden[-1]

        pair_mass = base_weight.new_zeros(pair_count)
        pair_origin_sum = base_weight.new_zeros(pair_count)
        pair_sensitivity_sum = base_weight.new_zeros(pair_count)
        pair_mass.index_add_(0, inverse, base_weight)
        pair_origin_sum.index_add_(0, inverse, base_weight * origin)
        pair_sensitivity_sum.index_add_(0, inverse, base_weight * sensitivity)
        pair_origin = pair_origin_sum / pair_mass.clamp_min(1e-8)
        pair_sensitivity = pair_sensitivity_sum / pair_mass.clamp_min(1e-8)
        pair_observed_mass = observed_weight.new_zeros(pair_count)
        pair_observed_mass.index_add_(0, inverse, observed_weight)
        pair_relation = (unique_source >= graph.response_idx).float()

        if refine:
            gate_input = torch.cat(
                (
                    pair_embedding,
                    pair_origin[:, None],
                    pair_sensitivity[:, None],
                    torch.log1p(
                        pair_mass / max(graph.attention_floor, 1e-8)
                    )[:, None],
                    pair_relation[:, None],
                ),
                dim=-1,
            )
            gate = torch.sigmoid(self.gate(gate_input).squeeze(-1))
        else:
            gate = pair_mass.new_ones(pair_count)

        return PairEncoding(
            sources=unique_source,
            embedding=pair_embedding,
            mass=pair_observed_mass,
            origin=pair_origin.clamp(0.0, 1.0),
            sensitivity=pair_sensitivity,
            gate=gate,
        )
