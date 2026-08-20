"""Causal routing features that preserve every layer and attention head."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import PhenomenologyConfig
from .routing import build_routing_state, collect_routing_edges
from .sources import response_lag_statistics, summarize_exact_sources


BASE_FEATURE_NAMES = (
    "prompt_mass",
    "response_mass",
    "self_mass",
    "unresolved_mass",
    "history_edge_fraction",
    "response_takeover",
    "prompt_effective_sources",
    "prompt_top1_share",
    "prompt_source_velocity",
    "prompt_active",
    "response_effective_sources",
    "response_top1_share",
    "recent_response_share",
    "response_mean_lag",
    "response_source_velocity",
    "response_active",
)


@dataclass(frozen=True)
class HeadResolvedFeatures:
    """One label-free tensor with shape ``[token, layer, head, feature]``."""

    values: torch.Tensor
    feature_names: tuple[str, ...]

    def feature(self, name: str) -> torch.Tensor:
        """Return one ``[token, layer, head]`` field by its scientific name."""

        try:
            index = self.feature_names.index(name)
        except ValueError as error:
            raise KeyError(name) from error
        return self.values[..., index]


class HeadResolvedFeatureExtractor:
    """Extract direct routes and cumulative response-source reuse per head."""

    def __init__(
        self,
        *,
        reuse_top_k: int = 5,
        recent_response_tokens: int = 4,
        block_rows: int = 8192,
        epsilon: float = 1e-8,
    ) -> None:
        if reuse_top_k < 0:
            raise ValueError("reuse_top_k cannot be negative")
        self.reuse_top_k = int(reuse_top_k)
        self.config = PhenomenologyConfig(
            recent_response_tokens=recent_response_tokens,
            block_rows=block_rows,
            epsilon=epsilon,
        )
        self.feature_names = BASE_FEATURE_NAMES + tuple(
            f"response_reuse_rank_{rank}" for rank in range(1, reuse_top_k + 1)
        )

    def extract(self, sample) -> HeadResolvedFeatures:
        """Read one sample and return finite, post-emission causal features."""

        edges = collect_routing_edges(sample, config=self.config)
        routing = build_routing_state(edges)
        prompt = summarize_exact_sources(
            routing,
            role="prompt",
            epsilon=self.config.epsilon,
        )
        response = summarize_exact_sources(
            routing,
            role="response",
            epsilon=self.config.epsilon,
        )
        recent_share, mean_lag = response_lag_statistics(
            routing,
            response,
            recent_tokens=self.config.recent_response_tokens,
            epsilon=self.config.epsilon,
        )

        off_diagonal_mass = routing.prompt_mass + routing.response_mass
        response_takeover = torch.where(
            off_diagonal_mass > self.config.epsilon,
            routing.response_mass / off_diagonal_mass.clamp_min(self.config.epsilon),
            torch.zeros_like(off_diagonal_mass),
        )
        fields = (
            routing.prompt_mass,
            routing.response_mass,
            routing.self_mass,
            routing.unresolved_mass,
            self._history_edge_fraction(edges),
            response_takeover,
            prompt.effective_sources,
            prompt.top1_share,
            prompt.velocity,
            prompt.valid.float(),
            response.effective_sources,
            response.top1_share,
            recent_share,
            mean_lag,
            response.velocity,
            response.valid.float(),
        )
        direct = torch.stack(fields, dim=-1)
        reuse = self._cumulative_response_reuse(routing)
        values = torch.cat((direct, reuse), dim=-1)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("head-resolved features contain non-finite values")
        return HeadResolvedFeatures(values=values, feature_names=self.feature_names)

    def _history_edge_fraction(self, edges) -> torch.Tensor:
        shape = (edges.num_response_tokens, edges.num_layers, edges.num_heads)
        prompt_count = torch.zeros(shape, dtype=torch.float32, device=edges.device)
        response_count = torch.zeros_like(prompt_count)
        if edges.weight.numel():
            is_prompt = edges.source < edges.response_idx
            ones = torch.ones_like(edges.weight)
            prompt_count.index_put_(
                (
                    edges.query[is_prompt],
                    edges.layer[is_prompt],
                    edges.head[is_prompt],
                ),
                ones[is_prompt],
                accumulate=True,
            )
            is_response = ~is_prompt
            response_count.index_put_(
                (
                    edges.query[is_response],
                    edges.layer[is_response],
                    edges.head[is_response],
                ),
                ones[is_response],
                accumulate=True,
            )
        total = prompt_count + response_count
        return torch.where(
            total > 0,
            response_count / total.clamp_min(1.0),
            torch.zeros_like(total),
        )

    def _cumulative_response_reuse(self, routing) -> torch.Tensor:
        """Return old ``received_topk`` coordinates with their real meaning."""

        edges = routing.edges
        tokens = edges.num_response_tokens
        channels = edges.num_layers * edges.num_heads
        output = torch.zeros(
            (tokens, channels, self.reuse_top_k),
            dtype=routing.edge_weight.dtype,
            device=edges.device,
        )
        response_edge = edges.source >= edges.response_idx
        query = edges.query[response_edge]
        source = edges.source[response_edge] - edges.response_idx
        channel = (
            edges.layer[response_edge] * edges.num_heads + edges.head[response_edge]
        )
        weight = routing.edge_weight[response_edge]
        received = torch.zeros(
            (channels, tokens),
            dtype=routing.edge_weight.dtype,
            device=edges.device,
        )

        for token in range(tokens):
            current = query == token
            if bool(current.any()):
                received.index_put_(
                    (channel[current], source[current]),
                    weight[current],
                    accumulate=True,
                )
            keep = min(self.reuse_top_k, token + 1)
            if keep:
                age = token - torch.arange(token + 1, device=edges.device) + 1
                age_normalized = received[:, : token + 1] / age
                output[token, :, :keep] = torch.topk(
                    age_normalized,
                    k=keep,
                    dim=1,
                    largest=True,
                    sorted=True,
                ).values

        return output.reshape(
            tokens,
            edges.num_layers,
            edges.num_heads,
            self.reuse_top_k,
        )
