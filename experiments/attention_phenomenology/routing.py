"""Attention routing tensors built only through the canonical research interface.

The representation has two complementary views:

* a fixed role simplex shared across samples, used for head-set geometry;
* exact response-source endpoints, used for provenance and topology controls.

Censored attention is represented by one explicit ``unresolved`` role. It is
never treated as an observed zero or distributed over invented source edges.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from .config import PhenomenologyConfig


@dataclass(frozen=True)
class RoutingEdges:
    """One sample's retained causal edges and exact self-attention diagonals."""

    num_layers: int
    num_heads: int
    num_response_tokens: int
    num_tokens: int
    response_idx: int
    attention_floor: float
    layer: torch.Tensor
    head: torch.Tensor
    query: torch.Tensor
    source: torch.Tensor
    weight: torch.Tensor
    diagonal: torch.Tensor  # [response, layer, head]

    @property
    def device(self):
        return self.diagonal.device


@dataclass(frozen=True)
class RoutingTensor:
    """Role-aligned and endpoint-preserving routing fields for one sample."""

    edges: RoutingEdges
    role_probability: torch.Tensor  # [response, layer, head, role]
    prompt_mass: torch.Tensor  # [response, layer, head]
    rr_mass: torch.Tensor  # retained off-diagonal RR mass
    self_mass: torch.Tensor
    unresolved_mass: torch.Tensor
    known_mass: torch.Tensor
    source_mass: torch.Tensor  # [response, layer, response source], summed over heads
    rr_query: torch.Tensor
    rr_layer: torch.Tensor
    rr_head: torch.Tensor
    rr_source: torch.Tensor
    rr_weight: torch.Tensor


def collect_routing_edges(
    sample,
    *,
    config: PhenomenologyConfig | None = None,
) -> RoutingEdges:
    """Collect one sample through ``ResearchSample.iter_sparse_attention_blocks``."""

    config = PhenomenologyConfig() if config is None else config
    attention = sample.attention()
    layers = int(attention.num_layers)
    heads = int(attention.num_heads)
    response_count = int(attention.num_response_tokens)
    response_idx = int(attention.response_idx)

    layer_parts: list[torch.Tensor] = []
    head_parts: list[torch.Tensor] = []
    query_parts: list[torch.Tensor] = []
    source_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []

    for block in sample.iter_sparse_attention_blocks(block_rows=config.block_rows):
        target = response_idx + block.query
        off_diagonal = block.source < target
        if not bool(off_diagonal.any()):
            continue
        layer_parts.append(block.layer[off_diagonal].long())
        head_parts.append(block.head[off_diagonal].long())
        query_parts.append(block.query[off_diagonal].long())
        source_parts.append(block.source[off_diagonal].long())
        weight_parts.append(block.weight[off_diagonal].float().clamp_min(0.0))

    device = attention.response_values.device
    if layer_parts:
        layer = torch.cat(layer_parts)
        head = torch.cat(head_parts)
        query = torch.cat(query_parts)
        source = torch.cat(source_parts)
        weight = torch.cat(weight_parts)
    else:
        layer = torch.empty(0, dtype=torch.long, device=device)
        head = torch.empty_like(layer)
        query = torch.empty_like(layer)
        source = torch.empty_like(layer)
        weight = torch.empty(0, dtype=torch.float32, device=device)

    diagonal = (
        attention.attention_diagonal[:, :, response_idx:]
        .float()
        .permute(2, 0, 1)
        .contiguous()
    )
    return RoutingEdges(
        num_layers=layers,
        num_heads=heads,
        num_response_tokens=response_count,
        num_tokens=int(attention.num_tokens),
        response_idx=response_idx,
        attention_floor=float(attention.attention_floor),
        layer=layer,
        head=head,
        query=query,
        source=source,
        weight=weight,
        diagonal=diagonal,
    )


def _prompt_bins(source: torch.Tensor, prompt_count: int, bins: int) -> torch.Tensor:
    return torch.div(source * bins, max(prompt_count, 1), rounding_mode="floor").clamp_max(
        bins - 1
    )


def _lag_bins(lag: torch.Tensor, bins: int) -> torch.Tensor:
    return torch.floor(torch.log2(lag.float())).long().clamp_max(bins - 1)


def build_routing_tensor(
    edges: RoutingEdges,
    *,
    config: PhenomenologyConfig | None = None,
) -> RoutingTensor:
    """Build role probabilities and exact RR source mass from retained edges."""

    config = PhenomenologyConfig() if config is None else config
    r = edges.num_response_tokens
    l = edges.num_layers
    h = edges.num_heads
    device = edges.device

    role = torch.zeros(
        (r, l, h, config.role_count), dtype=torch.float32, device=device
    )
    role[:, :, :, config.self_role] = edges.diagonal

    if edges.weight.numel():
        is_prompt = edges.source < edges.response_idx
        role_index = torch.empty_like(edges.source)
        if bool(is_prompt.any()):
            role_index[is_prompt] = _prompt_bins(
                edges.source[is_prompt], edges.response_idx, config.prompt_bins
            )
        history = ~is_prompt
        if bool(history.any()):
            response_source = edges.source[history] - edges.response_idx
            lag = edges.query[history] - response_source
            role_index[history] = config.prompt_bins + _lag_bins(
                lag, config.rr_lag_bins
            )
        role.index_put_(
            (edges.query, edges.layer, edges.head, role_index),
            edges.weight,
            accumulate=True,
        )

    known_mass = role[:, :, :, : config.unresolved_role].sum(dim=3)
    row_scale = torch.where(
        known_mass > 1.0,
        known_mass.reciprocal(),
        torch.ones_like(known_mass),
    )
    role[:, :, :, : config.unresolved_role] *= row_scale[:, :, :, None]
    known_mass = known_mass * row_scale
    unresolved = (1.0 - known_mass).clamp_min(0.0)
    role[:, :, :, config.unresolved_role] = unresolved

    prompt_mass = role[:, :, :, : config.prompt_bins].sum(dim=3)
    rr_mass = role[
        :, :, :, config.prompt_bins : config.prompt_bins + config.rr_lag_bins
    ].sum(dim=3)
    self_mass = role[:, :, :, config.self_role]

    if edges.weight.numel():
        scaled_weight = edges.weight * row_scale[
            edges.query, edges.layer, edges.head
        ]
        rr = edges.source >= edges.response_idx
        rr_query = edges.query[rr]
        rr_layer = edges.layer[rr]
        rr_head = edges.head[rr]
        rr_source = edges.source[rr] - edges.response_idx
        rr_weight = scaled_weight[rr]
    else:
        rr_query = edges.query
        rr_layer = edges.layer
        rr_head = edges.head
        rr_source = edges.source
        rr_weight = edges.weight

    source_mass = torch.zeros((r, l, r), dtype=torch.float32, device=device)
    if rr_weight.numel():
        source_mass.index_put_(
            (rr_query, rr_layer, rr_source), rr_weight, accumulate=True
        )

    return RoutingTensor(
        edges=edges,
        role_probability=role,
        prompt_mass=prompt_mass,
        rr_mass=rr_mass,
        self_mass=self_mass,
        unresolved_mass=unresolved,
        known_mass=known_mass,
        source_mass=source_mass,
        rr_query=rr_query,
        rr_layer=rr_layer,
        rr_head=rr_head,
        rr_source=rr_source,
        rr_weight=rr_weight,
    )


def _prompt_candidates(prompt_count: int, prompt_bin: int, bins: int) -> np.ndarray:
    if prompt_count < 1:
        return np.empty(0, dtype=np.int64)
    start = min((prompt_bin * prompt_count) // bins, prompt_count - 1)
    stop = min(
        max(((prompt_bin + 1) * prompt_count) // bins, start + 1), prompt_count
    )
    return np.arange(start, stop, dtype=np.int64)


def _response_candidates(query: int, lag_bin: int, bins: int) -> np.ndarray:
    if query < 1:
        return np.empty(0, dtype=np.int64)
    source = np.arange(query, dtype=np.int64)
    lag = query - source
    current_bin = np.minimum(np.floor(np.log2(lag)).astype(np.int64), bins - 1)
    return source[current_bin == lag_bin]


def rewire_exact_endpoints(
    edges: RoutingEdges,
    *,
    config: PhenomenologyConfig | None = None,
    seed: int | None = None,
) -> RoutingEdges:
    """Destroy exact ancestry while preserving channel, target, role, and coarse lag.

    The null keeps every edge weight and every ``(layer, head, query)`` row. Prompt
    sources are resampled inside the same prompt-position bin; RR sources are
    resampled inside the same log2 lag bin. Diagonal attention is unchanged.
    """

    config = PhenomenologyConfig() if config is None else config
    if edges.weight.numel() == 0:
        return edges

    rng = np.random.default_rng(config.random_seed if seed is None else seed)
    layer = edges.layer.detach().cpu().numpy()
    head = edges.head.detach().cpu().numpy()
    query = edges.query.detach().cpu().numpy()
    source = edges.source.detach().cpu().numpy()
    rewired = source.copy()

    groups: dict[tuple[int, int, int, int, int], list[int]] = {}
    for index, (current_layer, current_head, current_query, current_source) in enumerate(
        zip(layer, head, query, source)
    ):
        if current_source < edges.response_idx:
            role = 0
            bin_index = min(
                (int(current_source) * config.prompt_bins)
                // max(edges.response_idx, 1),
                config.prompt_bins - 1,
            )
        else:
            role = 1
            lag = int(current_query) - (int(current_source) - edges.response_idx)
            bin_index = min(int(np.floor(np.log2(lag))), config.rr_lag_bins - 1)
        key = (
            int(current_layer),
            int(current_head),
            int(current_query),
            role,
            bin_index,
        )
        groups.setdefault(key, []).append(index)

    for (_, _, current_query, role, bin_index), indices in groups.items():
        if role == 0:
            candidates = _prompt_candidates(
                edges.response_idx, bin_index, config.prompt_bins
            )
        else:
            candidates = _response_candidates(
                current_query, bin_index, config.rr_lag_bins
            )
            candidates = candidates + edges.response_idx
        if len(candidates) <= 1 or len(candidates) < len(indices):
            continue
        chosen = rng.choice(candidates, size=len(indices), replace=False)
        original = source[np.asarray(indices)]
        if np.array_equal(chosen, original):
            chosen = np.roll(chosen, 1)
        rewired[np.asarray(indices)] = chosen

    return replace(
        edges,
        source=torch.as_tensor(rewired, dtype=edges.source.dtype, device=edges.device),
    )
