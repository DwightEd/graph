"""Causal graph construction, structural motifs, and masked recoverability.

The audit deliberately avoids a learned detector. Each response is treated as a
causal multiplex graph whose parallel edges retain layer, head, and attention
weight. Statistics are computed from prefix topology only, and recovery masks
current endpoints/channels before ranking them with historical graph structure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from experiments.source_reuse_contrast.data import SourceReuseGraph
from .config import GraphAuditConfig


STRUCTURAL_METRICS = (
    "unique_sources",
    "active_incidences",
    "retained_mass",
    "diagonal_mass",
    "prompt_mass",
    "response_mass",
    "prompt_source_fraction",
    "reuse_degree_mean",
    "reuse_degree_max",
    "reuse_span_mean",
    "novel_source_mass_fraction",
    "coalition_mass_coverage",
    "coalition_co_use_strength",
    "coalition_novel_pair_fraction",
    "coalition_component_count",
    "coalition_giant_fraction",
    "shared_consumer_jaccard_mean",
    "shared_consumer_jaccard_max",
    "prompt_reachability",
    "prompt_profile_entropy",
    "prompt_profile_effective_bins",
    "widest_prompt_path",
    "expected_response_hops",
    "two_hop_prompt_relay_mass",
    "two_hop_response_echo_mass",
    "response_unresolved_or_unsupported_mass",
    "prompt_path_redundancy",
    "layer_js_mean",
    "layer_js_max",
    "layer_support_jaccard_mean",
    "layer_support_jaccard_min",
    "head_source_components_mean",
    "head_source_giant_fraction_mean",
    "head_source_four_cycle_density_mean",
    "head_shared_source_density_mean",
    "layer_prompt_shift",
    "channel_history_coverage",
    "channel_reuse_similarity",
    "channel_novelty",
)

RECOVERY_METRICS = (
    "endpoint_mrr",
    "endpoint_hits1",
    "endpoint_hits5",
    "endpoint_mean_percentile",
    "endpoint_recovery_error",
    "endpoint_masked_sources",
    "endpoint_candidate_mean",
    "channel_mrr",
    "channel_hits1",
    "channel_hits5",
    "channel_mean_percentile",
    "channel_recovery_error",
    "channel_masked_edges",
    "channel_weight_mae",
)


@dataclass(frozen=True)
class TokenPairs:
    sources: np.ndarray
    mass: np.ndarray
    channels: np.ndarray


@dataclass(frozen=True)
class GraphAuditResult:
    structural: np.ndarray
    recovery: np.ndarray
    valid_recovery: np.ndarray


class PrefixState:
    """Graph state available strictly before the current response token."""

    def __init__(self, graph: SourceReuseGraph, prompt_bins: int):
        nodes = graph.num_tokens
        channels = graph.num_layers * graph.num_heads
        self.response_idx = graph.response_idx
        self.prompt_bins = prompt_bins
        self.consumer_sets = [set() for _ in range(nodes)]
        self.consumer_count = np.zeros(nodes, dtype=np.int32)
        self.first_consumer = np.full(nodes, -1, dtype=np.int32)
        self.last_consumer = np.full(nodes, -1, dtype=np.int32)
        self.cumulative_mass = np.zeros(nodes, dtype=np.float64)
        self.channel_mass = np.zeros((nodes, channels), dtype=np.float64)
        self.channel_count = np.zeros((nodes, channels), dtype=np.float64)
        self.global_channel_mass = np.zeros(channels, dtype=np.float64)
        self.global_channel_count = np.zeros(channels, dtype=np.float64)
        self.co_use: dict[tuple[int, int], float] = {}

        tokens = graph.num_response_tokens
        self.prompt_profiles = np.zeros((tokens, prompt_bins), dtype=np.float64)
        self.prompt_strength = np.zeros(tokens, dtype=np.float64)
        self.direct_prompt_mass = np.zeros(tokens, dtype=np.float64)
        self.response_mass = np.zeros(tokens, dtype=np.float64)
        self.expected_hops = np.zeros(tokens, dtype=np.float64)
        self.widest_prompt_path = np.zeros(tokens, dtype=np.float64)

    def source_profile(self, source: int) -> np.ndarray:
        if source < self.response_idx:
            profile = np.zeros(self.prompt_bins, dtype=np.float64)
            bin_index = min(
                source * self.prompt_bins // max(self.response_idx, 1),
                self.prompt_bins - 1,
            )
            profile[bin_index] = 1.0
            return profile
        return self.prompt_profiles[source - self.response_idx]

    def update(
        self,
        token: int,
        pairs: TokenPairs,
        *,
        prompt_profile: np.ndarray,
        prompt_strength: float,
        direct_prompt_mass: float,
        response_mass: float,
        expected_hops: float,
        widest_prompt_path: float,
        coalition_top_sources: int,
    ) -> None:
        order = np.argsort(-pairs.mass)[:coalition_top_sources]
        selected = pairs.sources[order]
        selected_mass = pairs.mass[order]
        for left in range(len(selected)):
            for right in range(left + 1, len(selected)):
                a, b = sorted((int(selected[left]), int(selected[right])))
                increment = math.sqrt(float(selected_mass[left] * selected_mass[right]))
                self.co_use[(a, b)] = self.co_use.get((a, b), 0.0) + increment

        flat_channels = pairs.channels.reshape(len(pairs.sources), -1)
        for row, source in enumerate(pairs.sources.tolist()):
            self.consumer_sets[source].add(token)
            self.consumer_count[source] += 1
            if self.first_consumer[source] < 0:
                self.first_consumer[source] = token
            self.last_consumer[source] = token
            self.cumulative_mass[source] += float(pairs.mass[row])
            self.channel_mass[source] += flat_channels[row]
            self.channel_count[source] += (flat_channels[row] > 0).astype(np.float64)
        self.global_channel_mass += flat_channels.sum(axis=0)
        self.global_channel_count += (flat_channels > 0).sum(axis=0)

        self.prompt_profiles[token] = prompt_profile
        self.prompt_strength[token] = prompt_strength
        self.direct_prompt_mass[token] = direct_prompt_mass
        self.response_mass[token] = response_mass
        self.expected_hops[token] = expected_hops
        self.widest_prompt_path[token] = widest_prompt_path


def _token_pairs(graph: SourceReuseGraph, token: int) -> TokenPairs:
    current = graph.token_slice(token)
    source = graph.source[current].detach().cpu().numpy().astype(np.int64)
    if source.size == 0:
        return TokenPairs(
            sources=np.empty(0, dtype=np.int64),
            mass=np.empty(0, dtype=np.float64),
            channels=np.empty((0, graph.num_layers, graph.num_heads), dtype=np.float64),
        )
    layer = graph.layer[current].detach().cpu().numpy().astype(np.int64)
    head = graph.head[current].detach().cpu().numpy().astype(np.int64)
    weight = graph.weight[current].detach().cpu().numpy().astype(np.float64)
    sources, inverse = np.unique(source, return_inverse=True)
    channels = np.zeros(
        (len(sources), graph.num_layers, graph.num_heads), dtype=np.float64
    )
    np.add.at(channels, (inverse, layer, head), weight)
    mass = channels.sum(axis=(1, 2)) / float(graph.num_layers * graph.num_heads)
    return TokenPairs(sources=sources, mass=mass, channels=channels)


def _safe_weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    total = float(weight.sum())
    if value.size == 0 or total <= 0:
        return 0.0
    return float(np.dot(value, weight) / total)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1e-12 else float(np.dot(a, b) / denominator)


def _js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    a_sum, b_sum = float(a.sum()), float(b.sum())
    if a_sum <= 0 or b_sum <= 0:
        return 0.0
    p, q = a / a_sum, b / b_sum
    midpoint = 0.5 * (p + q)

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        selected = x > 0
        return float(np.sum(x[selected] * np.log(x[selected] / y[selected])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def _jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    left, right = set(a), set(b)
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def _percentile_score(value: np.ndarray) -> np.ndarray:
    if value.size <= 1:
        return np.full(value.shape, 0.5, dtype=np.float64)
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(len(value), dtype=np.float64)
    ranks[order] = np.arange(len(value), dtype=np.float64)
    return ranks / float(len(value) - 1)


def _rank_of_true(score: np.ndarray, true_index: int) -> tuple[float, float, float, float]:
    true_score = float(score[true_index])
    rank = 1 + int(np.sum(score > true_score))
    count = len(score)
    percentile = 1.0 - (rank - 1) / float(max(count - 1, 1))
    return 1.0 / rank, float(rank == 1), float(rank <= 5), percentile


def _component_statistics(
    nodes: np.ndarray,
    co_use: dict[tuple[int, int], float],
) -> tuple[float, float]:
    if len(nodes) == 0:
        return 0.0, 0.0
    index = {int(node): position for position, node in enumerate(nodes.tolist())}
    parent = list(range(len(nodes)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            a, b = sorted((int(nodes[left]), int(nodes[right])))
            if co_use.get((a, b), 0.0) > 0:
                union(index[a], index[b])
    counts: dict[int, int] = {}
    for item in range(len(nodes)):
        root = find(item)
        counts[root] = counts.get(root, 0) + 1
    return float(len(counts)), max(counts.values()) / float(len(nodes))


def _hypergraph_metrics(
    state: PrefixState,
    pairs: TokenPairs,
    config: GraphAuditConfig,
) -> dict[str, float]:
    names = (
        "reuse_degree_mean",
        "reuse_degree_max",
        "reuse_span_mean",
        "novel_source_mass_fraction",
        "coalition_mass_coverage",
        "coalition_co_use_strength",
        "coalition_novel_pair_fraction",
        "coalition_component_count",
        "coalition_giant_fraction",
        "shared_consumer_jaccard_mean",
        "shared_consumer_jaccard_max",
    )
    if len(pairs.sources) == 0:
        return {name: 0.0 for name in names}
    count = state.consumer_count[pairs.sources].astype(np.float64)
    span = np.where(
        count > 1,
        state.last_consumer[pairs.sources] - state.first_consumer[pairs.sources] + 1,
        0,
    ).astype(np.float64)
    total_mass = float(pairs.mass.sum())
    order = np.argsort(-pairs.mass)[: config.coalition_top_sources]
    sources = pairs.sources[order]
    mass = pairs.mass[order]
    coverage = float(mass.sum() / total_mass) if total_mass > 0 else 0.0

    co_use_values: list[float] = []
    novel_pairs = 0
    jaccard_values: list[float] = []
    for left in range(len(sources)):
        for right in range(left + 1, len(sources)):
            a, b = sorted((int(sources[left]), int(sources[right])))
            value = state.co_use.get((a, b), 0.0)
            co_use_values.append(math.log1p(value))
            novel_pairs += int(value == 0.0)
            jaccard_values.append(_jaccard(state.consumer_sets[a], state.consumer_sets[b]))
    pair_count = len(co_use_values)
    components, giant = _component_statistics(sources, state.co_use)
    return {
        "reuse_degree_mean": _safe_weighted_mean(count, pairs.mass),
        "reuse_degree_max": float(count.max(initial=0.0)),
        "reuse_span_mean": _safe_weighted_mean(span, pairs.mass),
        "novel_source_mass_fraction": (
            float(pairs.mass[count == 0].sum() / total_mass) if total_mass > 0 else 0.0
        ),
        "coalition_mass_coverage": coverage,
        "coalition_co_use_strength": float(np.mean(co_use_values)) if pair_count else 0.0,
        "coalition_novel_pair_fraction": novel_pairs / float(pair_count) if pair_count else 0.0,
        "coalition_component_count": components,
        "coalition_giant_fraction": giant,
        "shared_consumer_jaccard_mean": float(np.mean(jaccard_values)) if jaccard_values else 0.0,
        "shared_consumer_jaccard_max": float(np.max(jaccard_values)) if jaccard_values else 0.0,
    }


def _prompt_path_metrics(
    state: PrefixState,
    pairs: TokenPairs,
) -> tuple[dict[str, float], np.ndarray, float, float, float, float, float]:
    profile = np.zeros(state.prompt_bins, dtype=np.float64)
    direct_prompt = 0.0
    response_mass = 0.0
    hop_numerator = 0.0
    widest = 0.0
    two_hop_prompt = 0.0
    response_echo = 0.0
    grounded_response = 0.0

    for source, mass in zip(pairs.sources.tolist(), pairs.mass.tolist()):
        if source < state.response_idx:
            direct_prompt += mass
            profile += mass * state.source_profile(source)
            widest = max(widest, mass)
            continue
        response_mass += mass
        local = source - state.response_idx
        source_strength = state.prompt_strength[local]
        profile += mass * state.prompt_profiles[local]
        grounded_response += mass * source_strength
        hop_numerator += mass * source_strength * (1.0 + state.expected_hops[local])
        widest = max(widest, min(mass, state.widest_prompt_path[local]))
        two_hop_prompt += mass * state.direct_prompt_mass[local]
        response_echo += mass * state.response_mass[local]

    strength = float(profile.sum())
    expected_hops = hop_numerator / strength if strength > 1e-12 else 0.0
    normalized = profile / strength if strength > 1e-12 else profile
    selected = normalized > 0
    entropy = (
        float(-np.sum(normalized[selected] * np.log(normalized[selected])))
        if selected.any()
        else 0.0
    )
    normalized_entropy = entropy / math.log(state.prompt_bins) if state.prompt_bins > 1 else 0.0
    effective_bins = math.exp(entropy) if strength > 1e-12 else 0.0
    unsupported_or_unknown = max(response_mass - grounded_response, 0.0)
    redundancy = strength / max(widest, 1e-12) if strength > 0 else 0.0
    metrics = {
        "prompt_reachability": strength,
        "prompt_profile_entropy": normalized_entropy,
        "prompt_profile_effective_bins": effective_bins,
        "widest_prompt_path": widest,
        "expected_response_hops": expected_hops,
        "two_hop_prompt_relay_mass": two_hop_prompt,
        "two_hop_response_echo_mass": response_echo,
        "response_unresolved_or_unsupported_mass": unsupported_or_unknown,
        "prompt_path_redundancy": redundancy,
    }
    return metrics, profile, strength, direct_prompt, response_mass, expected_hops, widest


def _bipartite_components(binary: np.ndarray) -> tuple[float, float]:
    active_heads = np.flatnonzero(binary.any(axis=1))
    active_sources = np.flatnonzero(binary.any(axis=0))
    total = len(active_heads) + len(active_sources)
    if total == 0:
        return 0.0, 0.0
    head_map = {int(head): index for index, head in enumerate(active_heads.tolist())}
    source_map = {
        int(source): len(active_heads) + index
        for index, source in enumerate(active_sources.tolist())
    }
    parent = list(range(total))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: int, b: int) -> None:
        left, right = find(a), find(b)
        if left != right:
            parent[right] = left

    rows, cols = np.nonzero(binary)
    for head, source in zip(rows.tolist(), cols.tolist()):
        union(head_map[head], source_map[source])
    counts: dict[int, int] = {}
    for item in range(total):
        root = find(item)
        counts[root] = counts.get(root, 0) + 1
    return float(len(counts)), max(counts.values()) / float(total)


def _layer_head_metrics(
    graph: SourceReuseGraph,
    pairs: TokenPairs,
) -> dict[str, float]:
    names = (
        "layer_js_mean",
        "layer_js_max",
        "layer_support_jaccard_mean",
        "layer_support_jaccard_min",
        "head_source_components_mean",
        "head_source_giant_fraction_mean",
        "head_source_four_cycle_density_mean",
        "head_shared_source_density_mean",
        "layer_prompt_shift",
    )
    layers, heads = graph.num_layers, graph.num_heads
    if len(pairs.sources) == 0:
        return {name: 0.0 for name in names}
    layer_source = pairs.channels.sum(axis=2).T
    js_values: list[float] = []
    jaccard_values: list[float] = []
    for layer in range(layers - 1):
        js_values.append(_js_divergence(layer_source[layer], layer_source[layer + 1]))
        left = np.flatnonzero(layer_source[layer] > 0)
        right = np.flatnonzero(layer_source[layer + 1] > 0)
        jaccard_values.append(_jaccard(left, right))

    component_values: list[float] = []
    giant_values: list[float] = []
    cycle_values: list[float] = []
    shared_values: list[float] = []
    for layer in range(layers):
        binary = pairs.channels[:, layer, :].T > 0
        components, giant = _bipartite_components(binary)
        component_values.append(components)
        giant_values.append(giant)
        active_heads = int(binary.any(axis=1).sum())
        active_sources = int(binary.any(axis=0).sum())
        shared = binary.astype(np.int32) @ binary.astype(np.int32).T
        upper = shared[np.triu_indices(heads, k=1)]
        shared_values.append(
            float(upper.mean() / max(active_sources, 1)) if upper.size else 0.0
        )
        four_cycles = float(np.sum(upper * (upper - 1) / 2.0))
        possible = (
            active_heads * (active_heads - 1) / 2.0
            * active_sources * (active_sources - 1) / 2.0
        )
        cycle_values.append(four_cycles / possible if possible > 0 else 0.0)

    prompt = pairs.sources < graph.response_idx
    prompt_mass = layer_source[:, prompt].sum(axis=1) if prompt.any() else np.zeros(layers)
    total_mass = layer_source.sum(axis=1)
    prompt_fraction = np.divide(
        prompt_mass,
        total_mass,
        out=np.zeros_like(prompt_mass),
        where=total_mass > 0,
    )
    width = max(layers // 4, 1)
    shift = float(prompt_fraction[-width:].mean() - prompt_fraction[:width].mean())
    return {
        "layer_js_mean": float(np.mean(js_values)) if js_values else 0.0,
        "layer_js_max": float(np.max(js_values)) if js_values else 0.0,
        "layer_support_jaccard_mean": float(np.mean(jaccard_values)) if jaccard_values else 0.0,
        "layer_support_jaccard_min": float(np.min(jaccard_values)) if jaccard_values else 0.0,
        "head_source_components_mean": float(np.mean(component_values)),
        "head_source_giant_fraction_mean": float(np.mean(giant_values)),
        "head_source_four_cycle_density_mean": float(np.mean(cycle_values)),
        "head_shared_source_density_mean": float(np.mean(shared_values)),
        "layer_prompt_shift": shift,
    }


def _channel_history_metrics(state: PrefixState, pairs: TokenPairs) -> dict[str, float]:
    if len(pairs.sources) == 0:
        return {
            "channel_history_coverage": 0.0,
            "channel_reuse_similarity": 0.0,
            "channel_novelty": 0.0,
        }
    current = pairs.channels.reshape(len(pairs.sources), -1)
    similarity = np.zeros(len(pairs.sources), dtype=np.float64)
    has_history = np.zeros(len(pairs.sources), dtype=bool)
    for row, source in enumerate(pairs.sources.tolist()):
        history = state.channel_mass[source]
        has_history[row] = np.linalg.norm(history) > 1e-12
        similarity[row] = _cosine(current[row], history)
    total = float(pairs.mass.sum())
    coverage = float(pairs.mass[has_history].sum() / total) if total > 0 else 0.0
    mean_similarity = _safe_weighted_mean(similarity, pairs.mass)
    return {
        "channel_history_coverage": coverage,
        "channel_reuse_similarity": mean_similarity,
        "channel_novelty": 1.0 - mean_similarity if coverage > 0 else 0.0,
    }


def _candidate_context_profile(
    state: PrefixState,
    observed_sources: np.ndarray,
    observed_mass: np.ndarray,
) -> np.ndarray:
    profile = np.zeros(state.prompt_bins, dtype=np.float64)
    for source, mass in zip(observed_sources.tolist(), observed_mass.tolist()):
        profile += mass * state.source_profile(source)
    total = float(profile.sum())
    return profile / total if total > 0 else profile


def _endpoint_recovery(
    graph: SourceReuseGraph,
    state: PrefixState,
    pairs: TokenPairs,
    token: int,
    config: GraphAuditConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    source_count = len(pairs.sources)
    empty = {
        "endpoint_mrr": np.nan,
        "endpoint_hits1": np.nan,
        "endpoint_hits5": np.nan,
        "endpoint_mean_percentile": np.nan,
        "endpoint_recovery_error": np.nan,
        "endpoint_masked_sources": 0.0,
        "endpoint_candidate_mean": np.nan,
    }
    if source_count < config.minimum_sources_for_recovery:
        return empty
    mask_count = min(
        max(1, int(round(source_count * config.source_mask_fraction))),
        source_count - 1,
    )
    masked_rows = np.sort(rng.choice(source_count, size=mask_count, replace=False))
    observed_mask = np.ones(source_count, dtype=bool)
    observed_mask[masked_rows] = False
    observed_sources = pairs.sources[observed_mask]
    observed_mass = pairs.mass[observed_mask]
    current_set = set(pairs.sources.tolist())
    context_profile = _candidate_context_profile(state, observed_sources, observed_mass)
    channel_context = pairs.channels[observed_mask].sum(axis=0).reshape(-1)

    reciprocal: list[float] = []
    hits1: list[float] = []
    hits5: list[float] = []
    percentiles: list[float] = []
    candidate_counts: list[int] = []
    for masked_row in masked_rows.tolist():
        true_source = int(pairs.sources[masked_row])
        domain = (
            list(range(graph.response_idx))
            if true_source < graph.response_idx
            else list(range(graph.response_idx, graph.response_idx + token))
        )
        candidates = [
            source for source in domain if source == true_source or source not in current_set
        ]
        if len(candidates) < 2:
            continue
        popularity = np.log1p(state.consumer_count[candidates].astype(np.float64))
        co_use = np.asarray(
            [
                sum(
                    math.log1p(
                        state.co_use.get(tuple(sorted((candidate, int(other)))), 0.0)
                    )
                    for other in observed_sources.tolist()
                    if candidate != int(other)
                )
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        channel_similarity = np.asarray(
            [_cosine(state.channel_mass[candidate], channel_context) for candidate in candidates],
            dtype=np.float64,
        )
        profile_similarity = np.asarray(
            [_cosine(state.source_profile(candidate), context_profile) for candidate in candidates],
            dtype=np.float64,
        )
        combined = np.mean(
            np.stack(
                [
                    _percentile_score(popularity),
                    _percentile_score(co_use),
                    _percentile_score(channel_similarity),
                    _percentile_score(profile_similarity),
                ]
            ),
            axis=0,
        )
        true_index = candidates.index(true_source)
        rr, h1, h5, percentile = _rank_of_true(combined, true_index)
        reciprocal.append(rr)
        hits1.append(h1)
        hits5.append(h5)
        percentiles.append(percentile)
        candidate_counts.append(len(candidates))

    if not reciprocal:
        return empty
    mrr = float(np.mean(reciprocal))
    return {
        "endpoint_mrr": mrr,
        "endpoint_hits1": float(np.mean(hits1)),
        "endpoint_hits5": float(np.mean(hits5)),
        "endpoint_mean_percentile": float(np.mean(percentiles)),
        "endpoint_recovery_error": 1.0 - mrr,
        "endpoint_masked_sources": float(len(reciprocal)),
        "endpoint_candidate_mean": float(np.mean(candidate_counts)),
    }


def _channel_recovery(
    graph: SourceReuseGraph,
    state: PrefixState,
    pairs: TokenPairs,
    config: GraphAuditConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    empty = {
        "channel_mrr": np.nan,
        "channel_hits1": np.nan,
        "channel_hits5": np.nan,
        "channel_mean_percentile": np.nan,
        "channel_recovery_error": np.nan,
        "channel_masked_edges": 0.0,
        "channel_weight_mae": np.nan,
    }
    channel_count = graph.num_layers * graph.num_heads
    if len(pairs.sources) == 0:
        return empty
    flat = pairs.channels.reshape(len(pairs.sources), channel_count)
    reciprocal: list[float] = []
    hits1: list[float] = []
    hits5: list[float] = []
    percentiles: list[float] = []
    absolute_error: list[float] = []

    for row, source in enumerate(pairs.sources.tolist()):
        active = np.flatnonzero(flat[row] > 0)
        if len(active) < config.minimum_channels_for_recovery:
            continue
        mask_count = min(
            max(1, int(round(len(active) * config.channel_mask_fraction))),
            len(active) - 1,
        )
        masked = np.sort(rng.choice(active, size=mask_count, replace=False))
        observed = flat[row].copy()
        observed[masked] = 0.0
        candidates = np.flatnonzero(observed <= 0)
        if len(candidates) < 2:
            continue

        source_history = state.channel_mass[source]
        token_context = flat.sum(axis=0) - flat[row] + observed
        global_history = state.global_channel_mass
        neighbor = np.zeros(channel_count, dtype=np.float64)
        for channel in range(channel_count):
            layer, head = divmod(channel, graph.num_heads)
            values = []
            if layer > 0:
                values.append(observed[(layer - 1) * graph.num_heads + head] > 0)
            if layer + 1 < graph.num_layers:
                values.append(observed[(layer + 1) * graph.num_heads + head] > 0)
            neighbor[channel] = float(np.mean(values)) if values else 0.0
        score_all = np.mean(
            np.stack(
                [
                    _percentile_score(source_history),
                    _percentile_score(token_context),
                    _percentile_score(global_history),
                    _percentile_score(neighbor),
                ]
            ),
            axis=0,
        )
        score = score_all[candidates]
        source_mean = np.divide(
            state.channel_mass[source],
            state.channel_count[source],
            out=np.zeros(channel_count, dtype=np.float64),
            where=state.channel_count[source] > 0,
        )
        global_mean = np.divide(
            state.global_channel_mass,
            state.global_channel_count,
            out=np.zeros(channel_count, dtype=np.float64),
            where=state.global_channel_count > 0,
        )
        predicted_weight = 0.5 * (source_mean + global_mean)
        for true_channel in masked.tolist():
            true_index = int(np.flatnonzero(candidates == true_channel)[0])
            rr, h1, h5, percentile = _rank_of_true(score, true_index)
            reciprocal.append(rr)
            hits1.append(h1)
            hits5.append(h5)
            percentiles.append(percentile)
            absolute_error.append(
                abs(predicted_weight[true_channel] - flat[row, true_channel])
            )

    if not reciprocal:
        return empty
    mrr = float(np.mean(reciprocal))
    return {
        "channel_mrr": mrr,
        "channel_hits1": float(np.mean(hits1)),
        "channel_hits5": float(np.mean(hits5)),
        "channel_mean_percentile": float(np.mean(percentiles)),
        "channel_recovery_error": 1.0 - mrr,
        "channel_masked_edges": float(len(reciprocal)),
        "channel_weight_mae": float(np.mean(absolute_error)),
    }


def audit_graph(
    graph: SourceReuseGraph,
    config: GraphAuditConfig | None = None,
) -> GraphAuditResult:
    """Compute prefix-causal structural and recovery profiles for one sample."""

    config = GraphAuditConfig() if config is None else config
    config.validate()
    state = PrefixState(graph, config.prompt_bins)
    structural_rows: list[list[float]] = []
    recovery_rows: list[list[float]] = []
    valid_recovery: list[bool] = []

    for token in range(graph.num_response_tokens):
        pairs = _token_pairs(graph, token)
        hypergraph = _hypergraph_metrics(state, pairs, config)
        (
            path,
            prompt_profile,
            prompt_strength,
            direct_prompt_mass,
            response_mass,
            expected_hops,
            widest_prompt_path,
        ) = _prompt_path_metrics(state, pairs)
        layer = _layer_head_metrics(graph, pairs)
        channel = _channel_history_metrics(state, pairs)

        total_mass = float(pairs.mass.sum())
        prompt_mask = pairs.sources < graph.response_idx
        prompt_mass = float(pairs.mass[prompt_mask].sum()) if prompt_mask.any() else 0.0
        base = {
            "unique_sources": float(len(pairs.sources)),
            "active_incidences": float(np.count_nonzero(pairs.channels)),
            "retained_mass": total_mass,
            "diagonal_mass": float(graph.diagonal[token].detach().cpu().mean()),
            "prompt_mass": prompt_mass,
            "response_mass": total_mass - prompt_mass,
            "prompt_source_fraction": float(prompt_mask.mean()) if len(pairs.sources) else 0.0,
        }
        metrics = {**base, **hypergraph, **path, **layer, **channel}
        structural_rows.append([float(metrics[name]) for name in STRUCTURAL_METRICS])

        seed = config.random_seed + token * 1000003
        endpoint = _endpoint_recovery(
            graph,
            state,
            pairs,
            token,
            config,
            np.random.default_rng(seed),
        )
        channel_recovery = _channel_recovery(
            graph,
            state,
            pairs,
            config,
            np.random.default_rng(seed + 17),
        )
        recovery = {**endpoint, **channel_recovery}
        recovery_rows.append([float(recovery[name]) for name in RECOVERY_METRICS])
        valid_recovery.append(
            np.isfinite(recovery["endpoint_mrr"])
            or np.isfinite(recovery["channel_mrr"])
        )

        state.update(
            token,
            pairs,
            prompt_profile=prompt_profile,
            prompt_strength=prompt_strength,
            direct_prompt_mass=direct_prompt_mass,
            response_mass=response_mass,
            expected_hops=expected_hops,
            widest_prompt_path=widest_prompt_path,
            coalition_top_sources=config.coalition_top_sources,
        )

    return GraphAuditResult(
        structural=np.asarray(structural_rows, dtype=np.float32),
        recovery=np.asarray(recovery_rows, dtype=np.float32),
        valid_recovery=np.asarray(valid_recovery, dtype=bool),
    )
