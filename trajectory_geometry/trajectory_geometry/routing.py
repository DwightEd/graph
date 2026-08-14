"""Fixed-anchor route construction and label-free dynamics encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import SparseAttentionSample


@dataclass(frozen=True)
class AnchorSpec:
    prompt_bins: int = 8
    history_lag_edges: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    def validate(self) -> None:
        if self.prompt_bins < 1:
            raise ValueError("prompt_bins must be positive")
        if not self.history_lag_edges:
            raise ValueError("history_lag_edges cannot be empty")
        if any(value < 1 for value in self.history_lag_edges):
            raise ValueError("history lag edges must be positive")
        if tuple(sorted(set(self.history_lag_edges))) != self.history_lag_edges:
            raise ValueError("history lag edges must be unique and increasing")

    @property
    def history_bins(self) -> int:
        return len(self.history_lag_edges) + 1

    @property
    def anchors(self) -> int:
        return self.prompt_bins + self.history_bins + 2

    @property
    def self_index(self) -> int:
        return self.prompt_bins + self.history_bins

    @property
    def unresolved_index(self) -> int:
        return self.self_index + 1

    def names(self) -> list[str]:
        prompt = [f"prompt_bin_{index}" for index in range(self.prompt_bins)]
        history = [f"response_lag_le_{edge}" for edge in self.history_lag_edges]
        history.append("response_lag_far")
        return prompt + history + ["self", "unresolved"]


@dataclass(frozen=True)
class RouteDynamics:
    route_mass: np.ndarray
    route_distribution: np.ndarray
    route_embedding: np.ndarray
    temporal_js: np.ndarray
    depth_js: np.ndarray
    head_js: np.ndarray
    route_acceleration: np.ndarray
    prompt_mass: np.ndarray
    history_mass: np.ndarray
    self_mass: np.ndarray
    unresolved_mass: np.ndarray
    mass_overflow: np.ndarray
    anchor_names: tuple[str, ...]


def _normalize(values: np.ndarray) -> np.ndarray:
    total = values.sum(axis=-1, keepdims=True)
    return np.divide(values, total, out=np.zeros_like(values), where=total > 0)


def _entropy(probability: np.ndarray) -> np.ndarray:
    safe = np.where(probability > 0, probability, 1.0)
    return -np.sum(np.where(probability > 0, probability * np.log(safe), 0.0), axis=-1)


def _js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = _normalize(left)
    right = _normalize(right)
    middle = 0.5 * (left + right)

    def kl(value: np.ndarray, base: np.ndarray) -> np.ndarray:
        ratio = np.divide(value, base, out=np.ones_like(value), where=(value > 0) & (base > 0))
        return np.sum(np.where(value > 0, value * np.log(ratio), 0.0), axis=-1)

    return 0.5 * (kl(left, middle) + kl(right, middle))


def _build_route_mass(
    sample: SparseAttentionSample, spec: AnchorSpec, csr_row_block: int
) -> tuple[np.ndarray, np.ndarray]:
    spec.validate()
    shape = (sample.layers, sample.heads, sample.response_tokens, spec.anchors)
    route = np.zeros(shape, dtype=np.float32)
    overflow = np.zeros(shape[:-1], dtype=np.float32)
    for block in sample.iter_sparse_row_blocks(csr_row_block):
        if not block.source.size:
            continue
        anchor = np.empty(block.source.shape, dtype=np.int64)
        prompt = block.source < sample.response_idx
        anchor[prompt] = np.minimum(
            block.source[prompt] * spec.prompt_bins // sample.response_idx,
            spec.prompt_bins - 1,
        )
        history = ~prompt
        lag = block.target[history] - block.source[history]
        anchor[history] = spec.prompt_bins + np.searchsorted(
            np.asarray(spec.history_lag_edges), lag, side="left"
        )
        np.add.at(
            route,
            (block.layer, block.head, block.query, anchor),
            block.weight,
        )

    diagonal = sample.diagonal[:, :, sample.response_idx :]
    route[..., spec.self_index] = diagonal
    known = route.sum(axis=-1)
    overflow = np.maximum(known - 1.0, 0.0)
    route[..., spec.unresolved_index] = np.maximum(1.0 - known, 0.0)

    if np.any(overflow > 0.02):
        maximum = float(overflow.max())
        raise ValueError(f"attention row mass exceeds one by {maximum:.6f}")
    return route, overflow


def _countsketch_add(
    result: np.ndarray,
    values: np.ndarray,
    rng: np.random.Generator,
) -> None:
    if values.ndim != 2:
        raise ValueError("CountSketch input must be [rows, features]")
    output_dim = result.shape[1]
    buckets = rng.integers(0, output_dim, size=values.shape[1])
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=values.shape[1])
    for bucket in range(output_dim):
        selected = buckets == bucket
        if np.any(selected):
            result[:, bucket] += values[:, selected] @ signs[selected]


def _route_embedding(
    probability: np.ndarray, output_dim: int, seed: int
) -> np.ndarray:
    if output_dim < 1:
        raise ValueError("embedding_dim must be positive")
    layers, heads, response, anchors = probability.shape
    result = np.zeros((response, output_dim), dtype=np.float64)
    rng = np.random.default_rng(seed)

    def add(values: np.ndarray) -> None:
        flattened = values.transpose(2, 0, 1, 3).reshape(
            response, layers * heads * anchors
        )
        _countsketch_add(result, flattened, rng)

    add(probability)
    delta = np.zeros_like(probability)
    if response > 1:
        delta[:, :, 1:] = probability[:, :, 1:] - probability[:, :, :-1]
    add(delta)

    acceleration = np.zeros_like(probability)
    if response > 2:
        acceleration[:, :, 2:] = (
            probability[:, :, 2:]
            - 2 * probability[:, :, 1:-1]
            + probability[:, :, :-2]
        )
    add(acceleration)

    depth = np.zeros_like(probability)
    if layers > 1:
        depth[1:] = probability[1:] - probability[:-1]
    add(depth)
    return result.astype(np.float32)


def encode_route_dynamics(
    sample: SparseAttentionSample,
    *,
    spec: AnchorSpec | None = None,
    embedding_dim: int = 256,
    seed: int = 20260814,
    csr_row_block: int = 4096,
) -> RouteDynamics:
    spec = spec or AnchorSpec()
    route_mass, overflow = _build_route_mass(sample, spec, csr_row_block)
    probability = _normalize(route_mass)
    layers, heads, response, anchors = probability.shape

    temporal_js = np.zeros((layers, heads, response), dtype=np.float64)
    if response > 1:
        temporal_js[:, :, 1:] = _js(probability[:, :, 1:], probability[:, :, :-1])

    depth_js = np.zeros((layers, heads, response), dtype=np.float64)
    if layers > 1:
        depth_js[1:] = _js(probability[1:], probability[:-1])

    mean_head = probability.mean(axis=1)
    head_js = _entropy(mean_head) - _entropy(probability).mean(axis=1)

    acceleration = np.zeros((layers, heads, response), dtype=np.float64)
    if response > 2:
        second = probability[:, :, 2:] - 2 * probability[:, :, 1:-1] + probability[:, :, :-2]
        acceleration[:, :, 2:] = np.linalg.norm(second, axis=-1)

    embedding = _route_embedding(probability, embedding_dim, seed)

    prompt_slice = slice(0, spec.prompt_bins)
    history_slice = slice(spec.prompt_bins, spec.self_index)
    return RouteDynamics(
        route_mass=route_mass.astype(np.float32),
        route_distribution=probability.astype(np.float32),
        route_embedding=embedding,
        temporal_js=np.median(temporal_js, axis=(0, 1)).astype(np.float32),
        depth_js=np.median(depth_js, axis=(0, 1)).astype(np.float32),
        head_js=np.median(head_js, axis=0).astype(np.float32),
        route_acceleration=np.median(acceleration, axis=(0, 1)).astype(np.float32),
        prompt_mass=np.median(route_mass[..., prompt_slice].sum(axis=-1), axis=(0, 1)).astype(np.float32),
        history_mass=np.median(route_mass[..., history_slice].sum(axis=-1), axis=(0, 1)).astype(np.float32),
        self_mass=np.median(route_mass[..., spec.self_index], axis=(0, 1)).astype(np.float32),
        unresolved_mass=np.median(route_mass[..., spec.unresolved_index], axis=(0, 1)).astype(np.float32),
        mass_overflow=np.max(overflow, axis=(0, 1)).astype(np.float32),
        anchor_names=tuple(spec.names()),
    )
