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
    sample: SparseAttentionSample, spec: AnchorSpec
) -> tuple[np.ndarray, np.ndarray]:
    spec.validate()
    shape = (sample.layers, sample.heads, sample.response_tokens, spec.anchors)
    route = np.zeros(shape, dtype=np.float64)
    overflow = np.zeros(shape[:-1], dtype=np.float64)
    response_tokens = sample.response_tokens
    rows = sample.layers * sample.heads * response_tokens

    for row in range(rows):
        layer = row // (sample.heads * response_tokens)
        within_layer = row % (sample.heads * response_tokens)
        head = within_layer // response_tokens
        query = within_layer % response_tokens
        target = sample.response_idx + query
        start, end = int(sample.row_ptr[row]), int(sample.row_ptr[row + 1])
        for source, weight in zip(sample.columns[start:end], sample.values[start:end]):
            if source < sample.response_idx:
                anchor = min(
                    int(source) * spec.prompt_bins // sample.response_idx,
                    spec.prompt_bins - 1,
                )
            else:
                lag = target - int(source)
                history_bin = int(np.searchsorted(spec.history_lag_edges, lag, side="left"))
                anchor = spec.prompt_bins + history_bin
            route[layer, head, query, anchor] += float(weight)

        diagonal = float(sample.diagonal[layer, head, target])
        route[layer, head, query, spec.self_index] = diagonal
        known = float(route[layer, head, query].sum())
        overflow[layer, head, query] = max(known - 1.0, 0.0)
        route[layer, head, query, spec.unresolved_index] = max(1.0 - known, 0.0)

    if np.any(overflow > 0.02):
        maximum = float(overflow.max())
        raise ValueError(f"attention row mass exceeds one by {maximum:.6f}")
    return route, overflow


def _countsketch(values: np.ndarray, output_dim: int, seed: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("CountSketch input must be [rows, features]")
    if output_dim < 1:
        raise ValueError("embedding_dim must be positive")
    rng = np.random.default_rng(seed)
    buckets = rng.integers(0, output_dim, size=values.shape[1])
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=values.shape[1])
    result = np.zeros((values.shape[0], output_dim), dtype=np.float64)
    for bucket in range(output_dim):
        selected = buckets == bucket
        if np.any(selected):
            result[:, bucket] = values[:, selected] @ signs[selected]
    # CountSketch preserves the squared norm in expectation without an
    # additional 1/sqrt(output_dim) scaling.
    return result.astype(np.float32)


def encode_route_dynamics(
    sample: SparseAttentionSample,
    *,
    spec: AnchorSpec | None = None,
    embedding_dim: int = 256,
    seed: int = 20260814,
) -> RouteDynamics:
    spec = spec or AnchorSpec()
    route_mass, overflow = _build_route_mass(sample, spec)
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

    time_delta = np.zeros_like(probability)
    time_acceleration = np.zeros_like(probability)
    depth_delta = np.zeros_like(probability)
    if response > 1:
        time_delta[:, :, 1:] = probability[:, :, 1:] - probability[:, :, :-1]
    if response > 2:
        time_acceleration[:, :, 2:] = (
            probability[:, :, 2:]
            - 2 * probability[:, :, 1:-1]
            + probability[:, :, :-2]
        )
    if layers > 1:
        depth_delta[1:] = probability[1:] - probability[:-1]

    state = np.concatenate(
        [
            probability.transpose(2, 0, 1, 3).reshape(response, -1),
            time_delta.transpose(2, 0, 1, 3).reshape(response, -1),
            time_acceleration.transpose(2, 0, 1, 3).reshape(response, -1),
            depth_delta.transpose(2, 0, 1, 3).reshape(response, -1),
        ],
        axis=1,
    )
    embedding = _countsketch(state, embedding_dim, seed)

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
