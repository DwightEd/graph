"""Train-only affine transport operators for depth, relay and query-set audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import GraphConfig, TransportConfig
from .geometry import clr_to_profile, event_profiles
from .graph import AttentionEventGraph, PROMPT_EVENT, RESPONSE_EVENT
from .ridge import AffineMap, RidgeAccumulator

LOCAL_FEATURE_DIM = 5


def _local_features(graph: AttentionEventGraph, event_index: np.ndarray) -> np.ndarray:
    event_index = np.asarray(event_index, dtype=np.int64)
    role = graph.event_role[event_index].detach().cpu().numpy().astype(np.float64)
    lag = graph.event_lag[event_index].detach().cpu().numpy().astype(np.float64)
    query = graph.event_query[event_index].detach().cpu().numpy().astype(np.float64)
    mass = graph.event_mass[event_index].detach().cpu().numpy().astype(np.float64)
    observed = (
        graph.event_head_observed[event_index]
        .float()
        .mean(dim=-1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    return np.stack(
        (
            role,
            np.log1p(lag) / np.log1p(max(graph.num_tokens, 2)),
            query / max(graph.num_response_tokens - 1, 1),
            np.log1p(mass),
            observed,
        ),
        axis=1,
    )


def query_training_rows(
    graph: AttentionEventGraph,
    clr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return event, layer, local/set features, and CLR targets."""

    local_rows: list[np.ndarray] = []
    full_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    layers: list[int] = []
    events: list[int] = []
    mass = graph.event_mass.detach().cpu().numpy().astype(np.float64)
    role = graph.event_role.detach().cpu().numpy()
    index = graph.query_event_index.detach().cpu().numpy()
    pointer = graph.query_ptr.detach().cpu().numpy()

    for group in range(len(pointer) - 1):
        selected = index[pointer[group] : pointer[group + 1]]
        if len(selected) < 2:
            continue
        for event in selected:
            other = selected[selected != event]
            weight = mass[other]
            weight = weight / max(weight.sum(), 1e-12)
            mean = (clr[other] * weight[:, None]).sum(axis=0)
            variance = (np.square(clr[other] - mean) * weight[:, None]).sum(axis=0)
            prompt_fraction = float(np.mean(role[other] == PROMPT_EVENT))
            response_fraction = float(np.mean(role[other] == RESPONSE_EVENT))
            context = np.concatenate(
                (
                    mean,
                    np.sqrt(np.maximum(variance, 0.0)),
                    np.asarray(
                        [
                            prompt_fraction,
                            response_fraction,
                            np.log1p(len(other)),
                            np.log1p(mass[other].sum()),
                        ],
                        dtype=np.float64,
                    ),
                )
            )
            local = _local_features(graph, np.asarray([event]))[0]
            local_rows.append(local)
            full_rows.append(np.concatenate((local, context)))
            targets.append(clr[event])
            layers.append(int(graph.event_layer[event]))
            events.append(int(event))

    if not targets:
        heads = graph.num_heads
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, LOCAL_FEATURE_DIM), dtype=np.float64),
            np.empty((0, LOCAL_FEATURE_DIM + 2 * heads + 4), dtype=np.float64),
            np.empty((0, heads), dtype=np.float64),
        )
    return (
        np.asarray(events, dtype=np.int64),
        np.asarray(layers, dtype=np.int64),
        np.asarray(local_rows, dtype=np.float64),
        np.asarray(full_rows, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )


@dataclass(frozen=True)
class TransportReference:
    depth: tuple[AffineMap, ...]
    relay: tuple[tuple[AffineMap, AffineMap], ...]
    query_local: tuple[AffineMap, ...]
    query_full: tuple[AffineMap, ...]
    num_layers: int
    num_heads: int
    graph_config: GraphConfig
    transport_config: TransportConfig

    def depth_profile(self, layer: int, clr) -> np.ndarray:
        return clr_to_profile(self.depth[int(layer)].predict(clr))

    def depth_baseline(self, layer: int, rows: int = 1) -> np.ndarray:
        mean = self.depth[int(layer)].target_mean
        return np.repeat(clr_to_profile(mean[None]), rows, axis=0)

    def relay_profile(self, layer: int, role: int, clr) -> np.ndarray:
        return clr_to_profile(self.relay[int(layer)][int(role)].predict(clr))

    def relay_baseline(self, layer: int, role: int, rows: int = 1) -> np.ndarray:
        mean = self.relay[int(layer)][int(role)].target_mean
        return np.repeat(clr_to_profile(mean[None]), rows, axis=0)

    def query_profiles(
        self,
        layer: int,
        local: np.ndarray,
        full: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            clr_to_profile(self.query_local[int(layer)].predict(local)),
            clr_to_profile(self.query_full[int(layer)].predict(full)),
        )


class TransportFitter:
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        *,
        graph_config: GraphConfig,
        transport_config: TransportConfig,
    ) -> None:
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.graph_config = graph_config
        self.transport_config = transport_config
        self.depth = [
            RidgeAccumulator(num_heads, num_heads) for _ in range(max(num_layers - 1, 0))
        ]
        self.relay = [
            (
                RidgeAccumulator(num_heads, num_heads),
                RidgeAccumulator(num_heads, num_heads),
            )
            for _ in range(max(num_layers - 1, 0))
        ]
        full_dim = LOCAL_FEATURE_DIM + 2 * num_heads + 4
        self.query_local = [
            RidgeAccumulator(LOCAL_FEATURE_DIM, num_heads) for _ in range(num_layers)
        ]
        self.query_full = [RidgeAccumulator(full_dim, num_heads) for _ in range(num_layers)]

    @torch.no_grad()
    def update(self, graph: AttentionEventGraph) -> None:
        _, clr_tensor = event_profiles(
            graph.event_head_value,
            graph.event_head_observed,
            attention_floor=graph.attention_floor,
            censored_fill_ratio=self.graph_config.censored_fill_ratio,
        )
        clr = clr_tensor.detach().cpu().numpy().astype(np.float64)

        if graph.depth_edge_index.numel():
            left, right = graph.depth_edge_index.detach().cpu().numpy()
            left_layer = graph.event_layer[left].detach().cpu().numpy()
            for layer in range(self.num_layers - 1):
                selected = left_layer == layer
                if np.any(selected):
                    self.depth[layer].add(clr[left[selected]], clr[right[selected]])

        if graph.relay_edge_index.numel():
            left, right = graph.relay_edge_index.detach().cpu().numpy()
            left_layer = graph.event_layer[left].detach().cpu().numpy()
            left_role = graph.event_role[left].detach().cpu().numpy()
            for layer in range(self.num_layers - 1):
                for role in (PROMPT_EVENT, RESPONSE_EVENT):
                    selected = (left_layer == layer) & (left_role == role)
                    if np.any(selected):
                        self.relay[layer][role].add(
                            clr[left[selected]], clr[right[selected]]
                        )

        _events, layers, local, full, target = query_training_rows(graph, clr)
        for layer in range(self.num_layers):
            selected = layers == layer
            if np.any(selected):
                self.query_local[layer].add(local[selected], target[selected])
                self.query_full[layer].add(full[selected], target[selected])

    def freeze(self) -> TransportReference:
        alpha = self.transport_config.ridge_alpha
        minimum = self.transport_config.minimum_pairs
        return TransportReference(
            depth=tuple(model.freeze(alpha, minimum) for model in self.depth),
            relay=tuple(
                tuple(model.freeze(alpha, minimum) for model in pair)
                for pair in self.relay
            ),
            query_local=tuple(
                model.freeze(alpha, minimum) for model in self.query_local
            ),
            query_full=tuple(model.freeze(alpha, minimum) for model in self.query_full),
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            graph_config=self.graph_config,
            transport_config=self.transport_config,
        )
