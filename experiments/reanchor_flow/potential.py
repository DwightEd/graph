"""Target-conditioned path potential and flow on a causal DAG."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import EPS


@dataclass(frozen=True)
class PathMass:
    total: float
    through_anchor: float
    closure: float


def target_potential(transition: np.ndarray, sink: int) -> np.ndarray:
    graph = np.asarray(transition, dtype=np.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("transition must be square")
    if not 0 <= sink < len(graph):
        raise ValueError("sink is outside the graph")
    potential = np.zeros(len(graph), dtype=np.float64)
    potential[sink] = 1.0
    for source in range(sink - 1, -1, -1):
        potential[source] = np.dot(
            graph[source, source + 1 : sink + 1],
            potential[source + 1 : sink + 1],
        )
    return potential


def first_hit_path_mass(transition, sink: int, sources, anchors) -> PathMass:
    """Path mass reaching ``sink`` after first passing an anchor set."""

    graph = np.asarray(transition, dtype=np.float64)
    source = np.unique(np.asarray(sources, dtype=np.int64))
    anchor = set(np.asarray(anchors, dtype=np.int64).tolist())
    source = source[(source >= 0) & (source < sink)]
    if not len(source):
        return PathMass(0.0, 0.0, 0.0)
    potential = target_potential(graph, sink)
    hit = np.zeros_like(potential)
    hit[sink] = potential[sink] if sink in anchor else 0.0
    for node in range(sink - 1, -1, -1):
        hit[node] = (
            potential[node]
            if node in anchor
            else np.dot(
                graph[node, node + 1 : sink + 1],
                hit[node + 1 : sink + 1],
            )
        )
    total = float(potential[source].mean())
    through = float(hit[source].mean())
    return PathMass(total, through, through / total if total > EPS else 0.0)


def conditioned_flow(transition, sink: int, sources) -> tuple[np.ndarray, np.ndarray]:
    """Return sink-conditioned edge and node flow from uniform sources."""

    graph = np.asarray(transition, dtype=np.float64)
    potential = target_potential(graph, sink)
    source = np.unique(np.asarray(sources, dtype=np.int64))
    source = source[(source >= 0) & (source < sink) & (potential[source] > EPS)]
    edge_flow = np.zeros_like(graph)
    node_flow = np.zeros(len(graph), dtype=np.float64)
    if not len(source):
        return edge_flow, node_flow
    node_flow[source] += 1.0 / len(source)
    for node in range(sink):
        if node_flow[node] <= 0 or potential[node] <= EPS:
            continue
        target = np.arange(node + 1, sink + 1)
        probability = graph[node, target] * potential[target] / potential[node]
        flow = node_flow[node] * probability
        edge_flow[node, target] = flow
        node_flow[target] += flow
    return edge_flow, node_flow
