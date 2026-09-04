"""Generation-token DAG construction and structure-preserving controls."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class TokenDAG:
    """Causal graph with edges ``source token -> predicted response token``."""

    capacity: np.ndarray
    transition: np.ndarray
    response_start: int

    @property
    def token_count(self) -> int:
        return int(self.transition.shape[0])


def build_token_dag(route_rows, response_start: int) -> TokenDAG:
    """Lift predictor rows into an acyclic generation-dependency graph.

    Route row ``t`` is observed at query ``response_start - 1 + t`` and predicts
    token ``response_start + t``. Mapping it to the predicted token preserves
    the immediate previous response token as a causal source and avoids a fake
    self-loop at the query coordinate.
    """

    rows = np.asarray(route_rows, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("route_rows must have shape [response, source]")
    events, source_count = rows.shape
    token_count = source_count + 1
    if events > token_count - response_start:
        raise ValueError("route rows exceed the response token range")

    capacity = np.zeros((token_count, token_count), dtype=np.float64)
    transition = np.zeros_like(capacity)
    for event in range(events):
        target = response_start + event
        weight = np.clip(rows[event, :target], 0.0, None)
        capacity[:target, target] = weight
        total = float(weight.sum())
        if total > EPS:
            transition[:target, target] = weight / total
    return TokenDAG(capacity, transition, response_start)


def source_roles(token_count: int, response_start: int, evidence_mask) -> np.ndarray:
    """Return 0=evidence, 1=other prompt, 2=response."""

    evidence = np.zeros(token_count, dtype=bool)
    supplied = np.asarray(evidence_mask, dtype=bool).reshape(-1)
    evidence[: min(token_count, len(supplied))] = supplied[:token_count]
    role = np.full(token_count, 2, dtype=np.int8)
    role[:response_start] = 1
    role[evidence] = 0
    return role


def lag_bucket(lag: int) -> int:
    return int(math.floor(math.log2(max(lag, 1))))


def rewire_by_role_lag(
    transition: np.ndarray,
    response_start: int,
    evidence_mask,
    *,
    seed: int,
) -> np.ndarray:
    """Permute edge weights within target, source role, and lag bucket."""

    graph = np.asarray(transition, dtype=np.float64)
    rewired = graph.copy()
    role = source_roles(len(graph), response_start, evidence_mask)
    random = np.random.default_rng(seed)
    for target in range(response_start, len(graph)):
        groups: dict[tuple[int, int], list[int]] = {}
        for source in range(target):
            groups.setdefault(
                (int(role[source]), lag_bucket(target - source)), []
            ).append(source)
        for sources in groups.values():
            if len(sources) < 2:
                continue
            index = np.asarray(sources, dtype=np.int64)
            shift = int(random.integers(1, len(index)))
            rewired[index, target] = np.roll(graph[index, target], shift)
    return rewired


def role_inflow(
    transition: np.ndarray,
    response_start: int,
    evidence_mask,
) -> dict[str, np.ndarray]:
    graph = np.asarray(transition, dtype=np.float64)
    evidence = np.asarray(evidence_mask, dtype=bool)[:response_start]
    events = len(graph) - response_start
    result = {
        "evidence": np.zeros(events),
        "other_prompt": np.zeros(events),
        "history": np.zeros(events),
    }
    for event, target in enumerate(range(response_start, len(graph))):
        prompt = graph[:response_start, target]
        result["evidence"][event] = prompt[evidence].sum()
        result["other_prompt"][event] = prompt[~evidence].sum()
        result["history"][event] = graph[response_start:target, target].sum()
    return result


def capacity_bag(transition: np.ndarray, sink: int, edge_count: int) -> np.ndarray:
    """Select top individual capacities without enforcing path connectivity."""

    graph = np.asarray(transition, dtype=np.float64)
    source, target = np.nonzero(np.triu(graph[: sink + 1, : sink + 1], 1) > 0)
    mask = np.zeros_like(graph, dtype=bool)
    if edge_count <= 0 or not len(source):
        return mask
    value = graph[source, target]
    for position in np.argsort(-value, kind="stable")[:edge_count]:
        mask[source[position], target[position]] = True
    return mask


def matched_endpoint_mask(
    selected: np.ndarray,
    transition: np.ndarray,
    response_start: int,
    evidence_mask,
) -> np.ndarray:
    """Replace selected sources by same-role, similar-lag endpoints."""

    chosen = np.asarray(selected, dtype=bool)
    graph = np.asarray(transition, dtype=np.float64)
    role = source_roles(len(graph), response_start, evidence_mask)
    matched = np.zeros_like(chosen)
    for source, target in zip(*np.nonzero(chosen), strict=True):
        candidates = [
            other
            for other in range(target)
            if other != source
            and role[other] == role[source]
            and lag_bucket(target - other) == lag_bucket(target - source)
            and not chosen[other, target]
            and not matched[other, target]
        ]
        if not candidates:
            candidates = [
                other
                for other in range(target)
                if other != source
                and role[other] == role[source]
                and not chosen[other, target]
                and not matched[other, target]
            ]
        if candidates:
            replacement = min(
                candidates,
                key=lambda other: abs(graph[other, target] - graph[source, target]),
            )
            matched[replacement, target] = True
    return matched


def token_edges_to_query_mask(edges: np.ndarray) -> np.ndarray:
    """Map ``source -> predicted token`` edges to ``query, source`` gates."""

    selected = np.asarray(edges, dtype=bool)
    if selected.ndim != 2 or selected.shape[0] != selected.shape[1]:
        raise ValueError("edge mask must be square")
    token_count = len(selected)
    mask = np.zeros((token_count - 1, token_count - 1), dtype=bool)
    for source, prediction in zip(*np.nonzero(selected), strict=True):
        query = prediction - 1
        if 0 <= source <= query < token_count - 1:
            mask[query, source] = True
    return mask
