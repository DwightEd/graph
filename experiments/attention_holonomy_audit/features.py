"""Mechanism features for depth/relay transport, query sets and causal diamonds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .config import CONTROL_FEATURES, PRIMARY_FEATURES
from .geometry import event_profiles, hellinger_squared
from .graph import AttentionEventGraph
from .transport import TransportReference, query_training_rows


@dataclass(frozen=True)
class MechanismAudit:
    primary: np.ndarray
    controls: np.ndarray
    primary_maps: np.ndarray
    control_maps: np.ndarray
    nuisance: np.ndarray
    nuisance_names: tuple[str, ...]
    coverage: np.ndarray


class _MapAccumulator:
    def __init__(self, tokens: int, layers: int, features: int) -> None:
        self.total = np.zeros((tokens, layers, features), dtype=np.float64)
        self.weight = np.zeros_like(self.total)

    def add(self, query: int, layer: int, feature: int, value: float, weight: float = 1.0) -> None:
        if np.isfinite(value) and weight > 0:
            self.total[query, layer, feature] += float(value) * float(weight)
            self.weight[query, layer, feature] += float(weight)

    def values(self) -> np.ndarray:
        result = np.full_like(self.total, np.nan, dtype=np.float64)
        selected = self.weight > 0
        result[selected] = self.total[selected] / self.weight[selected]
        return result.astype(np.float32)


def _profile_to_clr(profile: np.ndarray) -> np.ndarray:
    values = np.log(np.clip(profile, 1e-30, 1.0))
    return values - values.mean(axis=-1, keepdims=True)


def _top_quartile_mean(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tokens, _, features = values.shape
    output = np.full((tokens, features), np.nan, dtype=np.float32)
    coverage = np.zeros((tokens, features), dtype=np.float32)
    for token in range(tokens):
        for feature in range(features):
            current = values[token, :, feature]
            current = current[np.isfinite(current)]
            coverage[token, feature] = len(current)
            if not len(current):
                continue
            count = max(1, int(np.ceil(len(current) / 4.0)))
            output[token, feature] = float(np.mean(np.partition(current, -count)[-count:]))
    return output, coverage


def _null_predecessors(
    graph: AttentionEventGraph,
    left: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Rotate predecessors inside layer/role/lag/observation strata."""

    layer = graph.event_layer[left].detach().cpu().numpy()
    role = graph.event_role[left].detach().cpu().numpy()
    lag = graph.event_lag[left].detach().cpu().numpy()
    observed = graph.event_head_observed[left].sum(dim=-1).detach().cpu().numpy()
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for row in range(len(left)):
        key = (
            int(layer[row]),
            int(role[row]),
            int(max(lag[row], 1)).bit_length() - 1,
            int(observed[row]),
        )
        groups.setdefault(key, []).append(row)

    output = left.copy()
    for key, rows in groups.items():
        if len(rows) < 2:
            continue
        payload = f"{seed}\0{graph.sample_id}\0{key}".encode()
        offset = 1 + int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (len(rows) - 1)
        source = [left[row] for row in rows]
        rotated = source[offset:] + source[:offset]
        for row, value in zip(rows, rotated, strict=True):
            output[row] = value
    return output


def compute_mechanism_audit(
    graph: AttentionEventGraph,
    reference: TransportReference,
    *,
    seed: int,
) -> MechanismAudit:
    """Compute label-free mechanism maps and token summaries for one sample."""

    profile_tensor, clr_tensor = event_profiles(
        graph.event_head_value,
        graph.event_head_observed,
        attention_floor=graph.attention_floor,
        censored_fill_ratio=reference.graph_config.censored_fill_ratio,
    )
    profile = profile_tensor.detach().cpu().numpy().astype(np.float64)
    clr = clr_tensor.detach().cpu().numpy().astype(np.float64)
    event_mass = graph.event_mass.detach().cpu().numpy().astype(np.float64)
    event_query = graph.event_query.detach().cpu().numpy().astype(np.int64)
    event_layer = graph.event_layer.detach().cpu().numpy().astype(np.int64)
    event_role = graph.event_role.detach().cpu().numpy().astype(np.int64)

    primary = _MapAccumulator(
        graph.num_response_tokens, graph.num_layers, len(PRIMARY_FEATURES)
    )
    controls = _MapAccumulator(
        graph.num_response_tokens, graph.num_layers, len(CONTROL_FEATURES)
    )

    depth_prediction: dict[int, np.ndarray] = {}
    if graph.depth_edge_index.numel():
        left, right = graph.depth_edge_index.detach().cpu().numpy()
        for predecessor, successor in zip(left, right, strict=True):
            layer = int(event_layer[predecessor])
            predicted = reference.depth_profile(layer, clr[predecessor : predecessor + 1])[0]
            baseline = reference.depth_baseline(layer)[0]
            error = float(hellinger_squared(predicted, profile[successor]))
            baseline_error = float(hellinger_squared(baseline, profile[successor]))
            query = int(event_query[successor])
            target_layer = int(event_layer[successor])
            weight = max(event_mass[successor], 1e-12)
            primary.add(query, target_layer, 0, error, weight)
            controls.add(query, target_layer, 0, baseline_error - error, weight)
            depth_prediction[int(successor)] = predicted

    if graph.relay_edge_index.numel():
        left, right = graph.relay_edge_index.detach().cpu().numpy()
        null_left = _null_predecessors(graph, left, seed=seed)
        successor_rows: dict[int, list[int]] = {}
        for row, successor in enumerate(right):
            successor_rows.setdefault(int(successor), []).append(row)

        for successor, rows in successor_rows.items():
            real_predictions = []
            null_predictions = []
            baseline_predictions = []
            weights = []
            for row in rows:
                predecessor = int(left[row])
                null_predecessor = int(null_left[row])
                layer = int(event_layer[predecessor])
                role = int(event_role[predecessor])
                real_predictions.append(
                    reference.relay_profile(layer, role, clr[predecessor : predecessor + 1])[0]
                )
                null_role = int(event_role[null_predecessor])
                null_predictions.append(
                    reference.relay_profile(
                        layer,
                        null_role,
                        clr[null_predecessor : null_predecessor + 1],
                    )[0]
                )
                baseline_predictions.append(reference.relay_baseline(layer, role)[0])
                weights.append(max(event_mass[predecessor], 1e-12))

            weights_array = np.asarray(weights, dtype=np.float64)
            weights_array /= weights_array.sum()

            def aggregate(values: list[np.ndarray]) -> np.ndarray:
                result = (np.asarray(values) * weights_array[:, None]).sum(axis=0)
                return result / result.sum()

            real = aggregate(real_predictions)
            null = aggregate(null_predictions)
            baseline = aggregate(baseline_predictions)
            actual = profile[successor]
            real_error = float(hellinger_squared(real, actual))
            null_error = float(hellinger_squared(null, actual))
            baseline_error = float(hellinger_squared(baseline, actual))
            dispersion = float(
                np.sum(
                    weights_array
                    * hellinger_squared(np.asarray(real_predictions), real[None])
                )
            )
            query = int(event_query[successor])
            layer = int(event_layer[successor])
            weight = max(event_mass[successor], 1e-12)
            primary.add(query, layer, 1, real_error, weight)
            primary.add(query, layer, 2, dispersion, weight)
            controls.add(query, layer, 1, baseline_error - real_error, weight)
            controls.add(query, layer, 3, null_error - real_error, weight)
            if successor in depth_prediction:
                disagreement = float(hellinger_squared(depth_prediction[successor], real))
                primary.add(query, layer, 3, disagreement, weight)

    events, layers, local, full, _target = query_training_rows(graph, clr)
    if len(events):
        for layer in range(graph.num_layers):
            selected = np.flatnonzero(layers == layer)
            if not len(selected):
                continue
            local_prediction, full_prediction = reference.query_profiles(
                layer, local[selected], full[selected]
            )
            local_error = hellinger_squared(local_prediction, profile[events[selected]])
            full_error = hellinger_squared(full_prediction, profile[events[selected]])
            for row, event in enumerate(events[selected]):
                query = int(event_query[event])
                weight = max(event_mass[event], 1e-12)
                primary.add(query, layer, 4, float(full_error[row]), weight)
                controls.add(
                    query, layer, 2, float(local_error[row] - full_error[row]), weight
                )

    if graph.diamond_index.numel():
        start, _depth_middle, _relay_middle, end = graph.diamond_index.detach().cpu().numpy()
        for first, fourth in zip(start, end, strict=True):
            layer = int(event_layer[first])
            role = int(event_role[first])
            relay_first = reference.relay_profile(layer, role, clr[first : first + 1])[0]
            path_a = reference.depth_profile(
                layer + 1, _profile_to_clr(relay_first[None])
            )[0]
            depth_first = reference.depth_profile(layer, clr[first : first + 1])[0]
            path_b = reference.relay_profile(
                layer + 1, role, _profile_to_clr(depth_first[None])
            )[0]
            holonomy = float(hellinger_squared(path_a, path_b))
            target_error = float(
                0.5
                * (
                    hellinger_squared(path_a, profile[fourth])
                    + hellinger_squared(path_b, profile[fourth])
                )
            )
            query = int(event_query[fourth])
            target_layer = int(event_layer[fourth])
            weight = max(event_mass[fourth], 1e-12)
            primary.add(query, target_layer, 5, holonomy, weight)
            controls.add(query, target_layer, 4, target_error, weight)

    primary_maps = primary.values()
    control_maps = controls.values()
    token_primary, coverage = _top_quartile_mean(primary_maps)
    token_controls, _ = _top_quartile_mean(control_maps)

    tokens = graph.num_response_tokens
    event_count = np.bincount(event_query, minlength=tokens).astype(np.float32)
    retained_mass = np.bincount(event_query, weights=event_mass, minlength=tokens).astype(
        np.float32
    )
    observed_fraction = np.zeros(tokens, dtype=np.float32)
    if graph.num_events:
        event_observed = graph.event_head_observed.float().mean(dim=-1).detach().cpu().numpy()
        observed_total = np.bincount(event_query, weights=event_observed, minlength=tokens)
        observed_fraction = (observed_total / np.maximum(event_count, 1)).astype(np.float32)

    relay_count = np.zeros(tokens, dtype=np.float32)
    if graph.relay_edge_index.numel():
        relay_target = event_query[graph.relay_edge_index[1].detach().cpu().numpy()]
        relay_count = np.bincount(relay_target, minlength=tokens).astype(np.float32)
    diamond_count = np.zeros(tokens, dtype=np.float32)
    if graph.diamond_index.numel():
        diamond_target = event_query[graph.diamond_index[3].detach().cpu().numpy()]
        diamond_count = np.bincount(diamond_target, minlength=tokens).astype(np.float32)

    absolute_position = np.arange(tokens, dtype=np.float32)
    relative_position = absolute_position / max(tokens - 1, 1)
    unresolved_mean = graph.unresolved_mass.mean(dim=(1, 2)).detach().cpu().numpy().astype(
        np.float32
    )
    nuisance_names = (
        "absolute_position",
        "relative_position",
        "response_length",
        "event_count",
        "relay_count",
        "diamond_count",
        "retained_mass",
        "observed_head_fraction",
        "unresolved_mean",
    )
    nuisance = np.stack(
        (
            absolute_position,
            relative_position,
            np.full(tokens, tokens, dtype=np.float32),
            event_count,
            relay_count,
            diamond_count,
            retained_mass,
            observed_fraction,
            unresolved_mean,
        ),
        axis=1,
    )

    return MechanismAudit(
        primary=token_primary,
        controls=token_controls,
        primary_maps=primary_maps,
        control_maps=control_maps,
        nuisance=nuisance,
        nuisance_names=nuisance_names,
        coverage=coverage,
    )
