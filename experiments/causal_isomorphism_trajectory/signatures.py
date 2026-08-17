"""Temporal-isomorphism signatures and two-axis attention trajectories.

The representation hashes rooted one-hop events, time-respecting two-hop paths
and source-sharing motifs. It is invariant to response-node renaming that
preserves causal order, roles and edge labels. It is a bounded temporal-WL-style
invariant, not a complete graph-isomorphism test.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from attention_graph.causal_events import (
    CausalMultiplexEvents,
    RP,
    RR,
    SUMMARY_NAMES,
    log_lag_bin,
)


ROLE_FEATURE_NAMES = (
    "log_rp_mass",
    "log_rr_mass",
    "prompt_mass_share",
    "rr_edge_fraction",
    "retained_entropy",
    "rr_retained_entropy",
    "retained_top1_share",
    "rr_retained_top1_share",
    "log_source_effective_number",
    "source_top1_share",
    "log_source_mean_lag",
    "log_source_lag_std",
    "active_channel_fraction",
    "route_effective_rank",
    "cross_channel_consensus",
    "anchor_turnover",
)

DEPTH_SUMMARY_NAMES = (
    "depth_mean",
    "late_depth_mean",
    "late_depth_max",
    "late_depth_slope",
    "depth_curvature",
)

MOTION_NAMES = (
    "time_signature_cosine_distance",
    "time_signature_l1_distance",
    "role_delta_l2",
    "depth_delta_l2",
)


@dataclass(frozen=True)
class SignatureConfig:
    hash_dim: int = 128
    lag_bins: int = 8
    weight_bins: int = 5
    position_buckets: int = 10
    late_band_transitions: int = 2
    source_anchor_count: int = 8
    max_parent_events: int = 8
    epsilon: float = 1e-8

    def validate(self) -> None:
        integers = (
            self.hash_dim,
            self.lag_bins,
            self.weight_bins,
            self.position_buckets,
            self.late_band_transitions,
            self.source_anchor_count,
            self.max_parent_events,
        )
        if min(map(int, integers)) < 1:
            raise ValueError("signature integer settings must be positive")
        if not np.isfinite(self.epsilon) or float(self.epsilon) <= 0:
            raise ValueError("epsilon must be positive and finite")


@dataclass(frozen=True)
class TrajectoryFeatureSet:
    """Per-token matrices for four preregistered variants."""

    full: np.ndarray
    static: np.ndarray
    topology: np.ndarray
    mass: np.ndarray
    position_bucket: np.ndarray
    full_feature_names: np.ndarray
    static_feature_names: np.ndarray
    topology_feature_names: np.ndarray
    mass_feature_names: np.ndarray
    role_state: np.ndarray
    global_signature: np.ndarray
    depth_transition: np.ndarray

    @property
    def response_count(self) -> int:
        return int(len(self.position_bucket))

    def variants(self) -> dict[str, np.ndarray]:
        return {
            "full": self.full,
            "static": self.static,
            "topology": self.topology,
            "mass": self.mass,
        }

    def names(self) -> dict[str, np.ndarray]:
        return {
            "full": self.full_feature_names,
            "static": self.static_feature_names,
            "topology": self.topology_feature_names,
            "mass": self.mass_feature_names,
        }


def causal_position_bucket(token_index: int, buckets: int) -> int:
    """Causal position bin that never uses final response length."""
    token_index = int(token_index)
    buckets = int(buckets)
    if token_index < 0 or buckets < 1:
        raise ValueError("invalid causal position bucket input")
    return min(int(np.floor(np.log2(token_index + 1))), buckets - 1)


def _normalized_entropy(
    mass: np.ndarray,
    count: np.ndarray,
    weight_log_weight: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    mass = np.asarray(mass, dtype=np.float64)
    count = np.asarray(count, dtype=np.float64)
    weight_log_weight = np.asarray(weight_log_weight, dtype=np.float64)
    entropy = np.zeros(len(mass), dtype=np.float64)
    valid = mass > epsilon
    entropy[valid] = (
        np.log(mass[valid])
        - weight_log_weight[valid] / mass[valid]
    )
    return np.divide(
        entropy,
        np.log(np.maximum(count, 2.0)),
        out=np.zeros_like(entropy),
        where=count > 1,
    ).astype(np.float32)


def _top1_share(
    maximum: np.ndarray,
    mass: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    return np.divide(
        maximum,
        mass,
        out=np.zeros_like(np.asarray(maximum), dtype=np.float32),
        where=np.asarray(mass) > epsilon,
    ).astype(np.float32)


def _effective_rank_and_consensus(
    channel_lag: np.ndarray,
    *,
    epsilon: float,
) -> tuple[float, float]:
    active = channel_lag.sum(axis=1) > epsilon
    matrix = channel_lag[active]
    if len(matrix) == 0:
        return 0.0, 0.0
    matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), epsilon)
    gram = matrix.T @ matrix
    eigenvalues = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    total = float(eigenvalues.sum())
    if total <= epsilon:
        effective_rank = 0.0
    else:
        probability = eigenvalues / total
        positive = probability > 0
        effective_rank = float(
            np.exp(
                -(probability[positive] * np.log(probability[positive])).sum()
            )
        )
        effective_rank /= float(max(1, min(matrix.shape)))
    if len(matrix) < 2:
        consensus = 0.0
    else:
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        unit = matrix / np.maximum(norm, epsilon)
        summed = unit.sum(axis=0)
        numerator = float(summed @ summed) - len(matrix)
        denominator = len(matrix) * (len(matrix) - 1)
        consensus = float(np.clip(numerator / denominator, -1.0, 1.0))
    return effective_rank, consensus


def _role_state(
    events: CausalMultiplexEvents,
    config: SignatureConfig,
) -> np.ndarray:
    summary = events.role_summary.detach().cpu().numpy().astype(np.float32)
    index = {name: offset for offset, name in enumerate(SUMMARY_NAMES)}
    rp_mass = summary[:, index["rp_mass"]]
    rr_mass = summary[:, index["rr_mass"]]
    rp_count = summary[:, index["rp_edge_count"]]
    rr_count = summary[:, index["rr_edge_count"]]
    total_mass = rp_mass + rr_mass
    total_count = rp_count + rr_count
    retained_entropy = _normalized_entropy(
        total_mass,
        total_count,
        summary[:, index["all_weight_log_weight"]],
        epsilon=config.epsilon,
    )
    rr_entropy = _normalized_entropy(
        rr_mass,
        rr_count,
        summary[:, index["rr_weight_log_weight"]],
        epsilon=config.epsilon,
    )
    retained_top1 = _top1_share(
        summary[:, index["all_max_weight"]],
        total_mass,
        epsilon=config.epsilon,
    )
    rr_top1 = _top1_share(
        summary[:, index["rr_max_weight"]],
        rr_mass,
        epsilon=config.epsilon,
    )

    source_effective = np.zeros(events.response_count, dtype=np.float32)
    source_top1 = np.zeros_like(source_effective)
    mean_lag = np.zeros_like(source_effective)
    lag_std = np.zeros_like(source_effective)
    active_channel = np.zeros_like(source_effective)
    route_rank = np.zeros_like(source_effective)
    consensus = np.zeros_like(source_effective)
    turnover = np.zeros_like(source_effective)

    relation = events.relation.detach().cpu().numpy()
    source = events.source.detach().cpu().numpy()
    weight = events.weight.detach().cpu().numpy().astype(np.float64)
    lag = events.lag.detach().cpu().numpy()
    channel = events.channel.detach().cpu().numpy()
    previous_anchors: set[int] = set()

    for token in range(events.response_count):
        current = events.target_slice(token)
        event_indices = np.arange(current.start, current.stop, dtype=np.int64)
        rr_indices = event_indices[relation[current] == RR]
        if len(rr_indices) == 0:
            if token > 0 and previous_anchors:
                turnover[token] = 1.0
            previous_anchors = set()
            continue

        current_sources = source[rr_indices]
        current_weights = weight[rr_indices]
        unique_source, inverse = np.unique(
            current_sources, return_inverse=True
        )
        source_weight = np.zeros(len(unique_source), dtype=np.float64)
        np.add.at(source_weight, inverse, current_weights)
        probability = source_weight / max(
            float(source_weight.sum()), config.epsilon
        )
        positive = probability > 0
        entropy = float(
            -(probability[positive] * np.log(probability[positive])).sum()
        )
        source_effective[token] = float(np.exp(entropy))
        source_top1[token] = float(probability.max(initial=0.0))
        source_lag = token - unique_source
        current_mean = float(probability @ source_lag)
        mean_lag[token] = current_mean
        lag_std[token] = float(
            np.sqrt(probability @ np.square(source_lag - current_mean))
        )

        unique_channel, inverse_channel = np.unique(
            channel[rr_indices], return_inverse=True
        )
        route = np.zeros(
            (len(unique_channel), config.lag_bins), dtype=np.float64
        )
        lag_bucket = np.minimum(
            np.floor(np.log2(np.maximum(lag[rr_indices], 1))).astype(np.int64),
            config.lag_bins - 1,
        )
        np.add.at(route, (inverse_channel, lag_bucket), current_weights)
        active_channel[token] = len(unique_channel) / float(events.num_channels)
        route_rank[token], consensus[token] = _effective_rank_and_consensus(
            route, epsilon=config.epsilon
        )

        keep = min(config.source_anchor_count, len(unique_source))
        order = np.argsort(source_weight, kind="stable")[-keep:]
        anchors = set(map(int, unique_source[order].tolist()))
        if token > 0:
            union = anchors | previous_anchors
            turnover[token] = (
                0.0
                if not union
                else 1.0 - len(anchors & previous_anchors) / len(union)
            )
        previous_anchors = anchors

    prompt_share = np.divide(
        rp_mass,
        total_mass,
        out=np.zeros_like(rp_mass),
        where=total_mass > config.epsilon,
    )
    rr_edge_fraction = np.divide(
        rr_count,
        total_count,
        out=np.zeros_like(rr_count),
        where=total_count > 0,
    )
    return np.column_stack(
        (
            np.log1p(rp_mass),
            np.log1p(rr_mass),
            prompt_share,
            rr_edge_fraction,
            retained_entropy,
            rr_entropy,
            retained_top1,
            rr_top1,
            np.log1p(source_effective),
            source_top1,
            np.log1p(mean_lag),
            np.log1p(lag_std),
            active_channel,
            route_rank,
            consensus,
            turnover,
        )
    ).astype(np.float32)


def _weight_bin(weight: float, floor: float, count: int) -> int:
    ratio = max(float(weight) / max(float(floor), 1e-12), 1.0)
    return min(int(np.floor(np.log2(ratio))), int(count) - 1)


def _event_labels(
    events: CausalMultiplexEvents,
    config: SignatureConfig,
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    relation = events.relation.detach().cpu().numpy()
    band = events.band.detach().cpu().numpy()
    head = events.head.detach().cpu().numpy()
    lag = events.lag.detach().cpu().numpy()
    weight = events.weight.detach().cpu().numpy()
    magnitude = np.log1p(
        weight / max(float(events.attention_floor), config.epsilon)
    ).astype(np.float32)
    labels: list[tuple[int, ...]] = []
    for event_index in range(events.num_events):
        lag_bucket = (
            0
            if int(relation[event_index]) == RP
            else min(
                log_lag_bin(int(lag[event_index])),
                config.lag_bins - 1,
            )
        )
        labels.append(
            (
                int(relation[event_index]),
                int(band[event_index]),
                int(head[event_index]),
                int(lag_bucket),
                _weight_bin(
                    float(weight[event_index]),
                    float(events.attention_floor),
                    config.weight_bins,
                ),
            )
        )
    return labels, magnitude


class _Hasher:
    def __init__(self, dimension: int):
        self.dimension = int(dimension)
        self.cache: dict[tuple, int] = {}

    def index(self, key: tuple) -> int:
        value = self.cache.get(key)
        if value is not None:
            return value
        digest = hashlib.blake2b(
            repr(key).encode("utf-8"), digest_size=8
        ).digest()
        value = int.from_bytes(digest, "big") % self.dimension
        self.cache[key] = value
        return value


def _normalize_histogram(values: np.ndarray, epsilon: float) -> np.ndarray:
    total = values.sum(axis=-1, keepdims=True)
    return np.divide(
        values,
        np.maximum(total, epsilon),
        out=np.zeros_like(values),
        where=total > epsilon,
    )


def _event_graph_signatures(
    events: CausalMultiplexEvents,
    config: SignatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = events.response_count
    bands = events.layer_bands
    dimension = config.hash_dim
    global_count = np.zeros((tokens, dimension), dtype=np.float32)
    global_mass = np.zeros_like(global_count)
    band_count = np.zeros((tokens, bands, dimension), dtype=np.float32)
    band_mass = np.zeros_like(band_count)
    labels, magnitude = _event_labels(events, config)
    relation = events.relation.detach().cpu().numpy()
    source = events.source.detach().cpu().numpy()
    band = events.band.detach().cpu().numpy()
    hasher = _Hasher(dimension)

    for token in range(tokens):
        current = events.target_slice(token)
        outer = np.arange(current.start, current.stop, dtype=np.int64)
        for edge_index in outer:
            edge_label = labels[edge_index]
            feature_index = hasher.index(("edge", *edge_label))
            current_band = int(band[edge_index])
            edge_magnitude = float(magnitude[edge_index])
            global_count[token, feature_index] += 1.0
            global_mass[token, feature_index] += edge_magnitude
            band_count[token, current_band, feature_index] += 1.0
            band_mass[token, current_band, feature_index] += edge_magnitude

            if int(relation[edge_index]) != RR:
                continue
            parent = int(source[edge_index])
            parent_slice = events.target_slice(parent)
            inner = np.arange(
                parent_slice.start, parent_slice.stop, dtype=np.int64
            )
            if len(inner) > config.max_parent_events:
                selected = np.argsort(
                    magnitude[inner], kind="stable"
                )[-config.max_parent_events :]
                inner = inner[selected]
            for inner_index in inner:
                inner_label = labels[inner_index]
                path_index = hasher.index(
                    ("path2", *inner_label, *edge_label)
                )
                path_mass = float(
                    np.sqrt(
                        max(float(magnitude[inner_index]), 0.0)
                        * max(edge_magnitude, 0.0)
                    )
                )
                global_count[token, path_index] += 1.0
                global_mass[token, path_index] += path_mass
                if int(band[inner_index]) == current_band:
                    band_count[token, current_band, path_index] += 1.0
                    band_mass[token, current_band, path_index] += path_mass

        rr_outer = outer[relation[current] == RR]
        if len(rr_outer):
            current_sources = source[rr_outer]
            for source_id in np.unique(current_sources):
                selected = rr_outer[current_sources == source_id]
                multiplicity = min(
                    int(np.floor(np.log2(max(len(selected), 1)))),
                    5,
                )
                age = min(
                    log_lag_bin(max(token - int(source_id), 1)),
                    config.lag_bins - 1,
                )
                parent = events.target_slice(int(source_id))
                parent_bin = min(
                    int(np.floor(np.log2(parent.stop - parent.start + 1))),
                    5,
                )
                band_count_value = len(set(map(int, band[selected].tolist())))
                motif_index = hasher.index(
                    (
                        "source_motif",
                        multiplicity,
                        age,
                        parent_bin,
                        min(band_count_value, bands),
                    )
                )
                global_count[token, motif_index] += 1.0
                global_mass[token, motif_index] += float(len(selected))
                for current_band in set(map(int, band[selected].tolist())):
                    local = selected[band[selected] == current_band]
                    local_index = hasher.index(
                        (
                            "band_source_motif",
                            current_band,
                            min(
                                int(
                                    np.floor(
                                        np.log2(max(len(local), 1))
                                    )
                                ),
                                5,
                            ),
                            age,
                            parent_bin,
                        )
                    )
                    band_count[token, current_band, local_index] += 1.0
                    band_mass[token, current_band, local_index] += float(
                        len(local)
                    )

    global_signature = np.concatenate(
        (
            _normalize_histogram(global_count, config.epsilon),
            _normalize_histogram(global_mass, config.epsilon),
        ),
        axis=1,
    )
    band_signature = np.concatenate(
        (
            _normalize_histogram(band_count, config.epsilon),
            _normalize_histogram(band_mass, config.epsilon),
        ),
        axis=2,
    )
    return global_signature.astype(np.float32), band_signature.astype(np.float32)


def _cosine_distance(
    left: np.ndarray,
    right: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    left_norm = np.linalg.norm(left, axis=-1)
    right_norm = np.linalg.norm(right, axis=-1)
    denominator = left_norm * right_norm
    result = np.zeros_like(numerator, dtype=np.float32)
    both = denominator > epsilon
    result[both] = 1.0 - np.clip(
        numerator[both] / denominator[both], -1.0, 1.0
    )
    one_zero = (left_norm <= epsilon) ^ (right_norm <= epsilon)
    result[one_zero] = 1.0
    return result


def _depth_transition(
    band_signature: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    if band_signature.shape[1] < 2:
        return np.zeros((len(band_signature), 0), dtype=np.float32)
    left = band_signature[:, :-1]
    right = band_signature[:, 1:]
    cosine = _cosine_distance(left, right, epsilon)
    l1 = 0.5 * np.abs(right - left).sum(axis=2)
    return np.concatenate((cosine, l1), axis=1).astype(np.float32)


def _depth_summaries(
    depth: np.ndarray,
    *,
    bands: int,
    late_count: int,
) -> np.ndarray:
    tokens = len(depth)
    transitions = max(int(bands) - 1, 0)
    if transitions == 0:
        return np.zeros(
            (tokens, len(DEPTH_SUMMARY_NAMES)), dtype=np.float32
        )
    cosine = depth[:, :transitions]
    late = cosine[:, -min(int(late_count), transitions) :]
    slope = (
        cosine[:, -1] - cosine[:, 0]
        if transitions > 1
        else np.zeros(tokens, dtype=np.float32)
    )
    curvature = (
        np.abs(np.diff(cosine, n=2, axis=1)).mean(axis=1)
        if transitions > 2
        else np.zeros(tokens, dtype=np.float32)
    )
    return np.column_stack(
        (
            cosine.mean(axis=1),
            late.mean(axis=1),
            late.max(axis=1),
            slope,
            curvature,
        )
    ).astype(np.float32)


def _prepend_zero_delta(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    if len(values) > 1:
        result[1:] = values[1:] - values[:-1]
    return result


def _time_signature_motion(
    signature: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.zeros(len(signature), dtype=np.float32)
    l1 = np.zeros(len(signature), dtype=np.float32)
    if len(signature) > 1:
        cosine[1:] = _cosine_distance(
            signature[:-1], signature[1:], epsilon
        )
        l1[1:] = 0.5 * np.abs(
            signature[1:] - signature[:-1]
        ).sum(axis=1)
    return cosine, l1


def _hash_names(prefix: str, dimension: int) -> list[str]:
    return [f"{prefix}_{index:04d}" for index in range(int(dimension))]


def extract_trajectory_features(
    events: CausalMultiplexEvents,
    *,
    config: SignatureConfig | None = None,
) -> TrajectoryFeatureSet:
    """Build state, generation-time and model-depth trajectory features."""
    config = SignatureConfig() if config is None else config
    config.validate()
    role = _role_state(events, config)
    global_signature, band_signature = _event_graph_signatures(events, config)
    depth = _depth_transition(band_signature, config.epsilon)
    depth_summary = _depth_summaries(
        depth,
        bands=events.layer_bands,
        late_count=config.late_band_transitions,
    )

    delta_role = _prepend_zero_delta(role)
    delta_signature = _prepend_zero_delta(global_signature)
    delta_depth = _prepend_zero_delta(depth)
    time_cosine, time_l1 = _time_signature_motion(
        global_signature, config.epsilon
    )
    role_delta_l2 = np.linalg.norm(
        delta_role, axis=1
    ).astype(np.float32)
    depth_delta_l2 = np.linalg.norm(
        delta_depth, axis=1
    ).astype(np.float32)
    motion = np.column_stack(
        (time_cosine, time_l1, role_delta_l2, depth_delta_l2)
    ).astype(np.float32)

    static = np.concatenate(
        (role, global_signature, depth, depth_summary), axis=1
    ).astype(np.float32)
    full = np.concatenate(
        (
            static,
            delta_role,
            delta_signature,
            delta_depth,
            motion,
        ),
        axis=1,
    ).astype(np.float32)
    topology = np.concatenate(
        (
            global_signature,
            depth,
            depth_summary,
            delta_signature,
            delta_depth,
            motion[:, [0, 1, 3]],
        ),
        axis=1,
    ).astype(np.float32)
    mass = np.concatenate(
        (role, delta_role, role_delta_l2[:, None]), axis=1
    ).astype(np.float32)

    signature_names = _hash_names(
        "event_signature", global_signature.shape[1]
    )
    depth_names = _hash_names("depth_transition", depth.shape[1])
    delta_role_names = [f"delta_{name}" for name in ROLE_FEATURE_NAMES]
    delta_signature_names = [
        f"delta_{name}" for name in signature_names
    ]
    delta_depth_names = [f"delta_{name}" for name in depth_names]
    static_names = [
        *ROLE_FEATURE_NAMES,
        *signature_names,
        *depth_names,
        *DEPTH_SUMMARY_NAMES,
    ]
    full_names = [
        *static_names,
        *delta_role_names,
        *delta_signature_names,
        *delta_depth_names,
        *MOTION_NAMES,
    ]
    topology_names = [
        *signature_names,
        *depth_names,
        *DEPTH_SUMMARY_NAMES,
        *delta_signature_names,
        *delta_depth_names,
        MOTION_NAMES[0],
        MOTION_NAMES[1],
        MOTION_NAMES[3],
    ]
    mass_names = [
        *ROLE_FEATURE_NAMES,
        *delta_role_names,
        MOTION_NAMES[2],
    ]

    buckets = np.asarray(
        [
            causal_position_bucket(token, config.position_buckets)
            for token in range(events.response_count)
        ],
        dtype=np.int16,
    )
    return TrajectoryFeatureSet(
        full=full,
        static=static,
        topology=topology,
        mass=mass,
        position_bucket=buckets,
        full_feature_names=np.asarray(full_names, dtype=str),
        static_feature_names=np.asarray(static_names, dtype=str),
        topology_feature_names=np.asarray(topology_names, dtype=str),
        mass_feature_names=np.asarray(mass_names, dtype=str),
        role_state=role,
        global_signature=global_signature,
        depth_transition=depth,
    )
