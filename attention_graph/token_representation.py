"""Structure-preserving, label-blind token graph representations.

Exact graph statistics remain directly recoverable. Sparse RP/RR propagation
retains absolute edge/path mass and explicit self-versus-ancestor residuals.
PCA is visualization-only; scoring is fixed and one-sided, with no learned
graph weights, clustering detector, labels, or backpropagation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from .graph import GraphBuildConfig, RP, RR, build_attention_graph
from .statistics import (
    DIRECT_FEATURES,
    TOKEN_FEATURES,
    direct_lookback,
    direct_lookback_from_graph,
    token_statistics,
)


SCHEMA = "structure-preserving-token-graph-v1"
EXACT_FEATURES = TOKEN_FEATURES + DIRECT_FEATURES

# Compact mechanism hypothesis fixed before evaluation labels are read.
# Other exact features are preserved and evaluated separately, but not mixed
# into the primary score because their anomaly direction is uncertain.
FEATURE_DIRECTIONS = {
    "prompt_mass_fraction": -1.0,
    "edge_density": -1.0,
    "retained_concentration": 1.0,
    "mean_edge_strength": 1.0,
    "history_lag": -1.0,
}
SCORE_FEATURES = tuple(FEATURE_DIRECTIONS)
SCORE_INDEX = np.asarray([EXACT_FEATURES.index(name) for name in SCORE_FEATURES])
SCORE_DIRECTION = np.asarray(
    [FEATURE_DIRECTIONS[name] for name in SCORE_FEATURES], dtype=np.float32
)
VIEWS = ("token_only", "token_graph", "no_rp", "no_rr")


@dataclass(frozen=True)
class TokenRepresentationConfig:
    position_bins: int = 10
    diffusion_hops: int = 2
    csr_row_block: int = 4096
    display_mass_cover: float = 0.80
    display_edges_per_type: int = 2
    display_max_edges: int = 300
    visual_reference_size: int = 30_000
    sample_ids: tuple[str, ...] = ()
    seed: int = 42

    def validate(self):
        if min(
            self.position_bins, self.diffusion_hops, self.csr_row_block,
            self.display_edges_per_type, self.display_max_edges,
            self.visual_reference_size,
        ) < 1:
            raise ValueError("representation limits must be positive")
        if self.diffusion_hops < 2:
            raise ValueError("diffusion_hops must be at least two")
        if not 0.0 < float(self.display_mass_cover) <= 1.0:
            raise ValueError("display_mass_cover must be in (0,1]")


class _PositionRobustScaler:
    """Train-only position-conditioned median/MAD without label access."""

    def __init__(self, bins):
        self.bins = int(bins)

    @staticmethod
    def _fit_rows(values):
        center = np.median(values, axis=0)
        mad = 1.4826 * np.median(np.abs(values - center), axis=0)
        std = values.std(axis=0)
        scale = np.where(mad > 1e-8, mad, np.where(std > 1e-8, std, 1.0))
        return center, scale

    def _bin(self, position):
        return np.minimum(
            (np.asarray(position, dtype=np.float64) * self.bins).astype(int),
            self.bins - 1,
        )

    def fit(self, values, position, valid=None):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("scaler requires a non-empty finite matrix")
        valid = (
            np.ones_like(values, dtype=bool)
            if valid is None else np.asarray(valid, dtype=bool)
        )
        if valid.shape != values.shape:
            raise ValueError("scaler validity mask must match the value matrix")
        global_center = np.zeros(values.shape[1], dtype=np.float64)
        global_scale = np.ones(values.shape[1], dtype=np.float64)
        for column in range(values.shape[1]):
            selected = values[valid[:, column], column]
            if len(selected):
                center, scale = self._fit_rows(selected[:, None])
                global_center[column], global_scale[column] = center[0], scale[0]
        self.center = np.tile(global_center, (self.bins, 1))
        self.scale = np.tile(global_scale, (self.bins, 1))
        bins = self._bin(position)
        self.bin_count = []
        for bin_id in range(self.bins):
            counts = []
            for column in range(values.shape[1]):
                selected = values[(bins == bin_id) & valid[:, column], column]
                counts.append(int(len(selected)))
                if len(selected) >= 3:
                    center, scale = self._fit_rows(selected[:, None])
                    self.center[bin_id, column] = center[0]
                    self.scale[bin_id, column] = scale[0]
            self.bin_count.append(counts)
        return self

    def transform(self, values, position):
        values = np.asarray(values, dtype=np.float64)
        bins = self._bin(position)
        output = (values - self.center[bins]) / self.scale[bins]
        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def report(self):
        return {
            "type": "train_only_position_conditioned_median_mad",
            "position_bins": self.bins,
            "tokens_per_bin": self.bin_count,
            "fit_uses_labels": False,
        }


class _RobustScaler:
    """Train-only global median/MAD for graph mass or visualization."""

    def fit(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.center, self.scale = _PositionRobustScaler._fit_rows(values)
        return self

    def transform(self, values):
        output = (np.asarray(values, dtype=np.float64) - self.center) / self.scale
        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class _PositivePathReliability:
    """Train-only positive path reference with continuous zero reliability."""

    def __init__(self, bins):
        self.bins = int(bins)

    def _bin(self, position):
        return np.minimum(
            (np.asarray(position, dtype=np.float64) * self.bins).astype(int),
            self.bins - 1,
        )

    def fit(self, mass, position, eligible):
        mass = np.asarray(mass, dtype=np.float64)
        eligible = np.asarray(eligible, dtype=bool)
        if mass.ndim != 2 or mass.shape != eligible.shape:
            raise ValueError("path reliability requires aligned matrices")
        positive = eligible & (mass > 0)
        global_reference = np.ones(mass.shape[1], dtype=np.float64)
        for column in range(mass.shape[1]):
            selected = mass[positive[:, column], column]
            if len(selected):
                global_reference[column] = max(float(np.median(selected)), 1e-12)
        self.reference = np.tile(global_reference, (self.bins, 1))
        bins = self._bin(position)
        self.positive_count = []
        for bin_id in range(self.bins):
            counts = []
            for column in range(mass.shape[1]):
                selected = mass[(bins == bin_id) & positive[:, column], column]
                counts.append(int(len(selected)))
                if len(selected) >= 3:
                    self.reference[bin_id, column] = max(
                        float(np.median(selected)), 1e-12
                    )
            self.positive_count.append(counts)
        return self

    def transform(self, mass, position, eligible):
        mass = np.maximum(np.asarray(mass, dtype=np.float64), 0.0)
        eligible = np.asarray(eligible, dtype=bool)
        if mass.shape != eligible.shape:
            raise ValueError("path reliability requires aligned matrices")
        reference = self.reference[self._bin(position)]
        reliability = mass / (mass + reference)
        return np.where(eligible, reliability, 0.0).astype(np.float32)

    def report(self):
        return {
            "type": "train_only_positive_path_median_q_over_q_plus_q0",
            "position_bins": self.bins,
            "positive_paths_per_bin_and_hop": self.positive_count,
            "zero_path_reliability": 0.0,
            "fit_uses_labels": False,
        }


def exact_token_features(attention, graph=None, *, csr_row_block=4096):
    """Historical exact scalars plus the exact direct Lookback baseline."""
    graph = build_attention_graph(attention) if graph is None else graph
    scalar = token_statistics(graph)
    lookback = direct_lookback(attention, csr_row_block=csr_row_block)[:, None]
    output = torch.cat((scalar, lookback), dim=1)
    if output.shape[1] != len(EXACT_FEATURES):
        raise ValueError("exact feature schema and matrix width differ")
    return torch.nan_to_num(output)


def _raw_relation_matrices(graph):
    """Sparse RP/RR matrices using absolute pair mass, never row-normalized."""
    response_count = graph.num_nodes - graph.response_idx
    device = graph.node_attr.device
    matrices = {}
    for relation, width, source_offset in (
        (RP, graph.response_idx, 0),
        (RR, response_count, graph.response_idx),
    ):
        selected = torch.nonzero(graph.edge_type == relation, as_tuple=False).flatten()
        if selected.numel():
            source = graph.edge_index[0, selected].long() - source_offset
            target = graph.edge_index[1, selected].long() - graph.response_idx
            value = graph.edge_score[selected].float().clamp_min(0.0)
            index = torch.stack((target, source))
        else:
            index = torch.empty((2, 0), dtype=torch.long, device=device)
            value = torch.empty(0, dtype=torch.float32, device=device)
        matrices[relation] = torch.sparse_coo_tensor(
            index, value, size=(response_count, width), device=device
        ).coalesce()
    return matrices[RP], matrices[RR]


def _without_relation(graph, relation):
    """Return the exact counterfactual graph after deleting one edge type."""
    keep_edge = graph.edge_type != int(relation)
    old_to_new = torch.full(
        (graph.num_edges,), -1, dtype=torch.long, device=graph.edge_type.device
    )
    old_to_new[keep_edge] = torch.arange(
        int(keep_edge.sum()), device=graph.edge_type.device
    )
    keep_trace = keep_edge[graph.trace_edge_id]
    return graph.__class__(**{
        **graph.__dict__,
        "edge_index": graph.edge_index[:, keep_edge],
        "edge_type": graph.edge_type[keep_edge],
        "edge_score": graph.edge_score[keep_edge],
        "trace_edge_id": old_to_new[graph.trace_edge_id[keep_trace]],
        "trace_channel": graph.trace_channel[keep_trace],
        "trace_value": graph.trace_value[keep_trace],
    })


def _exact_hop_reach_count(graph, hops):
    response_count = graph.num_nodes - graph.response_idx
    selected = graph.edge_type == RR
    sources = (graph.edge_index[0, selected] - graph.response_idx).cpu().tolist()
    targets = (graph.edge_index[1, selected] - graph.response_idx).cpu().tolist()
    parents = [set() for _ in range(response_count)]
    for source, target in zip(sources, targets):
        parents[int(target)].add(int(source))
    current = [set(row) for row in parents]
    output = []
    for _ in range(int(hops)):
        output.append([len(row) for row in current])
        current = [
            set().union(*(parents[source] for source in row)) if row else set()
            for row in current
        ]
    return torch.tensor(output, dtype=torch.int32).T.contiguous()


def structure_preserving_messages(graph, base_z, *, diffusion_hops=2):
    """Retain raw path mass, conditional ancestors and local innovations.

    ``M_k=A_RR M_(k-1)`` is saved before dividing by path mass. Therefore two
    rows with equal normalized neighbor distributions but different absolute
    mass remain distinguishable. Residuals preserve local changes rather than
    replacing the token by a smoothed neighbor average.
    """
    base = torch.as_tensor(base_z, dtype=torch.float32, device=graph.node_attr.device)
    response_count = graph.num_nodes - graph.response_idx
    if tuple(base.shape) != (response_count, len(EXACT_FEATURES)):
        raise ValueError("base_z does not align with response nodes")
    rp, rr = _raw_relation_matrices(graph)
    rp_direct = torch.sparse.mm(
        rp, torch.ones((graph.response_idx, 1), device=base.device)
    )[:, 0]
    raw_state = base
    rr_mass_state = torch.ones((response_count, 1), device=base.device)
    rp_mass_state = rp_direct[:, None]
    raw_messages, conditional, residuals = [], [], []
    rr_mass, rp_mass = [], [rp_direct]
    for _ in range(int(diffusion_hops)):
        raw_state = torch.sparse.mm(rr, raw_state)
        rr_mass_state = torch.sparse.mm(rr, rr_mass_state)
        rp_mass_state = torch.sparse.mm(rr, rp_mass_state)
        mass = rr_mass_state[:, 0]
        mean = torch.where(
            mass[:, None] > 1e-12,
            raw_state / mass[:, None].clamp_min(1e-12),
            torch.zeros_like(raw_state),
        )
        residual = torch.where(
            mass[:, None] > 1e-12, base - mean, torch.zeros_like(base)
        )
        raw_messages.append(raw_state)
        conditional.append(mean)
        residuals.append(residual)
        rr_mass.append(mass)
        rp_mass.append(rp_mass_state[:, 0])
    return {
        "raw_message": torch.stack(raw_messages, dim=1),
        "conditional_neighbor": torch.stack(conditional, dim=1),
        "self_neighbor_residual": torch.stack(residuals, dim=1),
        "rr_path_mass": torch.stack(rr_mass, dim=1),
        "rp_path_mass": torch.stack(rp_mass, dim=1),
        "rr_hop_reach_count": _exact_hop_reach_count(graph, diffusion_hops),
    }


def _metadata_template():
    return {name: [] for name in (
        "sample_id", "source_id", "token_index", "token_id",
        "relative_position", "task_type", "data_source", "generator_model",
    )}


def _append_metadata(metadata, sample, attention, count):
    metadata["sample_id"].extend([str(sample.sample_id)] * count)
    metadata["source_id"].extend([str(sample.source_id)] * count)
    metadata["token_index"].extend(range(count))
    metadata["token_id"].extend(
        attention.token_ids[attention.response_idx:].detach().cpu().tolist()
    )
    metadata["relative_position"].extend(
        np.arange(count, dtype=np.float32) / max(count - 1, 1)
    )
    for field in ("task_type", "data_source", "generator_model"):
        metadata[field].extend([str(getattr(sample, field))] * count)


def _metadata_arrays(metadata):
    return {
        name: np.asarray(values, dtype=dtype)
        for name, values, dtype in (
            ("sample_id", metadata["sample_id"], str),
            ("source_id", metadata["source_id"], str),
            ("token_index", metadata["token_index"], np.int32),
            ("token_id", metadata["token_id"], np.int32),
            ("relative_position", metadata["relative_position"], np.float32),
            ("task_type", metadata["task_type"], str),
            ("data_source", metadata["data_source"], str),
            ("generator_model", metadata["generator_model"], str),
        )
    }


def _path_eligibility(token_index, hops):
    """Causal eligibility, distinct from whether a retained path exists."""
    token_index = np.asarray(token_index, dtype=np.int64)
    rr = np.column_stack([
        token_index >= hop for hop in range(1, int(hops) + 1)
    ])
    rp = np.column_stack((
        np.ones(len(token_index), dtype=bool), rr,
    ))
    return rp, rr


def _extract_exact_split(dataset, config, split_name):
    rows, metadata = [], _metadata_template()
    for sample_id in tqdm(
        dataset.sample_ids, desc=f"[1/7] {split_name} exact graph scalars", unit="sample"
    ):
        sample = dataset[sample_id]
        attention = sample.attention()
        graph = build_attention_graph(attention, GraphBuildConfig(selection="threshold"))
        values = exact_token_features(
            attention, graph, csr_row_block=config.csr_row_block
        ).detach().cpu().numpy().astype(np.float32)
        rows.append(values)
        _append_metadata(metadata, sample, attention, len(values))
        sample.release_attention()
    if not rows:
        raise ValueError(f"{split_name} split contains no response tokens")
    return np.concatenate(rows), _metadata_arrays(metadata)


def _extract_graph_split(
    dataset, base_z, base_scaler, relative_position, config, split_name, *, keep_graphs
):
    buffers = {name: [] for name in (
        "raw_message", "conditional_neighbor", "self_neighbor_residual",
        "rr_path_mass", "rp_path_mass", "rr_hop_reach_count",
        "no_rp_exact", "no_rr_exact", "no_rp_self_neighbor_residual",
    )}
    graphs, offset = [], 0
    for sample_id in tqdm(
        dataset.sample_ids, desc=f"[2/7] {split_name} mass-preserving propagation", unit="sample"
    ):
        sample = dataset[sample_id]
        attention = sample.attention()
        graph = build_attention_graph(attention, GraphBuildConfig(selection="threshold"))
        count = attention.num_response_tokens
        messages = structure_preserving_messages(
            graph, base_z[offset:offset + count], diffusion_hops=config.diffusion_hops
        )
        for name in (
            "raw_message", "conditional_neighbor", "self_neighbor_residual",
            "rr_path_mass", "rp_path_mass", "rr_hop_reach_count",
        ):
            buffers[name].append(messages[name].detach().cpu().numpy())
        for name, relation in (("no_rp_exact", RP), ("no_rr_exact", RR)):
            counterfactual_graph = _without_relation(graph, relation)
            counterfactual = torch.cat((
                token_statistics(counterfactual_graph),
                direct_lookback_from_graph(counterfactual_graph)[:, None],
            ), dim=1)
            buffers[name].append(counterfactual.detach().cpu().numpy())
            if relation == RP:
                no_rp_z = base_scaler.transform(
                    counterfactual.detach().cpu().numpy(),
                    relative_position[offset:offset + count],
                )
                no_rp_messages = structure_preserving_messages(
                    counterfactual_graph, no_rp_z,
                    diffusion_hops=config.diffusion_hops,
                )
                buffers["no_rp_self_neighbor_residual"].append(
                    no_rp_messages["self_neighbor_residual"].detach().cpu().numpy()
                )
        if keep_graphs:
            graphs.append({
                "sample_id": str(sample.sample_id), "start": offset, "end": offset + count,
                "token_ids": graph.token_ids.cpu().numpy().astype(np.int32),
                "response_idx": int(graph.response_idx),
                "edge_index": graph.edge_index.cpu().numpy().astype(np.int32),
                "edge_type": graph.edge_type.cpu().numpy().astype(np.int8),
                "edge_score": graph.edge_score.cpu().numpy().astype(np.float32),
            })
        offset += count
        sample.release_attention()
    if offset != len(base_z):
        raise ValueError("graph traversal and exact scalar rows do not align")
    output = {
        name: np.concatenate(rows).astype(
            np.int32 if name == "rr_hop_reach_count" else np.float32
        )
        for name, rows in buffers.items()
    }
    return output, graphs


def _directed_evidence(z, history_present=None):
    selected = np.asarray(z, dtype=np.float32)[..., SCORE_INDEX]
    evidence = np.maximum(selected * SCORE_DIRECTION, 0.0)
    if history_present is not None:
        lag_column = SCORE_FEATURES.index("history_lag")
        evidence[..., lag_column] = np.where(history_present, evidence[..., lag_column], 0.0)
    return evidence.astype(np.float32)


def _directed_score(z, history_present=None, excluded=()):
    evidence = _directed_evidence(z, history_present)
    for name in excluded:
        evidence[..., SCORE_FEATURES.index(name)] = 0.0
    # Keep the denominator fixed so ablations are literal zeroed blocks under
    # one scoring formula rather than newly reweighted detectors.
    return evidence.mean(axis=-1).astype(np.float32)


def _masked_mean(values, available):
    """Mean only over eligible entries; empty rows stay zero."""
    values = np.asarray(values, dtype=np.float32)
    available = np.asarray(available, dtype=bool)
    count = available.sum(axis=1)
    total = np.where(available, values, 0.0).sum(axis=1)
    return np.divide(
        total, count, out=np.zeros_like(total, dtype=np.float32), where=count > 0
    ).astype(np.float32)


def _build_scores(
    base_z, base_exact, no_rp_z, no_rp_exact, no_rr_z, no_rr_exact,
    relative_position, rp_eligible, rr_eligible, graph, rp_scaler, rr_scaler,
    rr_reliability
):
    history_present = (
        np.asarray(base_exact)[:, EXACT_FEATURES.index("history_edge_fraction")] > 0
    )
    token = _directed_score(base_z, history_present)
    base_no_rp = _directed_score(
        no_rp_z,
        np.asarray(no_rp_exact)[:, EXACT_FEATURES.index("history_edge_fraction")] > 0,
        excluded=("prompt_mass_fraction",),
    )
    base_no_rr = _directed_score(
        no_rr_z,
        np.asarray(no_rr_exact)[:, EXACT_FEATURES.index("history_edge_fraction")] > 0,
        excluded=("history_lag",),
    )
    hop_present = history_present[:, None] & (graph["rr_path_mass"] > 1e-12)
    innovation_by_hop = _directed_score(
        graph["self_neighbor_residual"], hop_present
    )
    reachable = graph["rr_hop_reach_count"] > 0
    no_rp_innovation_by_hop = _directed_score(
        graph["no_rp_self_neighbor_residual"], reachable
    )
    rp_log = np.log1p(np.maximum(graph["rp_path_mass"], 0.0))
    rp_z = rp_scaler.transform(rp_log, relative_position)
    rp_z = np.where(rp_eligible, rp_z, 0.0).astype(np.float32)
    rp_direct_weakness = np.maximum(-rp_z[:, 0], 0.0).astype(np.float32)
    rr_log = np.log1p(np.maximum(graph["rr_path_mass"], 0.0))
    rr_z = rr_scaler.transform(rr_log, relative_position)
    rr_z = np.where(rr_eligible, rr_z, 0.0).astype(np.float32)
    reliability = rr_reliability.transform(
        graph["rr_path_mass"], relative_position, rr_eligible
    )
    innovation = _masked_mean(
        innovation_by_hop * reliability, rr_eligible
    )
    no_rp_innovation = _masked_mean(
        no_rp_innovation_by_hop * reliability, rr_eligible
    )
    rp_inherited_weakness = _masked_mean(
        np.maximum(-rp_z[:, 1:], 0.0), rr_eligible
    )
    rp_weakness = _masked_mean(
        np.maximum(-rp_z, 0.0), rp_eligible
    )
    rr_evidence = np.maximum(-rr_z, 0.0)
    rr_path_deficit = _masked_mean(rr_evidence, rr_eligible)
    graph_evidence = np.mean(
        np.stack((innovation, rp_weakness), axis=1), axis=1
    ).astype(np.float32)
    return {
        "token_only": token,
        "token_graph": np.mean(
            np.stack((token, innovation, rp_weakness), axis=1), axis=1
        ).astype(np.float32),
        "no_rp": np.mean(
            np.stack((base_no_rp, no_rp_innovation, np.zeros_like(token)), axis=1), axis=1
        ).astype(np.float32),
        "no_rr": np.mean(
            np.stack((base_no_rr, np.zeros_like(token), rp_direct_weakness), axis=1), axis=1
        ).astype(np.float32),
        "graph_innovation": innovation.astype(np.float32),
        "graph_evidence": graph_evidence,
        "rp_weakness": rp_weakness,
        "rp_direct_weakness": rp_direct_weakness,
        "rp_inherited_weakness": rp_inherited_weakness,
        "rr_path_deficit": rr_path_deficit,
        "innovation_by_hop": innovation_by_hop.astype(np.float32),
        "innovation_reliability": reliability,
        "rp_path_z": rp_z,
        "rr_path_z": rr_z,
    }


def _representation(base_z, graph, rp_z, rr_z):
    parts = [base_z]
    names = [f"base:{name}" for name in EXACT_FEATURES]
    for hop in range(graph["raw_message"].shape[1]):
        parts.extend((
            graph["raw_message"][:, hop],
            graph["conditional_neighbor"][:, hop],
            graph["self_neighbor_residual"][:, hop],
        ))
        names.extend(f"hop{hop + 1}:raw:{name}" for name in EXACT_FEATURES)
        names.extend(f"hop{hop + 1}:neighbor:{name}" for name in EXACT_FEATURES)
        names.extend(f"hop{hop + 1}:residual:{name}" for name in EXACT_FEATURES)
    parts.extend((
        rr_z,
        rp_z,
        np.log1p(graph["rr_hop_reach_count"].astype(np.float32)),
    ))
    names.extend(f"rr_path_mass_z:hop{hop + 1}" for hop in range(rr_z.shape[1]))
    names.extend(f"rp_path_mass_z:hop{hop}" for hop in range(rp_z.shape[1]))
    names.extend(f"rr_reach:hop{hop + 1}" for hop in range(graph["rr_hop_reach_count"].shape[1]))
    return np.concatenate(parts, axis=1).astype(np.float32), tuple(names)


def _visual_coordinates(
    train_representation, test_representation, seed, reference_size
):
    rng = np.random.default_rng(seed)
    if len(train_representation) > int(reference_size):
        reference_ids = np.sort(
            rng.choice(len(train_representation), int(reference_size), replace=False)
        )
    else:
        reference_ids = np.arange(len(train_representation))
    reference = train_representation[reference_ids]
    scaler = _RobustScaler().fit(reference)
    train = scaler.transform(reference)
    test = scaler.transform(test_representation)
    active = train.std(0) > 1e-8
    if not bool(active.any()):
        return np.zeros((len(test), 2), dtype=np.float32), {
            "type": "constant_zero_coordinates", "fit_uses_labels": False
        }
    count = min(2, int(active.sum()), max(1, len(train) - 1))
    pca = PCA(n_components=count, random_state=seed).fit(train[:, active])
    coordinates = np.zeros((len(test), 2), dtype=np.float32)
    coordinates[:, :count] = pca.transform(test[:, active]).astype(np.float32)
    return coordinates, {
        "type": "visualization_only_train_robust_pca", "components": count,
        "train_reference_nodes": int(len(reference_ids)),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "changes_detector_score": False, "fit_uses_labels": False,
    }


def _ranking(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    result = {"n": int(len(labels)), "positives": int(labels.sum()),
              "prevalence": float(labels.mean()) if len(labels) else None}
    if len(np.unique(labels)) < 2:
        return {**result, "auroc": None, "auprc": None}
    return {
        **result, "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "correct_median": float(np.median(scores[labels == 0])),
        "hallucination_median": float(np.median(scores[labels == 1])),
    }


def _feature_separation(labels, values, fixed_direction=None):
    labels = np.asarray(labels, dtype=np.int8)
    values = np.asarray(values, dtype=np.float64)
    base = {
        "n": int(len(labels)), "positives": int(labels.sum()),
        "used_to_fit_representation": False,
    }
    if len(np.unique(labels)) < 2:
        return {
            **base,
            "raw_auroc_higher_is_anomaly": None,
            "separability": None,
            "post_hoc_direction": None,
            "post_hoc_oriented_auprc": None,
            "correct_median": (
                float(np.median(values[labels == 0])) if np.any(labels == 0) else None
            ),
            "hallucination_median": (
                float(np.median(values[labels == 1])) if np.any(labels == 1) else None
            ),
        }
    raw_auc = float(roc_auc_score(labels, values))
    correct = float(np.median(values[labels == 0]))
    hallucination = float(np.median(values[labels == 1]))
    if raw_auc >= .5:
        direction, oriented = "higher_for_hallucination", values
    else:
        direction, oriented = "lower_for_hallucination", -values
    result = {
        "raw_auroc_higher_is_anomaly": raw_auc,
        "separability": max(raw_auc, 1.0 - raw_auc),
        "post_hoc_direction": direction,
        "post_hoc_oriented_auprc": float(average_precision_score(labels, oriented)),
        "correct_median": correct, "hallucination_median": hallucination,
        **base,
    }
    if fixed_direction is not None:
        frozen = values * float(fixed_direction)
        result["fixed_direction"] = (
            "higher" if float(fixed_direction) > 0 else "lower"
        )
        result["fixed_direction_auroc"] = float(roc_auc_score(labels, frozen))
        result["fixed_direction_auprc"] = float(
            average_precision_score(labels, frozen)
        )
    return result


def _grouped_ranking(labels, scores, metadata, field):
    groups = metadata[field].astype(str)
    return {
        group: _ranking(labels[groups == group], scores[groups == group])
        for group in sorted(set(groups))
    }


def _metric_delta(left, right, metric):
    if left.get(metric) is None or right.get(metric) is None:
        return None
    return float(left[metric] - right[metric])


def _response_feature_evaluation(labels, exact, metadata):
    sample_ids = metadata["sample_id"].astype(str)
    rows = {f"{summary}_{name}": [] for name in EXACT_FEATURES
            for summary in ("mean", "std")}
    response_labels = []
    for sample_id in dict.fromkeys(sample_ids):
        selected = sample_ids == sample_id
        response_labels.append(int(labels[selected].max()))
        for index, name in enumerate(EXACT_FEATURES):
            rows[f"mean_{name}"].append(float(exact[selected, index].mean()))
            rows[f"std_{name}"].append(float(exact[selected, index].std()))
    response_labels = np.asarray(response_labels, dtype=np.int8)
    return {
        name: _feature_separation(response_labels, values)
        for name, values in rows.items()
    }


def _read_labels(evaluation_dataset, metadata):
    for sample_id in tqdm(
        evaluation_dataset.sample_ids, desc="[6/7] load sealed labels", unit="sample"
    ):
        sample = evaluation_dataset[sample_id]
        sample.attention()
        sample.release_attention()
    store, rows = evaluation_dataset.labels(), []
    for sample_id in evaluation_dataset.sample_ids:
        sample = evaluation_dataset[sample_id]
        rows.extend(store.response_labels(sample).cpu().tolist())
        sample.release_attention()
    labels = np.asarray(rows, dtype=np.int8)
    if len(labels) != len(metadata["sample_id"]):
        raise ValueError("evaluation labels do not align with frozen token rows")
    return labels


def _sample_selection(config, metadata, coordinates):
    available = set(metadata["sample_id"].astype(str))
    if config.sample_ids:
        requested = list(dict.fromkeys(map(str, config.sample_ids)))
        missing = [value for value in requested if value not in available]
        if missing:
            raise ValueError(f"sample IDs are absent from test split: {missing}")
        return requested, "user_requested_before_labels"
    candidates = []
    for sample_id in dict.fromkeys(metadata["sample_id"].astype(str)):
        selected = metadata["sample_id"].astype(str) == sample_id
        values = coordinates[selected]
        dispersion = float(np.linalg.norm(values - values.mean(0), axis=1).mean())
        candidates.append((dispersion, len(values), sample_id))
    return [max(candidates)[2]], "label_free_max_representation_dispersion"


def _safe_filename(value):
    value = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))
    return (value.strip("._") or "sample")[:120]


def _display_edges(graph, config):
    target = graph["edge_index"][1]
    chosen = []
    for node in range(graph["response_idx"], len(graph["token_ids"])):
        incoming = np.flatnonzero(target == node)
        for relation in (RP, RR):
            ids = incoming[graph["edge_type"][incoming] == relation]
            if not len(ids):
                continue
            ranked = ids[np.argsort(-graph["edge_score"][ids], kind="stable")]
            mass = graph["edge_score"][ranked]
            reached = np.flatnonzero(np.cumsum(mass) >= config.display_mass_cover * mass.sum())
            count = int(reached[0]) + 1 if len(reached) else len(ranked)
            chosen.extend(ranked[:min(count, config.display_edges_per_type)].tolist())
    chosen = np.asarray(sorted(set(chosen)), dtype=np.int64)
    if len(chosen) > config.display_max_edges:
        order = np.argsort(-graph["edge_score"][chosen], kind="stable")
        chosen = np.sort(chosen[order[:config.display_max_edges]])
    return chosen


def _effective_relations(graph, hops, per_target=1):
    response_idx = int(graph["response_idx"])
    response_count = len(graph["token_ids"]) - response_idx
    rr = np.zeros((response_count, response_count), dtype=np.float64)
    rp = np.zeros((response_count, response_idx), dtype=np.float64)
    source, target = graph["edge_index"]
    target = target - response_idx
    for relation, matrix, offset in ((RR, rr, response_idx), (RP, rp, 0)):
        selected = graph["edge_type"] == relation
        np.add.at(matrix, (target[selected], source[selected] - offset),
                  graph["edge_score"][selected].astype(np.float64))
    rr_rows, rp_rows, power = [], [], rr.copy()
    for hop in range(2, int(hops) + 1):
        power = rr @ power
        for row in range(response_count):
            ids = np.flatnonzero((power[row] > 1e-12) & (rr[row] == 0))
            ids = ids[np.argsort(-power[row, ids], kind="stable")]
            rr_rows.extend((int(i), row, hop, float(power[row, i])) for i in ids[:per_target])
    inherited = rr @ rp
    for rr_hops in range(1, int(hops) + 1):
        for row in range(response_count):
            ids = np.flatnonzero((inherited[row] > 1e-12) & (rp[row] == 0))
            ids = ids[np.argsort(-inherited[row, ids], kind="stable")]
            rp_rows.extend((int(i), row, rr_hops + 1, float(inherited[row, i])) for i in ids[:per_target])
        inherited = rr @ inherited
    return rr_rows, rp_rows


def _save_sample_graphs(output, graphs, exact, base_z, representation, coordinates, scores, messages):
    directory = output / "sample_graphs"
    directory.mkdir(parents=True, exist_ok=False)
    paths = {}
    for graph in tqdm(graphs, desc="[5/7] save every sample graph", unit="sample"):
        start, end = graph["start"], graph["end"]
        path = directory / f"sample_{_safe_filename(graph['sample_id'])}.npz"
        np.savez_compressed(
            path, schema=np.asarray(SCHEMA), labels_included=np.asarray(False),
            sample_id=np.asarray(graph["sample_id"]), token_ids=graph["token_ids"],
            response_idx=np.asarray(graph["response_idx"], dtype=np.int32),
            edge_index=graph["edge_index"], edge_type=graph["edge_type"], edge_score=graph["edge_score"],
            exact_feature_names=np.asarray(EXACT_FEATURES), exact_token_features=exact[start:end],
            no_rp_exact_token_features=messages["no_rp_exact"][start:end],
            no_rr_exact_token_features=messages["no_rr_exact"][start:end],
            position_adjusted_features=base_z[start:end],
            token_graph_representation=representation[start:end],
            visualization_coordinates=coordinates[start:end],
            mechanism_coordinates=np.column_stack((
                scores["token_only"][start:end],
                scores["graph_evidence"][start:end],
            )).astype(np.float32),
            rr_raw_message=messages["raw_message"][start:end],
            rr_conditional_neighbor=messages["conditional_neighbor"][start:end],
            self_neighbor_residual=messages["self_neighbor_residual"][start:end],
            no_rp_self_neighbor_residual=messages[
                "no_rp_self_neighbor_residual"
            ][start:end],
            rr_path_mass=messages["rr_path_mass"][start:end],
            rp_path_mass=messages["rp_path_mass"][start:end],
            rr_hop_reach_count=messages["rr_hop_reach_count"][start:end],
            **{f"{name}_score": scores[name][start:end] for name in VIEWS},
            graph_innovation_score=scores["graph_innovation"][start:end],
            graph_evidence_score=scores["graph_evidence"][start:end],
            rp_weakness_score=scores["rp_weakness"][start:end],
            rp_direct_weakness_score=scores["rp_direct_weakness"][start:end],
            rp_inherited_weakness_score=scores[
                "rp_inherited_weakness"
            ][start:end],
            rr_path_deficit_diagnostic=scores["rr_path_deficit"][start:end],
            innovation_reliability=scores["innovation_reliability"][start:end],
        )
        paths[graph["sample_id"]] = path
    return paths


def _render_population(output, coordinates, scores, labels):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for label, color, name, size, alpha in (
        (0, "#2ca02c", "correct", 4, .12),
        (1, "#d62728", "hallucination", 10, .65),
    ):
        selected = labels == label
        axes[0, 0].scatter(
            scores["token_only"][selected], scores["graph_evidence"][selected],
            c=color, s=size, alpha=alpha, label=name, rasterized=True,
        )
        axes[0, 1].scatter(
            coordinates[selected, 0], coordinates[selected, 1],
            c=color, s=size, alpha=alpha, label=name, rasterized=True,
        )
    axes[0, 0].set(title="Interpretable mechanism map",
                   xlabel="fixed directional token score",
                   ylabel="RR/RP multi-hop graph evidence")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(title="Visualization-only train PCA of graph vectors",
                   xlabel="component 1", ylabel="component 2")
    axes[0, 1].legend(frameon=False)
    for axis, view in zip(axes[1], ("token_only", "token_graph")):
        for label, color, name in ((0, "#2ca02c", "correct"), (1, "#d62728", "hallucination")):
            axis.hist(scores[view][labels == label], bins=60, density=True,
                      alpha=.55, color=color, label=name)
        metric = _ranking(labels, scores[view])
        auc = "N/A" if metric["auroc"] is None else f"{metric['auroc']:.3f}"
        axis.set(title=f"{view}: AUROC={auc}",
                 xlabel="frozen score", ylabel="density")
        axis.legend(frameon=False)
    path = output / "population_token_representations.png"
    figure.savefig(path, dpi=240)
    plt.close(figure)
    return path


def _render_sample(output, graph, mechanism_coordinates, exact_z, labels, config):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected = _display_edges(graph, config)
    edge_index = graph["edge_index"][:, selected]
    edge_type = graph["edge_type"][selected]
    edge_score = graph["edge_score"][selected]
    response_idx = int(graph["response_idx"])
    count = len(mechanism_coordinates)
    response_nodes = np.arange(response_idx, response_idx + count)
    prompt_nodes = np.unique(edge_index[0, edge_index[0] < response_idx])
    colors = np.where(labels == 1, "#d62728", "#2ca02c")
    width = max(22, min(36, 18 + count * .12))
    figure, axes = plt.subplots(1, 4, figsize=(width, 6), constrained_layout=True)
    position = {
        **{int(node): (float(node), 1.0) for node in prompt_nodes},
        **{int(node): (float(node), 0.0) for node in response_nodes},
    }
    maximum = max(float(edge_score.max()) if len(edge_score) else 0.0, 1e-12)
    for edge, relation, weight in zip(edge_index.T, edge_type, edge_score):
        source, target = map(int, edge)
        if source not in position:
            continue
        axes[0].annotate("", xy=position[target], xytext=position[source], arrowprops={
            "arrowstyle": "->", "color": "#1f77b4" if relation == RP else "#777777",
            "alpha": .15 + .55 * float(weight / maximum),
            "lw": .3 + 1.5 * float(weight / maximum), "connectionstyle": "arc3,rad=.08",
        })
    if len(prompt_nodes):
        axes[0].scatter(prompt_nodes, np.ones(len(prompt_nodes)), marker="s", s=22, c="#4c78a8")
    axes[0].scatter(response_nodes, np.zeros(count), c=colors, s=38,
                    edgecolors="black", linewidths=.25)
    axes[0].set(title="Direct typed attention graph", xlabel="absolute token position",
                yticks=(0, 1), yticklabels=("response", "prompt"))

    rr_rows, rp_rows = _effective_relations(graph, config.diffusion_hops)
    axes[1].scatter(response_nodes, np.zeros(count), c=colors, s=38,
                    edgecolors="black", linewidths=.25)
    for source, target, hop, _ in rr_rows:
        axes[1].annotate("", xy=(target + response_idx, 0),
                         xytext=(source + response_idx, 0), arrowprops={
            "arrowstyle": "->", "linestyle": "--", "color": "#9467bd",
            "alpha": .4, "lw": .7, "connectionstyle": f"arc3,rad={.08 + .02 * hop}"})
    for source, target, hop, _ in rp_rows:
        axes[1].annotate("", xy=(target + response_idx, 0), xytext=(source, 1), arrowprops={
            "arrowstyle": "->", "linestyle": ":", "color": "#1f77b4",
            "alpha": .4, "lw": .7, "connectionstyle": f"arc3,rad={.08 + .02 * hop}"})
    axes[1].set(title="Non-adjacent raw path influence", xlabel="absolute token position",
                yticks=(0, 1), yticklabels=("response", "inherited prompt"))

    for source, target in edge_index.T[edge_type == RR]:
        source_index, target_index = int(source) - response_idx, int(target) - response_idx
        axes[2].plot(mechanism_coordinates[[source_index, target_index], 0],
                     mechanism_coordinates[[source_index, target_index], 1],
                     color="#777777", alpha=.15, lw=.6)
    axes[2].scatter(mechanism_coordinates[:, 0], mechanism_coordinates[:, 1], c=colors, s=46,
                    edgecolors="black", linewidths=.3)
    for index, point in enumerate(mechanism_coordinates):
        axes[2].text(point[0], point[1], str(index), fontsize=5, ha="center", va="bottom")
    axes[2].set(title="Frozen mechanism-space token graph",
                xlabel="token mechanism evidence", ylabel="multi-hop graph evidence")
    axes[2].legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="#2ca02c", label="correct"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#d62728", label="hallucination"),
        Line2D([], [], color="#777777", label="RR edge"),
    ], frameon=False)

    heat = exact_z[:, SCORE_INDEX].T * SCORE_DIRECTION[:, None]
    limit = max(float(np.quantile(np.abs(heat), .98)), 1e-6)
    image = axes[3].imshow(heat, aspect="auto", cmap="coolwarm", vmin=-limit,
                           vmax=limit, interpolation="nearest")
    axes[3].set(title="Signed exact mechanisms", xlabel="response token index",
                yticks=np.arange(len(SCORE_FEATURES)), yticklabels=SCORE_FEATURES)
    figure.colorbar(image, ax=axes[3], label="positive = hypothesis-consistent anomaly")
    path = output / f"sample_{_safe_filename(graph['sample_id'])}_token_graph.png"
    figure.suptitle(f"Sample {graph['sample_id']}; labels only color frozen nodes")
    figure.savefig(path, dpi=240)
    plt.close(figure)
    return path, len(selected), len(rr_rows), len(rp_rows)


def _geometry(dataset):
    return {name: dataset.manifest.get(name) for name in (
        "schema", "num_layers", "num_heads", "alignment",
        "attention_floor", "observer_model",
    )}


def discover_token_representations(
    train_dataset, test_dataset, evaluation_dataset, *, output_dir, config=None
):
    """Freeze exact scalars, propagation and scores before opening labels."""
    config = TokenRepresentationConfig() if config is None else config
    config.validate()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("token representation output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    if _geometry(train_dataset) != _geometry(test_dataset):
        raise ValueError("train and test attention geometry differ")
    if list(map(str, test_dataset.sample_ids)) != list(map(str, evaluation_dataset.sample_ids)):
        raise ValueError("evaluation dataset does not match ordered test sample IDs")

    train_exact, train_meta = _extract_exact_split(train_dataset, config, "train")
    test_exact, metadata = _extract_exact_split(test_dataset, config, "test")
    if set(train_meta["source_id"].astype(str)) & set(metadata["source_id"].astype(str)):
        raise ValueError("train and test source groups overlap")
    base_scaler = _PositionRobustScaler(config.position_bins).fit(
        train_exact, train_meta["relative_position"]
    )
    train_z = base_scaler.transform(train_exact, train_meta["relative_position"])
    test_z = base_scaler.transform(test_exact, metadata["relative_position"])
    train_graph, _ = _extract_graph_split(
        train_dataset, train_z, base_scaler, train_meta["relative_position"],
        config, "train", keep_graphs=False
    )
    test_graph, graphs = _extract_graph_split(
        test_dataset, test_z, base_scaler, metadata["relative_position"],
        config, "test", keep_graphs=True
    )

    print("[3/7] fitting train-only position calibration and frozen scores", flush=True)
    train_no_rp_z = base_scaler.transform(
        train_graph["no_rp_exact"], train_meta["relative_position"]
    )
    train_no_rr_z = base_scaler.transform(
        train_graph["no_rr_exact"], train_meta["relative_position"]
    )
    test_no_rp_z = base_scaler.transform(
        test_graph["no_rp_exact"], metadata["relative_position"]
    )
    test_no_rr_z = base_scaler.transform(
        test_graph["no_rr_exact"], metadata["relative_position"]
    )
    train_rp_eligible, train_rr_eligible = _path_eligibility(
        train_meta["token_index"], config.diffusion_hops
    )
    test_rp_eligible, test_rr_eligible = _path_eligibility(
        metadata["token_index"], config.diffusion_hops
    )
    rp_scaler = _PositionRobustScaler(config.position_bins).fit(
        np.log1p(np.maximum(train_graph["rp_path_mass"], 0.0)),
        train_meta["relative_position"],
        train_rp_eligible,
    )
    rr_scaler = _PositionRobustScaler(config.position_bins).fit(
        np.log1p(np.maximum(train_graph["rr_path_mass"], 0.0)),
        train_meta["relative_position"],
        train_rr_eligible,
    )
    rr_reliability = _PositivePathReliability(config.position_bins).fit(
        train_graph["rr_path_mass"], train_meta["relative_position"],
        train_rr_eligible,
    )
    train_scores = _build_scores(
        train_z, train_exact,
        train_no_rp_z, train_graph["no_rp_exact"],
        train_no_rr_z, train_graph["no_rr_exact"],
        train_meta["relative_position"],
        train_rp_eligible, train_rr_eligible,
        train_graph, rp_scaler, rr_scaler, rr_reliability
    )
    scores = _build_scores(
        test_z, test_exact,
        test_no_rp_z, test_graph["no_rp_exact"],
        test_no_rr_z, test_graph["no_rr_exact"],
        metadata["relative_position"],
        test_rp_eligible, test_rr_eligible,
        test_graph, rp_scaler, rr_scaler, rr_reliability
    )
    train_representation, representation_names = _representation(
        train_z, train_graph, train_scores["rp_path_z"], train_scores["rr_path_z"]
    )
    representation, test_names = _representation(
        test_z, test_graph, scores["rp_path_z"], scores["rr_path_z"]
    )
    if representation_names != test_names:
        raise ValueError("train and test representation schemas differ")
    coordinates, projection = _visual_coordinates(
        train_representation, representation, config.seed,
        config.visual_reference_size,
    )
    mechanism_coordinates = np.column_stack((
        scores["token_only"], scores["graph_evidence"]
    )).astype(np.float32)
    selected_samples, selection_rule = _sample_selection(
        config, metadata, mechanism_coordinates
    )

    print("[4/7] freezing label-free representations and scores", flush=True)
    np.savez_compressed(
        output / "token_representations_label_free.npz",
        schema=np.asarray(SCHEMA), labels_included=np.asarray(False),
        exact_feature_names=np.asarray(EXACT_FEATURES),
        score_feature_names=np.asarray(SCORE_FEATURES),
        score_feature_direction=SCORE_DIRECTION,
        representation_feature_names=np.asarray(representation_names),
        exact_token_features=test_exact, position_adjusted_features=test_z,
        no_rp_exact_token_features=test_graph["no_rp_exact"],
        no_rr_exact_token_features=test_graph["no_rr_exact"],
        rr_path_mass=test_graph["rr_path_mass"],
        rp_path_mass=test_graph["rp_path_mass"],
        rr_hop_reach_count=test_graph["rr_hop_reach_count"],
        innovation_reliability=scores["innovation_reliability"],
        token_graph_representation=representation,
        visualization_coordinates=coordinates,
        mechanism_coordinates=mechanism_coordinates,
        sample_id=metadata["sample_id"], source_id=metadata["source_id"],
        token_index=metadata["token_index"], token_id=metadata["token_id"],
        relative_position=metadata["relative_position"],
        task_type=metadata["task_type"], data_source=metadata["data_source"],
        generator_model=metadata["generator_model"],
        **{f"{name}_score": scores[name] for name in VIEWS},
        graph_innovation_score=scores["graph_innovation"],
        graph_evidence_score=scores["graph_evidence"],
        rp_weakness_score=scores["rp_weakness"],
        rp_direct_weakness_score=scores["rp_direct_weakness"],
        rp_inherited_weakness_score=scores["rp_inherited_weakness"],
        rr_path_deficit_diagnostic=scores["rr_path_deficit"],
    )
    graph_paths = _save_sample_graphs(
        output, graphs, test_exact, test_z, representation,
        coordinates, scores, test_graph
    )
    label_free_report = {
        "schema": SCHEMA, "labels_read": False,
        "exact_feature_names": list(EXACT_FEATURES),
        "exact_scalar_block_recoverable_without_projection": True,
        "fixed_score_hypothesis": FEATURE_DIRECTIONS,
        "score_formula": "mean positive signed train-position-MAD deviation",
        "base_scaler": base_scaler.report(),
        "path_mass_scalers": {
            "rp": rp_scaler.report(), "rr": rr_scaler.report(),
            "rr_innovation_reliability": rr_reliability.report(),
        },
        "graph_propagation": {
            "trainable": False, "rr_matrix_row_normalized": False,
            "raw_recurrence": "M_k = A_RR @ M_(k-1)",
            "path_mass": "q_k = A_RR^k @ 1",
            "neighbor_mean": "mu_k = M_k / q_k; raw M_k and q_k both retained",
            "innovation": "delta_k = base_z - mu_k",
            "innovation_reliability": "q_k/(q_k+q0), where q0 is the train-positive position/hop median",
            "prompt_mass": "p_0 = A_RP @ 1; p_k = A_RR @ p_(k-1)",
            "diffusion_hops": config.diffusion_hops,
            "hop_eligibility": "response token index t is eligible for RR hop k iff t >= k",
            "structurally_ineligible_hops_contribute_to_score": False,
        },
        "view_components": {
            "token_only": ["fixed base mechanism"],
            "token_graph": ["fixed base mechanism", "reliability-gated RR innovation", "RP path weakness"],
            "no_rp": ["base recomputed after deleting RP edges", "recomputed reliability-gated RR innovation"],
            "no_rr": ["base recomputed after deleting RR edges", "direct RP weakness only"],
        },
        "ablation_semantics": "delete relation edges, recompute exact base features, keep the same train-fitted calibrators and scoring formula",
        "protocol_status": (
            "exploratory_same_benchmark_hypotheses; confirm on an independent "
            "held-out test after freezing directions and hop count"
        ),
        "visual_projection": projection,
        "labels_read_during_fit": False,
        "train_tokens": int(len(train_exact)), "test_tokens": int(len(test_exact)),
        "sample_selection": {
            "sample_ids": selected_samples, "rule": selection_rule, "labels_used": False
        },
        "config": asdict(config),
    }
    (output / "label_free_report.json").write_text(
        json.dumps(label_free_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("[6/7] opening labels only after artifacts and scores are frozen", flush=True)
    labels = _read_labels(evaluation_dataset, metadata)
    view_metrics = {view: _ranking(labels, scores[view]) for view in VIEWS}
    exact_metrics = {
        name: _feature_separation(
            labels, test_exact[:, index], FEATURE_DIRECTIONS.get(name)
        )
        for index, name in enumerate(EXACT_FEATURES)
    }
    response_exact_metrics = _response_feature_evaluation(
        labels, test_exact, metadata
    )
    report = {
        **label_free_report, "labels_read": True,
        "labels_read_during": "evaluation_and_plot_coloring_only",
        "exact_feature_evaluation": exact_metrics,
        "response_exact_feature_evaluation": response_exact_metrics,
        "views": {
            view: {
                "evaluation": view_metrics[view],
                "by_task_type": _grouped_ranking(
                    labels, scores[view], metadata, "task_type"
                ),
                "by_data_source": _grouped_ranking(
                    labels, scores[view], metadata, "data_source"
                ),
            }
            for view in VIEWS
        },
        "score_components": {
            "combined_graph_evidence": _ranking(labels, scores["graph_evidence"]),
            "rr_innovation": _ranking(labels, scores["graph_innovation"]),
            "rp_path_weakness": _ranking(labels, scores["rp_weakness"]),
            "rp_direct_weakness": _ranking(labels, scores["rp_direct_weakness"]),
            "rp_inherited_weakness": _ranking(labels, scores["rp_inherited_weakness"]),
            "rr_path_deficit_diagnostic_only": _ranking(
                labels, scores["rr_path_deficit"]
            ),
        },
        "graph_gain_over_token_only": {
            metric: _metric_delta(
                view_metrics["token_graph"], view_metrics["token_only"], metric
            )
            for metric in ("auroc", "auprc")
        },
        "relation_ablation": {
            "full_minus_no_rp": {
                metric: _metric_delta(
                    view_metrics["token_graph"], view_metrics["no_rp"], metric
                )
                for metric in ("auroc", "auprc")
            },
            "full_minus_no_rr": {
                metric: _metric_delta(
                    view_metrics["token_graph"], view_metrics["no_rr"], metric
                )
                for metric in ("auroc", "auprc")
            },
        },
        "warning": "separability=max(raw_auc,1-raw_auc) is diagnostic, not raw AUROC",
    }
    report_path = output / "token_representation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("[7/7] rendering label-colored diagnostics", flush=True)
    population = _render_population(output, coordinates, scores, labels)
    graph_by_id = {graph["sample_id"]: graph for graph in graphs}
    sample_rows = []
    for sample_id in selected_samples:
        graph = graph_by_id[sample_id]
        start, end = graph["start"], graph["end"]
        figure, edges, rr_effective, rp_effective = _render_sample(
            output, graph, mechanism_coordinates[start:end], test_z[start:end],
            labels[start:end], config,
        )
        sample_rows.append({
            "sample_id": sample_id, "selection_rule": selection_rule,
            "response_nodes": int(end - start),
            "hallucination_tokens": int(labels[start:end].sum()),
            "display_edges": int(edges),
            "display_nonadjacent_rr_relations": int(rr_effective),
            "display_inherited_rp_relations": int(rp_effective),
            "figure": str(figure), "label_free_data": str(graph_paths[sample_id]),
        })
    report["population_figure"] = str(population)
    report["sample_visualizations"] = sample_rows
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ranked_exact = {
        name: row["separability"]
        for name, row in exact_metrics.items()
        if row["separability"] is not None
    }
    return {
        "output_dir": str(output), "report": str(report_path),
        "label_free_embeddings": str(output / "token_representations_label_free.npz"),
        "sample_graph_directory": str(output / "sample_graphs"),
        "population_figure": str(population), "sample_visualizations": sample_rows,
        "test_nodes": int(len(test_exact)), "view_metrics": view_metrics,
        "best_exact_feature_by_separability": (
            max(ranked_exact, key=ranked_exact.get) if ranked_exact else None
        ),
    }
