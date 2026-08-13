"""Label-blind token representations from sparse layer-head attention.

The primary representation is not a scalar Lookback score.  Every response
token first receives a ``[layer, head, mechanism]`` tensor.  A train-only
robust PCA keeps informative layer/head directions without using hallucination
labels.  Exact RP/RR endpoints are then added by fixed (non-neural) message
passing and evaluated as an explicit ablation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from .graph import GraphBuildConfig, RP, RR, build_attention_graph


SCHEMA = "attention-mechanism-token-representation-v1"
MECHANISMS = (
    "routing_balance",
    "effective_support_fraction",
    "dominant_edge_strength",
    "response_locality",
)
VIEWS = ("token_only", "token_graph", "no_rp", "no_rr")


@dataclass(frozen=True)
class TokenRepresentationConfig:
    base_dim: int = 32
    embedding_dim: int = 32
    source_sketch_dim: int = 16
    fit_reference_size: int = 30_000
    detector_reference_size: int = 100_000
    prototypes: int = 256
    diffusion_hops: int = 3
    csr_row_block: int = 4096
    display_mass_cover: float = 0.80
    display_edges_per_type: int = 2
    sample_ids: tuple[str, ...] = ()
    seed: int = 42

    def validate(self):
        positive = (
            self.base_dim,
            self.embedding_dim,
            self.source_sketch_dim,
            self.fit_reference_size,
            self.detector_reference_size,
            self.prototypes,
            self.diffusion_hops,
            self.csr_row_block,
            self.display_edges_per_type,
        )
        if any(int(value) < 1 for value in positive):
            raise ValueError("representation dimensions and limits must be positive")
        if self.source_sketch_dim % 2:
            raise ValueError("source_sketch_dim must be even")
        if self.diffusion_hops < 2:
            raise ValueError("diffusion_hops must be at least two to model non-adjacent nodes")
        if not 0.0 < float(self.display_mass_cover) <= 1.0:
            raise ValueError("display_mass_cover must be in (0,1]")


class _ReferenceSampler:
    """Bounded random reference with vectorized random-priority sampling."""

    def __init__(self, capacity, seed):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.values = None
        self.keys = np.empty(0, dtype=np.float64)

    def add(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or not len(values):
            return
        keys = self.rng.random(len(values))
        if len(values) > self.capacity:
            keep = np.argpartition(keys, self.capacity - 1)[: self.capacity]
            values, keys = values[keep], keys[keep]
        self.values = values.copy() if self.values is None else np.concatenate((self.values, values))
        self.keys = np.concatenate((self.keys, keys))
        if len(self.keys) > 2 * self.capacity:
            self._prune()

    def _prune(self):
        if len(self.keys) <= self.capacity:
            return
        keep = np.argpartition(self.keys, self.capacity - 1)[: self.capacity]
        order = np.argsort(self.keys[keep])
        keep = keep[order]
        self.values = self.values[keep]
        self.keys = self.keys[keep]

    def get(self):
        self._prune()
        if self.values is None or not len(self.values):
            raise ValueError("cannot fit a representation on an empty reference")
        return self.values


class _RobustProjector:
    """Median/MAD scaling and PCA fitted on train tokens only."""

    def __init__(self, output_dim, seed):
        self.output_dim = int(output_dim)
        self.seed = int(seed)
        self.center = None
        self.scale = None
        self.active = None
        self.pca = None
        self.kept_components = None

    def fit(self, reference):
        reference = np.asarray(reference, dtype=np.float64)
        if reference.ndim != 2 or not np.isfinite(reference).all():
            raise ValueError("projector reference must be a finite matrix")
        self.center = np.median(reference, axis=0)
        mad = 1.4826 * np.median(np.abs(reference - self.center), axis=0)
        std = reference.std(axis=0)
        raw_scale = np.where(mad > 1e-8, mad, std)
        self.active = raw_scale > 1e-8
        if not bool(self.active.any()):
            raise ValueError("all representation dimensions are constant on train")
        self.scale = np.where(self.active, raw_scale, 1.0)
        scaled = (reference[:, self.active] - self.center[self.active]) / self.scale[self.active]
        count = min(self.output_dim, scaled.shape[1], max(1, scaled.shape[0] - 1))
        self.pca = PCA(n_components=count, whiten=True, svd_solver="auto", random_state=self.seed)
        self.pca.fit(scaled)
        variance = np.asarray(self.pca.explained_variance_, dtype=np.float64)
        self.kept_components = np.isfinite(variance) & (variance > 1e-10)
        if not bool(self.kept_components.any()):
            raise ValueError("train reference has no non-degenerate PCA component")
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        if self.pca is None or values.ndim != 2 or values.shape[1] != len(self.center):
            raise ValueError("projector is unfitted or received the wrong input dimension")
        scaled = (values[:, self.active] - self.center[self.active]) / self.scale[self.active]
        output = self.pca.transform(scaled)[:, self.kept_components]
        return np.nan_to_num(
            output, copy=False, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

    def report(self):
        return {
            "input_dimensions": int(len(self.center)),
            "active_dimensions": int(self.active.sum()),
            "output_dimensions": int(self.kept_components.sum()),
            "explained_variance_ratio": self.pca.explained_variance_ratio_[
                self.kept_components
            ].tolist(),
            "fit_uses_labels": False,
        }

    def structured_loading_report(self, num_layers, num_heads, mechanisms):
        expected = int(num_layers) * int(num_heads) * int(mechanisms)
        if self.pca is None or len(self.center) != expected:
            raise ValueError("projector inputs do not match the requested layer-head shape")
        full = np.zeros((int(self.kept_components.sum()), expected), dtype=np.float64)
        full[:, self.active] = self.pca.components_[self.kept_components]
        weight = self.pca.explained_variance_ratio_[self.kept_components, None]
        importance = np.square(full) * weight
        importance = importance.sum(0).reshape(num_layers, num_heads, mechanisms)
        total = max(float(importance.sum()), 1e-12)
        return {
            "mechanism_fraction": (importance.sum((0, 1)) / total).tolist(),
            "layer_fraction": (importance.sum((1, 2)) / total).tolist(),
            "head_fraction": (importance.sum((0, 2)) / total).tolist(),
            "computed_without_labels": True,
        }


class _PrototypeDetector:
    """Fast train-only local-scale prototype novelty detector."""

    def __init__(self, prototypes, reference_size, seed):
        self.prototypes = int(prototypes)
        self.reference_size = int(reference_size)
        self.seed = int(seed)
        self.model = None
        self.scale = None
        self.fit_count = 0

    def fit(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("detector training values must be a non-empty finite matrix")
        rng = np.random.default_rng(self.seed)
        if len(values) > self.reference_size:
            ids = np.sort(rng.choice(len(values), self.reference_size, replace=False))
            reference = values[ids]
        else:
            reference = values
        unique = len(np.unique(reference, axis=0))
        clusters = min(self.prototypes, len(reference), unique)
        if clusters < 1:
            raise ValueError("detector reference has no usable rows")
        self.model = MiniBatchKMeans(
            n_clusters=clusters,
            batch_size=min(4096, max(256, len(reference))),
            n_init=5,
            random_state=self.seed,
        ).fit(reference)
        assignment = self.model.predict(reference)
        distance = np.linalg.norm(reference - self.model.cluster_centers_[assignment], axis=1)
        global_scale = max(float(np.median(distance)), 1e-6)
        scale = np.full(clusters, global_scale, dtype=np.float32)
        for cluster in range(clusters):
            selected = distance[assignment == cluster]
            if len(selected) >= 4:
                scale[cluster] = max(float(np.median(selected)), 1e-6)
        self.scale = scale
        self.fit_count = len(reference)
        return self

    def score(self, values):
        values = np.asarray(values, dtype=np.float32)
        assignment = self.model.predict(values)
        distance = np.linalg.norm(values - self.model.cluster_centers_[assignment], axis=1)
        return (distance / self.scale[assignment]).astype(np.float32)

    def report(self):
        return {
            "type": "train_only_minibatch_prototype_distance",
            "prototypes": int(self.model.n_clusters),
            "fit_tokens": int(self.fit_count),
            "fit_uses_labels": False,
        }


def mechanism_tensor(attention, *, csr_row_block=4096):
    """Return ``[response, layer, head, 4]`` without layer/head averaging.

    The compressed cache contains strict causal entries plus a separately saved
    diagonal.  Missing mass below the cache floor is not redistributed.
    """
    response_count = int(attention.num_response_tokens)
    prompt_count = int(attention.response_idx)
    if response_count < 1 or prompt_count < 1:
        raise ValueError("mechanism tensor requires a non-empty prompt and response")
    device = attention.response_values.device
    rows_count = int(attention.num_channels) * response_count
    row_ptr = attention.response_row_ptr.long()
    prompt_mass = torch.zeros(rows_count, dtype=torch.float32, device=device)
    history_mass = torch.zeros_like(prompt_mass)
    squared_mass = torch.zeros_like(prompt_mass)
    dominant = torch.zeros_like(prompt_mass)
    local_mass = torch.zeros_like(prompt_mass)

    for row_start in range(0, rows_count, int(csr_row_block)):
        row_end = min(row_start + int(csr_row_block), rows_count)
        starts = row_ptr[row_start:row_end]
        lengths = row_ptr[row_start + 1 : row_end + 1] - starts
        entry_count = int(lengths.sum())
        if not entry_count:
            continue
        repeated_starts = torch.repeat_interleave(starts, lengths)
        prefix = torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        positions = repeated_starts + torch.arange(entry_count, device=device) - prefix
        local_row = torch.repeat_interleave(
            torch.arange(row_end - row_start, device=device), lengths
        )
        global_row = local_row + row_start
        token_row = global_row.remainder(response_count)
        source = attention.response_column_indices[positions].long()
        value = attention.response_values[positions].float().clamp_min(0.0)
        is_prompt = source < prompt_count

        local_prompt = torch.zeros(row_end - row_start, dtype=torch.float32, device=device)
        local_history = torch.zeros_like(local_prompt)
        local_squared = torch.zeros_like(local_prompt)
        local_dominant = torch.zeros_like(local_prompt)
        local_locality = torch.zeros_like(local_prompt)
        local_squared.index_add_(0, local_row, value.square())
        local_dominant.scatter_reduce_(0, local_row, value, reduce="amax", include_self=True)
        if bool(is_prompt.any()):
            local_prompt.index_add_(0, local_row[is_prompt], value[is_prompt])
        is_history = ~is_prompt
        if bool(is_history.any()):
            history_row = local_row[is_history]
            history_value = value[is_history]
            local_history.index_add_(0, history_row, history_value)
            lag = token_row[is_history] - (source[is_history] - prompt_count)
            locality = lag.float().clamp_min(1.0).reciprocal()
            local_locality.index_add_(0, history_row, history_value * locality)
        prompt_mass[row_start:row_end] = local_prompt
        history_mass[row_start:row_end] = local_history
        squared_mass[row_start:row_end] = local_squared
        dominant[row_start:row_end] = local_dominant
        local_mass[row_start:row_end] = local_locality

    diagonal = (
        attention.attention_diagonal.float()[:, :, prompt_count:]
        .reshape(-1)
        .clamp_min(0.0)
    )
    token_row = torch.arange(rows_count, device=device).remainder(response_count)
    prompt_mean = prompt_mass / float(prompt_count)
    response_mean = (history_mass + diagonal) / (token_row + 1).float()
    routing_denominator = prompt_mean + response_mean
    routing = torch.where(
        routing_denominator > 0,
        prompt_mean / routing_denominator,
        torch.zeros_like(routing_denominator),
    )

    retained = prompt_mass + history_mass + diagonal
    hhi = torch.where(
        retained > 0,
        (squared_mass + diagonal.square()) / retained.square().clamp_min(1e-12),
        torch.zeros_like(retained),
    )
    possible_sources = (prompt_count + token_row + 1).float()
    effective_support = torch.where(hhi > 0, hhi.reciprocal(), torch.zeros_like(hhi))
    support_fraction = (effective_support / possible_sources).clamp(0.0, 1.0)
    dominant = torch.maximum(dominant, diagonal)
    response_total = history_mass + diagonal
    locality = torch.where(
        response_total > 0,
        (local_mass + diagonal) / response_total,
        torch.zeros_like(response_total),
    )
    tensor = torch.stack((routing, support_fraction, dominant, locality), dim=-1)
    tensor = tensor.reshape(
        attention.num_layers, attention.num_heads, response_count, len(MECHANISMS)
    ).permute(2, 0, 1, 3)
    unresolved = (1.0 - retained).clamp(0.0, 1.0).reshape(
        attention.num_layers, attention.num_heads, response_count
    ).permute(2, 0, 1)
    return torch.nan_to_num(tensor), torch.nan_to_num(unresolved)


def _source_codes(token_count, response_idx, dimension, device):
    position = torch.arange(token_count, dtype=torch.float32, device=device)
    prompt_denominator = max(response_idx - 1, 1)
    response_denominator = max(token_count - response_idx - 1, 1)
    relative = torch.where(
        position < response_idx,
        position / float(prompt_denominator),
        (position - response_idx) / float(response_denominator),
    )
    frequencies = (2.0 ** torch.arange(dimension // 2, device=device)) * torch.pi
    phase = relative[:, None] * frequencies[None, :]
    code = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)
    role = torch.where(position < response_idx, -1.0, 1.0)[:, None]
    code[:, 0::2] *= role
    return code


def fixed_graph_messages(
    graph, token_embedding, *, source_sketch_dim=16, diffusion_hops=3
):
    """Multi-scale causal diffusion over the exact retained RP/RR graph.

    ``H[k] = P_RR @ H[k-1]`` transports token state through paths of exactly
    ``k`` RR edges.  Prompt provenance is seeded by direct RP edges and follows
    the same recurrence, so a later token can inherit prompt evidence through
    response nodes even without a direct RP connection.
    """
    token_embedding = torch.as_tensor(
        token_embedding, dtype=torch.float32, device=graph.node_attr.device
    )
    response_count, base_dim = token_embedding.shape
    if response_count != graph.num_nodes - graph.response_idx:
        raise ValueError("token embedding does not cover the graph response nodes")
    sketch = _source_codes(
        graph.num_nodes, graph.response_idx, int(source_sketch_dim), graph.node_attr.device
    )
    if int(diffusion_hops) < 1:
        raise ValueError("diffusion_hops must be positive")
    rp_direct = torch.zeros((response_count, source_sketch_dim), device=sketch.device)
    rr_source = torch.empty(0, dtype=torch.long, device=sketch.device)
    rr_target = torch.empty(0, dtype=torch.long, device=sketch.device)
    rr_weight = torch.empty(0, dtype=torch.float32, device=sketch.device)

    for relation in (RP, RR):
        edge_ids = torch.nonzero(graph.edge_type == relation, as_tuple=False).flatten()
        if not edge_ids.numel():
            continue
        source = graph.edge_index[0, edge_ids].long()
        target = graph.edge_index[1, edge_ids].long() - graph.response_idx
        weight = graph.edge_score[edge_ids].float().clamp_min(0.0)
        denominator = torch.zeros(response_count, dtype=torch.float32, device=weight.device)
        denominator.index_add_(0, target, weight)
        normalized = weight / denominator[target].clamp_min(1e-12)
        if relation == RP:
            rp_direct.index_add_(0, target, normalized[:, None] * sketch[source])
        else:
            rr_source = source - graph.response_idx
            rr_target = target
            rr_weight = normalized
    rr_transition = torch.sparse_coo_tensor(
        torch.stack((rr_target, rr_source)),
        rr_weight,
        size=(response_count, response_count),
        dtype=torch.float32,
        device=sketch.device,
    ).coalesce()
    blocks = [token_embedding, rp_direct]
    slices = {
        "token": (0, base_dim),
        "rp_direct": (base_dim, base_dim + source_sketch_dim),
    }
    offset = base_dim + source_sketch_dim
    token_state = token_embedding
    position_state = sketch[graph.response_idx :]
    prompt_state = rp_direct
    reach = _exact_hop_reach_count(
        rr_source.detach().cpu().tolist(),
        rr_target.detach().cpu().tolist(),
        response_count,
        int(diffusion_hops),
    )
    influence_state = torch.ones(
        (response_count, 1), dtype=torch.float32, device=sketch.device
    )
    influence = []
    for hop in range(1, int(diffusion_hops) + 1):
        token_state = torch.sparse.mm(rr_transition, token_state)
        position_state = torch.sparse.mm(rr_transition, position_state)
        prompt_state = torch.sparse.mm(rr_transition, prompt_state)
        influence_state = torch.sparse.mm(rr_transition, influence_state)
        for name, block in (
            (f"rr_token_hop_{hop}", token_state),
            (f"rr_position_hop_{hop}", position_state),
            (f"rp_diffusion_hop_{hop}", prompt_state),
        ):
            width = int(block.shape[1])
            slices[name] = (offset, offset + width)
            offset += width
            blocks.append(block)
        influence.append(influence_state[:, 0])
    fused = torch.cat(blocks, dim=1)
    structure = {
        "hop_reach_count": reach,
        "hop_influence_mass": torch.stack(influence, dim=1),
    }
    return fused, slices, structure


def _exact_hop_reach_count(sources, targets, node_count, hops):
    parents = [set() for _ in range(int(node_count))]
    for source, target in zip(sources, targets):
        parents[int(target)].add(int(source))
    current = [set(value) for value in parents]
    rows = []
    for _ in range(int(hops)):
        rows.append([len(value) for value in current])
        current = [
            set().union(*(parents[source] for source in ancestors))
            if ancestors else set()
            for ancestors in current
        ]
    return torch.tensor(rows, dtype=torch.int32).T.contiguous()


def _geometry(dataset):
    return {
        name: dataset.manifest.get(name)
        for name in (
            "schema",
            "num_layers",
            "num_heads",
            "alignment",
            "attention_floor",
            "observer_model",
        )
    }


def _fit_base_projector(dataset, config):
    sampler = _ReferenceSampler(config.fit_reference_size, config.seed)
    expected = (
        int(dataset.manifest["num_layers"])
        * int(dataset.manifest["num_heads"])
        * len(MECHANISMS)
    )
    for sample_id in tqdm(dataset.sample_ids, desc="[1/6] train mechanism reference", unit="sample"):
        sample = dataset[sample_id]
        tensor, _ = mechanism_tensor(
            sample.attention(), csr_row_block=config.csr_row_block
        )
        flat = tensor.reshape(tensor.shape[0], -1).detach().cpu().numpy()
        if flat.shape[1] != expected:
            raise ValueError("train attention geometry changed across samples")
        sampler.add(flat)
        sample.release_attention()
    reference = sampler.get()
    return _RobustProjector(config.base_dim, config.seed).fit(reference), len(reference)


def _extract_split(dataset, base_projector, config, split_name, *, keep_graphs):
    fused_rows, unresolved_rows = [], []
    graphs = []
    metadata = {
        name: []
        for name in (
            "sample_id",
            "source_id",
            "token_index",
            "token_id",
            "task_type",
            "data_source",
            "generator_model",
        )
    }
    slices = None
    offset = 0
    for sample_id in tqdm(
        dataset.sample_ids,
        desc=f"[2/6] {split_name} token and graph views",
        unit="sample",
    ):
        sample = dataset[sample_id]
        attention = sample.attention()
        tensor, unresolved = mechanism_tensor(
            attention, csr_row_block=config.csr_row_block
        )
        flat = tensor.reshape(tensor.shape[0], -1).detach().cpu().numpy()
        token_embedding = base_projector.transform(flat)
        graph = build_attention_graph(attention, GraphBuildConfig(selection="threshold"))
        fused, current_slices, structure = fixed_graph_messages(
            graph,
            token_embedding,
            source_sketch_dim=config.source_sketch_dim,
            diffusion_hops=config.diffusion_hops,
        )
        if slices is None:
            slices = current_slices
        elif slices != current_slices:
            raise ValueError("graph message block layout changed across samples")
        fused_np = fused.detach().cpu().numpy().astype(np.float32)
        unresolved_np = unresolved.detach().cpu().numpy().astype(np.float32)
        count = int(attention.num_response_tokens)
        if len(fused_np) != count or len(unresolved_np) != count:
            raise ValueError("sample representation does not cover every response token")
        fused_rows.append(fused_np)
        unresolved_rows.append(unresolved_np)
        metadata["sample_id"].extend([str(sample.sample_id)] * count)
        metadata["source_id"].extend([str(sample.source_id)] * count)
        metadata["token_index"].extend(range(count))
        metadata["token_id"].extend(
            attention.token_ids[attention.response_idx :].detach().cpu().tolist()
        )
        for field in ("task_type", "data_source", "generator_model"):
            metadata[field].extend([str(getattr(sample, field))] * count)
        if keep_graphs:
            graphs.append(
                {
                    "sample_id": str(sample.sample_id),
                    "start": offset,
                    "end": offset + count,
                    "token_ids": graph.token_ids.detach().cpu().numpy().astype(np.int32),
                    "response_idx": int(graph.response_idx),
                    "edge_index": graph.edge_index.detach().cpu().numpy().astype(np.int32),
                    "edge_type": graph.edge_type.detach().cpu().numpy().astype(np.int8),
                    "edge_score": graph.edge_score.detach().cpu().numpy().astype(np.float32),
                    "hop_reach_count": structure["hop_reach_count"].detach().cpu().numpy().astype(np.int32),
                    "hop_influence_mass": structure["hop_influence_mass"].detach().cpu().numpy().astype(np.float32),
                }
            )
        offset += count
        sample.release_attention()
    if not fused_rows or slices is None:
        raise ValueError(f"{split_name} split contains no response tokens")
    arrays = {
        name: np.asarray(values, dtype=dtype)
        for name, values, dtype in (
            ("sample_id", metadata["sample_id"], str),
            ("source_id", metadata["source_id"], str),
            ("token_index", metadata["token_index"], np.int32),
            ("token_id", metadata["token_id"], np.int32),
            ("task_type", metadata["task_type"], str),
            ("data_source", metadata["data_source"], str),
            ("generator_model", metadata["generator_model"], str),
        )
    }
    return (
        np.concatenate(fused_rows).astype(np.float32),
        np.concatenate(unresolved_rows).astype(np.float32),
        arrays,
        graphs,
        slices,
    )


def _view_matrix(values, slices, view):
    if view == "token_only":
        start, end = slices["token"]
        return values[:, start:end]
    if view == "token_graph":
        return values
    output = values.copy()
    if view == "no_rp":
        for name, (start, end) in slices.items():
            if name == "rp_direct" or name.startswith("rp_diffusion_hop_"):
                output[:, start:end] = 0.0
    elif view == "no_rr":
        for name, (start, end) in slices.items():
            if name.startswith("rr_") or name.startswith("rp_diffusion_hop_"):
                output[:, start:end] = 0.0
    else:
        raise ValueError(f"unknown representation view: {view}")
    return output


def _fit_views(train, test, slices, config):
    embeddings, scores, diagnostics = {}, {}, {}
    rng = np.random.default_rng(config.seed + 10_000)
    train_limit = min(
        len(train), max(config.fit_reference_size, config.detector_reference_size)
    )
    train_ids = (
        np.arange(len(train))
        if train_limit == len(train)
        else np.sort(rng.choice(len(train), train_limit, replace=False))
    )
    for view in VIEWS:
        print(f"[3/6] fitting label-blind view {view}", flush=True)
        train_view = _view_matrix(train[train_ids], slices, view)
        test_view = _view_matrix(test, slices, view)
        if view == "token_only":
            train_embedding = train_view.astype(np.float32, copy=False)
            test_embedding = test_view.astype(np.float32, copy=False)
            projection = {
                "type": "base_train_only_robust_pca",
                "output_dimensions": int(train_embedding.shape[1]),
                "fit_uses_labels": False,
            }
        else:
            sampler = _ReferenceSampler(
                config.fit_reference_size, config.seed + 100
            )
            sampler.add(train_view)
            projector = _RobustProjector(
                config.embedding_dim, config.seed + 100
            ).fit(sampler.get())
            train_embedding = projector.transform(train_view)
            test_embedding = projector.transform(test_view)
            projection = {"type": "train_only_robust_pca", **projector.report()}
        detector = _PrototypeDetector(
            config.prototypes,
            config.detector_reference_size,
            config.seed + 1000,
        ).fit(train_embedding)
        embeddings[view] = test_embedding
        scores[view] = detector.score(test_embedding)
        diagnostics[view] = {
            "projection": projection,
            "detector": detector.report(),
        }
    return embeddings, scores, diagnostics


def _ranking(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    result = {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else None,
    }
    if len(np.unique(labels)) < 2:
        return {**result, "auroc": None, "auprc": None}
    return {
        **result,
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "correct_median": float(np.median(scores[labels == 0])),
        "hallucination_median": float(np.median(scores[labels == 1])),
    }


def _grouped_ranking(labels, scores, metadata, field):
    output = {}
    for value in sorted(set(metadata[field].astype(str))):
        selected = metadata[field].astype(str) == value
        output[value] = _ranking(labels[selected], scores[selected])
    return output


def _metric_delta(left, right, metric):
    left_value, right_value = left.get(metric), right.get(metric)
    return (
        None if left_value is None or right_value is None
        else float(left_value - right_value)
    )


def _read_labels(evaluation_dataset, metadata):
    # Formal cache labels are deliberately sealed until every test attention
    # sample has been visited.  This separate pass happens only after all
    # projections, detector fits, scores, and example choices are frozen.
    for sample_id in tqdm(
        evaluation_dataset.sample_ids,
        desc="[5/6] load sealed evaluation labels",
        unit="sample",
    ):
        sample = evaluation_dataset[sample_id]
        sample.attention()
        sample.release_attention()
    store = evaluation_dataset.labels()
    rows = []
    for sample_id in tqdm(
        evaluation_dataset.sample_ids,
        desc="[5/6] align evaluation labels",
        unit="sample",
    ):
        sample = evaluation_dataset[sample_id]
        labels = store.response_labels(sample).detach().cpu().numpy().astype(np.int8)
        rows.extend(labels.tolist())
        sample.release_attention()
    output = np.asarray(rows, dtype=np.int8)
    expected_sample = []
    expected_index = []
    for sample_id in evaluation_dataset.sample_ids:
        count = int((metadata["sample_id"].astype(str) == str(sample_id)).sum())
        expected_sample.extend([str(sample_id)] * count)
        expected_index.extend(range(count))
    if not (
        len(output) == len(metadata["sample_id"])
        and np.array_equal(metadata["sample_id"].astype(str), np.asarray(expected_sample))
        and np.array_equal(metadata["token_index"], np.asarray(expected_index, dtype=np.int32))
    ):
        raise ValueError("evaluation labels do not align with frozen token rows")
    return output


def _label_free_sample_selection(config, metadata, embedding):
    available = set(metadata["sample_id"].astype(str))
    if config.sample_ids:
        requested = list(dict.fromkeys(map(str, config.sample_ids)))
        missing = [sample_id for sample_id in requested if sample_id not in available]
        if missing:
            raise ValueError(f"sample IDs are absent from test split: {missing}")
        return requested, "user_requested_before_labels"
    candidates = []
    for sample_id in dict.fromkeys(metadata["sample_id"].astype(str)):
        selected = metadata["sample_id"].astype(str) == sample_id
        values = embedding[selected]
        dispersion = float(np.linalg.norm(values - values.mean(0), axis=1).mean())
        candidates.append((dispersion, len(values), sample_id))
    chosen = max(candidates, key=lambda row: (row[0], row[1], row[2]))
    return [chosen[2]], "label_free_max_token_embedding_dispersion"


def _save_sample_graphs(
    output, graphs, metadata, embeddings, scores, graph_features, feature_slices
):
    directory = output / "sample_graphs"
    directory.mkdir(parents=True, exist_ok=False)
    paths = {}
    for graph in tqdm(graphs, desc="[4/6] save per-sample graph artifacts", unit="sample"):
        start, end = graph["start"], graph["end"]
        path = directory / f"sample_{_safe_filename(graph['sample_id'])}.npz"
        np.savez_compressed(
            path,
            schema=np.asarray(SCHEMA),
            labels_included=np.asarray(False),
            sample_id=np.asarray(graph["sample_id"]),
            token_ids=graph["token_ids"],
            response_idx=np.asarray(graph["response_idx"], dtype=np.int32),
            edge_index=graph["edge_index"],
            edge_type=graph["edge_type"],
            edge_score=graph["edge_score"],
            diffusion_hop=np.arange(
                1, graph["hop_reach_count"].shape[1] + 1, dtype=np.int16
            ),
            rr_hop_reach_count=graph["hop_reach_count"],
            rr_hop_influence_mass=graph["hop_influence_mass"],
            multiscale_graph_features=graph_features[start:end],
            feature_block_name=np.asarray(list(feature_slices)),
            feature_block_start=np.asarray(
                [bounds[0] for bounds in feature_slices.values()], dtype=np.int32
            ),
            feature_block_end=np.asarray(
                [bounds[1] for bounds in feature_slices.values()], dtype=np.int32
            ),
            response_token_id=metadata["token_id"][start:end],
            token_index=metadata["token_index"][start:end],
            token_only_embedding=embeddings["token_only"][start:end],
            token_graph_embedding=embeddings["token_graph"][start:end],
            token_only_score=scores["token_only"][start:end],
            token_graph_score=scores["token_graph"][start:end],
        )
        paths[graph["sample_id"]] = path
    return paths


def _safe_filename(value):
    text = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    ).strip("._")
    return (text or "sample")[:120]


def _display_edges(graph, config):
    edge_index = graph["edge_index"]
    edge_type = graph["edge_type"]
    edge_score = graph["edge_score"]
    selected = []
    for target in range(graph["response_idx"], len(graph["token_ids"])):
        incoming = np.flatnonzero(edge_index[1] == target)
        for relation in (RP, RR):
            ids = incoming[edge_type[incoming] == relation]
            if not len(ids):
                continue
            ranked = ids[np.argsort(-edge_score[ids], kind="stable")]
            mass = edge_score[ranked]
            if float(mass.sum()) > 0:
                reached = np.flatnonzero(
                    np.cumsum(mass) >= config.display_mass_cover * mass.sum()
                )
                count = int(reached[0]) + 1 if len(reached) else len(ranked)
                ranked = ranked[:count]
            selected.extend(ranked[: config.display_edges_per_type].tolist())
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _relation_matrices(graph):
    response_idx = int(graph["response_idx"])
    response_count = len(graph["token_ids"]) - response_idx
    rr = np.zeros((response_count, response_count), dtype=np.float64)
    rp = np.zeros((response_count, response_idx), dtype=np.float64)
    source, target = graph["edge_index"]
    target = target - response_idx
    for relation, matrix, offset in ((RR, rr, response_idx), (RP, rp, 0)):
        selected = graph["edge_type"] == relation
        if not selected.any():
            continue
        relation_target = target[selected]
        relation_source = source[selected] - offset
        weight = graph["edge_score"][selected].astype(np.float64)
        denominator = np.zeros(response_count, dtype=np.float64)
        np.add.at(denominator, relation_target, weight)
        normalized = weight / np.maximum(denominator[relation_target], 1e-12)
        np.add.at(matrix, (relation_target, relation_source), normalized)
    return rr, rp


def _effective_relations(graph, hops, per_target=2):
    """Non-adjacent RR and inherited RP influences from matrix powers."""
    rr, rp = _relation_matrices(graph)
    rr_rows, rp_rows = [], []
    power = rr.copy()
    for hop in range(2, int(hops) + 1):
        power = rr @ power
        for target in range(len(rr)):
            candidates = np.flatnonzero((power[target] > 1e-8) & (rr[target] == 0))
            ranked = candidates[np.argsort(-power[target, candidates], kind="stable")]
            for source in ranked[:per_target]:
                rr_rows.append((int(source), target, hop, float(power[target, source])))
    inherited = rr @ rp
    for rr_hops in range(1, int(hops) + 1):
        for target in range(len(rr)):
            candidates = np.flatnonzero((inherited[target] > 1e-8) & (rp[target] == 0))
            ranked = candidates[np.argsort(-inherited[target, candidates], kind="stable")]
            for source in ranked[:1]:
                rp_rows.append(
                    (int(source), target, rr_hops + 1, float(inherited[target, source]))
                )
        inherited = rr @ inherited
    return rr_rows, rp_rows


def _render_population(output, embeddings, scores, labels):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for column, view in enumerate(("token_only", "token_graph")):
        values = embeddings[view]
        coordinates = np.zeros((len(values), 2), dtype=np.float32)
        coordinates[:, : min(2, values.shape[1])] = values[:, : min(2, values.shape[1])]
        for label, color, name, size, alpha in (
            (0, "#2ca02c", "correct", 4, 0.16),
            (1, "#d62728", "hallucination", 10, 0.72),
        ):
            selected = labels == label
            axes[0, column].scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                c=color,
                s=size,
                alpha=alpha,
                label=name,
                rasterized=True,
            )
        axes[0, column].set(
            title=f"{view}: frozen train-PCA coordinates",
            xlabel="component 1",
            ylabel="component 2",
        )
        axes[0, column].legend(frameon=False)
        metric = _ranking(labels, scores[view])
        auc_text = "N/A" if metric["auroc"] is None else f"{metric['auroc']:.3f}"
        for label, color, name in (
            (0, "#2ca02c", "correct"),
            (1, "#d62728", "hallucination"),
        ):
            selected = scores[view][labels == label]
            if len(selected):
                axes[1, column].hist(
                    selected,
                    bins=60,
                    density=True,
                    alpha=0.55,
                    color=color,
                    label=name,
                )
        axes[1, column].set(
            title=f"prototype novelty: AUROC={auc_text}",
            xlabel="train-only anomaly score",
            ylabel="density",
        )
        axes[1, column].legend(frameon=False)
    path = output / "population_token_representations.png"
    figure.savefig(path, dpi=240)
    plt.close(figure)
    return path


def _render_sample(output, graph, embedding, labels, config):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected_edges = _display_edges(graph, config)
    edge_index = graph["edge_index"][:, selected_edges]
    edge_type = graph["edge_type"][selected_edges]
    edge_score = graph["edge_score"][selected_edges]
    response_idx = int(graph["response_idx"])
    response_count = len(embedding)
    response_nodes = np.arange(response_idx, response_idx + response_count)
    prompt_nodes = np.unique(edge_index[0, edge_index[0] < response_idx])
    colors = np.where(labels == 1, "#d62728", "#2ca02c")

    width = max(18, min(32, 15 + response_count * 0.15))
    figure, axes = plt.subplots(1, 3, figsize=(width, 6), constrained_layout=True)
    topology = axes[0]
    position = {
        **{int(node): (float(node), 1.0) for node in prompt_nodes},
        **{int(node): (float(node), 0.0) for node in response_nodes},
    }
    maximum = max(float(edge_score.max()) if len(edge_score) else 0.0, 1e-8)
    for edge, relation, weight in zip(edge_index.T, edge_type, edge_score):
        source, target = map(int, edge)
        if source not in position:
            continue
        topology.annotate(
            "",
            xy=position[target],
            xytext=position[source],
            arrowprops={
                "arrowstyle": "->",
                "color": "#1f77b4" if relation == RP else "#7f7f7f",
                "alpha": 0.20 + 0.55 * float(weight / maximum),
                "lw": 0.4 + 1.8 * float(weight / maximum),
                "connectionstyle": "arc3,rad=.08",
            },
        )
    if len(prompt_nodes):
        topology.scatter(prompt_nodes, np.ones(len(prompt_nodes)), marker="s", s=28, c="#4c78a8")
    topology.scatter(response_nodes, np.zeros(response_count), c=colors, s=42, edgecolors="black", linewidths=.3, zorder=3)
    topology.set(
        title="Typed causal attention graph (display-pruned only)",
        xlabel="absolute token position",
        yticks=(0, 1),
        yticklabels=("response", "prompt endpoints"),
    )
    topology.grid(alpha=.12)
    topology.legend(handles=[
        Line2D([], [], color="#1f77b4", label="direct RP"),
        Line2D([], [], color="#7f7f7f", label="direct RR"),
    ], frameon=False)

    effective = axes[1]
    rr_effective, rp_effective = _effective_relations(
        graph, config.diffusion_hops
    )
    effective_prompt = np.unique([row[0] for row in rp_effective]).astype(int)
    if len(effective_prompt):
        effective.scatter(
            effective_prompt, np.ones(len(effective_prompt)),
            marker="s", s=26, c="#4c78a8", zorder=3,
        )
    effective.scatter(
        response_nodes, np.zeros(response_count), c=colors,
        s=40, edgecolors="black", linewidths=.3, zorder=3,
    )
    rr_max = max([row[3] for row in rr_effective], default=1.0)
    rp_max = max([row[3] for row in rp_effective], default=1.0)
    for source, target, hop, weight in rr_effective:
        effective.annotate(
            "",
            xy=(float(target + response_idx), 0.0),
            xytext=(float(source + response_idx), 0.0),
            arrowprops={
                "arrowstyle": "->", "linestyle": "--", "color": "#9467bd",
                "alpha": .20 + .55 * weight / max(rr_max, 1e-12),
                "lw": .5 + 1.5 * weight / max(rr_max, 1e-12),
                "connectionstyle": f"arc3,rad={.08 + .025 * hop}",
            },
        )
    for source, target, hop, weight in rp_effective:
        effective.annotate(
            "",
            xy=(float(target + response_idx), 0.0),
            xytext=(float(source), 1.0),
            arrowprops={
                "arrowstyle": "->", "linestyle": ":", "color": "#1f77b4",
                "alpha": .20 + .55 * weight / max(rp_max, 1e-12),
                "lw": .5 + 1.5 * weight / max(rp_max, 1e-12),
                "connectionstyle": f"arc3,rad={.08 + .025 * hop}",
            },
        )
    effective.set(
        title=f"Non-adjacent effective relations from P²…P^{config.diffusion_hops}",
        xlabel="absolute token position",
        yticks=(0, 1),
        yticklabels=("response", "inherited prompt"),
    )
    effective.grid(alpha=.12)
    effective.legend(handles=[
        Line2D([], [], color="#9467bd", linestyle="--", label="non-adjacent RR influence"),
        Line2D([], [], color="#1f77b4", linestyle=":", label="inherited prompt provenance"),
    ], frameon=False)

    coordinates = np.zeros((response_count, 2), dtype=np.float32)
    coordinates[:, : min(2, embedding.shape[1])] = embedding[:, : min(2, embedding.shape[1])]
    for source, target in edge_index.T[edge_type == RR]:
        source_index, target_index = int(source) - response_idx, int(target) - response_idx
        axes[2].plot(
            coordinates[[source_index, target_index], 0],
            coordinates[[source_index, target_index], 1],
            color="#7f7f7f",
            alpha=.18,
            lw=.7,
            zorder=1,
        )
    axes[2].scatter(
        coordinates[:, 0], coordinates[:, 1], c=colors, s=52,
        edgecolors="black", linewidths=.35, zorder=2,
    )
    for index, (x, y) in enumerate(coordinates):
        axes[2].text(x, y, str(index), fontsize=6, ha="center", va="bottom")
    axes[2].set(
        title="Every response token in the frozen graph embedding",
        xlabel="embedding component 1",
        ylabel="embedding component 2",
    )
    axes[2].legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor="#2ca02c", label="correct"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#d62728", label="hallucination"),
        Line2D([], [], color="#7f7f7f", label="RR edge"),
    ], frameon=False)
    path = output / f"sample_{_safe_filename(graph['sample_id'])}_token_graph.png"
    figure.suptitle(f"Sample {graph['sample_id']}; labels used only as node colors")
    figure.savefig(path, dpi=240)
    plt.close(figure)
    return path, int(len(selected_edges)), int(len(rr_effective)), int(len(rp_effective))


def discover_token_representations(
    train_dataset,
    test_dataset,
    evaluation_dataset,
    *,
    output_dir,
    config=None,
):
    """Fit label-blind token representations, then evaluate frozen scores."""
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

    base_projector, base_reference_count = _fit_base_projector(train_dataset, config)
    train, _, train_metadata, _, train_slices = _extract_split(
        train_dataset, base_projector, config, "train", keep_graphs=False
    )
    test, unresolved, metadata, graphs, test_slices = _extract_split(
        test_dataset, base_projector, config, "test", keep_graphs=True
    )
    if train_slices != test_slices:
        raise ValueError("train and test graph-message layouts differ")
    if set(train_metadata["source_id"].astype(str)) & set(metadata["source_id"].astype(str)):
        raise ValueError("train and test source groups overlap")

    embeddings, scores, view_diagnostics = _fit_views(
        train, test, test_slices, config
    )
    selected_samples, selection_rule = _label_free_sample_selection(
        config, metadata, embeddings["token_graph"]
    )
    graph_paths = _save_sample_graphs(
        output, graphs, metadata, embeddings, scores, test, test_slices
    )
    np.savez_compressed(
        output / "token_representations_label_free.npz",
        schema=np.asarray(SCHEMA),
        labels_included=np.asarray(False),
        mechanisms=np.asarray(MECHANISMS),
        sample_id=metadata["sample_id"],
        source_id=metadata["source_id"],
        token_index=metadata["token_index"],
        token_id=metadata["token_id"],
        task_type=metadata["task_type"],
        data_source=metadata["data_source"],
        generator_model=metadata["generator_model"],
        unresolved_control=unresolved,
        **{f"{view}_embedding": embeddings[view] for view in VIEWS},
        **{f"{view}_score": scores[view] for view in VIEWS},
    )
    label_free_report = {
        "schema": SCHEMA,
        "labels_read": False,
        "representation": {
            "mechanisms": list(MECHANISMS),
            "mechanism_tensor": "response_token_by_layer_by_head_by_mechanism",
            "layer_head_aggregation_before_projection": False,
            "base_projection": base_projector.report(),
            "base_reference_tokens": int(base_reference_count),
            "graph_propagation": {
                "trainable": False,
                "support": "all retained threshold-cache RP/RR pair edges",
                "rp_message": "weighted source-specific positional sketch",
                "rr_recurrence": "H_k = P_RR @ H_(k-1)",
                "prompt_recurrence": "B_0 = direct_RP_provenance; B_k = P_RR @ B_(k-1)",
                "diffusion_hops": int(config.diffusion_hops),
                "saved_structure_diagnostics": [
                    "rr_hop_reach_count", "rr_hop_influence_mass"
                ],
            },
            "base_loading_summary": base_projector.structured_loading_report(
                int(train_dataset.manifest["num_layers"]),
                int(train_dataset.manifest["num_heads"]),
                len(MECHANISMS),
            ),
        },
        "views": view_diagnostics,
        "labels_read_during_fit": False,
        "train_tokens": int(len(train)),
        "test_tokens": int(len(test)),
        "sample_selection": {
            "sample_ids": selected_samples,
            "rule": selection_rule,
            "labels_used": False,
        },
        "config": asdict(config),
    }
    (output / "label_free_report.json").write_text(
        json.dumps(label_free_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("[5/6] opening labels after embeddings and scores are frozen", flush=True)
    labels = _read_labels(evaluation_dataset, metadata)
    view_results = {}
    for view in VIEWS:
        view_results[view] = {
            "overall": _ranking(labels, scores[view]),
            "by_task_type": _grouped_ranking(
                labels, scores[view], metadata, "task_type"
            ),
            "by_data_source": _grouped_ranking(
                labels, scores[view], metadata, "data_source"
            ),
        }
    full_metrics = view_results["token_graph"]["overall"]
    token_metrics = view_results["token_only"]["overall"]
    report = {
        **label_free_report,
        "labels_read": True,
        "labels_read_during": "evaluation_and_plot_coloring_only",
        "views": {
            view: {**view_diagnostics[view], "evaluation": view_results[view]}
            for view in VIEWS
        },
        "graph_gain_over_token_only": {
            "auroc": _metric_delta(full_metrics, token_metrics, "auroc"),
            "auprc": _metric_delta(full_metrics, token_metrics, "auprc"),
            "interpretation": "positive means exact RP/RR graph propagation adds value",
        },
        "relation_ablation": {
            "full_minus_no_rp": {
                metric: _metric_delta(
                    full_metrics, view_results["no_rp"]["overall"], metric
                )
                for metric in ("auroc", "auprc")
            },
            "full_minus_no_rr": {
                metric: _metric_delta(
                    full_metrics, view_results["no_rr"]["overall"], metric
                )
                for metric in ("auroc", "auprc")
            },
            "direction": "positive means that relation's direct and propagated blocks add value",
        },
        "unresolved_control": {
            "mean": float(unresolved.mean()),
            "included_in_representation": False,
        },
    }
    report_path = output / "token_representation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("[6/6] rendering frozen population and requested sample views", flush=True)
    population_path = _render_population(output, embeddings, scores, labels)
    graph_by_id = {graph["sample_id"]: graph for graph in graphs}
    sample_rows = []
    for sample_id in selected_samples:
        graph = graph_by_id[sample_id]
        start, end = graph["start"], graph["end"]
        figure, display_edges, effective_rr, inherited_rp = _render_sample(
            output,
            graph,
            embeddings["token_graph"][start:end],
            labels[start:end],
            config,
        )
        sample_rows.append({
            "sample_id": sample_id,
            "selection_rule": selection_rule,
            "response_nodes": int(end - start),
            "hallucination_tokens": int(labels[start:end].sum()),
            "display_edges": display_edges,
            "display_nonadjacent_rr_relations": effective_rr,
            "display_inherited_rp_relations": inherited_rp,
            "figure": str(figure),
            "label_free_data": str(graph_paths[sample_id]),
        })
    report["population_figure"] = str(population_path)
    report["sample_visualizations"] = sample_rows
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "output_dir": str(output),
        "report": str(report_path),
        "label_free_embeddings": str(output / "token_representations_label_free.npz"),
        "sample_graph_directory": str(output / "sample_graphs"),
        "population_figure": str(population_path),
        "sample_visualizations": sample_rows,
        "test_nodes": int(len(test)),
        "view_metrics": {
            view: view_results[view]["overall"] for view in VIEWS
        },
    }
