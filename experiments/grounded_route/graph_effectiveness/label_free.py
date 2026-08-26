"""Label-free sensitivity of independently encoded saved graph variants.

The audit consumes only ``EncodedTokenGraph`` bundles. It does not perform a
second message-passing step and does not produce an anomaly score. Instead it
asks whether each frozen embedding geometry aligns with its own exact typed
endpoints, and how embeddings change when GroundedRoute itself is separately
trained and encoded on a matched graph intervention.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiment_protocol import scalar_text

from ..artifacts import save_npz
from ..controls import apply_variant
from ..graph import TokenEdges, TokenGraph


TOPOLOGY_SCHEMA = "grounded-route-label-free-topology"
TOPOLOGY_VERSION = 1


@dataclass(frozen=True)
class ChannelAlignment:
    """Weighted endpoint alignment for every saved ``(layer, head)`` type."""

    value: np.ndarray
    numerator: np.ndarray
    weight: np.ndarray


def channel_cosine_alignment(
    graph,
    edges=None,
    *,
    chunk_size: int = 262_144,
) -> np.ndarray:
    """Return weighted source-target cosine alignment as an ``[L,H]`` tensor."""

    return channel_alignment_statistics(
        graph,
        edges,
        chunk_size=chunk_size,
    ).value


def channel_alignment_statistics(
    graph,
    edges=None,
    *,
    chunk_size: int = 262_144,
) -> ChannelAlignment:
    """Accumulate alignment without materializing all edge embeddings at once."""

    source, target, layer, head, edge_weight = _edge_columns(graph, edges)
    relations = int(graph.layer_count) * int(graph.head_count)
    numerator = torch.zeros(relations, dtype=torch.float64)
    mass = torch.zeros(relations, dtype=torch.float64)
    embedding = graph.node_embedding.detach().to(device="cpu", dtype=torch.float32)

    for start in range(0, len(edge_weight), int(chunk_size)):
        stop = min(start + int(chunk_size), len(edge_weight))
        selected_source = source[start:stop].to(device="cpu", dtype=torch.long)
        selected_target = target[start:stop].to(device="cpu", dtype=torch.long)
        selected_weight = edge_weight[start:stop].to(device="cpu", dtype=torch.float64)
        relation = (
            layer[start:stop].to(device="cpu", dtype=torch.long)
            * int(graph.head_count)
            + head[start:stop].to(device="cpu", dtype=torch.long)
        )
        cosine = F.cosine_similarity(
            embedding[selected_source],
            embedding[selected_target],
            dim=1,
        ).to(torch.float64)
        numerator.index_add_(0, relation, selected_weight * cosine)
        mass.index_add_(0, relation, selected_weight)

    value = torch.full_like(numerator, torch.nan)
    observed = mass > 0.0
    value[observed] = numerator[observed] / mass[observed]
    shape = (int(graph.layer_count), int(graph.head_count))
    return ChannelAlignment(
        value=value.reshape(shape).numpy(),
        numerator=numerator.reshape(shape).numpy(),
        weight=mass.reshape(shape).numpy(),
    )


def label_free_audit(
    embedding_views,
    output_path,
    seed: int = 20260825,
    *,
    endpoint_rewire_passes: int = 4,
    summary_path=None,
) -> dict[str, object]:
    """Audit one or more independently trained, row-aligned variants.

    ``embedding_views`` is ``AlignedEmbeddingViews`` from :mod:`views`. A
    single verified ``GraphBundle`` is also accepted for the integrity-only
    workflow; multi-variant endpoint claims require independently encoded
    bundles loaded through ``load_embedding_views``.
    """

    variants, reference_variant, views = _normalized_views(embedding_views)
    reference_view = views[reference_variant]
    reference_records = tuple(reference_view.bundle.records)
    if not reference_records:
        raise ValueError("label-free audit needs at least one encoded graph")
    sample_id = np.asarray([record.sample_id for record in reference_records])
    source_id = np.asarray([record.source_id for record in reference_records])

    alignment_by_variant: list[np.ndarray] = []
    posthoc_rewired_by_variant: list[np.ndarray] = []
    mass_by_variant: list[np.ndarray] = []
    channel_by_variant: list[np.ndarray] = []
    posthoc_channel_by_variant: list[np.ndarray] = []
    posthoc_changed_by_variant: list[np.ndarray] = []
    index_sha256: list[str] = []
    changed_fraction: list[float] = []
    expected_shape = None

    for variant in variants:
        view = views[variant]
        records = {record.sample_id: record for record in view.bundle.records}
        if set(records) != set(sample_id.tolist()):
            raise ValueError("variant bundles contain different graph samples")
        values: list[np.ndarray] = []
        rewired_values: list[np.ndarray] = []
        numerators: list[np.ndarray] = []
        rewired_numerators: list[np.ndarray] = []
        masses: list[np.ndarray] = []
        sample_changed: list[int] = []
        for expected_source, current_sample in zip(
            source_id.tolist(), sample_id.tolist(), strict=True
        ):
            record = records[current_sample]
            if record.source_id != expected_source:
                raise ValueError("variant graph source identity differs")
            graph = record.load()
            statistic = channel_alignment_statistics(graph)
            rewired_edges, changed = _posthoc_rewire(
                graph,
                _sample_seed(seed, current_sample),
                endpoint_rewire_passes,
            )
            rewired = channel_alignment_statistics(graph, rewired_edges)
            if expected_shape is None:
                expected_shape = statistic.value.shape
            if (
                statistic.value.shape != expected_shape
                or rewired.value.shape != expected_shape
            ):
                raise ValueError("variant graphs use different layer/head geometry")
            if not np.allclose(statistic.weight, rewired.weight, rtol=0.0, atol=1e-7):
                raise ValueError("post-hoc endpoint rewire changed layer/head mass")
            values.append(statistic.value)
            rewired_values.append(rewired.value)
            numerators.append(statistic.numerator)
            rewired_numerators.append(rewired.numerator)
            masses.append(statistic.weight)
            sample_changed.append(changed)
        alignment_by_variant.append(np.stack(values))
        posthoc_rewired_by_variant.append(np.stack(rewired_values))
        mass_by_variant.append(np.stack(masses))
        channel_by_variant.append(_aggregate_alignment(numerators, masses))
        posthoc_channel_by_variant.append(
            _aggregate_alignment(rewired_numerators, masses)
        )
        posthoc_changed_by_variant.append(np.asarray(sample_changed, dtype=np.int64))
        index_sha256.append(str(view.bundle.index_sha256))
        changed_fraction.append(float(view.changed_fraction))

    alignment = np.stack(alignment_by_variant).astype(np.float32)
    posthoc_rewired = np.stack(posthoc_rewired_by_variant).astype(np.float32)
    channel_mass = np.stack(mass_by_variant).astype(np.float32)
    channel_alignment = np.stack(channel_by_variant).astype(np.float32)
    posthoc_channel = np.stack(posthoc_channel_by_variant).astype(np.float32)
    posthoc_changed = np.stack(posthoc_changed_by_variant)
    reference_index = variants.index(reference_variant)
    for position, variant in enumerate(variants):
        if not np.allclose(
            channel_mass[position],
            channel_mass[reference_index],
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                f"{variant} changed per-sample layer/head retained mass"
            )
    alignment_delta = channel_alignment[reference_index][None] - channel_alignment

    reference_embedding = np.asarray(reference_view.embedding, dtype=np.float32)
    aligned_cosine: list[np.ndarray] = []
    procrustes_rmse: list[np.ndarray] = []
    for variant in variants:
        cosine, residual = procrustes_embedding_sensitivity(
            reference_embedding,
            np.asarray(views[variant].embedding, dtype=np.float32),
        )
        aligned_cosine.append(cosine)
        procrustes_rmse.append(residual)
    aligned_cosine_array = np.stack(aligned_cosine).astype(np.float32)
    procrustes_array = np.stack(procrustes_rmse).astype(np.float32)

    output_path = Path(output_path)
    save_npz(
        output_path,
        schema=np.asarray(TOPOLOGY_SCHEMA),
        version=np.asarray(TOPOLOGY_VERSION, dtype=np.int32),
        labels_read=np.asarray(False),
        claim_scope=np.asarray(
            "representation sensitivity only; no hallucination relevance claim"
        ),
        variant=np.asarray(variants),
        reference_variant=np.asarray(reference_variant),
        seed=np.asarray(seed, dtype=np.int64),
        index_sha256=np.asarray(index_sha256),
        sample_id=sample_id,
        source_id=source_id,
        token_sample_id=reference_view.bundle.index.sample_id.astype(str),
        token_index=reference_view.bundle.index.token_index.astype(np.int32),
        changed_fraction=np.asarray(changed_fraction, dtype=np.float32),
        channel_alignment_by_sample=alignment,
        posthoc_rewired_alignment_by_sample=posthoc_rewired,
        posthoc_endpoint_alignment_delta_by_sample=alignment - posthoc_rewired,
        channel_weight_mass=channel_mass,
        channel_alignment=channel_alignment,
        posthoc_rewired_channel_alignment=posthoc_channel,
        posthoc_endpoint_channel_delta=channel_alignment - posthoc_channel,
        posthoc_rewired_changed_edges=posthoc_changed,
        channel_alignment_delta_from_reference=alignment_delta.astype(np.float32),
        aligned_embedding_cosine=aligned_cosine_array,
        procrustes_embedding_rmse=procrustes_array,
    )

    comparison = {}
    for position, variant in enumerate(variants):
        observed = np.isfinite(alignment_delta[position])
        comparison[variant] = {
            "changed_fraction": float(changed_fraction[position]),
            "mean_aligned_embedding_cosine": float(
                aligned_cosine_array[position].mean()
            ),
            "mean_procrustes_embedding_rmse": float(
                procrustes_array[position].mean()
            ),
            "mean_absolute_channel_alignment_delta": (
                float(np.mean(np.abs(alignment_delta[position][observed])))
                if observed.any()
                else None
            ),
            "posthoc_rewired_changed_fraction": float(
                posthoc_changed[position].sum()
                / max(1, sum(record.edge_count for record in reference_records))
            ),
            "mean_absolute_posthoc_endpoint_alignment_delta": float(
                np.nanmean(
                    np.abs(channel_alignment[position] - posthoc_channel[position])
                )
            ),
        }

    summary: dict[str, object] = {
        "schema": "grounded-route-label-free-topology-summary",
        "version": TOPOLOGY_VERSION,
        "labels_read": False,
        "claim_scope": (
            "representation sensitivity only; no hallucination relevance claim"
        ),
        "reference_variant": reference_variant,
        "seed": int(seed),
        "variants": list(variants),
        "samples": len(reference_records),
        "response_tokens": int(len(reference_embedding)),
        "layer_count": int(expected_shape[0]),
        "head_count": int(expected_shape[1]),
        "topology_artifact": str(output_path.resolve()),
        "comparisons": comparison,
    }
    summary_path = (
        output_path.with_suffix(".json") if summary_path is None else Path(summary_path)
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**summary, "summary": str(summary_path.resolve())}


def procrustes_embedding_sensitivity(
    reference,
    candidate,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare paired embeddings after removing global shift, scale and rotation."""

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("paired embeddings must have the same [token,dimension] shape")
    reference = _standardize_space(reference)
    candidate = _standardize_space(candidate)
    left, _, right = np.linalg.svd(candidate.T @ reference, full_matrices=False)
    aligned = candidate @ (left @ right)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(aligned, axis=1)
    cosine = np.divide(
        np.sum(reference * aligned, axis=1),
        denominator,
        out=np.zeros(len(reference), dtype=np.float64),
        where=denominator > 0.0,
    )
    rmse = np.sqrt(np.mean((reference - aligned) ** 2, axis=1))
    return cosine.astype(np.float32), rmse.astype(np.float32)


def _normalized_views(embedding_views):
    if hasattr(embedding_views, "views") and hasattr(
        embedding_views, "reference_variant"
    ):
        return (
            tuple(embedding_views.variants),
            str(embedding_views.reference_variant),
            embedding_views.views,
        )

    bundle = embedding_views
    try:
        variant = scalar_text(bundle.metadata, "variant")
    except (KeyError, ValueError):
        variant = "real"

    @dataclass(frozen=True)
    class SingleView:
        bundle: object
        variant: str
        changed_fraction: float

        @property
        def embedding(self):
            return self.bundle.index.embedding

    changed = np.asarray(bundle.metadata.get("changed_fraction", 0.0))
    view = SingleView(bundle, variant, float(changed.item()))
    return (variant,), variant, {variant: view}


def _aggregate_alignment(numerators, masses) -> np.ndarray:
    numerator = np.sum(np.stack(numerators), axis=0, dtype=np.float64)
    mass = np.sum(np.stack(masses), axis=0, dtype=np.float64)
    return np.divide(
        numerator,
        mass,
        out=np.full_like(numerator, np.nan),
        where=mass > 0.0,
    )


def _edge_columns(graph, edges):
    if edges is None:
        return (
            graph.edge_index[0],
            graph.edge_index[1],
            graph.edge_layer,
            graph.edge_head,
            graph.edge_weight,
        )
    return edges.source, edges.target, edges.layer, edges.head, edges.weight


def _posthoc_rewire(graph, seed: int, passes: int) -> tuple[TokenEdges, int]:
    """Create a descriptive matched endpoint null without re-encoding nodes."""

    token_graph = _as_token_graph(graph)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rewired = apply_variant(
        token_graph,
        "endpoint_rewire",
        generator,
        endpoint_rewire_passes=passes,
    )
    changed = int((rewired.edges.source != token_graph.edges.source).sum().item())
    return rewired.edges, changed


def _as_token_graph(graph) -> TokenGraph:
    token_count = int(graph.token_ids.numel())
    return TokenGraph(
        sample_id=graph.sample_id,
        source_id=graph.source_id,
        task_type=graph.task_type,
        response_start=graph.response_start,
        token_count=token_count,
        response_count=graph.response_count,
        layer_count=graph.layer_count,
        head_count=graph.head_count,
        attention_floor=graph.attention_floor,
        edges=TokenEdges(
            source=graph.edge_index[0].cpu(),
            target=graph.edge_index[1].cpu(),
            layer=graph.edge_layer.cpu(),
            head=graph.edge_head.cpu(),
            weight=graph.edge_weight.cpu(),
        ),
        diagonal=graph.diagonal.cpu(),
        unresolved=graph.unresolved.cpu(),
        token_ids=graph.token_ids.cpu(),
    ).check().canonicalize()


def _sample_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _standardize_space(embedding: np.ndarray) -> np.ndarray:
    centered = embedding - embedding.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(centered**2)))
    if scale == 0.0:
        raise ValueError("embedding space has no variation")
    return centered / scale
