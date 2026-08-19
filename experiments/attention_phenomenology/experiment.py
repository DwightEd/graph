"""Label-free fit and score stages for the attention phenomenology audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from research_dataset import open_research_dataset

from .artifacts import MANIFEST_SCHEMA, SCORE_SCHEMA, save_npz, sha256_file, write_json
from .config import PhenomenologyConfig
from .features import SamplePhenomenology, analyze_routing
from .hypotheses import FAMILY_NAMES, FEATURE_NAMES
from .nulls import rewire_exact_endpoints
from .reference import (
    Reservoir,
    family_atypicality,
    family_layer_atypicality,
    fit_reference_from_reservoirs,
    load_reference,
    save_reference,
    standardize_features,
    token_buckets,
)
from .routing import collect_routing_edges


def _sample_seed(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "little")


def _samples(dataset, limit: int | None):
    sample_ids = dataset.sample_ids if limit is None else dataset.sample_ids[:limit]
    return (dataset[sample_id] for sample_id in sample_ids)


def fit_reference(
    *,
    train_split,
    output,
    device="cpu",
    config: PhenomenologyConfig | None = None,
    reservoir_rows: int = 2048,
    limit: int | None = None,
) -> None:
    """Fit robust task/causal-position references without opening labels."""

    config = PhenomenologyConfig() if config is None else config
    dataset = open_research_dataset(train_split, device=device)
    rng = np.random.default_rng(config.random_seed)
    reservoirs: dict[tuple[str, int], Reservoir] = {}

    for sample in _samples(dataset, limit):
        analysis = analyze_routing(
            collect_routing_edges(sample, config=config), config=config
        )
        values = analysis.layer_features.cpu().numpy().astype(np.float32)
        buckets = token_buckets(len(values), config.causal_position_bins)
        task = str(sample.task_type or "unknown")
        for token, bucket in enumerate(buckets):
            for condition_task in (task, "__all__"):
                key = (condition_task, int(bucket))
                reservoirs.setdefault(key, Reservoir(reservoir_rows, rng)).add(
                    values[token]
                )
        sample.release_attention()

    reference = fit_reference_from_reservoirs(
        reservoirs,
        config=config,
        config_json=json.dumps(config.to_dict(), sort_keys=True),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_reference(output, reference)


def _sample_scores(
    analysis: SamplePhenomenology,
    *,
    task: str,
    reference,
    config: PhenomenologyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    layer_features = analysis.layer_features.cpu().numpy().astype(np.float32)
    buckets = token_buckets(len(layer_features), config.causal_position_bins)
    standardized = standardize_features(
        layer_features,
        task=task,
        buckets=buckets,
        reference=reference,
    )
    family_layer = family_layer_atypicality(standardized)
    family = family_atypicality(family_layer)
    return layer_features, standardized, family_layer, family, buckets


def _detail_arrays(analysis: SamplePhenomenology) -> dict[str, np.ndarray]:
    routing = analysis.routing
    edges = routing.edges
    return {
        "role_probability": routing.role_probability.cpu().numpy().astype(np.float16),
        "known_role_probability": routing.known_role_probability.cpu()
        .numpy()
        .astype(np.float16),
        "known_persistence_deaths": analysis.known_geometry.persistence_deaths.cpu()
        .numpy()
        .astype(np.float16),
        "full_persistence_deaths": analysis.full_geometry.persistence_deaths.cpu()
        .numpy()
        .astype(np.float16),
        "head_grounding_lower": analysis.provenance.head_lower.cpu()
        .numpy()
        .astype(np.float16),
        "head_grounding_upper": analysis.provenance.head_upper.cpu()
        .numpy()
        .astype(np.float16),
        "aggregate_grounding_lower": analysis.provenance.aggregate_lower.cpu()
        .numpy()
        .astype(np.float16),
        "aggregate_grounding_upper": analysis.provenance.aggregate_upper.cpu()
        .numpy()
        .astype(np.float16),
        "source_mass": routing.source_mass.cpu().numpy().astype(np.float16),
        "edge_layer": edges.layer.cpu().numpy().astype(np.int16),
        "edge_head": edges.head.cpu().numpy().astype(np.int16),
        "edge_query": edges.query.cpu().numpy().astype(np.int32),
        "edge_source": edges.source.cpu().numpy().astype(np.int32),
        "edge_weight": edges.weight.cpu().numpy().astype(np.float32),
    }


def score_split(
    *,
    split_root,
    reference_path,
    output_dir,
    device="cpu",
    config: PhenomenologyConfig | None = None,
    rewire: bool = True,
    detail_sample_ids: tuple[str, ...] = (),
    limit: int | None = None,
) -> None:
    """Freeze mechanism fields and null-control scores before labels are opened."""

    config = PhenomenologyConfig() if config is None else config
    reference_path = Path(reference_path).resolve()
    reference = load_reference(reference_path)
    if json.loads(reference.config_json) != config.to_dict():
        raise ValueError("score configuration differs from the fitted reference")

    dataset = open_research_dataset(split_root, device=device)
    output_dir = Path(output_dir)
    sample_dir = output_dir / "samples"
    detail_dir = output_dir / "details"
    sample_dir.mkdir(parents=True, exist_ok=True)
    detail_dir.mkdir(parents=True, exist_ok=True)
    details = set(map(str, detail_sample_ids))
    manifest_rows = []

    for sample in _samples(dataset, limit):
        task = str(sample.task_type or "unknown")
        edges = collect_routing_edges(sample, config=config)
        analysis = analyze_routing(edges, config=config)
        layer, standardized, family_layer, family, buckets = _sample_scores(
            analysis, task=task, reference=reference, config=config
        )
        arrays = {
            "schema": np.asarray(SCORE_SCHEMA),
            "sample_id": np.asarray(sample.sample_id),
            "source_id": np.asarray(sample.source_id),
            "task_type": np.asarray(task),
            "token_index": np.arange(len(layer), dtype=np.int32),
            "causal_position_bucket": buckets,
            "feature_names": np.asarray(FEATURE_NAMES, dtype=str),
            "family_names": np.asarray(FAMILY_NAMES, dtype=str),
            "layer_features": layer,
            "standardized_features": standardized,
            "family_layer_scores": family_layer,
            "family_scores": family,
        }

        rewired_edges = None
        if rewire:
            null = rewire_exact_endpoints(
                edges,
                config=config,
                seed=_sample_seed(sample.sample_id, config.random_seed),
            )
            rewired_edges = null.edges
            rewired_analysis = analyze_routing(rewired_edges, config=config)
            rewired_layer, rewired_standardized, rewired_family_layer, rewired_family, _ = (
                _sample_scores(
                    rewired_analysis,
                    task=task,
                    reference=reference,
                    config=config,
                )
            )
            role_error = (
                analysis.routing.role_probability
                - rewired_analysis.routing.role_probability
            ).abs().max().item()
            arrays.update(
                rewired_layer_features=rewired_layer,
                rewired_standardized_features=rewired_standardized,
                rewired_family_layer_scores=rewired_family_layer,
                rewired_family_scores=rewired_family,
                rewire_role_max_abs_error=np.asarray(role_error, dtype=np.float32),
                rewire_changed_fraction=np.asarray(
                    null.changed_fraction, dtype=np.float32
                ),
            )

        sample_path = sample_dir / f"{sample.sample_id}.npz"
        save_npz(sample_path, **arrays)

        detail_path = None
        if sample.sample_id in details:
            detail_path = detail_dir / f"{sample.sample_id}.npz"
            detail = _detail_arrays(analysis)
            if rewired_edges is not None:
                detail["rewired_edge_source"] = rewired_edges.source.cpu().numpy().astype(
                    np.int32
                )
            save_npz(detail_path, **detail)

        manifest_rows.append(
            {
                "sample_id": sample.sample_id,
                "source_id": sample.source_id,
                "task_type": task,
                "response_length": len(layer),
                "score_path": str(sample_path.relative_to(output_dir)),
                "detail_path": None
                if detail_path is None
                else str(detail_path.relative_to(output_dir)),
            }
        )
        sample.release_attention()

    write_json(
        output_dir / "manifest.json",
        {
            "schema": MANIFEST_SCHEMA,
            "labels_read": False,
            "split_root": str(Path(split_root).resolve()),
            "reference_path": str(reference_path),
            "reference_sha256": sha256_file(reference_path),
            "config": config.to_dict(),
            "feature_names": list(FEATURE_NAMES),
            "family_names": list(FAMILY_NAMES),
            "role_names": list(config.role_names),
            "rewired_control": rewire,
            "samples": manifest_rows,
        },
    )
