"""Fit and score commands for the label-free attention phenomenology audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from research_dataset import open_research_dataset

from .config import FAMILY_NAMES, FEATURE_NAMES, PhenomenologyConfig
from .features import SamplePhenomenology, analyze_routing
from .reference import (
    Reservoir,
    family_atypicality,
    fit_reference_from_reservoirs,
    load_reference,
    save_reference,
    standardize_features,
    token_buckets,
)
from .routing import collect_routing_edges, rewire_exact_endpoints


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_seed(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return int(base_seed) + int.from_bytes(digest[:4], "little")


def _sample_iterator(dataset, limit: int | None):
    sample_ids = dataset.sample_ids if limit is None else dataset.sample_ids[: int(limit)]
    for sample_id in sample_ids:
        yield dataset[sample_id]


def fit_reference(
    *,
    train_split,
    output,
    device="cpu",
    config: PhenomenologyConfig | None = None,
    reservoir_rows: int = 2048,
    limit: int | None = None,
) -> None:
    """Fit task/causal-position robust references without opening labels."""

    config = PhenomenologyConfig() if config is None else config
    dataset = open_research_dataset(train_split, device=device)
    rng = np.random.default_rng(config.random_seed)
    reservoirs: dict[tuple[str, int], Reservoir] = {}

    for sample in _sample_iterator(dataset, limit):
        edges = collect_routing_edges(sample, config=config)
        analysis = analyze_routing(edges, config=config)
        values = analysis.layer_features.detach().cpu().numpy().astype(np.float32)
        buckets = token_buckets(len(values), config.causal_position_bins)
        task = str(sample.task_type or "unknown")
        for token, bucket in enumerate(buckets):
            for condition_task in (task, "__all__"):
                key = (condition_task, int(bucket))
                reservoirs.setdefault(key, Reservoir(reservoir_rows, rng)).add(values[token])
        sample.release_attention()

    reference = fit_reference_from_reservoirs(
        reservoirs,
        config=config,
        config_json=json.dumps(config.to_dict(), sort_keys=True),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_reference(output, reference)


def _core_sample_arrays(
    analysis: SamplePhenomenology,
    *,
    task: str,
    reference,
    config: PhenomenologyConfig,
):
    layer_features = analysis.layer_features.detach().cpu().numpy().astype(np.float32)
    buckets = token_buckets(len(layer_features), config.causal_position_bins)
    standardized = standardize_features(
        layer_features,
        task=task,
        buckets=buckets,
        reference=reference,
    )
    return layer_features, standardized, family_atypicality(standardized), buckets


def _detail_arrays(analysis: SamplePhenomenology) -> dict[str, np.ndarray]:
    routing = analysis.routing
    edges = routing.edges
    return {
        "role_probability": routing.role_probability.detach().cpu().numpy().astype(np.float16),
        "persistence_deaths": analysis.geometry.persistence_deaths.detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        "head_grounding_lower": analysis.provenance.head_lower.detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        "head_grounding_upper": analysis.provenance.head_upper.detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        "aggregate_grounding_lower": analysis.provenance.aggregate_lower.detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        "aggregate_grounding_upper": analysis.provenance.aggregate_upper.detach()
        .cpu()
        .numpy()
        .astype(np.float16),
        "edge_layer": edges.layer.detach().cpu().numpy().astype(np.int16),
        "edge_head": edges.head.detach().cpu().numpy().astype(np.int16),
        "edge_query": edges.query.detach().cpu().numpy().astype(np.int32),
        "edge_source": edges.source.detach().cpu().numpy().astype(np.int32),
        "edge_weight": edges.weight.detach().cpu().numpy().astype(np.float32),
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
    """Freeze raw mechanism fields and anomaly scores before labels are opened."""

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

    for sample in _sample_iterator(dataset, limit):
        task = str(sample.task_type or "unknown")
        edges = collect_routing_edges(sample, config=config)
        analysis = analyze_routing(edges, config=config)
        layer_features, standardized, family_scores, buckets = _core_sample_arrays(
            analysis,
            task=task,
            reference=reference,
            config=config,
        )

        arrays: dict[str, np.ndarray] = {
            "schema": np.asarray("attention-phenomenology-score-v1"),
            "sample_id": np.asarray(sample.sample_id),
            "source_id": np.asarray(sample.source_id),
            "task_type": np.asarray(task),
            "token_index": np.arange(len(layer_features), dtype=np.int32),
            "causal_position_bucket": buckets,
            "feature_names": np.asarray(FEATURE_NAMES, dtype=str),
            "family_names": np.asarray(FAMILY_NAMES, dtype=str),
            "layer_features": layer_features,
            "standardized_features": standardized,
            "family_scores": family_scores,
        }

        if rewire:
            rewired_edges = rewire_exact_endpoints(
                edges,
                config=config,
                seed=_sample_seed(sample.sample_id, config.random_seed),
            )
            rewired_analysis = analyze_routing(rewired_edges, config=config)
            rewired_features, rewired_standardized, rewired_scores, _ = _core_sample_arrays(
                rewired_analysis,
                task=task,
                reference=reference,
                config=config,
            )
            role_error = float(
                (
                    analysis.routing.role_probability
                    - rewired_analysis.routing.role_probability
                )
                .abs()
                .max()
                .item()
            )
            arrays.update(
                rewired_layer_features=rewired_features,
                rewired_standardized_features=rewired_standardized,
                rewired_family_scores=rewired_scores,
                rewire_role_max_abs_error=np.asarray(role_error, dtype=np.float32),
            )

        sample_path = sample_dir / f"{sample.sample_id}.npz"
        np.savez_compressed(sample_path, **arrays)

        if sample.sample_id in details:
            detail_path = detail_dir / f"{sample.sample_id}.npz"
            detail_arrays = _detail_arrays(analysis)
            if rewire:
                detail_arrays["rewired_edge_source"] = (
                    rewired_edges.source.detach().cpu().numpy().astype(np.int32)
                )
            np.savez_compressed(detail_path, **detail_arrays)
        else:
            detail_path = None

        manifest_rows.append(
            {
                "sample_id": sample.sample_id,
                "source_id": sample.source_id,
                "task_type": task,
                "response_length": int(len(layer_features)),
                "score_path": str(sample_path.relative_to(output_dir)),
                "detail_path": None
                if detail_path is None
                else str(detail_path.relative_to(output_dir)),
            }
        )
        sample.release_attention()

    manifest = {
        "schema": "attention-phenomenology-manifest-v1",
        "labels_read": False,
        "split_root": str(Path(split_root).resolve()),
        "reference_path": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "config": config.to_dict(),
        "feature_names": list(FEATURE_NAMES),
        "family_names": list(FAMILY_NAMES),
        "role_names": list(config.role_names),
        "rewired_control": bool(rewire),
        "samples": manifest_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
