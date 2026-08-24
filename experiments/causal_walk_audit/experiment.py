"""Label-free fitting and score freezing for the causal-walk audit."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import random

import numpy as np
from tqdm.auto import tqdm

from .anchors import anchors_for_sample, load_anchor_manifest
from .artifacts import save_npz, write_json
from .config import WalkAuditConfig
from .lineage import propagate_anchor_lineage
from .markov import NestedMarkovModel, RowReservoir
from .trajectory import (
    LayerTrajectory,
    TrajectoryReference,
    layer_trajectory,
    summarize_trajectory,
)
from .walks import build_layer_event_graph, build_nested_features

MODEL_SCHEMA = "causal-walk-audit-model-v1"
SCORE_SCHEMA = "causal-walk-audit-score-v1"
MANIFEST_SCHEMA = "causal-walk-audit-manifest-v1"


def _open_dataset(root, *, device: str):
    from research_dataset import open_research_dataset

    return open_research_dataset(root, device=device)


def _selected_ids(dataset, task_type: str, limit: int | None) -> list[str]:
    selected = [
        str(sample_id)
        for sample_id in dataset.sample_ids
        if task_type.casefold() == "all"
        or str(dataset[sample_id].task_type).casefold() == task_type.casefold()
    ]
    return selected if limit is None else selected[:limit]


def _source_split(dataset, sample_ids: list[str], fraction: float, seed: int):
    groups: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        groups.setdefault(str(dataset[sample_id].source_id), []).append(sample_id)
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    count = max(1, min(len(names) - 1, round(len(names) * fraction)))
    validation = set(names[:count])
    fit = [
        item
        for name in names
        if name not in validation
        for item in groups[name]
    ]
    heldout = [
        item
        for name in names
        if name in validation
        for item in groups[name]
    ]
    return fit, heldout


def _sample_seed(sample_id: str, seed: int, offset: int = 0) -> int:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    return seed + offset + int.from_bytes(digest[:4], "little")


def _routing_for_sample(sample, config: WalkAuditConfig):
    from attention_lifecycle import loaded_attention
    from experiments.attention_phenomenology.config import PhenomenologyConfig
    from experiments.attention_phenomenology.routing import (
        build_routing_state,
        collect_routing_edges,
    )

    with loaded_attention(sample) as attention:
        token_ids = (
            attention.token_ids[attention.response_idx :]
            .cpu()
            .numpy()
            .astype(np.int32)
            .copy()
        )
        edges = collect_routing_edges(
            sample,
            config=PhenomenologyConfig(block_rows=config.block_rows),
        )
    return build_routing_state(edges), token_ids


def _sample_features(sample, manifest, config: WalkAuditConfig):
    routing, token_ids = _routing_for_sample(sample, config)
    anchors = anchors_for_sample(
        str(sample.sample_id),
        routing.edges.response_idx,
        manifest=manifest,
        max_anchors=config.max_anchors,
        chunk_tokens=config.prompt_chunk_tokens,
        device=routing.edges.device,
    )
    lineage = propagate_anchor_lineage(routing, anchors)
    events = build_layer_event_graph(routing)
    nested = build_nested_features(
        routing,
        lineage,
        events,
        max_anchors=config.max_anchors,
    )
    layer = layer_trajectory(
        lineage,
        minimum_anchor_mass=config.minimum_anchor_mass,
    )
    return routing, anchors, nested, layer, token_ids


class _ScalarReservoir:
    def __init__(self, capacity: int, seed: int):
        self.js = RowReservoir(capacity, seed)
        self.evidence = RowReservoir(capacity, seed + 1)
        self.response = RowReservoir(capacity, seed + 2)

    def add(self, layer: LayerTrajectory) -> None:
        js = layer.anchor_js[np.isfinite(layer.anchor_js)].reshape(-1, 1)
        if len(js):
            self.js.add(value=js)
        self.evidence.add(value=layer.known_anchor.reshape(-1, 1))
        self.response.add(value=layer.response_base.reshape(-1, 1))

    def reference(self) -> TrajectoryReference:
        js = self.js.matrix("value").ravel() if self.js.rows else np.zeros(1)
        evidence = self.evidence.matrix("value").ravel()
        response = self.response.matrix("value").ravel()
        return TrajectoryReference(
            js_high=float(np.quantile(js, 0.9)),
            js_low=float(np.quantile(js, 0.5)),
            evidence_high=float(np.quantile(evidence, 0.75)),
            evidence_low=float(np.quantile(evidence, 0.25)),
            response_high=float(np.quantile(response, 0.75)),
        )


def _add_nested(reservoir: RowReservoir, nested) -> None:
    reservoir.add(
        order1=nested.order1.detach().cpu().numpy(),
        order2=nested.order2.detach().cpu().numpy(),
        order3=nested.order3.detach().cpu().numpy(),
        target=nested.target.detach().cpu().numpy(),
    )


def _fit_from_reservoir(
    reservoir: RowReservoir,
    config: WalkAuditConfig,
    *,
    seed: int,
) -> NestedMarkovModel:
    return NestedMarkovModel.fit(
        reservoir.matrix("order1"),
        reservoir.matrix("order2"),
        reservoir.matrix("order3"),
        reservoir.matrix("target"),
        alpha=config.ridge_alpha,
        seed=seed,
    )


def fit_walk_audit(
    *,
    train_split,
    output_dir,
    device="cpu",
    config: WalkAuditConfig | None = None,
    task_type="QA",
    limit=None,
    anchor_manifest=None,
):
    config = WalkAuditConfig() if config is None else config
    manifest = load_anchor_manifest(anchor_manifest)
    dataset = _open_dataset(train_split, device=device)
    sample_ids = _selected_ids(dataset, task_type, limit)
    fit_ids, validation_ids = _source_split(
        dataset,
        sample_ids,
        0.1,
        config.random_seed,
    )

    fit_rows = RowReservoir(config.train_reservoir_rows, config.random_seed)
    validation_rows = RowReservoir(
        max(config.train_reservoir_rows // 4, 1_000),
        config.random_seed + 1,
    )
    all_rows = RowReservoir(
        config.train_reservoir_rows,
        config.random_seed + 2,
    )
    scalar = _ScalarReservoir(
        config.train_reservoir_rows,
        config.random_seed + 3,
    )

    validation_set = set(validation_ids)
    iterator = tqdm(
        sample_ids,
        desc="fit causal-walk audit",
        unit="sample",
        dynamic_ncols=True,
        disable=not config.show_progress,
    )
    anchor_modes: dict[str, int] = {}
    for sample_id in iterator:
        sample = dataset[sample_id]
        _, anchors, nested, layer, _ = _sample_features(
            sample,
            manifest,
            config,
        )
        _add_nested(all_rows, nested)
        destination = validation_rows if sample_id in validation_set else fit_rows
        _add_nested(destination, nested)
        scalar.add(layer)
        anchor_modes[anchors.mode] = anchor_modes.get(anchors.mode, 0) + 1

    provisional = _fit_from_reservoir(
        fit_rows,
        config,
        seed=config.random_seed,
    )
    validation = provisional.validation_summary(
        validation_rows.matrix("order1"),
        validation_rows.matrix("order2"),
        validation_rows.matrix("order3"),
        validation_rows.matrix("target"),
        seed=config.random_seed + 10,
    )
    final = _fit_from_reservoir(
        all_rows,
        config,
        seed=config.random_seed + 20,
    )
    trajectory_reference = scalar.reference()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.npz"
    metadata = {
        "schema": MODEL_SCHEMA,
        "labels_read": False,
        "train_split": str(Path(train_split).resolve()),
        "task_type": task_type,
        "config": asdict(config),
        "trajectory_reference": trajectory_reference.to_dict(),
        "anchor_modes": anchor_modes,
        "fit_samples": len(fit_ids),
        "validation_samples": len(validation_ids),
        "fit_rows": fit_rows.rows,
        "validation_rows": validation_rows.rows,
        "validation": validation,
    }
    final.save(model_path, metadata)
    write_json(output_dir / "training.json", metadata | {"model": "model.npz"})
    return model_path


def _aggregate_rows(
    value: np.ndarray,
    token: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.bincount(token, weights=value, minlength=count).astype(np.float32)
    rows = np.bincount(token, minlength=count).astype(np.float32)
    return total / np.maximum(rows, 1.0), rows


def _direct_role(routing) -> np.ndarray:
    prompt = routing.prompt_mass.mean(dim=(1, 2)).cpu().numpy()
    response = routing.response_mass.mean(dim=(1, 2)).cpu().numpy()
    return (
        response / np.maximum(prompt + response, 1e-8)
    ).astype(np.float32)


def _score_sample(
    sample,
    manifest,
    model,
    metadata,
    config: WalkAuditConfig,
):
    routing, anchors, nested, layer, token_ids = _sample_features(
        sample,
        manifest,
        config,
    )
    errors = model.errors(
        nested.order1.cpu().numpy(),
        nested.order2.cpu().numpy(),
        nested.order3.cpu().numpy(),
        nested.target.cpu().numpy(),
        seed=_sample_seed(str(sample.sample_id), config.random_seed),
    )
    token = nested.token_index.cpu().numpy().astype(np.int32)
    token_count = routing.edges.num_response_tokens
    aggregated = {
        name: _aggregate_rows(value, token, token_count)[0]
        for name, value in errors.items()
    }
    valid_rows = _aggregate_rows(
        np.ones_like(token, dtype=np.float32),
        token,
        token_count,
    )[1]

    reference = TrajectoryReference.from_dict(metadata["trajectory_reference"])
    trajectory = summarize_trajectory(
        layer,
        reference,
        horizon=config.score_horizon,
    )
    generator = __import__("torch").Generator(device=routing.edges.device)
    null_js = []
    for replicate in range(config.anchor_shuffle_replicates):
        generator.manual_seed(
            _sample_seed(
                str(sample.sample_id),
                config.random_seed,
                replicate + 1,
            )
        )
        shuffled = anchors.permuted(generator)
        shuffled_lineage = propagate_anchor_lineage(routing, shuffled)
        shuffled_layer = layer_trajectory(
            shuffled_lineage,
            minimum_anchor_mass=config.minimum_anchor_mass,
        )
        finite = np.isfinite(shuffled_layer.anchor_js)
        null_js.append(
            np.where(finite, shuffled_layer.anchor_js, 0.0).sum(axis=1)
            / np.maximum(finite.sum(axis=1), 1)
        )
    if null_js:
        anchor_null = np.stack(null_js).mean(axis=0).astype(np.float32)
    else:
        anchor_null = np.zeros(token_count, dtype=np.float32)

    scores = {
        **aggregated,
        "direct_role": _direct_role(routing),
        "anchor_js_mean": trajectory.anchor_js_mean,
        "anchor_js_peak": trajectory.anchor_js_peak,
        "anchor_js_excess": trajectory.anchor_js_mean - anchor_null,
        "recoupling_depth": trajectory.recoupling_depth,
        "recoupling_failure": trajectory.recoupling_failure,
        "response_persistence": trajectory.response_persistence,
        "evidence_escape": trajectory.evidence_escape,
        "lock_in": trajectory.lock_in,
        "known_anchor_mass": layer.known_anchor.mean(axis=1).astype(np.float32),
        "response_base_mass": layer.response_base.mean(axis=1).astype(np.float32),
    }
    return {
        "schema": np.asarray(SCORE_SCHEMA),
        "labels_included": np.asarray(False),
        "sample_id": np.asarray(str(sample.sample_id)),
        "source_id": np.asarray(str(sample.source_id)),
        "task_type": np.asarray(str(sample.task_type)),
        "response_token_ids": token_ids,
        "token_index": np.arange(token_count, dtype=np.int32),
        "valid_rows": valid_rows.astype(np.int16),
        "score_names": np.asarray(tuple(scores), dtype=str),
        "scores": np.stack([scores[name] for name in scores], axis=1),
        "anchor_names": np.asarray(anchors.names, dtype=str),
        "anchor_kinds": np.asarray(anchors.kinds, dtype=str),
        "anchor_mode": np.asarray(anchors.mode),
        "anchor_js_map": layer.anchor_js.astype(np.float16),
        "known_anchor_map": layer.known_anchor.astype(np.float16),
        "response_base_map": layer.response_base.astype(np.float16),
    }


def score_walk_audit(
    *,
    split_root,
    model_path,
    output_dir,
    device="cpu",
    task_type="QA",
    limit=None,
    anchor_manifest=None,
):
    model, metadata = NestedMarkovModel.load(model_path)
    config = WalkAuditConfig(**metadata["config"])
    manifest = load_anchor_manifest(anchor_manifest)
    dataset = _open_dataset(split_root, device=device)
    sample_ids = _selected_ids(dataset, task_type, limit)
    output_dir = Path(output_dir)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    iterator = tqdm(
        sample_ids,
        desc="score causal-walk audit",
        unit="sample",
        dynamic_ncols=True,
        disable=not config.show_progress,
    )
    for sample_id in iterator:
        sample = dataset[sample_id]
        arrays = _score_sample(sample, manifest, model, metadata, config)
        filename = f"{hashlib.sha256(sample_id.encode()).hexdigest()[:20]}.npz"
        save_npz(sample_dir / filename, **arrays)
        rows.append(
            {
                "sample_id": sample_id,
                "source_id": str(sample.source_id),
                "task_type": str(sample.task_type),
                "score_path": f"samples/{filename}",
                "tokens": int(len(arrays["token_index"])),
                "anchor_mode": str(arrays["anchor_mode"].item()),
            }
        )

    write_json(
        output_dir / "manifest.json",
        {
            "schema": MANIFEST_SCHEMA,
            "labels_read": False,
            "split_root": str(Path(split_root).resolve()),
            "model_path": str(Path(model_path).resolve()),
            "task_type": task_type,
            "config": metadata["config"],
            "validation": metadata["validation"],
            "trajectory_reference": metadata["trajectory_reference"],
            "samples": rows,
        },
    )
    return output_dir / "manifest.json"
