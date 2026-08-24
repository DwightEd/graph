"""Label-free fitting and scoring for the attention holonomy mechanism audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from experiment_protocol import canonical_source_group

from .artifacts import (
    REFERENCE_SCHEMA,
    SCORE_SCHEMA,
    load_reference,
    save_npz,
    sha256,
    transport_arrays,
)
from .config import AuditConfig, CONTROL_FEATURES, PRIMARY_FEATURES
from .features import compute_mechanism_audit
from .graph import build_attention_event_graph
from .reference import AlignedReservoir, fit_nuisance_reference
from .transport import TransportFitter


@dataclass(frozen=True)
class SourceSplit:
    fit_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    fit_groups: tuple[str, ...]
    calibration_groups: tuple[str, ...]


def _require_split(dataset, expected: str) -> None:
    actual = str(dataset.manifest.get("split"))
    if actual != expected:
        raise ValueError(f"expected dataset split {expected!r}, got {actual!r}")


def _selected_ids(dataset, task_type: str, limit: int | None) -> tuple[str, ...]:
    selected: list[str] = []
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        try:
            if task_type.casefold() == "all" or str(sample.task_type).casefold() == task_type.casefold():
                selected.append(str(sample_id))
        finally:
            sample.release_attention()
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("no samples match the requested audit scope")
    return tuple(selected)


def _split_sources(
    dataset,
    sample_ids: tuple[str, ...],
    *,
    fraction: float,
    seed: int,
) -> SourceSplit:
    grouped: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        sample = dataset[sample_id]
        try:
            grouped.setdefault(canonical_source_group(sample), []).append(sample_id)
        finally:
            sample.release_attention()
    if len(grouped) < 2:
        raise ValueError("the audit requires at least two source groups")

    def order(name: str) -> bytes:
        return hashlib.sha256(f"holoroute-audit\0{seed}\0{name}".encode()).digest()

    names = sorted(grouped, key=order)
    calibration_count = max(1, round(len(names) * float(fraction)))
    calibration_count = min(calibration_count, len(names) - 1)
    calibration_groups = set(names[:calibration_count])
    fit_groups = set(names) - calibration_groups

    def samples(groups: set[str]) -> tuple[str, ...]:
        return tuple(
            sample_id
            for group, ids in grouped.items()
            if group in groups
            for sample_id in ids
        )

    return SourceSplit(
        fit_ids=samples(fit_groups),
        calibration_ids=samples(calibration_groups),
        fit_groups=tuple(sorted(fit_groups)),
        calibration_groups=tuple(sorted(calibration_groups)),
    )


def _structure_gate(controls: np.ndarray, coverage: np.ndarray) -> dict[str, float | bool]:
    result: dict[str, float | bool] = {}
    control_names = (
        "depth_transport_gain",
        "relay_transport_gain",
        "query_set_gain",
        "relay_rewire_gain",
        "diamond_target_error",
    )
    for index, name in enumerate(control_names):
        values = controls[:, index]
        values = values[np.isfinite(values)]
        result[f"{name}_mean"] = float(values.mean()) if len(values) else float("nan")
        result[f"{name}_positive_fraction"] = (
            float(np.mean(values > 0)) if len(values) else float("nan")
        )
    result["diamond_token_coverage"] = float(np.mean(coverage[:, 5] > 0))
    result["depth_gate_pass"] = bool(result["depth_transport_gain_mean"] > 0)
    result["relay_gate_pass"] = bool(result["relay_transport_gain_mean"] > 0)
    result["query_gate_pass"] = bool(result["query_set_gain_mean"] > 0)
    result["path_gate_pass"] = bool(result["relay_rewire_gain_mean"] > 0)
    result["diamond_gate_ready"] = bool(result["diamond_token_coverage"] >= 0.05)
    return result


def fit_reference(
    dataset,
    reference_path,
    *,
    config: AuditConfig | None = None,
    task_type: str = "QA",
    limit: int | None = None,
) -> dict[str, object]:
    """Fit graph transports and position-conditioned references without labels."""

    config = AuditConfig() if config is None else config
    _require_split(dataset, "train")
    sample_ids = _selected_ids(dataset, task_type, limit)
    split = _split_sources(
        dataset,
        sample_ids,
        fraction=config.reference.calibration_fraction,
        seed=config.reference.seed,
    )
    layers = int(dataset.manifest["num_layers"])
    heads = int(dataset.manifest["num_heads"])
    fitter = TransportFitter(
        layers,
        heads,
        graph_config=config.graph,
        transport_config=config.transport,
    )

    for sample_id in tqdm(split.fit_ids, desc="fit attention transports", unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_attention_event_graph(sample, config=config.graph)
            fitter.update(graph)
        finally:
            sample.release_attention()
    transport = fitter.freeze()

    reservoir = AlignedReservoir(
        config.reference.reservoir_rows,
        config.reference.seed + 1,
    )
    control_rows: list[np.ndarray] = []
    coverage_rows: list[np.ndarray] = []
    for sample_id in tqdm(
        split.calibration_ids,
        desc="calibrate mechanism residuals",
        unit="sample",
    ):
        sample = dataset[sample_id]
        try:
            graph = build_attention_event_graph(sample, config=config.graph)
            audit = compute_mechanism_audit(
                graph,
                transport,
                seed=config.reference.seed,
            )
            reservoir.add(
                primary=audit.primary.astype(np.float32),
                nuisance=audit.nuisance.astype(np.float32),
                task=np.repeat(str(sample.task_type or ""), graph.num_response_tokens),
            )
            control_rows.append(audit.controls)
            coverage_rows.append(audit.coverage)
        finally:
            sample.release_attention()

    calibration = reservoir.values()
    nuisance_reference = fit_nuisance_reference(
        calibration["primary"],
        calibration["nuisance"],
        calibration["task"],
        config=config.reference,
    )
    gates = _structure_gate(
        np.concatenate(control_rows),
        np.concatenate(coverage_rows),
    )

    manifest_path = Path(dataset.root) / "manifest.json"
    payload = {
        "schema": np.asarray(REFERENCE_SCHEMA),
        "labels_included": np.asarray(False),
        "train_manifest_sha256": np.asarray(sha256(manifest_path)),
        "config_json": np.asarray(
            json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
        ),
        "num_layers": np.asarray(layers, dtype=np.int32),
        "num_heads": np.asarray(heads, dtype=np.int32),
        "fit_group_id": np.asarray(split.fit_groups),
        "calibration_group_id": np.asarray(split.calibration_groups),
        "primary_feature_names": np.asarray(PRIMARY_FEATURES),
        "control_feature_names": np.asarray(CONTROL_FEATURES),
        "task_names": np.asarray(nuisance_reference.task_names),
        "nuisance_coefficient": nuisance_reference.coefficient,
        "residual_median": nuisance_reference.residual_median,
        "residual_scale": nuisance_reference.residual_scale,
        "position_degree": np.asarray(
            nuisance_reference.position_degree, dtype=np.int32
        ),
    }
    payload.update(transport_arrays(transport))
    for name, value in gates.items():
        payload[name] = np.asarray(value)
    save_npz(reference_path, **payload)
    return {
        "reference": str(Path(reference_path).resolve()),
        "labels_read": False,
        "fit_groups": len(split.fit_groups),
        "calibration_groups": len(split.calibration_groups),
        **gates,
    }


def score_split(
    dataset,
    reference_path,
    score_path,
    *,
    task_type: str = "QA",
    limit: int | None = None,
    sidecar_dir=None,
) -> dict[str, object]:
    """Freeze mechanism features, residualized scores and per-sample maps."""

    _require_split(dataset, "test")
    reference_arrays, config, transport, nuisance_reference = load_reference(
        reference_path
    )
    if int(reference_arrays["num_layers"]) != int(dataset.manifest["num_layers"]):
        raise ValueError("reference and test split use different layer geometry")
    if int(reference_arrays["num_heads"]) != int(dataset.manifest["num_heads"]):
        raise ValueError("reference and test split use different head geometry")
    sample_ids = _selected_ids(dataset, task_type, limit)
    sidecar_root = Path(sidecar_dir) if sidecar_dir is not None else Path(score_path).parent / "holonomy_maps"
    sidecar_root.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "sample_id",
            "source_id",
            "task_type",
            "token_index",
            "response_length",
            "response_token_id",
            "joint_score",
            "raw_primary",
            "standardized_primary",
            "controls",
            "nuisance",
            "coverage",
            "sidecar_path",
        )
    }

    for sample_id in tqdm(sample_ids, desc="score attention holonomy", unit="sample"):
        sample = dataset[sample_id]
        try:
            graph = build_attention_event_graph(sample, config=config.graph)
            audit = compute_mechanism_audit(
                graph,
                transport,
                seed=config.reference.seed,
            )
            task = np.repeat(str(sample.task_type or ""), graph.num_response_tokens)
            standardized, joint = nuisance_reference.transform(
                audit.primary,
                audit.nuisance,
                task,
            )
            safe_name = hashlib.sha256(str(sample_id).encode()).hexdigest()[:16]
            sidecar = sidecar_root / f"{safe_name}.npz"
            save_npz(
                sidecar,
                sample_id=np.asarray(str(sample_id)),
                primary_feature_names=np.asarray(PRIMARY_FEATURES),
                control_feature_names=np.asarray(CONTROL_FEATURES),
                primary_maps=audit.primary_maps,
                control_maps=audit.control_maps,
                event_count=np.asarray(graph.num_events, dtype=np.int32),
                depth_edge_count=np.asarray(graph.depth_edge_index.shape[1], dtype=np.int32),
                relay_edge_count=np.asarray(graph.relay_edge_index.shape[1], dtype=np.int32),
                diamond_count=np.asarray(graph.diamond_index.shape[1], dtype=np.int32),
            )
            tokens = graph.num_response_tokens
            rows["sample_id"].append(np.repeat(str(sample.sample_id), tokens))
            rows["source_id"].append(np.repeat(str(sample.source_id), tokens))
            rows["task_type"].append(task)
            rows["token_index"].append(np.arange(tokens, dtype=np.int32))
            rows["response_length"].append(np.full(tokens, tokens, dtype=np.int32))
            rows["response_token_id"].append(
                graph.response_token_ids.detach().cpu().numpy().astype(np.int64)
            )
            rows["joint_score"].append(joint)
            rows["raw_primary"].append(audit.primary)
            rows["standardized_primary"].append(standardized)
            rows["controls"].append(audit.controls)
            rows["nuisance"].append(audit.nuisance)
            rows["coverage"].append(audit.coverage)
            rows["sidecar_path"].append(np.repeat(str(sidecar.resolve()), tokens))
        finally:
            sample.release_attention()

    arrays = {name: np.concatenate(values) for name, values in rows.items()}
    save_npz(
        score_path,
        schema=np.asarray(SCORE_SCHEMA),
        labels_included=np.asarray(False),
        reference_path=np.asarray(str(Path(reference_path).resolve())),
        reference_sha256=np.asarray(sha256(reference_path)),
        dataset_manifest_sha256=np.asarray(sha256(Path(dataset.root) / "manifest.json")),
        primary_score=np.asarray("joint_score"),
        primary_feature_names=np.asarray(PRIMARY_FEATURES),
        control_feature_names=np.asarray(CONTROL_FEATURES),
        nuisance_names=np.asarray(
            (
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
        ),
        **arrays,
    )
    return {
        "scores": str(Path(score_path).resolve()),
        "labels_read": False,
        "samples": len(sample_ids),
        "tokens": len(arrays["joint_score"]),
    }
