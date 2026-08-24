"""Label-free fit and held-out scoring orchestration.

Mathematical operations live in their dedicated modules.  This file owns only
sample-group partitioning, bounded reservoirs, multi-pass orchestration, and
frozen artifact assembly.  It never asks a dataset for labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from experiment_protocol import (
    FrozenFile,
    HeldOutSourceAudit,
    canonical_source_group,
    dataset_manifest_sha256,
)

from .artifacts import (
    load_reference,
    load_score_artifact,
    save_reference,
    save_score_artifact,
    verify_score_provenance,
)
from .calibration import (
    AlignedFloat32Reservoir,
    CalibrationReference,
    build_calibration,
    score_channels,
)
from .change_lockin import (
    MedianMAD,
    RobustChangeStats,
    change_lockin_score,
    fit_robust_change_stats,
    prompt_lineage_drop,
)
from .config import (
    CalibrationConfig,
    ChangeConfig,
    DeBruijnConfig,
    GraphConfig,
)
from .debruijn import DeBruijnAccumulator, FrozenDeBruijn
from .graph_builder import CausalRoutingGraph, build_causal_routing_graph
from .layered_automaton import (
    R_PLUS,
    STATE_NAMES,
    LayeredAutomatonResult,
    layered_attention_automaton,
)
from .nulls import RewireResult, causal_endpoint_rewire


@dataclass(frozen=True)
class ExperimentConfig:
    """All pre-registered, label-free settings for one reference."""

    graph: GraphConfig = GraphConfig()
    debruijn: DeBruijnConfig = DeBruijnConfig()
    change: ChangeConfig = ChangeConfig()
    calibration: CalibrationConfig = CalibrationConfig()

    def validate(self) -> None:
        self.graph.validate()
        self.debruijn.validate()
        self.change.validate()
        self.calibration.validate()
        if self.debruijn.soft_top_k > len(STATE_NAMES):
            raise ValueError("soft_top_k exceeds the five-state automaton")


@dataclass(frozen=True)
class ThreeWaySourceSplit:
    fit_sample_ids: tuple[str, ...]
    channel_sample_ids: tuple[str, ...]
    fusion_sample_ids: tuple[str, ...]
    fit_group_ids: tuple[str, ...]
    channel_group_ids: tuple[str, ...]
    fusion_group_ids: tuple[str, ...]

    @property
    def calibration_group_ids(self) -> tuple[str, ...]:
        return tuple(sorted((*self.channel_group_ids, *self.fusion_group_ids)))


@dataclass(frozen=True)
class EncodedRouting:
    graph: CausalRoutingGraph
    automaton: LayeredAutomatonResult
    route: torch.Tensor
    prompt_lineage: torch.Tensor
    surprisal: torch.Tensor
    predicted_route: torch.Tensor
    phase: object | None


@dataclass(frozen=True)
class RuntimeReference:
    arrays: dict[str, np.ndarray]
    config: ExperimentConfig
    debruijn: FrozenDeBruijn
    change_stats: RobustChangeStats
    calibration: CalibrationReference


def _selected_sample_ids(dataset, sample_ids=None, limit=None) -> tuple[str, ...]:
    available = tuple(map(str, dataset.sample_ids))
    if sample_ids is None:
        selected = available
    else:
        selected = tuple(map(str, sample_ids))
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("sample_ids must be non-empty and unique")
        if not set(selected).issubset(available):
            raise ValueError("sample_ids contain entries outside the dataset")
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError("no samples were selected")
    return selected


def _require_dataset_split(dataset, expected: str) -> None:
    """Enforce the train/test role at the core API boundary."""

    manifest = getattr(dataset, "manifest", None)
    actual = manifest.get("split") if isinstance(manifest, dict) else None
    if str(actual) != str(expected):
        raise ValueError(
            f"typed-path {expected} operation requires manifest split "
            f"{expected!r}, got {actual!r}"
        )


def _capture_dataset_manifest(dataset) -> FrozenFile:
    """Capture the exact manifest bytes used throughout a long-running pass."""

    return FrozenFile.capture(Path(dataset.root) / "manifest.json")


def _group_order(group_id: str, seed: int) -> bytes:
    return hashlib.sha256(
        f"ctpdb-three-stream-v1\0{int(seed)}\0{group_id}".encode("utf-8")
    ).digest()


def split_three_source_streams(
    dataset,
    sample_ids,
    *,
    config: CalibrationConfig,
) -> ThreeWaySourceSplit:
    """Split complete source groups into fit/channel/fusion streams."""

    config.validate()
    selected = _selected_sample_ids(dataset, sample_ids)
    group_by_sample: dict[str, str] = {}
    grouped: dict[str, list[str]] = {}
    for sample_id in selected:
        sample = dataset[sample_id]
        try:
            group = canonical_source_group(sample)
            group_by_sample[sample_id] = group
            grouped.setdefault(group, []).append(sample_id)
        finally:
            sample.release_attention()
    if len(grouped) < 3:
        raise ValueError(
            "three-stream calibration needs at least three complete source groups"
        )

    ordered = sorted(grouped, key=lambda value: _group_order(value, config.seed))
    group_count = len(ordered)
    channel_count = max(1, round(group_count * config.channel_fraction))
    fusion_count = max(1, round(group_count * config.fusion_fraction))
    while channel_count + fusion_count > group_count - 1:
        if channel_count >= fusion_count and channel_count > 1:
            channel_count -= 1
        elif fusion_count > 1:
            fusion_count -= 1
        else:
            raise ValueError("source groups cannot form three non-empty streams")

    channel_groups = set(ordered[:channel_count])
    fusion_groups = set(ordered[channel_count : channel_count + fusion_count])
    fit_groups = set(ordered).difference(channel_groups | fusion_groups)

    def samples(groups: set[str]) -> tuple[str, ...]:
        return tuple(
            sample_id for sample_id in selected if group_by_sample[sample_id] in groups
        )

    return ThreeWaySourceSplit(
        fit_sample_ids=samples(fit_groups),
        channel_sample_ids=samples(channel_groups),
        fusion_sample_ids=samples(fusion_groups),
        fit_group_ids=tuple(sorted(fit_groups)),
        channel_group_ids=tuple(sorted(channel_groups)),
        fusion_group_ids=tuple(sorted(fusion_groups)),
    )


def _json_config(config, **extra) -> str:
    value = config.to_dict()
    value.update(extra)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _config_from_reference(reference) -> ExperimentConfig:
    def parsed(name: str) -> dict:
        value = json.loads(str(np.asarray(reference[name]).item()))
        if not isinstance(value, dict):
            raise ValueError(f"reference field {name} must encode an object")
        return value

    graph = GraphConfig(**parsed("graph_config_json"))
    debruijn_values = parsed("debruijn_config_json")
    debruijn_values.pop("num_states", None)
    debruijn = DeBruijnConfig(**debruijn_values)
    change = ChangeConfig(**parsed("change_config_json"))
    calibration = CalibrationConfig(**parsed("calibration_config_json"))
    result = ExperimentConfig(
        graph=graph,
        debruijn=debruijn,
        change=change,
        calibration=calibration,
    )
    result.validate()
    return result


def _print_progress(stage: str, index: int, total: int, sample_id: str) -> None:
    print(f"[{stage}] {index + 1}/{total} sample={sample_id}", flush=True)


@torch.no_grad()
def _encode_structure(
    sample,
    *,
    graph_config: GraphConfig,
    debruijn: FrozenDeBruijn | None = None,
    change_stats: RobustChangeStats | None = None,
    change_config: ChangeConfig | None = None,
) -> EncodedRouting:
    graph = build_causal_routing_graph(sample, config=graph_config)
    return _encode_graph(
        graph,
        debruijn=debruijn,
        change_stats=change_stats,
        change_config=change_config,
    )


@torch.no_grad()
def _encode_graph(
    graph: CausalRoutingGraph,
    *,
    debruijn: FrozenDeBruijn | None,
    change_stats: RobustChangeStats | None,
    change_config: ChangeConfig | None,
) -> EncodedRouting:
    automaton = layered_attention_automaton(graph)
    route = automaton.flat_route_distribution
    prompt_lineage = automaton.flat_prompt_lineage
    if debruijn is None:
        empty = torch.empty(
            (graph.num_response_tokens, graph.num_channels),
            dtype=route.dtype,
            device=route.device,
        )
        return EncodedRouting(
            graph,
            automaton,
            route,
            prompt_lineage,
            empty,
            empty,
            None,
        )
    debruijn = debruijn.to(route.device).validate()
    surprisal = debruijn.score(route).to(dtype=route.dtype)
    predicted = debruijn.predict_distribution(route).to(dtype=route.dtype)
    phase = None
    if change_stats is not None:
        phase = change_lockin_score(
            route,
            surprisal,
            prompt_lineage,
            stats=change_stats,
            detached_indices=R_PLUS,
            predicted_route_distribution=predicted,
            config=change_config,
        )
    return EncodedRouting(
        graph,
        automaton,
        route,
        prompt_lineage,
        surprisal,
        predicted,
        phase,
    )


def _reservoir(size: int, seed: int, group: str) -> AlignedFloat32Reservoir:
    del group  # Each source stream owns a separate reservoir instance.
    return AlignedFloat32Reservoir(size=int(size), seed=int(seed))


def _add_reservoir(
    reservoir: AlignedFloat32Reservoir,
    group: str,
    blocks: dict[str, np.ndarray],
) -> None:
    del group
    reservoir.add(blocks)


def _numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().to(device="cpu", dtype=torch.float32).numpy()


def _runtime_reference(reference_path, *, device) -> RuntimeReference:
    arrays = load_reference(reference_path)
    config = _config_from_reference(arrays)
    prior = torch.as_tensor(
        arrays["prior_probability"], dtype=torch.float32, device=device
    )
    transition = torch.as_tensor(
        arrays["transition_probability"], dtype=torch.float32, device=device
    )
    debruijn = FrozenDeBruijn(
        config=config.debruijn,
        prior=prior,
        transition=transition,
        token_count=int(np.asarray(arrays.get("debruijn_token_count", 1)).item()),
        transition_window_count=int(
            np.asarray(arrays.get("debruijn_transition_window_count", 0)).item()
        ),
    ).validate()
    stats = RobustChangeStats(
        surprisal=MedianMAD(
            torch.as_tensor(arrays["surprisal_median"], device=device),
            torch.as_tensor(arrays["surprisal_scale"], device=device),
        ),
        prompt_lineage_drop=MedianMAD(
            torch.as_tensor(
                arrays["prompt_lineage_drop_median"], device=device
            ),
            torch.as_tensor(
                arrays["prompt_lineage_drop_scale"], device=device
            ),
        ),
    ).validate()
    calibration = CalibrationReference(
        arrays["calibration_channel_score"],
        arrays["calibration_fusion_stat"],
        independent_fusion_reference=bool(
            np.asarray(arrays["calibration_independent_fusion"]).item()
        ),
    )
    return RuntimeReference(arrays, config, debruijn, stats, calibration)


def fit_reference(
    dataset,
    reference_path,
    *,
    config: ExperimentConfig | None = None,
    sample_ids=None,
    limit=None,
) -> dict[str, object]:
    """Fit all references from three disjoint unlabeled source streams."""

    config = ExperimentConfig() if config is None else config
    config.validate()
    _require_dataset_split(dataset, "train")
    train_manifest = _capture_dataset_manifest(dataset)
    selected = _selected_sample_ids(dataset, sample_ids, limit)
    split = split_three_source_streams(
        dataset,
        selected,
        config=config.calibration,
    )
    layers = int(dataset.manifest["num_layers"])
    heads = int(dataset.manifest["num_heads"])
    channels = layers * heads
    count_device = torch.device(str(getattr(dataset, "device", "cpu")))
    count_dtype = torch.float32 if count_device.type == "cuda" else torch.float64
    accumulator = DeBruijnAccumulator(
        num_channels=channels,
        num_states=len(STATE_NAMES),
        config=config.debruijn,
        device=count_device,
        dtype=count_dtype,
    )

    # Pass A1: expected De Bruijn counts. update() is called once per complete
    # response, so temporal contexts can never cross sample boundaries.
    total = len(split.fit_sample_ids)
    for index, sample_id in enumerate(split.fit_sample_ids):
        _print_progress("fit-grammar", index, total, sample_id)
        sample = dataset[sample_id]
        try:
            encoded = _encode_structure(sample, graph_config=config.graph)
            accumulator.update(encoded.route)
        finally:
            sample.release_attention()
    debruijn = accumulator.freeze()

    # Pass A2: robust change references under the now-frozen grammar.
    fit_reservoir = _reservoir(
        config.calibration.reference_size,
        config.calibration.seed + 1,
        "fit",
    )
    for index, sample_id in enumerate(split.fit_sample_ids):
        _print_progress("fit-phase-scale", index, total, sample_id)
        sample = dataset[sample_id]
        try:
            encoded = _encode_structure(
                sample,
                graph_config=config.graph,
                debruijn=debruijn,
            )
            drop = prompt_lineage_drop(encoded.prompt_lineage)
            _add_reservoir(
                fit_reservoir,
                "fit",
                {
                    "surprisal": _numpy(encoded.surprisal),
                    "prompt_lineage_drop": _numpy(drop),
                },
            )
        finally:
            sample.release_attention()
    fit_blocks = fit_reservoir.values()
    change_stats = fit_robust_change_stats(
        torch.from_numpy(fit_blocks["surprisal"]),
        torch.from_numpy(fit_blocks["prompt_lineage_drop"]),
        config=config.change,
    )

    # Pass B: independent per-channel score ECDF.
    channel_reservoir = _reservoir(
        config.calibration.reference_size,
        config.calibration.seed + 2,
        "channel",
    )
    total = len(split.channel_sample_ids)
    for index, sample_id in enumerate(split.channel_sample_ids):
        _print_progress("channel-calibration", index, total, sample_id)
        sample = dataset[sample_id]
        try:
            encoded = _encode_structure(
                sample,
                graph_config=config.graph,
                debruijn=debruijn,
                change_stats=change_stats,
                change_config=config.change,
            )
            _add_reservoir(
                channel_reservoir,
                "channel",
                {"raw_channel_score": _numpy(encoded.phase.raw_channel_score)},
            )
        finally:
            sample.release_attention()
    channel_blocks = channel_reservoir.values()
    channel_values = channel_blocks["raw_channel_score"]

    # Pass C: final fusion ECDF and exact-endpoint topology gate. True and
    # rewired rows share reservoir slots, so the gate is paired token-wise.
    fusion_reservoir = _reservoir(
        config.calibration.reference_size,
        config.calibration.seed + 3,
        "fusion",
    )
    changed_weighted = 0.0
    edge_count = 0
    total = len(split.fusion_sample_ids)
    for index, sample_id in enumerate(split.fusion_sample_ids):
        _print_progress("fusion-calibration", index, total, sample_id)
        sample = dataset[sample_id]
        try:
            encoded = _encode_structure(
                sample,
                graph_config=config.graph,
                debruijn=debruijn,
                change_stats=change_stats,
                change_config=config.change,
            )
            rewired: RewireResult = causal_endpoint_rewire(
                encoded.graph,
                seed=config.calibration.seed,
            )
            rewired_encoded = _encode_graph(
                rewired.graph,
                debruijn=debruijn,
                change_stats=change_stats,
                change_config=config.change,
            )
            _add_reservoir(
                fusion_reservoir,
                "fusion",
                {
                    "true_channel_score": _numpy(encoded.phase.raw_channel_score),
                    "rewired_channel_score": _numpy(
                        rewired_encoded.phase.raw_channel_score
                    ),
                },
            )
            changed_weighted += rewired.changed_fraction * encoded.graph.num_edges
            edge_count += encoded.graph.num_edges
        finally:
            sample.release_attention()
    fusion_blocks = fusion_reservoir.values()
    true_fusion_values = fusion_blocks["true_channel_score"]
    rewired_fusion_values = fusion_blocks["rewired_channel_score"]
    # CUSUM is intentionally unbounded before empirical calibration.  Keep the
    # reference in float32 so long responses cannot overflow a float16 sidecar.
    calibration = build_calibration(
        channel_values,
        true_fusion_values,
        storage_dtype=np.float32,
    )
    true_score = score_channels(calibration, true_fusion_values).score
    rewired_score = score_channels(calibration, rewired_fusion_values).score
    topology_gap = rewired_score.astype(np.float64) - true_score.astype(np.float64)
    mean_gap = float(topology_gap.mean())

    debruijn_config_json = _json_config(
        config.debruijn,
        num_states=len(STATE_NAMES),
    )
    payload = {
        "train_dataset_manifest_sha256": np.asarray(train_manifest.sha256),
        "num_layers": np.asarray(layers, dtype=np.int32),
        "num_heads": np.asarray(heads, dtype=np.int32),
        "graph_config_json": np.asarray(_json_config(config.graph)),
        "debruijn_config_json": np.asarray(debruijn_config_json),
        "change_config_json": np.asarray(_json_config(config.change)),
        "calibration_config_json": np.asarray(_json_config(config.calibration)),
        "fit_group_id": np.asarray(split.fit_group_ids, dtype=str),
        "channel_calibration_group_id": np.asarray(
            split.channel_group_ids, dtype=str
        ),
        "fusion_calibration_group_id": np.asarray(
            split.fusion_group_ids, dtype=str
        ),
        "calibration_group_id": np.asarray(
            split.calibration_group_ids, dtype=str
        ),
        "route_state_names": np.asarray(STATE_NAMES, dtype=str),
        "prior_probability": _numpy(debruijn.prior),
        "transition_probability": _numpy(debruijn.transition),
        "debruijn_token_count": np.asarray(
            debruijn.token_count, dtype=np.int64
        ),
        "debruijn_transition_window_count": np.asarray(
            debruijn.transition_window_count, dtype=np.int64
        ),
        "surprisal_median": _numpy(change_stats.surprisal.median),
        "surprisal_scale": _numpy(change_stats.surprisal.scale),
        "prompt_lineage_drop_median": _numpy(
            change_stats.prompt_lineage_drop.median
        ),
        "prompt_lineage_drop_scale": _numpy(
            change_stats.prompt_lineage_drop.scale
        ),
        "calibration_channel_score": calibration.calibration_channel_score,
        "calibration_fusion_stat": calibration.calibration_fusion_stat,
        "calibration_independent_fusion": np.asarray(True, dtype=np.bool_),
        "topology_gate_mean_gap": np.asarray(mean_gap, dtype=np.float32),
        "topology_gate_median_gap": np.asarray(
            float(np.median(topology_gap)), dtype=np.float32
        ),
        "topology_gate_positive_fraction": np.asarray(
            float(np.mean(topology_gap > 0.0)), dtype=np.float32
        ),
        "topology_gate_changed_edge_fraction": np.asarray(
            changed_weighted / max(edge_count, 1), dtype=np.float32
        ),
        "topology_gate_pass": np.asarray(mean_gap > 0.0, dtype=np.bool_),
    }
    train_manifest.verify(train_manifest.path)
    save_reference(reference_path, payload)
    train_manifest.verify(train_manifest.path)
    return {
        "reference": str(Path(reference_path).resolve()),
        "labels_read": False,
        "fit_groups": len(split.fit_group_ids),
        "channel_calibration_groups": len(split.channel_group_ids),
        "fusion_calibration_groups": len(split.fusion_group_ids),
        "fit_tokens": debruijn.token_count,
        "topology_gate_mean_gap": mean_gap,
        "topology_gate_pass": bool(mean_gap > 0.0),
    }


def _metadata_text(value) -> str:
    return "" if value is None else str(value)


def _safe_sidecar_name(sample_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)[:80] or "sample"
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{digest}.npz"


def _write_channel_sidecar(
    directory,
    sample_id: str,
    encoded: EncodedRouting,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / _safe_sidecar_name(sample_id)
    phase = encoded.phase
    layers, heads = encoded.graph.num_layers, encoded.graph.num_heads
    np.savez_compressed(
        destination,
        schema=np.asarray("causal-typed-path-debruijn-channel-sidecar-v1"),
        sample_id=np.asarray(sample_id),
        labels_included=np.asarray(False, dtype=np.bool_),
        route_distribution=_numpy(encoded.automaton.route_distribution),
        raw_channel_score=_numpy(phase.raw_channel_score).reshape(-1, layers, heads),
        transition_surprisal=_numpy(encoded.surprisal).reshape(-1, layers, heads),
        prompt_lineage=_numpy(encoded.prompt_lineage).reshape(-1, layers, heads),
        detached=_numpy(encoded.automaton.flat_detached).reshape(-1, layers, heads),
        rupture=_numpy(phase.rupture_memory).reshape(-1, layers, heads),
        lockin=_numpy(phase.lockin).reshape(-1, layers, heads),
        predictive_jsd=_numpy(phase.predictive_jsd).reshape(-1, layers, heads),
    )
    return destination


def score_split(
    dataset,
    reference_path,
    output_path,
    *,
    sample_ids=None,
    limit=None,
    save_channel_sidecars: bool = False,
    sidecar_dir=None,
) -> dict[str, object]:
    """Score a source-disjoint split and freeze complete token rows."""

    _require_dataset_split(dataset, "test")
    reference_file = FrozenFile.capture(reference_path)
    test_manifest = _capture_dataset_manifest(dataset)
    selected = _selected_sample_ids(dataset, sample_ids, limit)
    runtime = _runtime_reference(
        reference_file.path,
        device=getattr(dataset, "device", "cpu"),
    )
    reference_file.verify(reference_file.path)
    reference = runtime.arrays
    geometry = (
        int(dataset.manifest["num_layers"]),
        int(dataset.manifest["num_heads"]),
    )
    expected_geometry = (int(reference["num_layers"]), int(reference["num_heads"]))
    if geometry != expected_geometry:
        raise ValueError("test attention geometry differs from the reference")

    reserved = np.concatenate(
        (
            reference["fit_group_id"],
            reference["channel_calibration_group_id"],
            reference["fusion_calibration_group_id"],
        )
    )
    complete = sample_ids is None and limit is None
    audit = HeldOutSourceAudit(
        dataset,
        selected_sample_ids=selected,
        reserved_source_ids=reserved,
        require_complete_split=complete,
    )
    rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "sample_id",
            "source_id",
            "token_index",
            "response_length",
            "task_type",
            "data_source",
            "generator_model",
            "score",
            "fusion_stat",
            "channel_score_mean",
            "transition_surprisal_mean",
            "prompt_lineage_mean",
            "response_survival_mean",
            "rupture_mean",
            "lockin_mean",
            "conservation_error_max",
            "top_channel_index",
            "top_channel_value",
        )
    }
    written_sidecars: list[str] = []
    total = len(selected)
    for index, sample_id in enumerate(selected):
        _print_progress("score", index, total, sample_id)
        sample = dataset[sample_id]
        try:
            audit.observe(sample)
            encoded = _encode_structure(
                sample,
                graph_config=runtime.config.graph,
                debruijn=runtime.debruijn,
                change_stats=runtime.change_stats,
                change_config=runtime.config.change,
            )
            phase = encoded.phase
            raw = _numpy(phase.raw_channel_score)
            calibrated = score_channels(runtime.calibration, raw)
            response_count = encoded.graph.num_response_tokens
            top_k = min(runtime.config.calibration.top_channels, raw.shape[1])
            order = np.argsort(raw, axis=1, kind="stable")[:, -top_k:][:, ::-1]
            top_value = np.take_along_axis(raw, order, axis=1)
            source_id = canonical_source_group(sample)

            # np.full(..., dtype=str) creates a one-character Unicode dtype and
            # silently truncates identifiers. Build from a repeated list so
            # NumPy infers the complete string width.
            rows["sample_id"].append(
                np.asarray([sample_id] * response_count, dtype=str)
            )
            rows["source_id"].append(
                np.asarray([source_id] * response_count, dtype=str)
            )
            rows["token_index"].append(np.arange(response_count, dtype=np.int32))
            rows["response_length"].append(
                np.full(response_count, response_count, dtype=np.int32)
            )
            for field, value in (
                ("task_type", sample.task_type),
                ("data_source", sample.data_source),
                ("generator_model", sample.generator_model),
            ):
                rows[field].append(
                    np.asarray(
                        [_metadata_text(value)] * response_count,
                        dtype=str,
                    )
                )
            rows["score"].append(calibrated.score.astype(np.float32))
            rows["fusion_stat"].append(calibrated.fusion_stat.astype(np.float32))
            rows["channel_score_mean"].append(raw.mean(axis=1, dtype=np.float32))
            rows["transition_surprisal_mean"].append(
                _numpy(encoded.surprisal).mean(axis=1, dtype=np.float32)
            )
            rows["prompt_lineage_mean"].append(
                _numpy(encoded.prompt_lineage).mean(axis=1, dtype=np.float32)
            )
            rows["response_survival_mean"].append(
                _numpy(encoded.automaton.flat_detached).mean(axis=1, dtype=np.float32)
            )
            rows["rupture_mean"].append(
                _numpy(phase.rupture_memory).mean(axis=1, dtype=np.float32)
            )
            rows["lockin_mean"].append(
                _numpy(phase.lockin).mean(axis=1, dtype=np.float32)
            )
            rows["conservation_error_max"].append(
                _numpy(
                    encoded.automaton.conservation_error.reshape(
                        response_count, -1
                    )
                ).max(axis=1)
            )
            rows["top_channel_index"].append(order.astype(np.int32))
            rows["top_channel_value"].append(top_value.astype(np.float32))

            if save_channel_sidecars:
                target_dir = (
                    Path(sidecar_dir)
                    if sidecar_dir is not None
                    else Path(output_path).parent
                    / f"{Path(output_path).stem}_channel_sidecars"
                )
                written_sidecars.append(
                    str(_write_channel_sidecar(target_dir, sample_id, encoded))
                )
        finally:
            sample.release_attention()
    source_audit = audit.finish()
    combined = {name: np.concatenate(parts, axis=0) for name, parts in rows.items()}
    payload = {
        **combined,
        "reference_path": np.asarray(str(reference_file.path)),
        "reference_sha256": np.asarray(reference_file.sha256),
        "dataset_manifest_sha256": np.asarray(test_manifest.sha256),
        "fit_group_id": reference["fit_group_id"],
        "channel_calibration_group_id": reference[
            "channel_calibration_group_id"
        ],
        "fusion_calibration_group_id": reference["fusion_calibration_group_id"],
        "calibration_group_id": reference["calibration_group_id"],
        "test_group_id": np.asarray(source_audit.test_source_ids, dtype=str),
        "test_sample_id": np.asarray(source_audit.test_sample_ids, dtype=str),
        "audit_scope": np.asarray(source_audit.test_scope),
        "channel_sidecar_path": np.asarray(written_sidecars, dtype=str),
    }
    reference_file.verify(reference_file.path)
    test_manifest.verify(test_manifest.path)
    save_score_artifact(output_path, payload)
    load_score_artifact(output_path)
    reference_file.verify(reference_file.path)
    test_manifest.verify(test_manifest.path)
    return {
        "scores": str(Path(output_path).resolve()),
        "labels_read": False,
        "samples": len(selected),
        "tokens": len(combined["score"]),
        "audit_scope": source_audit.test_scope,
        "channel_sidecars": len(written_sidecars),
    }


def visualize_scored_sample(
    dataset,
    reference_path,
    score_path,
    *,
    sample_id: str,
    output_path,
    token_index: int | None = None,
) -> dict[str, object]:
    """Recompute one label-free route state and render its frozen score rows."""

    from .visualization import render_sample_diagnostics

    sample_id = str(sample_id)
    if sample_id not in dataset:
        raise ValueError("sample_id is outside the selected dataset")
    artifact = load_score_artifact(score_path)
    verify_score_provenance(artifact)
    recorded_reference = Path(str(np.asarray(artifact["reference_path"]).item())).resolve()
    if recorded_reference != Path(reference_path).resolve():
        raise ValueError("visualization reference differs from the score artifact")
    if str(np.asarray(artifact["dataset_manifest_sha256"]).item()) != dataset_manifest_sha256(dataset):
        raise ValueError("visualization dataset differs from the score artifact")
    row_sample = np.asarray(artifact["sample_id"], dtype=str)
    selected_rows = np.flatnonzero(row_sample == sample_id)
    if not len(selected_rows):
        raise ValueError("sample_id has no rows in the score artifact")
    order = np.argsort(np.asarray(artifact["token_index"])[selected_rows])
    selected_rows = selected_rows[order]
    runtime = _runtime_reference(
        reference_path,
        device=getattr(dataset, "device", "cpu"),
    )
    sample = dataset[sample_id]
    try:
        encoded = _encode_structure(
            sample,
            graph_config=runtime.config.graph,
            debruijn=runtime.debruijn,
            change_stats=runtime.change_stats,
            change_config=runtime.config.change,
        )
        output = render_sample_diagnostics(
            encoded.graph,
            encoded.automaton,
            encoded.phase,
            token_score=np.asarray(artifact["score"])[selected_rows],
            output_path=output_path,
            token_index=token_index,
        )
    finally:
        sample.release_attention()
    return {
        "sample_id": sample_id,
        "figure": str(output.resolve()),
        "labels_read": False,
    }


__all__ = [
    "ExperimentConfig",
    "RuntimeReference",
    "ThreeWaySourceSplit",
    "fit_reference",
    "score_split",
    "split_three_source_streams",
    "visualize_scored_sample",
]
