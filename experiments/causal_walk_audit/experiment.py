"""Label-free fitting and held-out scoring for typed route grammar."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from experiment_protocol import canonical_source_group

from .artifacts import (
    REFERENCE_SCHEMA,
    SCORE_SCHEMA,
    load_npz,
    save_npz,
    sha256,
)
from .automaton import STATE_NAMES, run_typed_automaton
from .calibration import HierarchicalCalibration, RowReservoir
from .config import (
    AuditConfig,
    CalibrationConfig,
    GrammarConfig,
    GraphConfig,
    PhaseConfig,
)
from .grammar import GrammarAccumulator, RouteGrammar
from .graph import build_routing_graph
from .nulls import rewire_endpoints
from .phase import ChannelStats, fit_channel_stats, score_phase


@dataclass(frozen=True)
class SourceSplit:
    fit_ids: tuple[str, ...]
    channel_ids: tuple[str, ...]
    fusion_ids: tuple[str, ...]
    fit_groups: tuple[str, ...]
    channel_groups: tuple[str, ...]
    fusion_groups: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeReference:
    arrays: dict[str, np.ndarray]
    config: AuditConfig
    grammar: RouteGrammar
    stats: ChannelStats
    calibration: HierarchicalCalibration
    order2_enabled: bool


def _selected_ids(dataset, task_type: str, limit: int | None) -> tuple[str, ...]:
    selected = []
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        try:
            if task_type.casefold() == "all" or str(sample.task_type).casefold() == task_type.casefold():
                selected.append(str(sample_id))
        finally:
            sample.release_attention()
    return tuple(selected if limit is None else selected[: int(limit)])


def _group_key(group: str, seed: int) -> bytes:
    return hashlib.sha256(f"typed-route-v2\0{seed}\0{group}".encode()).digest()


def _split_sources(
    dataset,
    sample_ids: tuple[str, ...],
    config: CalibrationConfig,
) -> SourceSplit:
    grouped: dict[str, list[str]] = {}
    for sample_id in sample_ids:
        sample = dataset[sample_id]
        try:
            grouped.setdefault(canonical_source_group(sample), []).append(sample_id)
        finally:
            sample.release_attention()
    if len(grouped) < 3:
        raise ValueError("typed route grammar requires at least three source groups")

    names = sorted(grouped, key=lambda value: _group_key(value, config.seed))
    channel_count = max(1, round(len(names) * config.channel_fraction))
    fusion_count = max(1, round(len(names) * config.fusion_fraction))
    while channel_count + fusion_count >= len(names):
        if channel_count >= fusion_count and channel_count > 1:
            channel_count -= 1
        elif fusion_count > 1:
            fusion_count -= 1
        else:
            raise ValueError("source groups cannot form three non-empty streams")

    channel_groups = set(names[:channel_count])
    fusion_groups = set(names[channel_count : channel_count + fusion_count])
    fit_groups = set(names) - channel_groups - fusion_groups

    def samples(groups: set[str]) -> tuple[str, ...]:
        return tuple(
            sample_id
            for group, ids in grouped.items()
            if group in groups
            for sample_id in ids
        )

    return SourceSplit(
        fit_ids=samples(fit_groups),
        channel_ids=samples(channel_groups),
        fusion_ids=samples(fusion_groups),
        fit_groups=tuple(sorted(fit_groups)),
        channel_groups=tuple(sorted(channel_groups)),
        fusion_groups=tuple(sorted(fusion_groups)),
    )


def _require_split(dataset, expected: str) -> None:
    actual = str(dataset.manifest.get("split"))
    if actual != expected:
        raise ValueError(f"expected dataset split {expected!r}, got {actual!r}")


def _encode(sample, config: AuditConfig):
    graph = build_routing_graph(sample, config=config.graph)
    trace = run_typed_automaton(graph)
    return graph, trace, trace.flat


def _score_structure(
    graph,
    route,
    grammar: RouteGrammar,
    stats: ChannelStats,
    config: AuditConfig,
    *,
    use_order2: bool,
):
    grammar = grammar.to(route.device)
    surprisal, order1_surprisal, predicted, order2_weight = grammar.score(
        route,
        use_order2=use_order2,
    )
    phase = score_phase(
        route,
        predicted,
        surprisal,
        stats=stats,
        config=config.phase,
    )
    return surprisal, order1_surprisal, predicted, order2_weight, phase


def _config_from_json(value: str) -> AuditConfig:
    payload = json.loads(value)
    return AuditConfig(
        graph=GraphConfig(**payload["graph"]),
        grammar=GrammarConfig(**payload["grammar"]),
        phase=PhaseConfig(**payload["phase"]),
        calibration=CalibrationConfig(**payload["calibration"]),
    )


def load_reference(path, *, device: str | torch.device = "cpu") -> RuntimeReference:
    arrays = load_npz(path)
    if str(arrays["schema"].item()) != REFERENCE_SCHEMA:
        raise ValueError("unsupported typed route reference")
    config = _config_from_json(str(arrays["config_json"].item()))
    grammar = RouteGrammar(
        prior=torch.as_tensor(arrays["prior"], device=device),
        order1=torch.as_tensor(arrays["order1"], device=device),
        order2=torch.as_tensor(arrays["order2"], device=device),
        order2_context_count=torch.as_tensor(
            arrays["order2_context_count"], device=device
        ),
        backoff_tau=float(arrays["backoff_tau"]),
        token_count=int(arrays["grammar_token_count"]),
    )
    stats = ChannelStats(
        median=torch.as_tensor(arrays["surprisal_median"], device=device),
        scale=torch.as_tensor(arrays["surprisal_scale"], device=device),
    )
    calibration = HierarchicalCalibration(
        channel_reference=arrays["channel_reference"],
        layer_reference=arrays["layer_reference"],
        global_reference=arrays["global_reference"],
        num_layers=int(arrays["num_layers"]),
        num_heads=int(arrays["num_heads"]),
    )
    return RuntimeReference(
        arrays,
        config,
        grammar,
        stats,
        calibration,
        bool(arrays["order2_enabled"].item()),
    )


def fit_reference(
    dataset,
    reference_path,
    *,
    config: AuditConfig | None = None,
    task_type: str = "QA",
    limit: int | None = None,
) -> dict[str, object]:
    """Fit grammar, robust phase scale and calibration without labels."""

    config = AuditConfig() if config is None else config
    _require_split(dataset, "train")
    sample_ids = _selected_ids(dataset, task_type, limit)
    split = _split_sources(dataset, sample_ids, config.calibration)
    layers = int(dataset.manifest["num_layers"])
    heads = int(dataset.manifest["num_heads"])
    channels = layers * heads

    accumulator = GrammarAccumulator(
        channels,
        len(STATE_NAMES),
        config=config.grammar,
        device=getattr(dataset, "device", "cpu"),
    )
    for sample_id in tqdm(split.fit_ids, desc="fit route grammar", unit="sample"):
        sample = dataset[sample_id]
        try:
            _, _, route = _encode(sample, config)
            accumulator.update(route)
        finally:
            sample.release_attention()
    grammar = accumulator.freeze()

    order2_gain_rows: list[np.ndarray] = []
    for sample_id in tqdm(
        split.channel_ids,
        desc="validate grammar order",
        unit="sample",
    ):
        sample = dataset[sample_id]
        try:
            _, _, route = _encode(sample, config)
            backoff, order1, _, _ = grammar.to(route.device).score(
                route,
                use_order2=True,
            )
            order2_gain_rows.append(
                (order1 - backoff).mean(dim=1).cpu().numpy()
            )
        finally:
            sample.release_attention()
    order2_gain = np.concatenate(order2_gain_rows)
    order2_mean_gain = float(order2_gain.mean())
    order2_positive_fraction = float(np.mean(order2_gain > 0.0))
    order2_enabled = bool(
        order2_mean_gain > 0.0 and order2_positive_fraction > 0.5
    )

    phase_rows = RowReservoir(
        config.calibration.reservoir_rows,
        config.calibration.seed + 1,
    )
    for sample_id in tqdm(split.fit_ids, desc="fit rupture scale", unit="sample"):
        sample = dataset[sample_id]
        try:
            _, _, route = _encode(sample, config)
            surprisal, _, _, _ = grammar.to(route.device).score(
                route,
                use_order2=order2_enabled,
            )
            phase_rows.add(surprisal.cpu().numpy())
        finally:
            sample.release_attention()
    stats = fit_channel_stats(
        torch.from_numpy(phase_rows.matrix()),
        scale_floor=config.phase.scale_floor,
    )

    channel_rows = RowReservoir(
        config.calibration.reservoir_rows,
        config.calibration.seed + 2,
    )
    for sample_id in tqdm(
        split.channel_ids,
        desc="calibrate channels",
        unit="sample",
    ):
        sample = dataset[sample_id]
        try:
            graph, _, route = _encode(sample, config)
            *_, phase = _score_structure(
                graph,
                route,
                grammar,
                stats,
                config,
                use_order2=order2_enabled,
            )
            channel_rows.add(phase.rupture.cpu().numpy())
        finally:
            sample.release_attention()

    fusion_rows = RowReservoir(
        config.calibration.reservoir_rows,
        config.calibration.seed + 3,
    )
    topology_gap: list[np.ndarray] = []
    changed_weight = 0.0
    edge_count = 0
    for sample_id in tqdm(
        split.fusion_ids,
        desc="calibrate fusion and topology",
        unit="sample",
    ):
        sample = dataset[sample_id]
        try:
            graph, _, route = _encode(sample, config)
            *_, phase = _score_structure(
                graph,
                route,
                grammar,
                stats,
                config,
                use_order2=order2_enabled,
            )
            fusion_rows.add(phase.rupture.cpu().numpy())

            rewired = rewire_endpoints(
                graph,
                seed=config.calibration.seed,
            )
            rewired_route = run_typed_automaton(rewired.graph).flat
            *_, rewired_phase = _score_structure(
                rewired.graph,
                rewired_route,
                grammar,
                stats,
                config,
                use_order2=order2_enabled,
            )
            topology_gap.append(
                (
                    rewired_phase.rupture.mean(dim=1)
                    - phase.rupture.mean(dim=1)
                ).cpu().numpy()
            )
            changed_weight += rewired.changed_fraction * max(graph.weight.numel(), 1)
            edge_count += max(graph.weight.numel(), 1)
        finally:
            sample.release_attention()

    calibration = HierarchicalCalibration.fit(
        channel_rows.matrix(),
        fusion_rows.matrix(),
        num_layers=layers,
        num_heads=heads,
    )
    gap = np.concatenate(topology_gap) if topology_gap else np.zeros(1)
    changed_fraction = changed_weight / max(edge_count, 1)
    topology_pass = bool(
        changed_fraction >= config.calibration.topology_min_changed_fraction
        and float(gap.mean()) > 0.0
        and float(np.mean(gap > 0.0)) > 0.5
    )

    manifest_path = Path(dataset.root) / "manifest.json"
    save_npz(
        reference_path,
        schema=np.asarray(REFERENCE_SCHEMA),
        labels_included=np.asarray(False),
        train_manifest_sha256=np.asarray(sha256(manifest_path)),
        config_json=np.asarray(
            json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
        ),
        num_layers=np.asarray(layers, dtype=np.int32),
        num_heads=np.asarray(heads, dtype=np.int32),
        state_names=np.asarray(STATE_NAMES),
        fit_group_id=np.asarray(split.fit_groups),
        channel_group_id=np.asarray(split.channel_groups),
        fusion_group_id=np.asarray(split.fusion_groups),
        prior=grammar.prior.cpu().numpy(),
        order1=grammar.order1.cpu().numpy(),
        order2=grammar.order2.cpu().numpy(),
        order2_context_count=grammar.order2_context_count.cpu().numpy(),
        backoff_tau=np.asarray(grammar.backoff_tau, dtype=np.float32),
        grammar_token_count=np.asarray(grammar.token_count, dtype=np.int64),
        order2_enabled=np.asarray(order2_enabled),
        order2_validation_mean_gain=np.asarray(
            order2_mean_gain, dtype=np.float32
        ),
        order2_validation_positive_fraction=np.asarray(
            order2_positive_fraction, dtype=np.float32
        ),
        surprisal_median=stats.median.cpu().numpy(),
        surprisal_scale=stats.scale.cpu().numpy(),
        channel_reference=calibration.channel_reference,
        layer_reference=calibration.layer_reference,
        global_reference=calibration.global_reference,
        topology_mean_gap=np.asarray(float(gap.mean()), dtype=np.float32),
        topology_median_gap=np.asarray(float(np.median(gap)), dtype=np.float32),
        topology_positive_fraction=np.asarray(
            float(np.mean(gap > 0.0)), dtype=np.float32
        ),
        topology_changed_fraction=np.asarray(
            changed_fraction, dtype=np.float32
        ),
        topology_gate_pass=np.asarray(topology_pass),
    )
    return {
        "reference": str(Path(reference_path).resolve()),
        "labels_read": False,
        "fit_groups": len(split.fit_groups),
        "channel_groups": len(split.channel_groups),
        "fusion_groups": len(split.fusion_groups),
        "grammar_tokens": grammar.token_count,
        "order2_enabled": order2_enabled,
        "order2_validation_mean_gain": order2_mean_gain,
        "order2_validation_positive_fraction": order2_positive_fraction,
        "topology_changed_fraction": changed_fraction,
        "topology_gate_pass": topology_pass,
    }


def score_split(
    dataset,
    reference_path,
    score_path,
    *,
    task_type: str = "QA",
    limit: int | None = None,
) -> dict[str, object]:
    """Freeze one primary grammar-rupture score and mechanism diagnostics."""

    _require_split(dataset, "test")
    runtime = load_reference(
        reference_path,
        device=getattr(dataset, "device", "cpu"),
    )
    sample_ids = _selected_ids(dataset, task_type, limit)
    rows: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "sample_id",
            "source_id",
            "token_index",
            "response_length",
            "task_type",
            "score",
            "grammar_surprisal_mean",
            "order1_surprisal_mean",
            "order2_gain_mean",
            "order2_weight_mean",
            "rupture_mean",
            "closure_mean",
            "rupture_closure_mean",
            "prompt_lineage_mean",
            "unresolved_mean",
        )
    }

    for sample_id in tqdm(sample_ids, desc="score typed route grammar", unit="sample"):
        sample = dataset[sample_id]
        try:
            graph, trace, route = _encode(sample, runtime.config)
            (
                surprisal,
                order1_surprisal,
                _,
                order2_weight,
                phase,
            ) = _score_structure(
                graph,
                route,
                runtime.grammar,
                runtime.stats,
                runtime.config,
                use_order2=runtime.order2_enabled,
            )
            score, _, _ = runtime.calibration.score(
                phase.rupture.cpu().numpy()
            )
            tokens = graph.num_response_tokens
            rows["sample_id"].append(np.repeat(str(sample.sample_id), tokens))
            rows["source_id"].append(np.repeat(str(sample.source_id), tokens))
            rows["token_index"].append(np.arange(tokens, dtype=np.int32))
            rows["response_length"].append(
                np.full(tokens, tokens, dtype=np.int32)
            )
            rows["task_type"].append(np.repeat(str(sample.task_type), tokens))
            rows["score"].append(score)
            rows["grammar_surprisal_mean"].append(
                surprisal.mean(dim=1).cpu().numpy()
            )
            rows["order1_surprisal_mean"].append(
                order1_surprisal.mean(dim=1).cpu().numpy()
            )
            rows["order2_gain_mean"].append(
                (order1_surprisal - surprisal).mean(dim=1).cpu().numpy()
            )
            rows["order2_weight_mean"].append(
                order2_weight.mean(dim=1).cpu().numpy()
            )
            rows["rupture_mean"].append(
                phase.rupture.mean(dim=1).cpu().numpy()
            )
            rows["closure_mean"].append(
                phase.closure_score.mean(dim=1).cpu().numpy()
            )
            rows["rupture_closure_mean"].append(
                phase.rupture_closure.mean(dim=1).cpu().numpy()
            )
            rows["prompt_lineage_mean"].append(
                phase.prompt_lineage.mean(dim=1).cpu().numpy()
            )
            rows["unresolved_mean"].append(
                route[..., -1].mean(dim=1).cpu().numpy()
            )
        finally:
            sample.release_attention()

    arrays = {
        name: np.concatenate(value)
        for name, value in rows.items()
    }
    save_npz(
        score_path,
        schema=np.asarray(SCORE_SCHEMA),
        labels_included=np.asarray(False),
        reference_path=np.asarray(str(Path(reference_path).resolve())),
        reference_sha256=np.asarray(sha256(reference_path)),
        dataset_manifest_sha256=np.asarray(
            sha256(Path(dataset.root) / "manifest.json")
        ),
        primary_score=np.asarray("score"),
        **arrays,
    )
    return {
        "scores": str(Path(score_path).resolve()),
        "labels_read": False,
        "samples": len(sample_ids),
        "tokens": len(arrays["score"]),
    }
