"""Position-matched post-hoc tests for the three saved mechanisms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from research_dataset import open_research_dataset

from .audit import load_index
from .capture import HISTORY, ROLE_NAMES, SELF
from .data import EVIDENCE
from .visualize import plot_population, plot_sample_dashboard


def _mean_or_none(value: np.ndarray) -> float | None:
    return float(value.mean()) if len(value) else None


def _layer_summary(value: torch.Tensor, name: str) -> dict[str, torch.Tensor]:
    width = max(value.shape[0] // 3, 1)
    early = value[:width].mean(0)
    late = value[-width:].mean(0)
    return {
        f"{name}_mean": value.mean(0),
        f"{name}_early": early,
        f"{name}_late": late,
        f"{name}_layer_shift": late - early,
    }


def layer_metrics(artifact: dict) -> dict[str, torch.Tensor]:
    """Return the registered layer-by-token mechanism measurements."""
    trace = artifact["trace"]
    edge = trace["role_edge_magnitude"].float()
    attention = trace["role_attention"].float()
    response_self = (
        torch.arange(edge.shape[1], dtype=torch.float32)[None, :, None] > 0
    )

    edge_role = edge.sum(2)
    edge_total = edge_role.sum(-1).clamp_min(1e-12)
    edge_evidence = edge_role[..., EVIDENCE] / edge_total
    edge_response = (
        edge_role[..., HISTORY]
        + edge_role[..., SELF] * response_self.squeeze(-1)
    ) / edge_total

    attention_role = attention.sum(2)
    attention_total = attention_role.sum(-1).clamp_min(1e-12)
    attention_evidence = attention_role[..., EVIDENCE] / attention_total
    attention_response = (
        attention_role[..., HISTORY]
        + attention_role[..., SELF] * response_self.squeeze(-1)
    ) / attention_total

    active_sources = (trace["source_role"] >= 0).sum(-1).float().clamp_min(2)
    dispersion = trace["source_message_entropy"].float() / active_sources.log()[None]
    coherence = trace["message_coherence"].float()

    head_roles = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
    mean_role = head_roles.mean(2)
    head_role_js = -(
        mean_role * mean_role.clamp_min(1e-12).log()
    ).sum(-1) + (
        head_roles * head_roles.clamp_min(1e-12).log()
    ).sum(-1).mean(2)

    return {
        "message_evidence_share": edge_evidence,
        "message_response_share": edge_response,
        "message_routing_drift": edge_response - edge_evidence,
        "attention_evidence_share": attention_evidence,
        "attention_response_share": attention_response,
        "attention_routing_drift": attention_response - attention_evidence,
        "message_source_dispersion": dispersion,
        "head_role_disagreement": head_role_js,
        "message_coherence": coherence,
    }


def token_metrics(
    artifact: dict,
    layers: dict[str, torch.Tensor] | None = None,
) -> dict[str, np.ndarray]:
    """Reduce layer routes while retaining the full tensors in each sample audit."""

    metrics = {}
    for name, value in (layers or layer_metrics(artifact)).items():
        metrics.update(_layer_summary(value, name))
    metrics.update({name: value.float() for name, value in artifact["mechanism"].items()})
    mechanism = artifact["mechanism"]
    metrics["message_independent_capture_signature"] = (
        (mechanism["evidence_response_removed_margin"] > 0)
        & (mechanism["full_margin"] > 0)
        & (mechanism["evidence_message_effect"] <= 0)
    ).float()
    return {name: value.cpu().numpy() for name, value in metrics.items()}


def _position_match_design(
    label: np.ndarray,
    sample: np.ndarray,
    source: np.ndarray,
    generator: np.ndarray,
    token_index: np.ndarray,
    response_length: np.ndarray,
    *,
    position_bin: int,
) -> dict[str, object]:
    relative_bin = np.minimum(
        ((token_index + 0.5) * 10 / response_length).astype(np.int16), 9
    )
    ordinal_bin = token_index // position_bin
    cells: dict[tuple[str, int, int], list[int]] = {}
    for index, key in enumerate(zip(sample, ordinal_bin, relative_bin)):
        cells.setdefault(key, []).append(index)
    sample_meta: dict[str, tuple[str, str]] = {}
    covered_hallucinated = 0
    matched = []
    for (sample_id, _ordinal, _relative), indices in cells.items():
        indices = np.asarray(indices)
        current_label = label[indices]
        if current_label.any() and (~current_label).any():
            positives = int(current_label.sum())
            negatives = int((~current_label).sum())
            weight = positives * negatives / (positives + negatives)
            matched.append(
                (
                    str(sample_id),
                    indices[current_label],
                    indices[~current_label],
                    float(weight),
                )
            )
            row = indices[0]
            sample_meta[str(sample_id)] = (str(source[row]), str(generator[row]))
            covered_hallucinated += positives
    return {
        "cells": matched,
        "sample_meta": sample_meta,
        "covered_hallucinated_tokens": covered_hallucinated,
        "hallucinated_token_coverage": float(covered_hallucinated / max(label.sum(), 1)),
    }


def _position_matched_difference(
    value: np.ndarray,
    label: np.ndarray,
    sample: np.ndarray,
    source: np.ndarray,
    generator: np.ndarray,
    token_index: np.ndarray,
    response_length: np.ndarray,
    *,
    position_bin: int,
    bootstrap: int,
    seed: int,
    design: dict[str, object] | None = None,
) -> dict[str, object]:
    """Within-response contrasts, then response/generator/source aggregation."""

    design = design or _position_match_design(
        label,
        sample,
        source,
        generator,
        token_index,
        response_length,
        position_bin=position_bin,
    )
    by_sample: dict[str, list[tuple[float, float]]] = {}
    for sample_id, positive, negative, weight in design["cells"]:
        effect = value[positive].mean() - value[negative].mean()
        by_sample.setdefault(sample_id, []).append((float(effect), weight))

    by_source_generator: dict[tuple[str, str], list[float]] = {}
    for sample_id, current in by_sample.items():
        effects = np.asarray([effect for effect, _weight in current])
        weights = np.asarray([weight for _effect, weight in current])
        by_source_generator.setdefault(design["sample_meta"][sample_id], []).append(
            float(np.average(effects, weights=weights))
        )
    by_source: dict[str, list[float]] = {}
    for (source_id, _generator), current in by_source_generator.items():
        by_source.setdefault(source_id, []).append(float(np.mean(current)))
    effects = np.asarray([np.mean(current) for current in by_source.values()])
    if not len(effects):
        return {
            "position_matched_source_equal_difference": None,
            "ci95": [None, None],
            "p_value": None,
            "sources": 0,
            "matched_samples": 0,
            "matched_cells": 0,
            "covered_hallucinated_tokens": 0,
            "hallucinated_token_coverage": 0.0,
        }
    random = np.random.default_rng(seed)
    draws = random.choice(effects, (bootstrap, len(effects)), replace=True).mean(1)
    null = (
        effects[None]
        * random.choice((-1.0, 1.0), (bootstrap, len(effects)), replace=True)
    ).mean(1)
    observed = float(effects.mean())
    return {
        "position_matched_source_equal_difference": observed,
        "ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))],
        "p_value": float((1 + np.sum(np.abs(null) >= abs(observed))) / (bootstrap + 1)),
        "sources": int(len(effects)),
        "matched_samples": int(len(by_sample)),
        "matched_cells": int(len(design["cells"])),
        "covered_hallucinated_tokens": int(design["covered_hallucinated_tokens"]),
        "hallucinated_token_coverage": float(design["hallucinated_token_coverage"]),
    }


def _onset_pairs(
    label: np.ndarray,
    sample_id: np.ndarray,
    source_id: np.ndarray,
    token_index: np.ndarray,
    *,
    radius: int = 8,
) -> list[list[tuple[int, int, np.ndarray, np.ndarray, str]]]:
    """Match every onset to nearby all-correct transitions in the same response."""

    sample_rows = {
        sample: np.flatnonzero(sample_id == sample) for sample in np.unique(sample_id)
    }
    pairs = [[] for _ in range(2 * radius + 1)]
    for rows in sample_rows.values():
        rows = rows[np.argsort(token_index[rows])]
        current_label = label[rows]
        starts = np.flatnonzero(current_label & ~np.r_[False, current_label[:-1]])
        for onset in starts:
            onset = int(onset)
            if onset < radius or onset + radius >= len(rows):
                continue
            source = str(source_id[rows[0]])
            controls = [
                center
                for center in range(radius, len(rows) - radius)
                if not current_label[center - radius : center + radius + 1].any()
            ]
            controls = sorted(controls, key=lambda center: abs(center - onset))[:8]
            if not controls:
                continue
            for offset in range(-radius, radius + 1):
                event_position = onset + offset
                pairs[offset + radius].append(
                    (
                        rows[event_position],
                        rows[onset - 1],
                        rows[np.asarray(controls) + offset],
                        rows[np.asarray(controls) - 1],
                        source,
                    )
                )
    return pairs


def _matched_onset(
    value: np.ndarray,
    pairs: list[list[tuple[int, int, np.ndarray, np.ndarray, str]]],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    radius = (len(pairs) - 1) // 2

    means, lows, highs, counts, sources = [], [], [], [], []
    random = np.random.default_rng(seed)
    for current_pairs in pairs:
        if not current_pairs:
            means.append(None)
            lows.append(None)
            highs.append(None)
            counts.append(0)
            sources.append(0)
            continue
        current = np.asarray(
            [
                value[event] - value[event_base]
                - np.mean(value[controls] - value[control_bases])
                for event, event_base, controls, control_bases, _source in current_pairs
            ]
        )
        current_sources = np.asarray([pair[-1] for pair in current_pairs])
        source_effects = np.asarray(
            [current[current_sources == source].mean() for source in np.unique(current_sources)]
        )
        draws = random.choice(
            source_effects, (bootstrap, len(source_effects)), replace=True
        ).mean(1)
        means.append(float(source_effects.mean()))
        lows.append(float(np.quantile(draws, 0.025)))
        highs.append(float(np.quantile(draws, 0.975)))
        counts.append(int(len(current)))
        sources.append(int(len(source_effects)))
    return {
        "offset": list(range(-radius, radius + 1)),
        "difference_in_difference": means,
        "ci95_low": lows,
        "ci95_high": highs,
        "events": counts,
        "sources": sources,
    }


SAMPLE_METRICS = (
    "message_evidence_share_mean",
    "message_response_share_mean",
    "message_routing_drift_mean",
    "message_source_dispersion_mean",
    "head_role_disagreement_mean",
    "evidence_message_effect",
    "response_message_effect",
    "evidence_response_removed_margin",
    "full_margin",
    "message_independent_capture_signature",
)

REGISTERED_INFERENCE_METRICS = {
    "message_routing_drift_mean",
    "message_source_dispersion_mean",
    "head_role_disagreement_mean",
    "evidence_message_effect",
    "message_independent_capture_signature",
}


def _positive_runs(label: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(label.astype(bool), (1, 1))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _class_mean(value: np.ndarray, mask: np.ndarray) -> float | None:
    return float(value[mask].mean()) if mask.any() else None


def _top_routes(artifact: dict, token: int, tokenizer, top_k: int = 8) -> tuple[list, list]:
    trace = artifact["trace"]
    layers = trace["role_edge_magnitude"].shape[0]
    late = 2 * layers // 3
    source_index = trace["top_source_index"][late:, token].numpy().reshape(-1)
    source_mass = trace["top_source_magnitude"][late:, token].numpy().reshape(-1)
    mass_by_source: dict[int, float] = {}
    for index, mass in zip(source_index, source_mass):
        if index >= 0:
            mass_by_source[int(index)] = mass_by_source.get(int(index), 0.0) + float(mass)
    total = float(trace["role_edge_magnitude"][late:, token].sum()) or 1.0
    token_ids = artifact["token_ids"].numpy()
    source_role = trace["source_role"][token].numpy()
    sources = [
        {
            "source_index": index,
            "token_id": int(token_ids[index]),
            "token": tokenizer.convert_ids_to_tokens(int(token_ids[index])),
            "role": ROLE_NAMES[int(source_role[index])],
            "late_retained_mass_over_total": mass / total,
        }
        for index, mass in sorted(
            mass_by_source.items(), key=lambda item: item[1], reverse=True
        )[:top_k]
    ]

    edge = trace["role_edge_magnitude"][late:, token].numpy()
    flat = np.argsort(edge.reshape(-1))[::-1][:top_k]
    head_routes = []
    for index in flat:
        layer, head, role = np.unravel_index(index, edge.shape)
        head_routes.append(
            {
                "layer": int(late + layer),
                "head": int(head),
                "role": ROLE_NAMES[int(role)],
                "edge_magnitude": float(edge[layer, head, role]),
            }
        )
    return sources, head_routes


def _sample_record(
    row: dict,
    artifact: dict,
    label: np.ndarray,
    metrics: dict[str, np.ndarray],
    tokenizer,
) -> dict:
    target_ids = artifact["target_ids"].numpy()
    token_text = tokenizer.convert_ids_to_tokens(target_ids.tolist())
    runs = _positive_runs(label)
    onsets = []
    for start, stop in runs:
        previous = start - 1
        top_sources, top_head_routes = _top_routes(artifact, start, tokenizer)
        changes = {}
        for name in (
            "message_routing_drift_mean",
            "message_evidence_share_mean",
            "message_source_dispersion_mean",
            "evidence_message_effect",
            "response_message_effect",
        ):
            changes[name] = (
                float(metrics[name][start] - metrics[name][previous])
                if previous >= 0
                else None
            )
        onsets.append(
            {
                "start": start,
                "stop": stop,
                "token": token_text[start],
                "span_text": tokenizer.decode(target_ids[start:stop].tolist()),
                "previous_token": token_text[previous] if previous >= 0 else None,
                "changes_from_previous_token": changes,
                "evidence_effect": float(metrics["evidence_message_effect"][start]),
                "response_effect": float(metrics["response_message_effect"][start]),
                "full_margin": float(metrics["full_margin"][start]),
                "top_source_mass": (
                    "retained top-k source mass divided by exact all-source mass "
                    "over the final layer third"
                ),
                "top_late_sources": top_sources,
                "top_late_head_routes": top_head_routes,
            }
        )

    summary = {}
    for name in SAMPLE_METRICS:
        value = metrics[name]
        summary[name] = {
            "all": float(value.mean()),
            "correct": _class_mean(value, ~label),
            "hallucinated": _class_mean(value, label),
        }
    return {
        "schema": "ragtruth-functional-message-sample-audit-v1",
        "sample_id": str(row["sample_id"]),
        "source_id": str(row["source_id"]),
        "split": row.get("split"),
        "generator_model": row.get("generator_model"),
        "response_text": tokenizer.decode(target_ids.tolist()),
        "token_ids": target_ids.tolist(),
        "token_text": token_text,
        "label": label.astype(np.int8).tolist(),
        "hallucinated_tokens": int(label.sum()),
        "hallucinated_fraction": float(label.mean()),
        "hallucination_runs": [[start, stop] for start, stop in runs],
        "summary": summary,
        "token_metrics": {
            name: metrics[name].astype(float).tolist() for name in SAMPLE_METRICS
        },
        "onsets": onsets,
    }


def _holm(report: dict) -> None:
    primary = report["statistical_design"]["primary_endpoints"]
    available = [
        (name, report["summaries"][name]["p_value"])
        for name in primary
        if report["summaries"][name]["p_value"] is not None
    ]
    ordered = sorted(available, key=lambda item: item[1])
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        adjusted = min(1.0, (len(ordered) - rank) * value)
        running = max(running, adjusted)
        report["summaries"][name]["holm_p_value"] = running


def _summarize_arrays(
    *,
    label: np.ndarray,
    sample_id: np.ndarray,
    source_id: np.ndarray,
    generator: np.ndarray,
    token_index: np.ndarray,
    response_length: np.ndarray,
    metrics: dict[str, np.ndarray],
    samples: int,
    position_bin: int,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    match_design = _position_match_design(
        label,
        sample_id,
        source_id,
        generator,
        token_index,
        response_length,
        position_bin=position_bin,
    )
    onset_pairs = _onset_pairs(
        label,
        sample_id,
        source_id,
        token_index,
    )
    summaries, onset = {}, {}
    for offset, (name, value) in enumerate(metrics.items()):
        current_bootstrap = (
            bootstrap
            if name in REGISTERED_INFERENCE_METRICS
            else min(bootstrap, 1000)
        )
        summaries[name] = {
            "correct_mean": _mean_or_none(value[~label]),
            "hallucinated_mean": _mean_or_none(value[label]),
            **_position_matched_difference(
                value,
                label,
                sample_id,
                source_id,
                generator,
                token_index,
                response_length,
                position_bin=position_bin,
                bootstrap=current_bootstrap,
                seed=seed + offset,
                design=match_design,
            ),
        }
        if name in {
            "message_routing_drift_mean",
            "message_source_dispersion_mean",
            "evidence_message_effect",
        }:
            onset[name] = _matched_onset(
                value,
                onset_pairs,
                bootstrap=bootstrap,
                seed=seed + offset,
            )
    report = {
        "schema": "ragtruth-functional-message-evaluation-v3",
        "samples": int(samples),
        "tokens": int(len(label)),
        "hallucinated_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        "position_bin": int(position_bin),
        "statistical_design": {
            "estimand": "hallucinated_minus_correct_within_mixed_responses",
            "matching": "sample_id + absolute_position_bin + relative_position_decile",
            "aggregation": "overlap-weighted cells -> response -> generator/source -> equal source",
            "bootstrap_unit": "source_id",
            "bootstrap_samples": int(bootstrap),
            "diagnostic_bootstrap_samples": int(min(bootstrap, 1000)),
            "multiplicity_correction": "Holm for three primary endpoints",
            "primary_endpoints": [
                "message_routing_drift_mean",
                "message_source_dispersion_mean",
                "evidence_message_effect",
            ],
            "onset": (
                "each hallucination-span onset vs up to eight nearest all-correct "
                "transitions in the same response"
            ),
        },
        "summaries": summaries,
        "matched_onset": onset,
    }
    preferred = metrics["full_margin"] > 0
    candidate = metrics["message_independent_capture_signature"] > 0

    def rate(mask: np.ndarray, event: np.ndarray) -> float | None:
        return float(event[mask].mean()) if mask.any() else None

    report["observer_readout"] = {
        "target_preferred_correct": rate(~label, preferred),
        "target_preferred_hallucinated": rate(label, preferred),
        "capture_given_preferred_correct": rate(~label & preferred, candidate),
        "capture_given_preferred_hallucinated": rate(label & preferred, candidate),
    }
    _holm(report)
    return report


def evaluate_saved(
    *,
    trace_root: str | Path,
    split_root: str | Path,
    output: str | Path,
    model_path: str | Path,
    split_name: str | None = None,
    sample_output: str | Path | None = None,
    figure_output: str | Path | None = None,
    position_bin: int = 16,
    bootstrap: int = 10000,
    seed: int = 20260828,
) -> dict[str, object]:
    from transformers import AutoTokenizer

    rows = load_index(trace_root)
    sample_ids = [row["sample_id"] for row in rows]
    dataset = open_research_dataset(
        split_root, device="cpu", retain_embedded_labels=True
    )
    labels = dataset.prepare_evaluation_labels(sample_ids)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    split_name = str(split_name or dataset.manifest.get("split") or "unknown")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_output = Path(sample_output or output.parent / "sample_audits")
    figure_output = Path(figure_output or output.parent / "figures")
    sample_output.mkdir(parents=True, exist_ok=True)
    (figure_output / "samples").mkdir(parents=True, exist_ok=True)

    collected: dict[str, list[np.ndarray]] = {}
    label_rows, sample_rows, source_rows = [], [], []
    generator_rows, split_rows = [], []
    token_rows, length_rows = [], []
    sample_records, sample_index = [], []
    for row in rows:
        artifact = torch.load(
            Path(trace_root) / "samples" / row["path"],
            map_location="cpu",
            weights_only=True,
        )
        sample = dataset[row["sample_id"]]
        attention = sample.attention()
        if str(sample.source_id) != str(row["source_id"]):
            raise ValueError("saved trace source does not match evaluation cache")
        response_start = int(attention.response_idx)
        cached_targets = attention.token_ids[response_start:].cpu()
        if not torch.equal(artifact["target_ids"], cached_targets):
            raise ValueError("saved trace targets do not match evaluation cache")
        current_layers = layer_metrics(artifact)
        current = token_metrics(artifact, current_layers)
        token_label = labels.response_labels(sample).cpu().numpy().astype(bool)
        sample.release_attention()
        count = len(token_label)
        record = _sample_record(
            {**row, "split": split_name},
            artifact,
            token_label,
            current,
            tokenizer,
        )
        record_path = sample_output / f"{row['sample_id']}.json"
        record_path.write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        figure_path = figure_output / "samples" / f"{row['sample_id']}.png"
        plot_sample_dashboard(
            record,
            {
                "routing_imbalance": current_layers["message_routing_drift"].numpy(),
                "source_dispersion": current_layers["message_source_dispersion"].numpy(),
                "evidence_share": current_layers["message_evidence_share"].numpy(),
                "response_share": current_layers["message_response_share"].numpy(),
            },
            figure_path,
        )
        population_row = {
            "sample_id": str(row["sample_id"]),
            "source_id": str(row["source_id"]),
            "split": split_name,
            "hallucinated_tokens": int(token_label.sum()),
            "response_tokens": int(count),
            "hallucinated_fraction": float(token_label.mean()),
            "routing_mean": float(current["message_routing_drift_mean"].mean()),
            "evidence_effect_mean": float(current["evidence_message_effect"].mean()),
            "audit": str(record_path),
            "figure": str(figure_path),
        }
        sample_records.append(population_row)
        sample_index.append(population_row)
        for name, value in current.items():
            collected.setdefault(name, []).append(value)
        label_rows.append(token_label)
        analysis_sample = f"{split_name}:{row['sample_id']}"
        sample_rows.append(np.repeat(np.asarray([analysis_sample]), count))
        source_rows.append(np.repeat(np.asarray([str(row["source_id"])]), count))
        generator_rows.append(
            np.repeat(np.asarray([str(row.get("generator_model"))]), count)
        )
        split_rows.append(np.repeat(np.asarray([split_name]), count))
        token_rows.append(np.arange(count, dtype=np.int32))
        length_rows.append(np.full(count, count, dtype=np.int32))

    label = np.concatenate(label_rows)
    sample_id = np.concatenate(sample_rows)
    source_id = np.concatenate(source_rows)
    generator = np.concatenate(generator_rows)
    split = np.concatenate(split_rows)
    token_index = np.concatenate(token_rows)
    response_length = np.concatenate(length_rows)
    metrics = {name: np.concatenate(values) for name, values in collected.items()}
    arrays_path = output.with_name("token_metrics.npz")
    np.savez_compressed(
        arrays_path,
        label=label,
        sample_id=sample_id,
        source_id=source_id,
        generator_model=generator,
        split=split,
        token_index=token_index,
        response_length=response_length,
        **metrics,
    )
    report = _summarize_arrays(
        label=label,
        sample_id=sample_id,
        source_id=source_id,
        generator=generator,
        token_index=token_index,
        response_length=response_length,
        metrics=metrics,
        samples=len(rows),
        position_bin=position_bin,
        bootstrap=bootstrap,
        seed=seed,
    )
    manifest_path = Path(trace_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = report["summaries"]["message_routing_drift_mean"]
    mixed_samples = sum(
        0 < record["hallucinated_tokens"] < record["response_tokens"]
        for record in sample_records
    )
    report.update(
        {
            "split": split_name,
            "generator_models": sorted(
                {str(row.get("generator_model")) for row in rows}
            ),
            "coverage": {
                "cached_candidates": int(
                    manifest.get("dataset_candidates", len(rows))
                ),
                "eligible_qa": manifest.get(
                    "eligible_qa", manifest.get("eligible_qa_seen", len(rows))
                ),
                "captured": int(manifest.get("samples", len(rows))),
                "evaluated": int(len(rows)),
                "complete": bool(manifest.get("complete", False)),
                "mixed_label_samples": int(mixed_samples),
                "matched_samples": int(reference["matched_samples"]),
                "matched_cells": int(reference["matched_cells"]),
                "covered_hallucinated_tokens": int(
                    reference["covered_hallucinated_tokens"]
                ),
                "hallucinated_token_coverage": float(
                    reference["hallucinated_token_coverage"]
                ),
            },
            "labels_used_during": "posthoc_evaluation_only",
            "analysis_scope": "mechanism_audit_not_hallucination_detector",
            "observer_scope": (
                "teacher-forced observer dynamics; formation claims require observer "
                "and generator checkpoints to match"
            ),
            "token_metrics": str(arrays_path),
            "sample_audits": {
                "directory": str(sample_output),
                "index": str(output.with_name("sample_audits.jsonl")),
                "count": len(sample_index),
            },
            "figures": str(figure_output),
        }
    )
    output.with_name("sample_audits.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sample_index),
        encoding="utf-8",
    )
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    plot_population(report, sample_records, figure_output)
    return report


def combine_saved(
    *,
    inputs: list[tuple[str, str | Path]],
    output: str | Path,
    figure_output: str | Path | None = None,
    position_bin: int = 16,
    bootstrap: int = 10000,
    seed: int = 20260828,
) -> dict[str, object]:
    """Recompute one coverage summary from split reports and token arrays."""

    metadata = {
        "label",
        "sample_id",
        "source_id",
        "generator_model",
        "split",
        "token_index",
        "response_length",
    }
    arrays, reports, sample_records, sample_rows = [], {}, [], []
    for name, path in inputs:
        current_report = json.loads(Path(path).read_text(encoding="utf-8"))
        reports[name] = current_report
        arrays.append(np.load(current_report["token_metrics"], allow_pickle=False))
        index_path = Path(current_report["sample_audits"]["index"])
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            sample_rows.append(row)
            sample_records.append(row)

    metric_names = sorted(set(arrays[0].files) - metadata)
    merged = {
        name: np.concatenate([current[name] for current in arrays])
        for name in metadata | set(metric_names)
    }
    metrics = {name: merged[name] for name in metric_names}
    report = _summarize_arrays(
        label=merged["label"].astype(bool),
        sample_id=merged["sample_id"],
        source_id=merged["source_id"],
        generator=merged["generator_model"],
        token_index=merged["token_index"],
        response_length=merged["response_length"],
        metrics=metrics,
        samples=sum(current["samples"] for current in reports.values()),
        position_bin=position_bin,
        bootstrap=bootstrap,
        seed=seed,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = output.with_name("token_metrics.npz")
    np.savez_compressed(arrays_path, **merged)
    sample_index_path = output.with_name("sample_audits.jsonl")
    sample_index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sample_rows),
        encoding="utf-8",
    )
    source_sets = {
        name: set(np.unique(current["source_id"])) for (name, _path), current in zip(inputs, arrays)
    }
    overlap = (
        [str(value) for value in sorted(set.intersection(*source_sets.values()))]
        if len(source_sets) > 1
        else []
    )
    report.update(
        {
            "scope": "all_available_qa_splits",
            "by_split": reports,
            "coverage": {
                "cached_candidates": sum(r["coverage"]["cached_candidates"] for r in reports.values()),
                "eligible_qa": (
                    sum(r["coverage"]["eligible_qa"] for r in reports.values())
                    if all(r["coverage"]["eligible_qa"] is not None for r in reports.values())
                    else None
                ),
                "captured": sum(r["coverage"]["captured"] for r in reports.values()),
                "evaluated": sum(r["coverage"]["evaluated"] for r in reports.values()),
                "complete": all(r["coverage"]["complete"] for r in reports.values()),
                "mixed_label_samples": sum(r["coverage"]["mixed_label_samples"] for r in reports.values()),
                "matched_samples": report["summaries"]["message_routing_drift_mean"]["matched_samples"],
                "matched_cells": report["summaries"]["message_routing_drift_mean"]["matched_cells"],
                "covered_hallucinated_tokens": report["summaries"]["message_routing_drift_mean"]["covered_hallucinated_tokens"],
                "hallucinated_token_coverage": report["summaries"]["message_routing_drift_mean"]["hallucinated_token_coverage"],
            },
            "source_overlap_between_splits": overlap,
            "token_metrics": str(arrays_path),
            "sample_audits": {"index": str(sample_index_path), "count": len(sample_rows)},
        }
    )
    figure_output = Path(figure_output or output.parent / "figures")
    report["figures"] = str(figure_output)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    plot_population(report, sample_records, figure_output)
    for current in arrays:
        current.close()
    return report


__all__ = ["combine_saved", "evaluate_saved", "layer_metrics", "token_metrics"]
