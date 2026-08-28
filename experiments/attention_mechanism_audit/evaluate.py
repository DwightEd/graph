"""Position-matched post-hoc tests for the three saved mechanisms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from research_dataset import open_research_dataset

from .audit import load_index
from .capture import HISTORY, SELF
from .data import EVIDENCE


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


def token_metrics(artifact: dict) -> dict[str, np.ndarray]:
    """Reduce raw layer/head routes without discarding layer evolution."""

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

    metrics = {}
    metrics.update(_layer_summary(edge_evidence, "message_evidence_share"))
    metrics.update(_layer_summary(edge_response, "message_response_share"))
    metrics.update(
        _layer_summary(edge_response - edge_evidence, "message_routing_drift")
    )
    metrics.update(_layer_summary(attention_evidence, "attention_evidence_share"))
    metrics.update(_layer_summary(attention_response, "attention_response_share"))
    metrics.update(
        _layer_summary(
            attention_response - attention_evidence, "attention_routing_drift"
        )
    )
    metrics.update(_layer_summary(dispersion, "message_source_dispersion"))
    metrics.update(_layer_summary(head_role_js, "head_role_disagreement"))
    metrics.update(_layer_summary(coherence, "message_coherence"))
    metrics.update({name: value.float() for name, value in artifact["mechanism"].items()})
    mechanism = artifact["mechanism"]
    metrics["message_independent_capture_signature"] = (
        (mechanism["evidence_response_removed_margin"] > 0)
        & (mechanism["full_margin"] > 0)
        & (mechanism["evidence_message_effect"] <= 0)
    ).float()
    return {name: value.cpu().numpy() for name, value in metrics.items()}


def _position_matched_difference(
    value: np.ndarray,
    label: np.ndarray,
    source: np.ndarray,
    token_index: np.ndarray,
    response_length: np.ndarray,
    *,
    position_bin: int,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    relative_bin = np.minimum(
        ((token_index + 0.5) * 10 / response_length).astype(np.int16), 9
    )
    ordinal_bin = token_index // position_bin
    cells: dict[tuple[str, int, int], list[int]] = {}
    for index, key in enumerate(zip(source, ordinal_bin, relative_bin)):
        cells.setdefault(key, []).append(index)
    by_source: dict[str, list[float]] = {}
    matched_cells = 0
    for (source_id, _ordinal, _relative), indices in cells.items():
        indices = np.asarray(indices)
        current_label = label[indices]
        if current_label.any() and (~current_label).any():
            effect = value[indices[current_label]].mean() - value[
                indices[~current_label]
            ].mean()
            by_source.setdefault(source_id, []).append(float(effect))
            matched_cells += 1
    effects = np.asarray(
        [np.mean(current) for current in by_source.values()], dtype=np.float64
    )
    if not len(effects):
        return {
            "position_matched_source_equal_difference": None,
            "ci95": [None, None],
            "sources": 0,
            "matched_cells": 0,
        }
    random = np.random.default_rng(seed)
    draws = random.choice(effects, (bootstrap, len(effects)), replace=True).mean(1)
    return {
        "position_matched_source_equal_difference": float(effects.mean()),
        "ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))],
        "sources": int(len(effects)),
        "matched_cells": int(matched_cells),
    }


def _onset_pairs(
    label: np.ndarray,
    sample_id: np.ndarray,
    source_id: np.ndarray,
    token_index: np.ndarray,
    response_length: np.ndarray,
    *,
    position_bin: int,
    radius: int = 8,
) -> list[list[tuple[int, int, np.ndarray, np.ndarray, str]]]:
    sample_rows = {
        sample: np.flatnonzero(sample_id == sample) for sample in np.unique(sample_id)
    }
    pairs = [[] for _ in range(2 * radius + 1)]
    for rows in sample_rows.values():
        rows = rows[np.argsort(token_index[rows])]
        current_label = label[rows]
        starts = np.flatnonzero(current_label & ~np.r_[False, current_label[:-1]])
        if (
            not len(starts)
            or starts[0] < radius
            or starts[0] + radius >= len(rows)
        ):
            continue
        onset = int(starts[0])
        source = source_id[rows[0]]
        ordinal = token_index[rows[onset]] // position_bin
        relative = min(
            int((token_index[rows[onset]] + 0.5) * 10 / response_length[rows[onset]]),
            9,
        )
        controls = []
        for candidate_rows in sample_rows.values():
            if source_id[candidate_rows[0]] != source:
                continue
            candidate_rows = candidate_rows[np.argsort(token_index[candidate_rows])]
            if label[candidate_rows].any():
                continue
            for center in range(radius, len(candidate_rows) - radius):
                row = candidate_rows[center]
                if token_index[row] // position_bin != ordinal:
                    continue
                candidate_relative = min(
                    int((token_index[row] + 0.5) * 10 / response_length[row]), 9
                )
                if candidate_relative == relative:
                    controls.append((candidate_rows, center))
        if not controls:
            continue
        for offset in range(-radius, radius + 1):
            event_position = onset + offset
            control_positions, control_bases = [], []
            for control_rows, center in controls:
                position = center + offset
                control_positions.append(control_rows[position])
                control_bases.append(control_rows[center - 1])
            pairs[offset + radius].append(
                (
                    rows[event_position],
                    rows[onset - 1],
                    np.asarray(control_positions),
                    np.asarray(control_bases),
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


def evaluate_saved(
    *,
    trace_root: str | Path,
    split_root: str | Path,
    output: str | Path,
    position_bin: int = 16,
    bootstrap: int = 1000,
    seed: int = 20260828,
) -> dict[str, object]:
    rows = load_index(trace_root)
    sample_ids = [row["sample_id"] for row in rows]
    dataset = open_research_dataset(
        split_root, device="cpu", retain_embedded_labels=True
    )
    labels = dataset.prepare_evaluation_labels(sample_ids)

    collected: dict[str, list[np.ndarray]] = {}
    label_rows, sample_rows, source_rows = [], [], []
    token_rows, length_rows = [], []
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
        current = token_metrics(artifact)
        token_label = labels.response_labels(sample).cpu().numpy().astype(bool)
        count = len(token_label)
        for name, value in current.items():
            collected.setdefault(name, []).append(value)
        label_rows.append(token_label)
        sample_rows.append(np.repeat(np.asarray([str(row["sample_id"])]), count))
        source_rows.append(np.repeat(np.asarray([str(row["source_id"])]), count))
        token_rows.append(np.arange(count, dtype=np.int32))
        length_rows.append(np.full(count, count, dtype=np.int32))
        sample.release_attention()

    label = np.concatenate(label_rows)
    sample_id = np.concatenate(sample_rows)
    source_id = np.concatenate(source_rows)
    token_index = np.concatenate(token_rows)
    response_length = np.concatenate(length_rows)
    metrics = {name: np.concatenate(values) for name, values in collected.items()}
    onset_pairs = _onset_pairs(
        label,
        sample_id,
        source_id,
        token_index,
        response_length,
        position_bin=position_bin,
    )
    summaries, onset = {}, {}
    for offset, (name, value) in enumerate(metrics.items()):
        summaries[name] = {
            "correct_mean": float(value[~label].mean()),
            "hallucinated_mean": float(value[label].mean()),
            **_position_matched_difference(
                value,
                label,
                source_id,
                token_index,
                response_length,
                position_bin=position_bin,
                bootstrap=bootstrap,
                seed=seed + offset,
            ),
        }
        onset[name] = _matched_onset(
            value,
            onset_pairs,
            bootstrap=bootstrap,
            seed=seed + offset,
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = output.with_name("token_metrics.npz")
    np.savez_compressed(
        arrays_path,
        label=label,
        sample_id=sample_id,
        source_id=source_id,
        token_index=token_index,
        response_length=response_length,
        **metrics,
    )
    report = {
        "schema": "ragtruth-functional-message-evaluation-v2",
        "samples": len(rows),
        "tokens": int(len(label)),
        "hallucinated_tokens": int(label.sum()),
        "generator_models": sorted(
            {str(row.get("generator_model")) for row in rows}
        ),
        "position_bin": int(position_bin),
        "labels_used_during": "posthoc_evaluation_only",
        "observer_scope": (
            "teacher-forced observer dynamics; formation claims require observer "
            "and generator checkpoints to match"
        ),
        "summaries": summaries,
        "matched_onset": onset,
        "token_metrics": str(arrays_path),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


__all__ = ["evaluate_saved", "token_metrics"]
