"""Pool saved traces by task and evaluate four fixed label-free scores."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from research_dataset import open_research_dataset

from .audit import SCHEMA, VERSION, load_index
from .data import canonical_task_type
from .visualize import plot_population, plot_sample_dashboard

SCORE_ORDER = (
    "causal_route_capture",
    "routing_imbalance",
    "source_dispersion",
    "message_independent_preference",
)

SCORE_DEFINITIONS = {
    "causal_route_capture": ("logp(no evidence messages) - logp(no response messages)"),
    "routing_imbalance": (
        "mean_layer(response functional-message share - evidence share)"
    ),
    "source_dispersion": (
        "mean_layer(normalized entropy of source-token message magnitudes)"
    ),
    "message_independent_preference": (
        "observed-token margin after evidence and response messages are removed"
    ),
}

AUDIT_BASES = (
    "evidence_share",
    "response_share",
    "routing_imbalance",
    "source_dispersion",
)


def _layer_reductions(value: torch.Tensor, name: str) -> dict[str, np.ndarray]:
    """Keep the old early/late audit without creating new detector features."""

    width = max(value.shape[0] // 3, 1)
    reduced = {
        f"{name}_mean": value.mean(0),
        f"{name}_early": value[:width].mean(0),
        f"{name}_late": value[-width:].mean(0),
        f"{name}_layer_shift": value[-width:].mean(0) - value[:width].mean(0),
    }
    return {key: current.cpu().numpy() for key, current in reduced.items()}


def _binary_metrics(label: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    prevalence = float(label.mean())
    auprc = float(average_precision_score(label, score))
    return {
        "auroc": float(roc_auc_score(label, score)),
        "auprc": auprc,
        "auprc_lift": auprc / prevalence,
    }


def _source_bootstrap(
    label: np.ndarray,
    score: np.ndarray,
    source_id: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    groups = np.unique(source_id)
    rows = {group: np.flatnonzero(source_id == group) for group in groups}
    random = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        chosen = random.choice(groups, len(groups), replace=True)
        index = np.concatenate([rows[group] for group in chosen])
        if np.unique(label[index]).size == 2:
            values.append(
                (
                    roc_auc_score(label[index], score[index]),
                    average_precision_score(label[index], score[index]),
                )
            )
    values = np.asarray(values)
    return {
        "replicates": int(len(values)),
        "auroc_low": float(np.quantile(values[:, 0], 0.025)),
        "auroc_high": float(np.quantile(values[:, 0], 0.975)),
        "auprc_low": float(np.quantile(values[:, 1], 0.025)),
        "auprc_high": float(np.quantile(values[:, 1], 0.975)),
    }


def layer_mechanisms(artifact: dict) -> dict[str, torch.Tensor]:
    """Return the two route mechanisms without expanding hand-built features."""

    trace = artifact["trace"]
    total = trace["total_message_magnitude"].float().clamp_min(1e-12)
    evidence_share = trace["evidence_message_magnitude"].float() / total
    response_share = trace["response_message_magnitude"].float() / total

    active_sources = torch.arange(total.shape[1]) + artifact["response_start"]
    active_sources = active_sources.float().clamp_min(2)
    dispersion = trace["source_message_entropy"].float() / active_sources.log()[None]
    return {
        "routing_imbalance": response_share - evidence_share,
        "source_dispersion": dispersion,
        "evidence_share": evidence_share,
        "response_share": response_share,
    }


def token_scores(artifact: dict) -> dict[str, np.ndarray]:
    """Compute the fixed primary score and its three mechanism components."""

    layers = layer_mechanisms(artifact)
    inputs = artifact["score_inputs"]
    values = {
        "causal_route_capture": (
            inputs["no_evidence_logprob"] - inputs["no_response_logprob"]
        ),
        "routing_imbalance": layers["routing_imbalance"].mean(0),
        "source_dispersion": layers["source_dispersion"].mean(0),
        "message_independent_preference": inputs["no_evidence_response_margin"],
    }
    return {name: values[name].float().cpu().numpy() for name in SCORE_ORDER}


def token_audit_metrics(artifact: dict) -> dict[str, np.ndarray]:
    """Measurements used only for hallucinated-vs-correct post-hoc auditing."""

    layers = layer_mechanisms(artifact)
    metrics = {}
    for name in AUDIT_BASES:
        metrics.update(_layer_reductions(layers[name], name))
    return metrics


def _position_match_design(
    label: np.ndarray,
    sample_id: np.ndarray,
    token_index: np.ndarray,
    response_length: np.ndarray,
    *,
    position_bin: int,
) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """Match labels inside one response at similar absolute/relative positions."""

    relative = np.minimum(
        ((token_index + 0.5) * 10 / response_length).astype(np.int16), 9
    )
    absolute = token_index // position_bin
    cells: dict[tuple[str, int, int], list[int]] = {}
    for index, key in enumerate(zip(sample_id, absolute, relative)):
        cells.setdefault(key, []).append(index)

    matched = []
    for (sample, _absolute, _relative), rows in cells.items():
        rows = np.asarray(rows)
        positive = rows[label[rows]]
        negative = rows[~label[rows]]
        if len(positive) and len(negative):
            weight = len(positive) * len(negative) / len(rows)
            matched.append((str(sample), positive, negative, float(weight)))
    return matched


def _position_matched_difference(
    value: np.ndarray,
    matched: list[tuple[str, np.ndarray, np.ndarray, float]],
    sample_source: dict[str, str],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Aggregate within-response contrasts, then weight every source equally."""

    by_sample: dict[str, list[tuple[float, float]]] = {}
    for sample, positive, negative, weight in matched:
        effect = float(value[positive].mean() - value[negative].mean())
        by_sample.setdefault(sample, []).append((effect, weight))

    by_source: dict[str, list[float]] = {}
    for sample, rows in by_sample.items():
        effects = np.asarray([effect for effect, _weight in rows])
        weights = np.asarray([weight for _effect, weight in rows])
        by_source.setdefault(sample_source[sample], []).append(
            float(np.average(effects, weights=weights))
        )
    source_effects = np.asarray(
        [np.mean(effects) for effects in by_source.values()], dtype=np.float64
    )
    if not len(source_effects):
        return {
            "hallucinated_minus_correct": None,
            "ci95": [None, None],
            "sources": 0,
            "matched_samples": 0,
            "matched_cells": 0,
        }

    interval = [None, None]
    if bootstrap:
        random = np.random.default_rng(seed)
        draws = random.choice(
            source_effects, (bootstrap, len(source_effects)), replace=True
        ).mean(1)
        interval = [float(value) for value in np.quantile(draws, (0.025, 0.975))]
    return {
        "hallucinated_minus_correct": float(source_effects.mean()),
        "ci95": interval,
        "sources": int(len(source_effects)),
        "matched_samples": int(len(by_sample)),
        "matched_cells": int(len(matched)),
    }


def group_difference_audit(
    arrays: dict[str, np.ndarray],
    audit_names: tuple[str, ...],
    *,
    position_bin: int,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Compare hallucinated and correct tokens after within-response matching."""

    label = arrays["label"]
    matched = _position_match_design(
        label,
        arrays["sample_id"],
        arrays["token_index"],
        arrays["response_length"],
        position_bin=position_bin,
    )
    sample_source = {
        str(sample): str(source)
        for sample, source in zip(arrays["sample_id"], arrays["source_id"])
    }
    metrics = {}
    for offset, name in enumerate(audit_names):
        value = arrays[name]
        metrics[name] = {
            "correct_mean": float(value[~label].mean()),
            "hallucinated_mean": float(value[label].mean()),
            **_position_matched_difference(
                value,
                matched,
                sample_source,
                bootstrap=bootstrap,
                seed=seed + offset,
            ),
        }
    covered = sum(len(positive) for _sample, positive, _negative, _weight in matched)
    return {
        "role": "posthoc_mechanism_audit_not_score_selection",
        "matching": "sample_id + absolute_position_bin + relative_position_decile",
        "aggregation": "matched cells -> response -> equal source",
        "bootstrap_unit": "source_id",
        "position_bin": int(position_bin),
        "covered_hallucinated_tokens": int(covered),
        "hallucinated_token_coverage": float(covered / max(label.sum(), 1)),
        "metrics": metrics,
    }


def detection_summary(
    label: np.ndarray,
    scores: dict[str, np.ndarray],
    source_id: np.ndarray,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, dict]:
    """Evaluate fixed score directions; labels never change a score or sign."""

    if np.unique(label).size != 2:
        return {
            name: {
                "auroc": None,
                "auprc": None,
                "auprc_lift": None,
                "auroc_ci95": [None, None],
                "auprc_ci95": [None, None],
            }
            for name in SCORE_ORDER
        }

    results = {}
    for offset, name in enumerate(SCORE_ORDER):
        result = _binary_metrics(label, scores[name])
        if bootstrap:
            interval = _source_bootstrap(
                label,
                scores[name],
                source_id,
                replicates=bootstrap,
                seed=seed + offset,
            )
            result.update(
                {
                    "auroc_ci95": [
                        interval["auroc_low"],
                        interval["auroc_high"],
                    ],
                    "auprc_ci95": [
                        interval["auprc_low"],
                        interval["auprc_high"],
                    ],
                    "bootstrap_replicates": interval["replicates"],
                }
            )
        else:
            result.update({"auroc_ci95": [None, None], "auprc_ci95": [None, None]})
        results[name] = result
    return results


def build_report(
    *,
    task_type: str,
    label: np.ndarray,
    sample_id: np.ndarray,
    source_id: np.ndarray,
    scores: dict[str, np.ndarray],
    bootstrap: int,
    seed: int,
) -> dict:
    """Build the single all-data report."""

    detection = detection_summary(
        label,
        scores,
        source_id,
        bootstrap=bootstrap,
        seed=seed,
    )
    return {
        "schema": "ragtruth-three-mechanism-detection-v2",
        "task_type": task_type,
        "samples": int(np.unique(sample_id).size),
        "sources": int(np.unique(source_id).size),
        "tokens": int(len(label)),
        "hallucinated_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        "primary_score": "causal_route_capture",
        "score_definitions": SCORE_DEFINITIONS,
        "score_direction": (
            "higher means more hallucination-like; source_dispersion keeps "
            "the original high-dispersion hypothesis"
        ),
        "detection": detection,
        "labels_used_during": "posthoc_evaluation_only",
        "analysis_scope": (
            "fixed-score exploratory audit over pooled captured task tokens"
        ),
    }


def _load_scores(
    trace_root: Path,
    task_type: str,
) -> list[tuple[dict, dict[str, np.ndarray]]]:
    scored = []
    for row in load_index(trace_root):
        if canonical_task_type(row["task_type"]) != task_type:
            continue
        artifact = torch.load(
            trace_root / "samples" / row["path"],
            map_location="cpu",
            weights_only=True,
        )
        scored.append(
            (row, {**token_scores(artifact), **token_audit_metrics(artifact)})
        )
    return scored


def _add_labels(
    scored: list[tuple[dict, dict[str, np.ndarray]]],
    split_root: Path,
) -> dict[str, np.ndarray]:
    dataset = open_research_dataset(
        split_root,
        device="cpu",
        retain_embedded_labels=True,
    )
    labels = dataset.prepare_evaluation_labels([row["sample_id"] for row, _ in scored])
    metric_names = tuple(scored[0][1])
    arrays: dict[str, list[np.ndarray]] = {
        "label": [],
        "sample_id": [],
        "source_id": [],
        "token_index": [],
        "response_length": [],
        **{name: [] for name in metric_names},
    }
    for row, current in scored:
        sample = dataset[row["sample_id"]]
        token_label = labels.response_labels(sample).cpu().numpy().astype(bool)
        sample.release_attention()
        count = len(token_label)
        if any(len(value) != count for value in current.values()):
            raise ValueError(
                f"trace/label length mismatch for sample {row['sample_id']}"
            )
        arrays["label"].append(token_label)
        arrays["sample_id"].append(np.repeat(str(row["sample_id"]), count))
        arrays["source_id"].append(np.repeat(str(row["source_id"]), count))
        arrays["token_index"].append(np.arange(count, dtype=np.int32))
        arrays["response_length"].append(np.full(count, count, dtype=np.int32))
        for name in metric_names:
            arrays[name].append(current[name])
    return {name: np.concatenate(value) for name, value in arrays.items()}


def _load_manifest(
    trace_root: Path,
    split_root: Path | None = None,
    task_type: str | None = None,
) -> dict:
    manifest = json.loads((trace_root / "manifest.json").read_text(encoding="utf-8"))
    valid = (
        manifest.get("schema") == SCHEMA
        and manifest.get("version") == VERSION
        and (
            split_root is None
            or manifest.get("split_root") == str(split_root.resolve())
        )
        and (task_type is None or task_type in manifest.get("task_types", []))
    )
    if not valid:
        raise ValueError(f"trace manifest does not match v{VERSION} or its cache")
    return manifest


def evaluate_all(
    *,
    inputs: Iterable[tuple[str | Path, str | Path]],
    task_type: str,
    output: str | Path,
    bootstrap: int = 1000,
    seed: int = 20260828,
    position_bin: int = 16,
) -> dict:
    """Pool physical cache shards first, then evaluate exactly once."""

    task_type = canonical_task_type(task_type)
    shards, manifests = [], []
    for trace_root, split_root in inputs:
        trace_root = Path(trace_root)
        split_root = Path(split_root)
        manifests.append(_load_manifest(trace_root, split_root, task_type))
        scores = _load_scores(trace_root, task_type)
        if not scores:
            raise ValueError(f"no {task_type} samples in {trace_root}")
        shards.append((scores, split_root))
    pieces = [_add_labels(scores, split_root) for scores, split_root in shards]
    merged = {
        name: np.concatenate([piece[name] for piece in pieces]) for name in pieces[0]
    }
    scores = {name: merged[name] for name in SCORE_ORDER}
    report = build_report(
        task_type=task_type,
        label=merged["label"],
        sample_id=merged["sample_id"],
        source_id=merged["source_id"],
        scores=scores,
        bootstrap=bootstrap,
        seed=seed,
    )
    audit_names = tuple(
        name
        for name in merged
        if any(name.startswith(f"{base}_") for base in AUDIT_BASES)
    )
    report["group_difference_audit"] = group_difference_audit(
        merged,
        audit_names,
        position_bin=position_bin,
        bootstrap=bootstrap,
        seed=seed + len(SCORE_ORDER),
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores_path = output.with_name("token_scores.npz")
    figures = output.parent / "figures"
    np.savez_compressed(scores_path, **merged)
    plot_population(
        merged["label"],
        scores,
        merged["token_index"],
        merged["response_length"],
        report,
        figures,
    )
    report.update(
        {
            "token_scores": str(scores_path),
            "figures": str(figures),
            "physical_cache_shards": len(pieces),
            "capture_complete": all(manifest["complete"] for manifest in manifests),
        }
    )
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


def plot_saved_sample(
    *,
    inputs: Iterable[str | Path],
    sample_id: str,
    model_path: str | Path,
    output: str | Path,
) -> dict:
    """Render one saved response without replaying the model."""

    from transformers import AutoTokenizer

    for trace_root_value in inputs:
        trace_root = Path(trace_root_value)
        _load_manifest(trace_root)
        row = next(
            (
                row
                for row in load_index(trace_root)
                if str(row["sample_id"]) == str(sample_id)
            ),
            None,
        )
        if row is None:
            continue
        artifact = torch.load(
            trace_root / "samples" / row["path"],
            map_location="cpu",
            weights_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        layers = layer_mechanisms(artifact)
        trace = artifact["trace"]
        top_index = trace["top_source_index"].numpy()
        top_mass = trace["top_source_magnitude"].numpy()
        response_tokens = top_index.shape[1]
        source_flow = np.zeros(
            (response_tokens, len(artifact["token_ids"]) - 1),
            dtype=np.float32,
        )
        response_index = np.broadcast_to(
            np.arange(response_tokens)[None, :, None],
            top_index.shape,
        )
        valid = top_index >= 0
        np.add.at(
            source_flow,
            (response_index[valid], top_index[valid]),
            top_mass[valid],
        )
        source_flow /= trace["total_message_magnitude"].sum(0).numpy()[:, None] + 1e-12
        shown = np.argsort(source_flow.sum(0))[-16:][::-1]
        score_inputs = artifact["score_inputs"]
        full = score_inputs["full_logprob"]
        record = {
            "sample_id": str(sample_id),
            "token_text": tokenizer.convert_ids_to_tokens(
                artifact["token_ids"][artifact["response_start"] :].tolist()
            ),
            "evidence_effect": (full - score_inputs["no_evidence_logprob"]).numpy(),
            "response_effect": (full - score_inputs["no_response_logprob"]).numpy(),
            "source_token_text": [
                f"{index}:"
                + tokenizer.convert_ids_to_tokens(int(artifact["token_ids"][index]))
                for index in shown
            ],
            "source_flow": source_flow[:, shown].T,
        }
        plot_sample_dashboard(
            record,
            {name: value.numpy() for name, value in layers.items()},
            Path(output),
        )
        return {"sample_id": str(sample_id), "output": str(output)}
    raise ValueError(f"sample {sample_id} was not found in the saved traces")
