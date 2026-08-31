"""Pool saved QA traces and evaluate four fixed label-free scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from research_dataset import open_research_dataset

from .audit import load_index
from .capture import HISTORY, SELF
from .data import EVIDENCE
from .visualize import plot_population, plot_sample_dashboard


SCORE_ORDER = (
    "causal_route_capture",
    "routing_imbalance",
    "source_dispersion",
    "message_independent_preference",
)

SCORE_DEFINITIONS = {
    "causal_route_capture": (
        "logp(no evidence messages) - logp(no response messages)"
    ),
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


def _binary_metrics(label: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    prevalence = float(label.mean())
    auprc = float(average_precision_score(label, score))
    return {
        "tokens": int(len(label)),
        "positive_tokens": int(label.sum()),
        "prevalence": prevalence,
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
    edge = trace["role_edge_magnitude"].float()
    role_mass = edge.sum(2)
    total = role_mass.sum(-1).clamp_min(1e-12)
    response_token_exists = (torch.arange(edge.shape[1]) > 0)[None]
    evidence_share = role_mass[..., EVIDENCE] / total
    response_share = (
        role_mass[..., HISTORY]
        + role_mass[..., SELF] * response_token_exists.to(role_mass.dtype)
    ) / total

    active_sources = (trace["source_role"] >= 0).sum(-1).float().clamp_min(2)
    dispersion = (
        trace["source_message_entropy"].float()
        / active_sources.log()[None]
    )
    return {
        "routing_imbalance": response_share - evidence_share,
        "source_dispersion": dispersion,
        "evidence_share": evidence_share,
        "response_share": response_share,
    }


def token_scores(
    artifact: dict,
    layers: dict[str, torch.Tensor] | None = None,
) -> dict[str, np.ndarray]:
    """Compute the fixed primary score and its three mechanism components."""

    layers = layers or layer_mechanisms(artifact)
    evidence_effect = artifact["mechanism"]["evidence_message_effect"].float()
    response_effect = artifact["mechanism"]["response_message_effect"].float()
    values = {
        "causal_route_capture": response_effect - evidence_effect,
        "routing_imbalance": layers["routing_imbalance"].mean(0),
        "source_dispersion": layers["source_dispersion"].mean(0),
        "message_independent_preference": artifact["mechanism"][
            "evidence_response_removed_margin"
        ].float(),
    }
    return {name: values[name].cpu().numpy() for name in SCORE_ORDER}


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
            result.update(
                {"auroc_ci95": [None, None], "auprc_ci95": [None, None]}
            )
        results[name] = result
    return results


def build_report(
    *,
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
    means = {
        name: {
            "correct": float(scores[name][~label].mean()) if (~label).any() else None,
            "hallucinated": float(scores[name][label].mean()) if label.any() else None,
        }
        for name in SCORE_ORDER[1:]
    }
    return {
        "schema": "ragtruth-three-mechanism-detection-v1",
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
        "mechanism_means": means,
        "labels_used_during": "posthoc_evaluation_only",
        "analysis_scope": (
            "fixed-score exploratory audit over every cached QA token"
        ),
    }


def _load_input(
    trace_root: Path,
    split_root: Path,
) -> dict[str, np.ndarray]:
    rows = load_index(trace_root)
    dataset = open_research_dataset(
        split_root,
        device="cpu",
        retain_embedded_labels=True,
    )
    labels = dataset.prepare_evaluation_labels([row["sample_id"] for row in rows])
    split = str(dataset.manifest.get("split") or split_root.name)
    arrays: dict[str, list[np.ndarray]] = {
        "label": [],
        "sample_id": [],
        "source_id": [],
        "split": [],
        "token_index": [],
        "response_length": [],
        **{name: [] for name in SCORE_ORDER},
    }
    for row in rows:
        artifact = torch.load(
            trace_root / "samples" / row["path"],
            map_location="cpu",
            weights_only=True,
        )
        sample = dataset[row["sample_id"]]
        token_label = labels.response_labels(sample).cpu().numpy().astype(bool)
        sample.release_attention()
        current = token_scores(artifact)
        count = len(token_label)
        if any(len(value) != count for value in current.values()):
            raise ValueError(
                f"trace/label length mismatch for sample {row['sample_id']}"
            )
        arrays["label"].append(token_label)
        arrays["sample_id"].append(np.repeat(str(row["sample_id"]), count))
        arrays["source_id"].append(np.repeat(str(row["source_id"]), count))
        arrays["split"].append(np.repeat(split, count))
        arrays["token_index"].append(np.arange(count, dtype=np.int32))
        arrays["response_length"].append(np.full(count, count, dtype=np.int32))
        for name in SCORE_ORDER:
            arrays[name].append(current[name])
    return {
        name: np.concatenate(value)
        for name, value in arrays.items()
    }


def evaluate_all(
    *,
    inputs: Iterable[tuple[str | Path, str | Path]],
    output: str | Path,
    bootstrap: int = 1000,
    seed: int = 20260828,
) -> dict:
    """Pool physical cache shards first, then evaluate exactly once."""

    pieces, manifests = [], []
    for trace_root, split_root in inputs:
        trace_root = Path(trace_root)
        pieces.append(_load_input(trace_root, Path(split_root)))
        manifests.append(
            json.loads((trace_root / "manifest.json").read_text(encoding="utf-8"))
        )
    merged = {
        name: np.concatenate([piece[name] for piece in pieces])
        for name in pieces[0]
    }
    scores = {name: merged[name] for name in SCORE_ORDER}
    report = build_report(
        label=merged["label"],
        sample_id=merged["sample_id"],
        source_id=merged["source_id"],
        scores=scores,
        bootstrap=bootstrap,
        seed=seed,
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
            "capture_complete": all(
                manifest["complete"] for manifest in manifests
            ),
        }
    )
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return report


def plot_saved_sample(
    *,
    inputs: Iterable[tuple[str | Path, str | Path]],
    sample_id: str,
    model_path: str | Path,
    output: str | Path,
) -> dict:
    """Render one saved response without replaying the model."""

    from transformers import AutoTokenizer

    for trace_root_value, split_root_value in inputs:
        trace_root = Path(trace_root_value)
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
        dataset = open_research_dataset(
            split_root_value,
            device="cpu",
            retain_embedded_labels=True,
        )
        labels = dataset.prepare_evaluation_labels([row["sample_id"]])
        sample = dataset[row["sample_id"]]
        token_label = labels.response_labels(sample).cpu().numpy().astype(bool)
        sample.release_attention()
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
        source_flow /= (
            trace["role_edge_magnitude"].sum((0, 2, 3)).numpy()[:, None]
            + 1e-12
        )
        shown = np.argsort(source_flow.sum(0))[-16:][::-1]
        mechanism = artifact["mechanism"]
        record = {
            "sample_id": str(sample_id),
            "token_text": tokenizer.convert_ids_to_tokens(
                artifact["target_ids"].tolist()
            ),
            "label": token_label,
            "evidence_effect": mechanism["evidence_message_effect"].numpy(),
            "response_effect": mechanism["response_message_effect"].numpy(),
            "source_token_text": [
                f"{index}:"
                + tokenizer.convert_ids_to_tokens(
                    int(artifact["token_ids"][index])
                )
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


__all__ = [
    "SCORE_DEFINITIONS",
    "SCORE_ORDER",
    "build_report",
    "detection_summary",
    "evaluate_all",
    "layer_mechanisms",
    "plot_saved_sample",
    "token_scores",
]
