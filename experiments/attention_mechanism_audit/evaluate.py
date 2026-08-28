"""Label-free evaluation of the fixed grounding-control audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audit import AUDIT_SCHEMA, MARGIN_DEFINITION, AuditArtifact, load_artifact


EVALUATION_SCHEMA = "grounding-control-chain-evaluation"


def source_mean_bootstrap(
    value: np.ndarray,
    source_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Average within source, then bootstrap equally weighted sources."""

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    value = np.asarray(value, dtype=np.float64)
    source_id = np.asarray(source_id).astype(str)
    finite = np.isfinite(value)
    value = value[finite]
    source_id = source_id[finite]
    if len(value) == 0:
        return {
            "available": False,
            "samples": 0,
            "sources": 0,
            "source_equal_mean": None,
            "ci_low": None,
            "ci_high": None,
        }

    source_mean = np.asarray(
        [value[source_id == source].mean() for source in np.unique(source_id)],
        dtype=np.float64,
    )
    random = np.random.default_rng(seed)
    draw = random.integers(0, len(source_mean), size=(replicates, len(source_mean)))
    bootstrapped = source_mean[draw].mean(axis=1)
    low, high = np.quantile(bootstrapped, (0.025, 0.975))
    return {
        "available": True,
        "samples": len(value),
        "sources": len(source_mean),
        "source_equal_mean": float(source_mean.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _summary(
    value: np.ndarray,
    artifact: AuditArtifact,
    *,
    replicates: int,
    seed: int,
    selected: np.ndarray | None = None,
) -> dict[str, object]:
    if selected is None:
        selected = np.ones(len(artifact.sample_id), dtype=np.bool_)
    selected = np.asarray(selected, dtype=np.bool_)
    return source_mean_bootstrap(
        np.asarray(value)[selected],
        artifact.source_id[selected],
        replicates=replicates,
        seed=seed,
    )


def evaluate(
    artifact: AuditArtifact,
    *,
    bootstrap_replicates: int = 1_000,
    seed: int = 20260828,
) -> dict[str, object]:
    """Evaluate the three pre-registered failure mechanisms without labels."""

    artifact = artifact.validate()
    select_success = (artifact.relevant_gain > 0.0) & (
        artifact.select_contrast > 0.0
    )
    self_lock = (artifact.history_prior_support > 0.0) & (
        artifact.history_evidence_relay <= 0.0
    )
    capture_failure = artifact.prior_capture > 0.0

    return {
        "schema": EVALUATION_SCHEMA,
        "audit_schema": AUDIT_SCHEMA,
        "margin_definition": MARGIN_DEFINITION,
        "samples": len(artifact.sample_id),
        "sources": len(np.unique(artifact.source_id.astype(str))),
        "labels_used": False,
        "source_aggregation": "mean_within_source_then_equal_weight_across_sources",
        "bootstrap_replicates": int(bootstrap_replicates),
        "mechanisms": {
            "select": {
                "intervention": "total_source_path_ablation",
                "relevant_gain_definition": (
                    "margin_counter_context-margin_no_relevant"
                ),
                "select_contrast_definition": (
                    "margin_no_irrelevant-margin_no_relevant"
                ),
                "success_definition": "relevant_gain>0 and select_contrast>0",
                "relevant_gain": _summary(
                    artifact.relevant_gain,
                    artifact,
                    replicates=bootstrap_replicates,
                    seed=seed,
                ),
                "select_contrast": _summary(
                    artifact.select_contrast,
                    artifact,
                    replicates=bootstrap_replicates,
                    seed=seed + 1,
                ),
                "success_rate": _summary(
                    select_success.astype(np.float64),
                    artifact,
                    replicates=bootstrap_replicates,
                    seed=seed + 2,
                ),
            },
            "relay": {
                "history_prior_support": _summary(
                    artifact.history_prior_support,
                    artifact,
                    replicates=bootstrap_replicates,
                    seed=seed + 3,
                    selected=select_success,
                ),
                "history_evidence_relay": _summary(
                    artifact.history_evidence_relay,
                    artifact,
                    replicates=bootstrap_replicates,
                    seed=seed + 4,
                    selected=select_success,
                ),
                "self_lock_definition": (
                    "history_prior_support>0 and history_evidence_relay<=0"
                ),
                "effective_domain": "select_success",
                "eligible_samples": int(select_success.sum()),
                "eligible_sources": int(
                    len(np.unique(artifact.source_id[select_success].astype(str)))
                ),
                "self_lock_rate": _summary(
                    self_lock.astype(np.float64),
                    artifact,
                    replicates=bootstrap_replicates,
                    seed=seed + 5,
                    selected=select_success,
                ),
            },
            "override": {
                "eligibility": "question-only prior identifiable and select_success",
                "eligible_samples": int(select_success.sum()),
                "eligible_sources": int(
                    len(np.unique(artifact.source_id[select_success].astype(str)))
                ),
                "question_prior_strength": _summary(
                    artifact.question_prior_strength,
                    artifact,
                    replicates=bootstrap_replicates,
                    seed=seed + 6,
                    selected=select_success,
                ),
                "prior_capture": _summary(
                    artifact.prior_capture,
                    artifact,
                    replicates=bootstrap_replicates,
                    seed=seed + 7,
                    selected=select_success,
                ),
                "capture_failure_definition": (
                    "prior_capture>0 inside the select_success domain"
                ),
                "capture_failure_rate": _summary(
                    capture_failure.astype(np.float64),
                    artifact,
                    replicates=bootstrap_replicates,
                    seed=seed + 8,
                    selected=select_success,
                ),
            },
        },
    }


def evaluate_artifact(
    artifact_path: str | Path,
    output_path: str | Path,
    *,
    bootstrap_replicates: int = 1_000,
    seed: int = 20260828,
) -> dict[str, object]:
    report = evaluate(
        load_artifact(artifact_path),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
