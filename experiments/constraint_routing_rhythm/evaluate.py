"""Post-hoc evaluation of the frozen constraint-deficit score.

This is the only module in the project that opens hallucination labels.  Route
construction and intervention artifacts contain no labels and are loaded in
full before the evaluation dataset is opened.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from research_dataset import open_research_dataset

from .artifacts import load_result
from .data import TASK_TYPES, canonical_task_type

MANIFEST_NAME = "run_manifest.json"
PRIMARY_SCORE = "constraint_deficit"
MIN_BOOTSTRAP_SUCCESS_RATE = 0.9
RESULT_FIELDS = {
    "sample_id",
    "source_id",
    "task_type",
    "query_position",
    "prediction_position",
    PRIMARY_SCORE,
    "baseline_margin",
    "baseline_target_logprob",
    "baseline_entropy",
    "functional_reach",
    "relay_capacity",
    "evidence_tokens",
    "model_id",
    "generator_model",
    "observer_model",
    "target_token_id",
    "valid",
    "control_audited",
    "matched_control_available",
    "relay_audited",
    "direct_response_cut_delta",
    "matched_non_evidence_cut_delta",
    "upstream_cut_delta",
    "downstream_cut_delta",
    "joint_cut_delta",
    "relay_interaction",
}


def metric_arrays(label, score) -> tuple[np.ndarray, np.ndarray]:
    label = np.asarray(label)
    score = np.asarray(score, dtype=np.float64)
    if label.ndim != 1 or score.shape != label.shape:
        raise ValueError("label and score must be aligned vectors")
    if not np.isin(label, (0, 1)).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(score).all():
        raise ValueError("metrics require finite scores")
    return label.astype(bool), score


def average_ranks(value: np.ndarray) -> np.ndarray:
    """Return one-based ranks with the mean rank assigned to ties."""

    order = np.argsort(value, kind="mergesort")
    sorted_value = value[order]
    ranks = np.empty(len(value), dtype=np.float64)
    starts = np.append(0, np.flatnonzero(sorted_value[:-1] != sorted_value[1:]) + 1)
    stops = np.append(starts[1:], len(value))
    tied_rank = 0.5 * (starts + stops - 1) + 1.0
    ranks[order] = np.repeat(tied_rank, stops - starts)
    return ranks


def binary_auroc(label, score) -> float | None:
    """Compute tie-aware binary AUROC without fitting a detector."""

    label, score = metric_arrays(label, score)
    if np.unique(label).size < 2:
        return None
    return float(roc_auc_score(label, score))


def average_precision(label, score) -> float | None:
    """Compute sklearn-equivalent AP, treating tied scores as one threshold."""

    label, score = metric_arrays(label, score)
    if np.unique(label).size < 2:
        return None
    return float(average_precision_score(label, score))


def metric_summary(
    label: np.ndarray, score: np.ndarray
) -> dict[str, float | int | None]:
    valid = np.isfinite(score)
    selected_label = label[valid]
    selected_score = score[valid]
    return {
        "tokens": int(valid.sum()),
        "coverage": float(valid.mean()) if len(valid) else 0.0,
        "auroc": binary_auroc(selected_label, selected_score),
        "average_precision": average_precision(selected_label, selected_score),
    }


def bootstrap_metrics(
    label,
    score,
    source_id,
    *,
    repeats: int = 400,
    seed: int = 2026,
) -> dict[str, object]:
    """Resample complete source clusters, keeping all samples and tokens.

    This preserves dependence among repeated generations and response tokens
    attached to one RAGTruth source.
    """

    label, score = metric_arrays(label, score)
    source_id = np.asarray(source_id).astype(str)
    if source_id.shape != label.shape:
        raise ValueError("bootstrap source IDs must align with label and score")
    if repeats < 0:
        raise ValueError("bootstrap repeats must be nonnegative")
    if not len(label) or repeats == 0:
        return {
            "bootstrap_replicates": 0,
            "bootstrap_reliable": False,
            "auroc_ci95": [None, None],
            "average_precision_ci95": [None, None],
        }

    sources = np.unique(source_id)
    if len(sources) < 2:
        return {
            "bootstrap_replicates": 0,
            "bootstrap_reliable": False,
            "auroc_ci95": [None, None],
            "average_precision_ci95": [None, None],
        }
    rows = {source: np.flatnonzero(source_id == source) for source in sources}
    random = np.random.default_rng(seed)
    estimates: list[tuple[float, float]] = []
    for _ in range(repeats):
        chosen_sources = random.choice(sources, len(sources), replace=True)
        index = np.concatenate([rows[source] for source in chosen_sources])
        auroc = binary_auroc(label[index], score[index])
        ap = average_precision(label[index], score[index])
        if auroc is not None and ap is not None:
            estimates.append((auroc, ap))

    enough = len(estimates) >= np.ceil(MIN_BOOTSTRAP_SUCCESS_RATE * repeats)
    if not enough:
        return {
            "bootstrap_replicates": len(estimates),
            "bootstrap_reliable": False,
            "auroc_ci95": [None, None],
            "average_precision_ci95": [None, None],
        }
    interval = np.quantile(np.asarray(estimates), (0.025, 0.975), axis=0)
    return {
        "bootstrap_replicates": len(estimates),
        "bootstrap_reliable": True,
        "auroc_ci95": interval[:, 0].tolist(),
        "average_precision_ci95": interval[:, 1].tolist(),
    }


def correlation(first: np.ndarray, second: np.ndarray) -> dict[str, float | None]:
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(first, second)[0, 1]),
        "spearman": float(
            np.corrcoef(average_ranks(first), average_ranks(second))[0, 1]
        ),
    }


def scalar_text(result: Mapping[str, np.ndarray], name: str) -> str:
    value = result[name]
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return str(value.item())


def finite_summary(
    values: np.ndarray,
    sample_id: np.ndarray,
    source_id: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Summarize finite audit values without treating missing reruns as zero."""

    values = np.asarray(values, dtype=np.float64)
    selected = np.isfinite(values)
    if mask is not None:
        selected &= np.asarray(mask, dtype=bool)
    finite = values[selected]
    if not len(finite):
        return {
            "tokens": 0,
            "samples": 0,
            "sources": 0,
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
        }
    return {
        "tokens": len(finite),
        "samples": int(np.unique(sample_id[selected]).size),
        "sources": int(np.unique(source_id[selected]).size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
    }


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def model_identity_matches(intervention: str, observed: str) -> bool:
    """Compare recorded IDs without requiring both sides to store full paths."""

    return intervention == observed or Path(intervention).name == Path(observed).name


def load_manifest(
    result_root: str | Path, dataset_root: str | Path
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Load one completed run and resolve only its explicitly listed results."""

    root = Path(result_root)
    output = root.parent
    path = output / MANIFEST_NAME
    if not path.is_file():
        raise ValueError(f"run manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("analysis_complete") is not True:
        raise ValueError("analysis is incomplete; resume it before evaluation")
    config = manifest["config"]
    if Path(config["dataset_root"]).resolve() != Path(dataset_root).resolve():
        raise ValueError("evaluation dataset differs from the run manifest")

    entries = list(manifest["samples"])
    if not entries or any(entry.get("complete") is not True for entry in entries):
        raise ValueError("run manifest has no complete sample set")
    max_events = config["max_events"]
    for entry in entries:
        full_events = int(entry["full_response_events"])
        intended_events = min(full_events, max_events or full_events)
        if int(entry["events"]) != intended_events:
            raise ValueError("result does not cover its intended response scope")
    names = [str(entry["result"]) for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("run manifest lists a result more than once")
    selected = {task: 0 for task in TASK_TYPES}
    for entry in entries:
        selected[canonical_task_type(entry["task_type"])] += 1
    if selected != manifest.get("selected_samples"):
        raise ValueError("run manifest selected-sample counts are inconsistent")
    for task in TASK_TYPES:
        task_entries = [
            entry
            for entry in entries
            if canonical_task_type(entry["task_type"]) == task
        ]
        audited = [entry for entry in task_entries if entry["audit_requested"]]
        audit_sources = manifest["audit_source_ids"][task]
        available_sources = {str(entry["source_id"]) for entry in task_entries}
        intended_audits = min(int(config["audit_limit"]), len(available_sources))
        if (
            len(audit_sources) != intended_audits
            or len(audit_sources) != len(set(audit_sources))
            or len(audited) != len(audit_sources)
            or not set(audit_sources) <= available_sources
        ):
            raise ValueError("run manifest audit source scope is inconsistent")
        if [str(entry["sample_id"]) for entry in audited] != manifest[
            "audit_sample_ids"
        ][task]:
            raise ValueError("run manifest audit sample IDs are inconsistent")
        if {str(entry["source_id"]) for entry in audited} != set(audit_sources):
            raise ValueError("run manifest audit source IDs are inconsistent")
    expected = {output / name for name in names}
    actual = {
        result
        for result in root.rglob("*.npz")
        if not (result.name.startswith(".") and result.name.endswith(".tmp.npz"))
    }
    if actual != expected:
        raise ValueError("result files do not exactly match the run manifest")
    return manifest, entries


def load_results(
    output_root: Path, entries: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Load the complete label-free result set before labels are opened."""

    rows = []
    seen = set()
    for entry in entries:
        path = output_root / str(entry["result"])
        result = load_result(path)
        missing = RESULT_FIELDS - result.keys()
        if missing:
            raise ValueError(f"{path.name} is missing: {', '.join(sorted(missing))}")
        sample_id = scalar_text(result, "sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate sample result: {sample_id}")
        seen.add(sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "source_id": scalar_text(result, "source_id"),
                "task_type": canonical_task_type(scalar_text(result, "task_type")),
                "model_id": scalar_text(result, "model_id"),
                "generator_model": scalar_text(result, "generator_model"),
                "observer_model": scalar_text(result, "observer_model"),
                "query_position": result["query_position"].astype(np.int64),
                "prediction_position": result["prediction_position"].astype(np.int64),
                "target_token_id": result["target_token_id"].astype(np.int64),
                PRIMARY_SCORE: result[PRIMARY_SCORE].astype(np.float64),
                "baseline_margin": result["baseline_margin"].astype(np.float64),
                "baseline_target_logprob": result["baseline_target_logprob"].astype(
                    np.float64
                ),
                "baseline_entropy": result["baseline_entropy"].astype(np.float64),
                "functional_reach": result["functional_reach"].astype(np.float64),
                "relay_capacity": result["relay_capacity"].astype(np.float64),
                "valid": result["valid"].astype(bool),
                "control_audited": bool(result["control_audited"].item()),
                "matched_control_available": bool(
                    result["matched_control_available"].item()
                ),
                "relay_audited": bool(result["relay_audited"].item()),
                "direct_response_cut_delta": result["direct_response_cut_delta"].astype(
                    np.float64
                ),
                "matched_non_evidence_cut_delta": result[
                    "matched_non_evidence_cut_delta"
                ].astype(np.float64),
                "upstream_cut_delta": result["upstream_cut_delta"].astype(np.float64),
                "downstream_cut_delta": result["downstream_cut_delta"].astype(
                    np.float64
                ),
                "joint_cut_delta": result["joint_cut_delta"].astype(np.float64),
                "relay_interaction": result["relay_interaction"].astype(np.float64),
                "evidence_tokens": int(result["evidence_tokens"].item()),
                "full_response_events": int(entry["full_response_events"]),
            }
        )
        row = rows[-1]
        if (
            sample_id != str(entry["sample_id"])
            or row["source_id"] != str(entry["source_id"])
            or row["task_type"] != canonical_task_type(entry["task_type"])
            or row["generator_model"] != str(entry["generator_model"])
            or row["observer_model"] != str(entry["observer_model"])
            or row["control_audited"] != bool(entry["audit_requested"])
            or row["evidence_tokens"] != len(entry["evidence_positions"])
            or len(row["prediction_position"]) != int(entry["events"])
        ):
            raise ValueError(f"result disagrees with run manifest: {sample_id}")
    return rows


def task_report(
    task_type: str,
    rows: list[dict[str, object]],
    *,
    bootstrap: int,
    seed: int,
    intended_scope: Mapping[str, object],
) -> dict[str, object]:
    label = np.concatenate([row["label"] for row in rows])
    score = np.concatenate([row[PRIMARY_SCORE] for row in rows])
    margin = np.concatenate([row["baseline_margin"] for row in rows])
    target_logprob = np.concatenate([row["baseline_target_logprob"] for row in rows])
    entropy = np.concatenate([row["baseline_entropy"] for row in rows])
    functional_reach = np.concatenate([row["functional_reach"] for row in rows])
    relay_capacity = np.concatenate([row["relay_capacity"] for row in rows])
    absolute_position = np.concatenate([row["absolute_position"] for row in rows])
    position = np.concatenate([row["relative_position"] for row in rows])
    response_length = np.concatenate([row["response_length"] for row in rows])
    evidence_tokens = np.concatenate([row["evidence_tokens"] for row in rows])
    source_id = np.concatenate([row["source_id"] for row in rows])
    sample_id = np.concatenate([row["sample_id"] for row in rows])
    declared_valid = np.concatenate([row["valid"] for row in rows])
    valid = declared_valid & np.isfinite(score)
    full_response_tokens = sum(int(row["full_response_events"]) for row in rows)
    complete_response_samples = sum(
        len(row[PRIMARY_SCORE]) == int(row["full_response_events"]) for row in rows
    )

    primary = metric_summary(label[valid], score[valid])
    primary["tokens"] = int(valid.sum())
    primary["coverage"] = float(valid.mean()) if len(valid) else 0.0
    if valid.any() and bootstrap:
        primary.update(
            bootstrap_metrics(
                label[valid],
                score[valid],
                source_id[valid],
                repeats=bootstrap,
                seed=seed,
            )
        )
    else:
        primary.update(
            bootstrap_replicates=0,
            bootstrap_reliable=False,
            auroc_ci95=[None, None],
            average_precision_ci95=[None, None],
        )

    negative_margin = -margin
    negative_target_logprob = -target_logprob
    control_sample = np.asarray(
        [bool(row["control_audited"]) for row in rows], dtype=bool
    )
    matched_sample = control_sample & np.asarray(
        [bool(row["matched_control_available"]) for row in rows], dtype=bool
    )
    relay_sample = np.asarray([bool(row["relay_audited"]) for row in rows], dtype=bool)
    control_token = np.concatenate(
        [
            np.repeat(control_sample[index], len(row[PRIMARY_SCORE]))
            for index, row in enumerate(rows)
        ]
    )
    matched_token = np.concatenate(
        [
            np.repeat(matched_sample[index], len(row[PRIMARY_SCORE]))
            for index, row in enumerate(rows)
        ]
    )
    relay_token = np.concatenate(
        [
            np.repeat(relay_sample[index], len(row[PRIMARY_SCORE]))
            for index, row in enumerate(rows)
        ]
    )
    direct = np.concatenate([row["direct_response_cut_delta"] for row in rows])
    matched = np.concatenate([row["matched_non_evidence_cut_delta"] for row in rows])
    upstream = np.concatenate([row["upstream_cut_delta"] for row in rows])
    downstream = np.concatenate([row["downstream_cut_delta"] for row in rows])
    joint = np.concatenate([row["joint_cut_delta"] for row in rows])
    interaction = np.concatenate([row["relay_interaction"] for row in rows])
    matched_pair = matched_token & valid & np.isfinite(matched)
    direct_pair = control_token & valid & np.isfinite(direct)
    relay_complete = (
        relay_token
        & valid
        & np.isfinite(upstream)
        & np.isfinite(downstream)
        & np.isfinite(joint)
        & np.isfinite(interaction)
    )
    if relay_complete.any() and not np.allclose(
        interaction[relay_complete],
        joint[relay_complete] - upstream[relay_complete] - downstream[relay_complete],
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("relay interaction disagrees with its four-cell contrasts")
    scheduled_audits = len(
        intended_scope.get("audit_sample_ids", {}).get(task_type, [])
    )
    generators = [str(row["generator_model"]) for row in rows]
    observers = [str(row["observer_model"]) for row in rows]
    known_generators = [identity for identity in generators if identity]
    known_observers = [identity for identity in observers if identity]
    route_valid = valid & np.isfinite(relay_capacity)
    route_quantile = float(intended_scope.get("carrier_quantile", 0.75))
    route_threshold = (
        float(np.quantile(relay_capacity[route_valid], route_quantile))
        if route_valid.any()
        else None
    )
    high_route = (
        route_valid & (relay_capacity >= route_threshold)
        if route_threshold is not None
        else np.zeros_like(valid)
    )
    return {
        "task_type": task_type,
        "model_id": str(rows[0]["model_id"]),
        "model_roles": {
            "intervention_model": str(rows[0]["model_id"]),
            "response_generator_models": sorted(set(known_generators)),
            "observer_models": sorted(set(known_observers)),
            "response_generator_known_samples": len(known_generators),
            "observer_known_samples": len(known_observers),
            "intervention_matches_all_response_generators": (
                all(
                    model_identity_matches(str(rows[0]["model_id"]), identity)
                    for identity in known_generators
                )
                if len(known_generators) == len(rows)
                else None
            ),
            "intervention_matches_all_observers": (
                all(
                    model_identity_matches(str(rows[0]["model_id"]), identity)
                    for identity in known_observers
                )
                if len(known_observers) == len(rows)
                else None
            ),
        },
        "samples": len(rows),
        "sources": int(np.unique(source_id).size),
        "tokens": len(label),
        "hallucinated_tokens": int(label.sum()),
        "prevalence": float(label.mean()) if len(label) else None,
        "valid_tokens": int(valid.sum()),
        "valid_hallucinated_tokens": int(label[valid].sum()),
        "valid_coverage": float(valid.mean()) if len(valid) else 0.0,
        "full_response_tokens": full_response_tokens,
        "analyzed_response_coverage": (
            len(label) / full_response_tokens if full_response_tokens else 0.0
        ),
        "complete_response_samples": complete_response_samples,
        "response_scope": (
            "full" if complete_response_samples == len(rows) else "prefix"
        ),
        "intended_scope": dict(intended_scope),
        "primary_score": PRIMARY_SCORE,
        "score_direction": (
            "higher means deleting evidence Value messages preserved or improved "
            "the fixed target-versus-runner margin"
        ),
        PRIMARY_SCORE: primary,
        "controls": {
            "absolute_response_position": metric_summary(
                label[valid], absolute_position[valid]
            ),
            "relative_response_position": metric_summary(label[valid], position[valid]),
            "response_length": metric_summary(label[valid], response_length[valid]),
            "evidence_tokens": metric_summary(label[valid], evidence_tokens[valid]),
            "negative_baseline_margin": metric_summary(
                label[valid], negative_margin[valid]
            ),
            "negative_baseline_target_logprob": metric_summary(
                label[valid], negative_target_logprob[valid]
            ),
            "baseline_entropy": metric_summary(label[valid], entropy[valid]),
        },
        "diagnostic_correlation_with_primary": {
            "functional_reach": correlation(score[valid], functional_reach[valid]),
            "relay_capacity": correlation(score[valid], relay_capacity[valid]),
            "absolute_response_position": correlation(
                score[valid], absolute_position[valid]
            ),
            "relative_response_position": correlation(score[valid], position[valid]),
            "response_length": correlation(score[valid], response_length[valid]),
            "evidence_tokens": correlation(score[valid], evidence_tokens[valid]),
            "negative_baseline_margin": correlation(
                score[valid], negative_margin[valid]
            ),
            "negative_baseline_target_logprob": correlation(
                score[valid], negative_target_logprob[valid]
            ),
            "baseline_entropy": correlation(score[valid], entropy[valid]),
        },
        "route_control_dissociation": {
            "definition": (
                "relay_capacity proposes an ordered route; constraint_deficit "
                "measures the total evidence-source cut. They are not combined."
            ),
            "finite_route_tokens": int(route_valid.sum()),
            "high_route_quantile": route_quantile,
            "high_route_threshold": route_threshold,
            "high_route_tokens": int(high_route.sum()),
            "weak_control_fraction_within_high_route": (
                float(np.mean(score[high_route] >= 0)) if high_route.any() else None
            ),
            "primary_within_high_route": metric_summary(
                label[high_route], score[high_route]
            ),
            "relay_vs_evidence_support": correlation(
                relay_capacity[route_valid], -score[route_valid]
            ),
        },
        "negative_baseline_margin_definition": "-baseline_margin from the artifact",
        "negative_baseline_target_logprob_definition": (
            "-baseline_target_logprob from the artifact"
        ),
        "bootstrap": {
            "unit": "source cluster; all samples and tokens retained",
            "requested_replicates": bootstrap,
            "successful_replicates": primary["bootstrap_replicates"],
            "seed": seed,
        },
        "audit_diagnostics": {
            "coverage": {
                "scheduled_samples": scheduled_audits,
                "control_audited_samples": int(control_sample.sum()),
                "control_sample_fraction_of_task": fraction(
                    int(control_sample.sum()), len(rows)
                ),
                "control_completion_fraction_of_scheduled": fraction(
                    int(control_sample.sum()), scheduled_audits
                ),
                "control_audited_tokens": int(control_token.sum()),
                "control_token_fraction_of_task": fraction(
                    int(control_token.sum()), len(label)
                ),
                "matched_available_samples": int(matched_sample.sum()),
                "matched_sample_fraction_of_control_audits": fraction(
                    int(matched_sample.sum()), int(control_sample.sum())
                ),
                "matched_opportunity_tokens": int(matched_token.sum()),
                "relay_audited_samples": int(relay_sample.sum()),
                "relay_sample_fraction_of_control_audits": fraction(
                    int(relay_sample.sum()), int(control_sample.sum())
                ),
                "relay_opportunity_tokens": int(relay_token.sum()),
            },
            "evidence_vs_matched_non_evidence": {
                "definition": (
                    "constraint_deficit - matched_non_evidence_cut_delta on "
                    "the same finite audited tokens"
                ),
                "finite_fraction_of_opportunity": fraction(
                    int(matched_pair.sum()), int(matched_token.sum())
                ),
                "total_evidence_cut_delta": finite_summary(
                    score, sample_id, source_id, matched_pair
                ),
                "matched_non_evidence_cut_delta": finite_summary(
                    matched, sample_id, source_id, matched_pair
                ),
                "paired_contrast": finite_summary(
                    score - matched, sample_id, source_id, matched_pair
                ),
            },
            "total_vs_direct_response": {
                "definition": (
                    "constraint_deficit - direct_response_cut_delta; this paired "
                    "nonlinear contrast is not an indirect effect"
                ),
                "finite_fraction_of_opportunity": fraction(
                    int(direct_pair.sum()), int(control_token.sum())
                ),
                "total_evidence_cut_delta": finite_summary(
                    score, sample_id, source_id, direct_pair
                ),
                "direct_response_cut_delta": finite_summary(
                    direct, sample_id, source_id, direct_pair
                ),
                "paired_contrast": finite_summary(
                    score - direct, sample_id, source_id, direct_pair
                ),
            },
            "relay": {
                "definition": (
                    "all cut deltas are intervention minus baseline; interaction "
                    "is joint - upstream - downstream"
                ),
                "complete_case_fraction_of_opportunity": fraction(
                    int(relay_complete.sum()), int(relay_token.sum())
                ),
                "upstream_cut_delta": finite_summary(
                    upstream, sample_id, source_id, relay_complete
                ),
                "downstream_cut_delta": finite_summary(
                    downstream, sample_id, source_id, relay_complete
                ),
                "joint_cut_delta": finite_summary(
                    joint, sample_id, source_id, relay_complete
                ),
                "relay_interaction": finite_summary(
                    interaction, sample_id, source_id, relay_complete
                ),
            },
        },
        "labels_used_during": "post-hoc evaluation only",
    }


def save_json(path: str | Path, value: Mapping[str, object]) -> None:
    """Write a human-readable report for the foreground CLI."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def evaluate_results(
    result_root: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    bootstrap: int = 400,
    seed: int = 2026,
) -> dict[str, dict[str, object]]:
    """Join frozen token scores to labels and report each task separately."""

    manifest, entries = load_manifest(result_root, dataset_root)
    results = load_results(Path(result_root).parent, entries)
    expected_model = str(manifest["config"]["model_id"])
    if any(str(result["model_id"]) != expected_model for result in results):
        raise ValueError("result model differs from the run manifest")
    sample_ids = [str(row["sample_id"]) for row in results]

    # Opening this dataset with retained labels is intentionally delayed until
    # every score artifact and score direction has been fixed above.
    dataset = open_research_dataset(
        dataset_root,
        device="cpu",
        retain_embedded_labels=True,
    )
    missing = [sample_id for sample_id in sample_ids if sample_id not in dataset]
    if missing:
        raise ValueError(f"dataset has no sample: {missing[0]}")
    # Canonical datasets open their sidecar here. Formal caches populate their
    # retained label one sample at a time below, avoiding a second large .pt load.
    label_store = dataset.prepare_evaluation_labels([])

    by_task: dict[str, list[dict[str, object]]] = {task: [] for task in TASK_TYPES}
    for result in results:
        sample_id = str(result["sample_id"])
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            label = np.asarray(label_store.response_labels(sample), dtype=np.int8)
            response_start = int(attention.response_idx)
            prediction = np.asarray(result["prediction_position"])
            expected = response_start + np.arange(len(prediction))
            if (
                len(label) != int(result["full_response_events"])
                or len(prediction) > len(label)
                or not np.array_equal(prediction, expected)
            ):
                raise ValueError(
                    f"result is not a contiguous response prefix: {sample_id}"
                )
            token_ids = np.asarray(attention.token_ids.detach().cpu(), dtype=np.int64)
            if not np.array_equal(result["target_token_id"], token_ids[prediction]):
                raise ValueError(f"response tokens changed for {sample_id}")
            full_response_events = len(label)
            label = label[: len(prediction)]
            if str(sample.source_id) != str(result["source_id"]):
                raise ValueError(f"source mismatch for {sample_id}")
            task = canonical_task_type(sample.task_type)
            if task != result["task_type"]:
                raise ValueError(f"task mismatch for {sample_id}")
            generator_model = getattr(sample, "generator_model", None)
            generator_model = "" if generator_model is None else str(generator_model)
            sample_observer_model = getattr(sample, "observer_model", None)
            sample_observer_model = (
                "" if sample_observer_model is None else str(sample_observer_model)
            )
            model_path = getattr(dataset, "spec", {}).get("model_path")
            observer_model = str(model_path) if model_path else sample_observer_model
            if (
                generator_model != result["generator_model"]
                or observer_model != result["observer_model"]
            ):
                raise ValueError(f"model-role metadata changed for {sample_id}")

            count = len(label)
            absolute_position = prediction.astype(np.float64)
            relative_position = np.arange(count, dtype=np.float64)
            relative_position /= max(full_response_events - 1, 1)
            by_task[task].append(
                {
                    "label": label,
                    PRIMARY_SCORE: np.asarray(result[PRIMARY_SCORE]),
                    "baseline_margin": np.asarray(result["baseline_margin"]),
                    "baseline_target_logprob": np.asarray(
                        result["baseline_target_logprob"]
                    ),
                    "baseline_entropy": np.asarray(result["baseline_entropy"]),
                    "functional_reach": np.asarray(result["functional_reach"]),
                    "relay_capacity": np.asarray(result["relay_capacity"]),
                    "valid": np.asarray(result["valid"]),
                    "absolute_position": absolute_position,
                    "relative_position": relative_position,
                    "response_length": np.repeat(float(full_response_events), count),
                    "evidence_tokens": np.repeat(
                        float(result["evidence_tokens"]), count
                    ),
                    "source_id": np.repeat(str(result["source_id"]), count),
                    "sample_id": np.repeat(str(result["sample_id"]), count),
                    "model_id": str(result["model_id"]),
                    "generator_model": str(result["generator_model"]),
                    "observer_model": str(result["observer_model"]),
                    "control_audited": bool(result["control_audited"]),
                    "matched_control_available": bool(
                        result["matched_control_available"]
                    ),
                    "relay_audited": bool(result["relay_audited"]),
                    "direct_response_cut_delta": np.asarray(
                        result["direct_response_cut_delta"]
                    ),
                    "matched_non_evidence_cut_delta": np.asarray(
                        result["matched_non_evidence_cut_delta"]
                    ),
                    "upstream_cut_delta": np.asarray(result["upstream_cut_delta"]),
                    "downstream_cut_delta": np.asarray(result["downstream_cut_delta"]),
                    "joint_cut_delta": np.asarray(result["joint_cut_delta"]),
                    "relay_interaction": np.asarray(result["relay_interaction"]),
                    "full_response_events": full_response_events,
                }
            )
        finally:
            sample.release_attention()

    reports = {}
    output_root = Path(output_root)
    for number, task in enumerate(TASK_TYPES):
        if not by_task[task]:
            continue
        report = task_report(
            task,
            by_task[task],
            bootstrap=bootstrap,
            seed=seed + number,
            intended_scope={
                **manifest["config"],
                "selected_samples": manifest["selected_samples"],
                "audit_source_ids": manifest["audit_source_ids"],
                "audit_sample_ids": manifest["audit_sample_ids"],
            },
        )
        save_json(output_root / task.casefold() / "report.json", report)
        reports[task] = report
    return reports
