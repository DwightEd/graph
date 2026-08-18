"""Post-hoc evaluation for frozen RR topology-dynamics features."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from experiment_protocol import FrozenEvaluation

from .artifacts import (
    EVALUATION_SCHEMA,
    load_topology_artifact,
    score_temporal_scope,
    verify_score_provenance,
)
from .features import LAYER_PROFILE_NAMES

CONVERGENCE_FEATURES = (
    "route_effective_rank",
    "route_participation_rank",
    "route_top1_energy_share",
    "cross_head_route_consensus",
    "source_effective_number",
    "source_entropy",
    "source_top1_share",
    "channel_route_velocity",
    "source_route_velocity",
    "anchor_turnover",
    "offline_route_distance_to_final",
    "offline_source_distance_to_final",
)

GROUNDING_FEATURES = (
    "direct_prompt_share",
    "prompt_groundedness",
    "grounded_rr_relay",
    "ungrounded_rr_feedback",
    "residual_grounded_source_share",
)

RESIDUAL_FEATURES = (
    "spectral_residual_energy",
    "residual_effective_channels",
    "residual_channel_entropy",
    "residual_channel_top1_share",
    "residual_channel_top5pct_share",
    "residual_weighted_lag",
    "residual_recent_lag_share",
    "residual_mid_lag_share",
    "residual_far_lag_share",
    "residual_source_effective_number",
    "residual_source_top1_share",
)

SETWALK_COORDINATION_FEATURES = (
    "local_rr_collapse_strength",
    "early_local_rr_collapse",
    "late_local_rr_collapse",
    "early_minus_late_local_rr_collapse",
    "local_rr_collapse_depth",
    "rp_rr_relation_effective_rank_mean",
    "rp_rr_head_consensus_mean",
    "rp_rr_head_specialization_mean",
    "cross_layer_relation_shift_mean",
    "cross_layer_relation_shift_max",
    "cross_layer_adjacency_gap_vs_all_pairs",
)


def _finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _binary_metrics(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(score)
    y = y[finite]
    score = score[finite]
    if len(y) == 0 or np.unique(y).size < 2:
        return None
    auc = float(roc_auc_score(y, score))
    auc_lower = float(roc_auc_score(y, -score))
    ap_higher = float(average_precision_score(y, score))
    ap_lower = float(average_precision_score(y, -score))
    normal = score[y == 0]
    positive = score[y == 1]
    normal_median = float(np.median(normal))
    positive_median = float(np.median(positive))
    normal_mad = 1.4826 * float(np.median(np.abs(normal - normal_median)))
    robust_effect = (positive_median - normal_median) / max(normal_mad, 1e-8)
    return {
        "tokens": len(y),
        "positive_tokens": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc_higher": auc,
        "auroc_lower": auc_lower,
        "orientation_free_auroc": max(auc, auc_lower),
        "direction": (
            "higher_in_hallucination" if auc >= auc_lower else "lower_in_hallucination"
        ),
        "auprc_higher": ap_higher,
        "auprc_lower": ap_lower,
        "orientation_free_auprc": max(ap_higher, ap_lower),
        "normal_median": normal_median,
        "hallucination_median": positive_median,
        "median_difference": positive_median - normal_median,
        "robust_effect_mad": robust_effect,
    }


def _bootstrap_mean_interval(values, *, replicates: int, seed: int):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    result = {"groups": len(values), "mean": float(values.mean())}
    if replicates < 1 or len(values) < 2:
        result.update({"ci_low": None, "ci_high": None})
        return result
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        estimates[replicate] = values[
            rng.integers(0, len(values), size=len(values))
        ].mean()
    result.update(
        {
            "ci_low": float(np.quantile(estimates, 0.025)),
            "ci_high": float(np.quantile(estimates, 0.975)),
        }
    )
    return result


def _within_sample_effect(values, y, sample_id):
    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    sample_id = np.asarray(sample_id, dtype=str)
    result = []
    for current in np.unique(sample_id):
        selected = sample_id == current
        finite = selected & np.isfinite(values)
        if not bool((finite & (y == 0)).any()) or not bool(
            (finite & (y == 1)).any()
        ):
            continue
        result.append(
            float(values[finite & (y == 1)].mean() - values[finite & (y == 0)].mean())
        )
    return np.asarray(result, dtype=np.float64)


def first_onset_effects(values, y, sample_id, token_index, window):
    """Return one standardized pre/post effect at each response's first 0->1."""

    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    sample_id = np.asarray(sample_id, dtype=str)
    token_index = np.asarray(token_index, dtype=np.int64)
    by_sample = []
    for current in np.unique(sample_id):
        selected = np.flatnonzero(sample_id == current)
        order = selected[np.argsort(token_index[selected])]
        local_y = y[order]
        local_value = values[order]
        local_token = token_index[order]
        transitions = np.flatnonzero((local_y[1:] == 1) & (local_y[:-1] == 0)) + 1
        if not len(transitions):
            continue
        local = int(transitions[0])
        run_end = local
        while run_end < len(order) and local_y[run_end] == 1:
            run_end += 1
        pre_start = max(0, local - int(window))
        post_end = min(run_end, local + int(window))
        pre = local_value[pre_start:local]
        post = local_value[local:post_end]
        pre_label = local_y[pre_start:local]
        contiguous = bool((np.diff(local_token[pre_start:post_end]) == 1).all())
        if (
            not contiguous
            or len(pre) == 0
            or len(post) == 0
            or not bool((pre_label == 0).all())
            or not np.isfinite(pre).all()
            or not np.isfinite(post).all()
        ):
            continue
        by_sample.append(float(post.mean() - pre.mean()))
    return np.asarray(by_sample, dtype=np.float64)


def _spearman(values, residual, mask=None):
    values = np.asarray(values, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(residual)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if int(finite.sum()) < 3:
        return None
    coefficient, p_value = spearmanr(values[finite], residual[finite])
    return {
        "rho": _finite_float(coefficient),
        "p_value": _finite_float(p_value),
        "tokens": int(finite.sum()),
    }


def _feature_metric_rows(feature_names, matrix, y, *, representation):
    rows = []
    mapping = {}
    for index, name in enumerate(feature_names):
        metric = _binary_metrics(y, matrix[:, index])
        mapping[str(name)] = metric
        if metric is not None:
            rows.append(
                {"representation": representation, "feature": str(name), **metric}
            )
    return mapping, rows


def _layer_metric_rows(y, artifact):
    rows = []
    report = {}
    for family in LAYER_PROFILE_NAMES:
        matrix = np.asarray(artifact[family], dtype=np.float32)
        current = []
        for layer in range(matrix.shape[1]):
            metric = _binary_metrics(y, matrix[:, layer])
            current.append(metric)
            if metric is not None:
                rows.append({"family": family, "layer": layer, **metric})
        report[family] = current
    return report, rows


def _rank_metric_rows(y, artifact):
    matrix = np.asarray(artifact["spectral_rank_residual_energy"], dtype=np.float32)
    rows = []
    report = []
    for rank in range(matrix.shape[1]):
        metric = _binary_metrics(y, matrix[:, rank])
        report.append(metric)
        if metric is not None:
            rows.append({"spectral_rank": rank, **metric})
    return report, rows


def _phase_curve_rows(feature_names, raw, y, relative_position, phase_bins):
    selected_names = tuple(
        name
        for name in (
            *CONVERGENCE_FEATURES,
            *GROUNDING_FEATURES,
            *SETWALK_COORDINATION_FEATURES,
            "spectral_residual_energy",
            "residual_effective_channels",
            "residual_weighted_lag",
        )
        if name in set(feature_names)
    )
    name_to_index = {str(name): index for index, name in enumerate(feature_names)}
    phase = np.minimum(
        (np.asarray(relative_position) * int(phase_bins)).astype(np.int64),
        int(phase_bins) - 1,
    )
    rows = []
    for name in selected_names:
        values = raw[:, name_to_index[name]]
        for phase_bin in range(int(phase_bins)):
            for label in (0, 1):
                selected = (phase == phase_bin) & (y == label) & np.isfinite(values)
                if not bool(selected.any()):
                    continue
                rows.append(
                    {
                        "feature": name,
                        "phase_bin": phase_bin,
                        "label": label,
                        "tokens": int(selected.sum()),
                        "mean": float(values[selected].mean()),
                        "median": float(np.median(values[selected])),
                    }
                )
    return rows


def _write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_topology_artifact(
    dataset,
    artifact_path,
    output_dir,
    *,
    bootstrap_replicates=None,
    onset_window=None,
    phase_bins=None,
    seed=None,
):
    """Open labels post-hoc and diagnose topology differences."""

    evaluation = FrozenEvaluation.capture(artifact_path, expected_split="test")
    artifact = load_topology_artifact(evaluation.artifact.path)
    reference = verify_score_provenance(artifact)
    aligned = evaluation.align_loaded(dataset, artifact)
    y = aligned.token_label
    feature_names = np.asarray(artifact["feature_names"], dtype=str)
    raw = np.asarray(artifact["features_raw"], dtype=np.float32)
    z = np.asarray(artifact["features_z"], dtype=np.float32)
    name_to_index = {str(name): index for index, name in enumerate(feature_names)}

    bootstrap_replicates = int(
        reference["bootstrap_replicates"]
        if bootstrap_replicates is None
        else bootstrap_replicates
    )
    onset_window = int(
        reference["onset_window"] if onset_window is None else onset_window
    )
    phase_bins = int(reference["phase_bins"] if phase_bins is None else phase_bins)
    seed = int(reference["seed"] if seed is None else seed)

    raw_metrics, feature_rows_raw = _feature_metric_rows(
        feature_names, raw, y, representation="raw"
    )
    z_metrics, feature_rows_z = _feature_metric_rows(
        feature_names, z, y, representation="train_standardized"
    )
    layer_metrics, layer_rows = _layer_metric_rows(y, artifact)
    rank_metrics, rank_rows = _rank_metric_rows(y, artifact)

    sample_effects = {}
    onset_effects = {}
    sample_rows = []
    onset_rows = []
    for index, name in enumerate(feature_names):
        within = _within_sample_effect(z[:, index], y, artifact["sample_id"])
        within_report = _bootstrap_mean_interval(
            within,
            replicates=bootstrap_replicates,
            seed=seed + index,
        )
        sample_effects[str(name)] = within_report
        if within_report is not None:
            sample_rows.append(
                {
                    "feature": str(name),
                    "representation": "train_standardized_features_z",
                    **within_report,
                }
            )

        onset = first_onset_effects(
            z[:, index],
            y,
            artifact["sample_id"],
            artifact["token_index"],
            onset_window,
        )
        onset_report = _bootstrap_mean_interval(
            onset,
            replicates=bootstrap_replicates,
            seed=seed + 10_000 + index,
        )
        onset_effects[str(name)] = onset_report
        if onset_report is not None:
            onset_rows.append(
                {
                    "feature": str(name),
                    "representation": "train_standardized_features_z",
                    "onset_definition": "first_0_to_1_transition_per_response",
                    **onset_report,
                }
            )

    residual = raw[:, name_to_index["spectral_residual_energy"]]
    correlations = {}
    correlation_rows = []
    for index, name in enumerate(feature_names):
        current = {
            "all": _spearman(raw[:, index], residual),
            "normal": _spearman(raw[:, index], residual, y == 0),
            "hallucination": _spearman(raw[:, index], residual, y == 1),
        }
        correlations[str(name)] = current
        for population, metric in current.items():
            if metric is not None:
                correlation_rows.append(
                    {"feature": str(name), "population": population, **metric}
                )

    phase_rows = _phase_curve_rows(
        feature_names,
        raw,
        y,
        artifact["relative_position"],
        phase_bins,
    )

    overall = {
        "tokens": len(y),
        "positive_tokens": int(y.sum()),
        "prevalence": float(y.mean()),
        "samples": len(np.unique(artifact["sample_id"])),
    }
    report = {
        "schema": EVALUATION_SCHEMA,
        "overall": overall,
        "feature_metrics_raw": raw_metrics,
        "feature_metrics_train_standardized": z_metrics,
        "within_sample_effects_train_standardized": sample_effects,
        "first_hallucination_onset_effects_train_standardized": onset_effects,
        "layer_metrics": layer_metrics,
        "spectral_rank_metrics": rank_metrics,
        "correlation_with_spectral_residual": correlations,
        "hypotheses": {
            "route_convergence": list(CONVERGENCE_FEATURES),
            "grounding_vs_feedback": list(GROUNDING_FEATURES),
            "setwalk_rp_rr_coordination": list(SETWALK_COORDINATION_FEATURES),
            "spectral_escape_localization": list(RESIDUAL_FEATURES),
        },
        "claim_boundaries": {
            "labels_used_during": "posthoc_evaluation_only",
            "effect_representation": "train_standardized_features_z",
            "onset_definition": "first_0_to_1_transition_per_response",
            **score_temporal_scope().as_dict(),
            "confidence_available": False,
            "confidence_reason": (
                "the canonical attention cache contains attention/metadata but no "
                "token logits, entropy, NLL, or calibrated confidence"
            ),
            "topology_scope": (
                "retained cache-censored causal response-query topology; prompt "
                "query rows and exact sub-floor attention values are unavailable"
            ),
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "feature_metrics.csv", feature_rows_raw + feature_rows_z)
    _write_csv(output_dir / "within_sample_effects.csv", sample_rows)
    _write_csv(output_dir / "onset_effects.csv", onset_rows)
    _write_csv(output_dir / "layer_metrics.csv", layer_rows)
    _write_csv(output_dir / "spectral_rank_metrics.csv", rank_rows)
    _write_csv(output_dir / "phase_curves.csv", phase_rows)
    _write_csv(output_dir / "residual_correlations.csv", correlation_rows)
    return report
