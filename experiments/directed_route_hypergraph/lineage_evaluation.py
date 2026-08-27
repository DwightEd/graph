"""Post-hoc label evaluation for frozen routing-lineage scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiment_protocol import FrozenEvaluation
from experiments.grounded_route.evaluate import metrics, source_bootstrap
from research_dataset import open_research_dataset

from .lineage_artifacts import ARTIFACT_VERSION, sample_level_values, require_artifact
from .lineage_scoring import SCORE_SCHEMA, validate_score_fields
from .routing_lineage import ENDOGENOUS, INDIRECT, UNRESOLVED


def load_frozen_labels(frozen: FrozenEvaluation, arrays, test_root):
    """Open labels after capture and bind them to the exact response tokens."""

    dataset = open_research_dataset(
        test_root,
        device="cpu",
        retain_embedded_labels=True,
    )
    sample_id = arrays["sample_id"].astype(str)
    token_index = arrays["token_index"].astype(np.int64)
    for current_id in dict.fromkeys(sample_id.tolist()):
        rows = np.flatnonzero(sample_id == current_id)
        sample = dataset[current_id]
        try:
            attention = sample.attention()
            response_token_id = (
                attention.token_ids[int(attention.response_idx) :]
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            if not np.all(arrays["response_length"][rows] == len(response_token_id)):
                raise ValueError("evaluation response length differs from score rows")
            if not np.array_equal(
                arrays["response_token_id"][rows],
                response_token_id[token_index[rows]],
            ):
                raise ValueError("evaluation token IDs differ from score rows")
        finally:
            sample.release_attention()
    return frozen.align_loaded(dataset, arrays)


def paired_source_delta(
    label: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    source_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Source-bootstrap the paired metric difference ``left - right``."""

    left_point = metrics(label, left)
    right_point = metrics(label, right)
    point = {
        "auroc_delta": None
        if left_point["auroc"] is None
        else float(left_point["auroc"] - right_point["auroc"]),
        "auprc_delta": None
        if left_point["auprc"] is None
        else float(left_point["auprc"] - right_point["auprc"]),
    }
    source_id = np.asarray(source_id).astype(str)
    groups = tuple(dict.fromkeys(source_id.tolist()))
    rows = {group: np.flatnonzero(source_id == group) for group in groups}
    random = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(replicates)):
        chosen = random.choice(groups, len(groups), replace=True)
        selected = np.concatenate([rows[group] for group in chosen])
        left_metric = metrics(label[selected], left[selected])
        right_metric = metrics(label[selected], right[selected])
        if left_metric["auroc"] is not None:
            estimates.append(
                (
                    left_metric["auroc"] - right_metric["auroc"],
                    left_metric["auprc"] - right_metric["auprc"],
                )
            )
    if not estimates:
        return {**point, "replicates_valid": 0}
    values = np.asarray(estimates)
    return {
        **point,
        "replicates_valid": len(values),
        "auroc_delta_ci_low": float(np.quantile(values[:, 0], 0.025)),
        "auroc_delta_ci_high": float(np.quantile(values[:, 0], 0.975)),
        "auprc_delta_ci_low": float(np.quantile(values[:, 1], 0.025)),
        "auprc_delta_ci_high": float(np.quantile(values[:, 1], 0.975)),
    }


def onset_diagnostics(
    label: np.ndarray,
    score: np.ndarray,
    sample_id: np.ndarray,
    token_index: np.ndarray,
    available: np.ndarray,
    *,
    alarm_threshold: float = 0.95,
    preceding_window: int = 4,
) -> dict[str, float | int | None]:
    """Compare the first labeled span onset in each response with its predecessor.

    Token zero has no cached predecessor query.  If a response begins inside a
    positive span, that response is recorded as unscorable and no later span is
    substituted for its true first onset.
    """

    onsets = []
    paired = []
    window_alarm = []
    first_onset_count = 0
    unavailable_first_onset_count = 0
    sample_id = np.asarray(sample_id).astype(str)
    for sample in dict.fromkeys(sample_id.tolist()):
        rows = np.flatnonzero(sample_id == sample)
        rows = rows[np.argsort(token_index[rows])]
        onset_position = None
        for position, row in enumerate(rows):
            previous_label = 0 if position == 0 else int(label[rows[position - 1]])
            if int(label[row]) == 1 and previous_label == 0:
                onset_position = position
                break
        if onset_position is None:
            continue
        first_onset_count += 1
        row = rows[onset_position]
        if not available[row]:
            unavailable_first_onset_count += 1
            continue
        onsets.append(float(score[row]))
        if onset_position > 0:
            previous = rows[onset_position - 1]
            if available[previous] and int(label[previous]) == 0:
                paired.append(float(score[row] - score[previous]))
        start = max(0, onset_position - int(preceding_window))
        window = rows[start : onset_position + 1]
        window = window[available[window]]
        window_alarm.append(bool((score[window] >= alarm_threshold).any()))
    return {
        "responses_with_first_onset": first_onset_count,
        "first_onsets_available": len(onsets),
        "first_onsets_unavailable_token_zero": unavailable_first_onset_count,
        "matched_correct_predecessors": len(paired),
        "mean_onset_score": None if not onsets else float(np.mean(onsets)),
        "mean_onset_minus_previous": None if not paired else float(np.mean(paired)),
        "median_onset_minus_previous": None if not paired else float(np.median(paired)),
        "positive_pair_fraction": None
        if not paired
        else float(np.mean(np.asarray(paired) > 0)),
        "alarm_threshold": float(alarm_threshold),
        "preceding_window": int(preceding_window),
        "alarm_in_window_rate": None
        if not window_alarm
        else float(np.mean(window_alarm)),
    }


def relay_rescue_diagnostics(
    label: np.ndarray,
    lineage: np.ndarray,
    direct_deficit_score: np.ndarray,
) -> dict[str, object]:
    """Inspect I versus E when matched calibration says direct lookback is low."""

    selected = direct_deficit_score >= 0.75
    result: dict[str, object] = {
        "selection": "direct_prompt_deficit_conditional_score >= 0.75",
        "tokens": int(selected.sum()),
    }
    for value, name in ((0, "correct"), (1, "hallucinated")):
        rows = selected & (label == value)
        result[name] = {
            "tokens": int(rows.sum()),
            "mean_indirect": None
            if not bool(rows.any())
            else float(lineage[rows, INDIRECT].mean()),
            "mean_endogenous": None
            if not bool(rows.any())
            else float(lineage[rows, ENDOGENOUS].mean()),
        }
    return result


def detector_result(
    label: np.ndarray,
    score: np.ndarray,
    source_id: np.ndarray,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    """Return point metrics and source-grouped confidence intervals."""

    return {
        **metrics(label, score),
        "source_bootstrap": source_bootstrap(
            label, score, source_id, bootstrap_replicates, seed
        ),
    }


def evaluate_scores(
    test_root,
    score_path,
    output_path,
    *,
    bootstrap_replicates: int = 500,
    seed: int = 20260827,
) -> dict[str, object]:
    """Capture scores, then load rows and labels, and mask unavailable tokens."""

    frozen = FrozenEvaluation.capture(score_path, expected_split="test")
    arrays = require_artifact(frozen.artifact.path, SCORE_SCHEMA)
    validate_score_fields(arrays)
    labels = load_frozen_labels(frozen, arrays, test_root)
    label_all = labels.token_label.astype(np.int8)
    available = arrays["available"].astype(bool)
    if not bool(available.any()):
        raise ValueError("score artifact has no predecessor-aligned rows")
    label = label_all[available]
    source_id = labels.source_id.astype(str)[available]
    controls = tuple(arrays["controls"].astype(str).tolist())
    score_fields = {
        control: f"{control}_conditional_score" for control in controls
    }
    score_fields.update(
        posthoc_same_token="posthoc_same_token_conditional_score",
        direct_prompt_deficit="direct_prompt_deficit_conditional_score",
        unresolved_mass="unresolved_mass_conditional_score",
    )
    dispersion_score_fields = {
        "entropy_lower_bound": "dispersion_entropy_lower_conditional_score",
        "entropy_upper_bound": "dispersion_entropy_upper_conditional_score",
        "head_role_js": "dispersion_role_js_conditional_score",
    }

    detector_report = {}
    for offset, (name, field) in enumerate(score_fields.items()):
        score = arrays[field].astype(np.float64)[available]
        detector_report[name] = detector_result(
            label,
            score,
            source_id,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + offset,
        )
    dispersion_report = {}
    for offset, (name, field) in enumerate(dispersion_score_fields.items()):
        dispersion_report[name] = detector_result(
            label,
            arrays[field].astype(np.float64)[available],
            source_id,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 40 + offset,
        )
    raw_scores = {
        control: arrays[f"{control}_raw_takeover"].astype(np.float64)[available]
        for control in controls
    }
    raw_scores.update(
        posthoc_same_token=arrays["posthoc_same_token_raw_takeover"].astype(
            np.float64
        )[available],
        direct_prompt_deficit=-arrays["direct_prompt_lookback"].astype(np.float64)[
            available
        ],
        unresolved_mass=1.0 - arrays["known_mass"].astype(np.float64)[available],
    )
    raw_report = {name: metrics(label, score) for name, score in raw_scores.items()}
    dispersion_raw_scores = {
        "entropy_lower_bound": arrays["dispersion_entropy_lower_raw"].astype(
            np.float64
        )[available],
        "entropy_upper_bound": arrays["dispersion_entropy_upper_raw"].astype(
            np.float64
        )[available],
        "head_role_js": arrays["dispersion_role_js_raw"].astype(np.float64)[
            available
        ],
    }
    dispersion_raw_report = {
        name: metrics(label, score) for name, score in dispersion_raw_scores.items()
    }
    position_scores = {
        "response_ordinal": arrays["absolute_position_score"].astype(np.float64)[
            available
        ],
        "absolute_sequence_position": arrays[
            "absolute_sequence_position_score"
        ].astype(np.float64)[available],
        "prompt_length": arrays["prompt_length_score"].astype(np.float64)[available],
        "relative_position_offline": arrays[
            "relative_position_offline_score"
        ].astype(np.float64)[available],
        "response_length_offline": arrays["response_length_offline_score"].astype(
            np.float64
        )[available],
    }
    position_report = {}
    for offset, (name, score) in enumerate(position_scores.items()):
        position_report[name] = detector_result(
            label,
            score,
            source_id,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + 20 + offset,
        )

    primary = arrays["ordered_conditional_score"].astype(np.float64)[available]
    comparison_scores = {
        name: arrays[field].astype(np.float64)[available]
        for name, field in score_fields.items()
        if name != "ordered"
    }
    comparison_scores.update(position_scores)
    paired = {
        f"ordered_minus_{name}": paired_source_delta(
            label,
            primary,
            right,
            source_id,
            replicates=bootstrap_replicates,
            seed=seed + 100 + offset,
        )
        for offset, (name, right) in enumerate(comparison_scores.items())
    }
    dispersion_paired = {
        f"{bound}_minus_{name}": paired_source_delta(
            label,
            arrays[field].astype(np.float64)[available],
            right,
            source_id,
            replicates=bootstrap_replicates,
            seed=seed + 200 + offset,
        )
        for offset, ((bound, field), (name, right)) in enumerate(
            (
                (dispersion, position)
                for dispersion in dispersion_score_fields.items()
                for position in position_scores.items()
            )
        )
    }

    ordered_lineage = arrays["ordered_lineage"].astype(np.float64)
    unresolved = ordered_lineage[:, UNRESOLVED]
    unresolved_available = unresolved[available]
    _, changed = sample_level_values(
        arrays["sample_id"], arrays["carrier_rewire_changed_fraction"]
    )
    report = {
        "schema": "attention-routing-lineage-evaluation",
        "version": ARTIFACT_VERSION,
        "labels_read": True,
        "labels_used_during": "posthoc_evaluation_only",
        "alignment": str(np.asarray(arrays["alignment"]).item()),
        "observability": {
            "prompt_partition": str(
                np.asarray(arrays["prompt_partition"]).item()
            ),
            "functional_contribution_observed": bool(
                np.asarray(arrays["functional_contribution_observed"]).item()
            ),
            "drift_observed": bool(
                np.asarray(arrays["drift_observed"]).item()
            ),
            "dispersion_observed": bool(
                np.asarray(arrays["dispersion_observed"]).item()
            ),
            "parametric_bias_observed": bool(
                np.asarray(arrays["parametric_bias_observed"]).item()
            ),
        },
        "primary": {
            "drift": "ordered_conditional_takeover",
            "dispersion": (
                "separately calibrated mean-layer entropy bounds and head-role JSD"
            ),
            "combined_score": None,
        },
        "samples": len(changed),
        "tokens_complete": len(label_all),
        "tokens_evaluated": int(available.sum()),
        "unavailable_boundary_tokens": int((~available).sum()),
        "positive_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        "token_detection": detector_report,
        "drift_detection": detector_report,
        "dispersion_detection": dispersion_report,
        "raw_mechanism_scores": raw_report,
        "raw_dispersion_scores": dispersion_raw_report,
        "score_orientation": {
            "drift": "larger means more response-rooted routing ancestry",
            "dispersion_entropy": "larger means more diffuse attention endpoints",
            "dispersion_head_role_js": (
                "larger means stronger between-head source-role disagreement"
            ),
        },
        "calibration_audit": {
            "statistic": str(
                np.asarray(arrays["calibration_statistic"]).item()
            ),
            "position_bin_width": int(
                np.asarray(arrays["position_bin_width"]).item()
            ),
            "minimum_reference_sources": int(
                np.asarray(arrays["minimum_reference_sources"]).item()
            ),
            "support_sources_min": int(
                arrays["calibration_support_sources"][available].min()
            ),
            "support_sources_median": float(
                np.median(arrays["calibration_support_sources"][available])
            ),
            "fallback_fraction": {
                "task_and_position_bin": float(
                    np.mean(arrays["calibration_fallback_level"][available] == 0)
                ),
                "task_wide": float(
                    np.mean(arrays["calibration_fallback_level"][available] == 1)
                ),
                "global": float(
                    np.mean(arrays["calibration_fallback_level"][available] == 2)
                ),
            },
        },
        "position_baselines": position_report,
        "position_baseline_scope": {
            "response_ordinal": "online causal position inside the response",
            "absolute_sequence_position": (
                "online causal prompt length plus response ordinal"
            ),
            "prompt_length": "known before generation",
            "relative_position_offline": "uses future response length",
            "response_length_offline": "uses future response length",
        },
        "paired_deltas": paired,
        "dispersion_position_deltas": dispersion_paired,
        "onset": onset_diagnostics(
            label_all,
            arrays["ordered_conditional_score"].astype(np.float64),
            arrays["sample_id"].astype(str),
            arrays["token_index"].astype(np.int32),
            available,
        ),
        "relay_rescue": relay_rescue_diagnostics(
            label,
            ordered_lineage[available],
            arrays["direct_prompt_deficit_conditional_score"].astype(np.float64)[
                available
            ],
        ),
        "dispersion_onset": {
            name: onset_diagnostics(
                label_all,
                arrays[field].astype(np.float64),
                arrays["sample_id"].astype(str),
                arrays["token_index"].astype(np.int32),
                available,
            )
            for name, field in dispersion_score_fields.items()
        },
        "dispersion_audit": {
            "layer_aggregation": str(
                np.asarray(arrays["dispersion_layer_aggregation"]).item()
            ),
            "entropy_lower_mean": float(
                arrays["routing_entropy_lower"][available].mean()
            ),
            "entropy_upper_mean": float(
                arrays["routing_entropy_upper"][available].mean()
            ),
            "role_js_mean": float(arrays["routing_role_js"][available].mean()),
            "role_mass_max_error": float(
                np.abs(
                    arrays["routing_role_mass"][available].sum(axis=-1) - 1.0
                ).max()
            ),
            "mechanisms_combined": bool(
                np.asarray(arrays["mechanisms_combined"]).item()
            ),
        },
        "mass_audit": {
            "max_absolute_total_error": float(
                np.max(np.abs(ordered_lineage.sum(axis=1) - 1.0))
            ),
            "unresolved_mean": float(unresolved_available.mean()),
            "unresolved_p90": float(np.quantile(unresolved_available, 0.90)),
        },
        "carrier_rewire": {
            "changed_fraction_sample_mean": float(changed.mean()),
            "nonzero_sample_fraction": float((changed > 0).mean()),
        },
        "score_artifact": str(frozen.artifact.path),
        "score_sha256": frozen.artifact.sha256,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**report, "evaluation": str(output_path.resolve())}
