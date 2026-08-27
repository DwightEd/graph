"""Source-balanced, label-free calibration of routing-lineage scores."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from experiment_protocol import file_sha256, scalar_text
from experiments.grounded_route.artifacts import save_npz

from .lineage_artifacts import (
    ARTIFACT_VERSION,
    ROW_FIELDS,
    TRACE_SCHEMA,
    copied_metadata,
    require_artifact,
)
from .lineage_controls import LINEAGE_CONTROLS


SCORE_SCHEMA = "attention-routing-lineage-score"
DEFAULT_POSITION_BIN_WIDTH = 16
DEFAULT_MINIMUM_REFERENCE_SOURCES = 10


def source_balanced_high_tail(
    reference_score: np.ndarray,
    reference_source: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    """Compute plus-one high-tail probabilities with equal source weight.

    Each source contributes one smoothed empirical survival function regardless
    of how many token rows it contains.  Averaging those functions prevents a
    long response or prolific source from dominating calibration.
    """

    reference_score = np.asarray(reference_score, dtype=np.float64)
    reference_source = np.asarray(reference_source).astype(str)
    value = np.asarray(value, dtype=np.float64)
    sources = tuple(dict.fromkeys(reference_source.tolist()))
    if not sources:
        raise ValueError("high-tail calibration requires reference rows")
    probability = np.zeros(len(value), dtype=np.float64)
    for source in sources:
        ordered = np.sort(reference_score[reference_source == source])
        position = np.searchsorted(ordered, value, side="left")
        probability += (len(ordered) - position + 1.0) / (len(ordered) + 1.0)
    return probability / len(sources)


def conditional_high_tail(
    calibration_score: np.ndarray,
    calibration_source: np.ndarray,
    calibration_task: np.ndarray,
    calibration_position: np.ndarray,
    test_score: np.ndarray,
    test_task: np.ndarray,
    test_position: np.ndarray,
    *,
    position_bin_width: int,
    minimum_reference_sources: int = DEFAULT_MINIMUM_REFERENCE_SOURCES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate by task and fixed absolute response-position bins.

    Exact ``(task, bin)`` reference rows are used only when they contain enough
    distinct source groups.  Sparse bins fall back to task-wide rows, then to
    the global reference.  Neither answer length nor future relative position
    enters a stratum.
    """

    if int(position_bin_width) < 1:
        raise ValueError("position_bin_width must be positive")
    if int(minimum_reference_sources) < 1:
        raise ValueError("minimum_reference_sources must be positive")
    calibration_score = np.asarray(calibration_score, dtype=np.float64)
    calibration_source = np.asarray(calibration_source).astype(str)
    calibration_task = np.asarray(calibration_task).astype(str)
    calibration_position = np.asarray(calibration_position, dtype=np.int64)
    test_score = np.asarray(test_score, dtype=np.float64)
    test_task = np.asarray(test_task).astype(str)
    test_position = np.asarray(test_position, dtype=np.int64)
    calibration_bin = calibration_position // int(position_bin_width)
    test_bin = test_position // int(position_bin_width)

    probability = np.ones(len(test_score), dtype=np.float64)
    support_rows = np.zeros(len(test_score), dtype=np.int32)
    support_sources = np.zeros(len(test_score), dtype=np.int32)
    fallback = np.full(len(test_score), 2, dtype=np.int8)
    for task, position_bin in dict.fromkeys(
        zip(test_task.tolist(), test_bin.tolist(), strict=True)
    ):
        selected_test = (test_task == task) & (test_bin == position_bin)
        exact = (calibration_task == task) & (calibration_bin == position_bin)
        task_wide = calibration_task == task
        exact_sources = len(set(calibration_source[exact].tolist()))
        task_sources = len(set(calibration_source[task_wide].tolist()))
        if exact_sources >= int(minimum_reference_sources):
            selected_reference = exact
            level = 0
        elif task_sources >= int(minimum_reference_sources):
            selected_reference = task_wide
            level = 1
        else:
            selected_reference = np.ones(len(calibration_score), dtype=bool)
            level = 2
        probability[selected_test] = source_balanced_high_tail(
            calibration_score[selected_reference],
            calibration_source[selected_reference],
            test_score[selected_test],
        )
        support_rows[selected_test] = int(selected_reference.sum())
        support_sources[selected_test] = len(
            set(calibration_source[selected_reference].tolist())
        )
        fallback[selected_test] = level
    return probability, support_rows, support_sources, fallback


def same_trace_geometry(
    calibration: Mapping[str, np.ndarray],
    test: Mapping[str, np.ndarray],
) -> None:
    """Require identical routing settings across calibration and test."""

    for field in (
        "layer_count",
        "head_count",
        "attention_floor",
        "seed",
        "carrier_rewire_passes",
        "takeover_epsilon",
    ):
        if np.asarray(calibration[field]).item() != np.asarray(test[field]).item():
            raise ValueError(f"calibration and test traces differ in {field}")
    if tuple(calibration["controls"].astype(str)) != tuple(test["controls"].astype(str)):
        raise ValueError("calibration and test traces use different controls")
    for field in (
        "alignment",
        "prompt_partition",
        "functional_contribution_observed",
        "drift_observed",
        "dispersion_observed",
        "parametric_bias_observed",
    ):
        if np.asarray(calibration[field]).item() != np.asarray(test[field]).item():
            raise ValueError(f"calibration and test traces differ in {field}")


def require_disjoint_sources(
    calibration: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]
) -> None:
    """Prevent the same canonical source from calibrating and testing itself."""

    calibration_sources = set(calibration["source_id"].astype(str).tolist())
    test_sources = set(test["source_id"].astype(str).tolist())
    overlap = calibration_sources.intersection(test_sources)
    if overlap:
        raise ValueError("calibration and test traces share source groups")


def require_calibration_test_scope(
    calibration: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]
) -> None:
    """Bind calibration to train rows and scoring to held-out test rows."""

    if (
        scalar_text(calibration, "split") != "train"
        or scalar_text(calibration, "scope") != "calibration"
    ):
        raise ValueError("lineage calibration requires train calibration rows")
    if (
        scalar_text(test, "split") != "test"
        or scalar_text(test, "scope") != "all"
    ):
        raise ValueError("lineage scoring requires all held-out test rows")


def calibrated_field(
    calibration: Mapping[str, np.ndarray],
    test: Mapping[str, np.ndarray],
    calibration_raw: np.ndarray,
    test_raw: np.ndarray,
    *,
    position_bin_width: int,
    minimum_reference_sources: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate one high-oriented raw statistic on available rows only."""

    calibration_available = calibration["available"].astype(bool)
    test_available = test["available"].astype(bool)
    probability = np.ones(len(test_raw), dtype=np.float64)
    support_rows = np.zeros(len(test_raw), dtype=np.int32)
    support_sources = np.zeros(len(test_raw), dtype=np.int32)
    fallback = np.full(len(test_raw), -1, dtype=np.int8)
    calibrated = conditional_high_tail(
        np.asarray(calibration_raw)[calibration_available],
        calibration["source_id"][calibration_available],
        calibration["task_type"][calibration_available],
        calibration["token_index"][calibration_available],
        np.asarray(test_raw)[test_available],
        test["task_type"][test_available],
        test["token_index"][test_available],
        position_bin_width=position_bin_width,
        minimum_reference_sources=minimum_reference_sources,
    )
    probability[test_available] = calibrated[0]
    support_rows[test_available] = calibrated[1]
    support_sources[test_available] = calibrated[2]
    fallback[test_available] = calibrated[3]
    return probability, support_rows, support_sources, fallback


def validate_score_fields(arrays: Mapping[str, np.ndarray]) -> None:
    """Validate every frozen comparator before evaluation can open labels."""

    row_count = len(arrays["sample_id"])
    controls = tuple(np.asarray(arrays["controls"]).astype(str).tolist())
    if controls != LINEAGE_CONTROLS:
        raise ValueError("score artifact has unexpected mechanism controls")

    families = (
        *controls,
        "posthoc_same_token",
        "direct_prompt_deficit",
        "unresolved_mass",
    )
    dispersion = (
        "dispersion_entropy_lower",
        "dispersion_entropy_upper",
        "dispersion_role_js",
    )
    bounded_fields = []
    raw_fields = []
    for name in (*families, *dispersion):
        bounded_fields.extend((f"{name}_tail_pvalue", f"{name}_conditional_score"))
        if name in dispersion:
            raw_fields.append(f"{name}_raw")
        elif name in controls or name == "posthoc_same_token":
            raw_fields.append(f"{name}_raw_takeover")

    required = (
        *bounded_fields,
        *raw_fields,
        "score",
        "drift_score",
        "dispersion_lower_score",
        "dispersion_upper_score",
        "absolute_position_score",
        "absolute_sequence_position_score",
        "prompt_length_score",
        "relative_position_offline_score",
        "response_length_offline_score",
        "calibration_support_rows",
        "calibration_support_sources",
        "calibration_fallback_level",
    )
    for name in required:
        if name not in arrays:
            raise ValueError(f"score artifact is missing {name}")
        value = np.asarray(arrays[name])
        if value.shape != (row_count,) or not np.isfinite(value).all():
            raise ValueError(f"score field {name} has invalid rows")
    if any(
        np.any(np.asarray(arrays[name]) < -1e-6)
        or np.any(np.asarray(arrays[name]) > 1.0 + 1e-6)
        for name in bounded_fields
    ):
        raise ValueError("conditional scores and tail probabilities must lie in [0, 1]")

    for name in (*controls, "posthoc_same_token"):
        lineage = np.asarray(arrays[f"{name}_lineage"])
        if lineage.shape != (row_count, 4) or not np.isfinite(lineage).all():
            raise ValueError(f"score artifact has invalid {name} lineage")

    if not np.array_equal(arrays["score"], arrays["ordered_conditional_score"]):
        raise ValueError("primary score alias differs from ordered drift")
    if not np.array_equal(arrays["drift_score"], arrays["score"]):
        raise ValueError("drift score alias differs from the primary score")
    if not np.array_equal(
        arrays["dispersion_lower_score"],
        arrays["dispersion_entropy_lower_conditional_score"],
    ) or not np.array_equal(
        arrays["dispersion_upper_score"],
        arrays["dispersion_entropy_upper_conditional_score"],
    ):
        raise ValueError("dispersion score aliases differ from entropy scores")

    token_index = np.asarray(arrays["token_index"], dtype=np.float64)
    response_length = np.asarray(arrays["response_length"], dtype=np.float64)
    prompt_length = np.asarray(arrays["prompt_length"], dtype=np.float64)
    expected_positions = {
        "absolute_position_score": token_index,
        "absolute_sequence_position_score": prompt_length + token_index,
        "prompt_length_score": prompt_length,
        "relative_position_offline_score": token_index
        / np.maximum(response_length - 1.0, 1.0),
        "response_length_offline_score": response_length,
    }
    if any(
        not np.allclose(np.asarray(arrays[name]), expected, atol=1e-7, rtol=0.0)
        for name, expected in expected_positions.items()
    ):
        raise ValueError("frozen position baselines differ from row identity")

    available = np.asarray(arrays["available"], dtype=bool)
    fallback = np.asarray(arrays["calibration_fallback_level"], dtype=np.int8)
    if np.any(fallback[~available] != -1) or not set(fallback[available]).issubset(
        {0, 1, 2}
    ):
        raise ValueError("calibration fallback levels are invalid")
    scalar_expectations = {
        "primary_score": "ordered_conditional_score",
        "primary_score_family": "routing_drift",
        "calibration_statistic": "source_balanced_empirical_high_tail_probability",
    }
    for name, expected in scalar_expectations.items():
        if scalar_text(arrays, name) != expected:
            raise ValueError(f"score artifact has invalid {name}")
    if bool(np.asarray(arrays["mechanisms_combined"]).item()):
        raise ValueError("drift and dispersion must remain separate mechanisms")
    expected_dispersion = (
        "dispersion_entropy_lower_conditional_score",
        "dispersion_entropy_upper_conditional_score",
        "dispersion_role_js_conditional_score",
    )
    if tuple(np.asarray(arrays["dispersion_scores"]).astype(str)) != expected_dispersion:
        raise ValueError("score artifact has unexpected dispersion readers")
    if int(np.asarray(arrays["position_bin_width"]).item()) < 1 or int(
        np.asarray(arrays["minimum_reference_sources"]).item()
    ) < 1:
        raise ValueError("score artifact has invalid calibration settings")
    if np.any(np.asarray(arrays["calibration_support_sources"])[available] < 1):
        raise ValueError("available scores require calibration source support")


def score_traces(
    calibration_trace_path,
    test_trace_path,
    output_path,
    *,
    position_bin_width: int = DEFAULT_POSITION_BIN_WIDTH,
    minimum_reference_sources: int = DEFAULT_MINIMUM_REFERENCE_SOURCES,
) -> dict[str, object]:
    """Freeze label-free conditional scores for every mechanism control."""

    calibration = require_artifact(calibration_trace_path, TRACE_SCHEMA)
    test = require_artifact(test_trace_path, TRACE_SCHEMA)
    same_trace_geometry(calibration, test)
    require_calibration_test_scope(calibration, test)
    require_disjoint_sources(calibration, test)
    if not bool(calibration["available"].any()):
        raise ValueError("calibration trace has no predecessor-aligned rows")

    controls = tuple(test["controls"].astype(str).tolist())
    raw_fields: dict[str, tuple[np.ndarray, np.ndarray]] = {
        control: (
            calibration[f"{control}_raw_takeover"],
            test[f"{control}_raw_takeover"],
        )
        for control in controls
    }
    raw_fields["posthoc_same_token"] = (
        calibration["posthoc_same_token_raw_takeover"],
        test["posthoc_same_token_raw_takeover"],
    )
    raw_fields["direct_prompt_deficit"] = (
        -calibration["direct_prompt_lookback"],
        -test["direct_prompt_lookback"],
    )
    raw_fields["unresolved_mass"] = (
        1.0 - calibration["known_mass"],
        1.0 - test["known_mass"],
    )
    raw_fields["dispersion_entropy_lower"] = (
        calibration["routing_entropy_lower"].mean(axis=1),
        test["routing_entropy_lower"].mean(axis=1),
    )
    raw_fields["dispersion_entropy_upper"] = (
        calibration["routing_entropy_upper"].mean(axis=1),
        test["routing_entropy_upper"].mean(axis=1),
    )
    raw_fields["dispersion_role_js"] = (
        calibration["routing_role_js"].mean(axis=1),
        test["routing_role_js"].mean(axis=1),
    )

    arrays: dict[str, np.ndarray] = {
        name: np.asarray(test[name]) for name in ROW_FIELDS
    }
    arrays.update(
        available=test["available"].astype(bool),
        predictor_token_index=test["predictor_token_index"].astype(np.int32),
        ordered_representation=test["ordered_representation"].astype(np.float32),
        routing_representation=test["routing_representation"].astype(np.float32),
        routing_entropy_lower=test["routing_entropy_lower"].astype(np.float32),
        routing_entropy_upper=test["routing_entropy_upper"].astype(np.float32),
        routing_concentration_lower=test["routing_concentration_lower"].astype(
            np.float32
        ),
        routing_concentration_upper=test["routing_concentration_upper"].astype(
            np.float32
        ),
        routing_role_mass=test["routing_role_mass"].astype(np.float32),
        routing_head_role_std=test["routing_head_role_std"].astype(np.float32),
        routing_role_js=test["routing_role_js"].astype(np.float32),
        direct_prompt_lookback=test["direct_prompt_lookback"].astype(np.float32),
        known_mass=test["known_mass"].astype(np.float32),
        carrier_rewire_changed_fraction=test[
            "carrier_rewire_changed_fraction"
        ].astype(np.float32),
        absolute_position_score=test["token_index"].astype(np.float32),
        absolute_sequence_position_score=(
            test["prompt_length"] + test["token_index"]
        ).astype(np.float32),
        prompt_length_score=test["prompt_length"].astype(np.float32),
        relative_position_offline_score=(
            test["token_index"]
            / np.maximum(test["response_length"] - 1, 1)
        ).astype(np.float32),
        response_length_offline_score=test["response_length"].astype(np.float32),
    )
    for control in controls:
        arrays[f"{control}_lineage"] = test[f"{control}_lineage"].astype(np.float32)
        arrays[f"{control}_raw_takeover"] = test[
            f"{control}_raw_takeover"
        ].astype(np.float32)
    arrays["posthoc_same_token_lineage"] = test[
        "posthoc_same_token_lineage"
    ].astype(np.float32)
    arrays["posthoc_same_token_raw_takeover"] = test[
        "posthoc_same_token_raw_takeover"
    ].astype(np.float32)

    shared_support = None
    for name, (calibration_raw, test_raw) in raw_fields.items():
        calibrated = calibrated_field(
            calibration,
            test,
            calibration_raw,
            test_raw,
            position_bin_width=position_bin_width,
            minimum_reference_sources=minimum_reference_sources,
        )
        probability, support_rows, support_sources, fallback = calibrated
        arrays[f"{name}_tail_pvalue"] = probability.astype(np.float32)
        arrays[f"{name}_conditional_score"] = (1.0 - probability).astype(np.float32)
        if name.startswith("dispersion_"):
            arrays[f"{name}_raw"] = np.asarray(test_raw, dtype=np.float32)
        if shared_support is None:
            shared_support = (support_rows, support_sources, fallback)

    arrays["score"] = arrays["ordered_conditional_score"]
    arrays["drift_score"] = arrays["ordered_conditional_score"]
    arrays["dispersion_lower_score"] = arrays[
        "dispersion_entropy_lower_conditional_score"
    ]
    arrays["dispersion_upper_score"] = arrays[
        "dispersion_entropy_upper_conditional_score"
    ]
    arrays["calibration_support_rows"] = shared_support[0]
    arrays["calibration_support_sources"] = shared_support[1]
    arrays["calibration_fallback_level"] = shared_support[2]
    arrays.update(copied_metadata(test))
    arrays.update(
        schema=np.asarray(SCORE_SCHEMA),
        version=np.asarray(ARTIFACT_VERSION, dtype=np.int32),
        labels_included=np.asarray(False),
        controls=np.asarray(controls),
        primary_score=np.asarray("ordered_conditional_score"),
        primary_score_family=np.asarray("routing_drift"),
        dispersion_scores=np.asarray(
            (
                "dispersion_entropy_lower_conditional_score",
                "dispersion_entropy_upper_conditional_score",
                "dispersion_role_js_conditional_score",
            )
        ),
        dispersion_layer_aggregation=np.asarray("arithmetic_mean"),
        mechanisms_combined=np.asarray(False),
        position_bin_width=np.asarray(position_bin_width, dtype=np.int32),
        minimum_reference_sources=np.asarray(
            minimum_reference_sources, dtype=np.int32
        ),
        conditioning_fields=np.asarray(("task_type", "absolute_position_bin")),
        calibration_statistic=np.asarray(
            "source_balanced_empirical_high_tail_probability"
        ),
        calibration_trace_sha256=np.asarray(file_sha256(calibration_trace_path)),
        test_trace_sha256=np.asarray(file_sha256(test_trace_path)),
    )
    validate_score_fields(arrays)
    save_npz(output_path, **arrays)
    return {
        "scores": str(Path(output_path).resolve()),
        "samples": len(set(arrays["sample_id"].astype(str).tolist())),
        "nodes": len(arrays["sample_id"]),
        "available_nodes": int(arrays["available"].sum()),
        "drift_score": "ordered_conditional_score",
        "dispersion_scores": (
            "dispersion_entropy_lower_conditional_score",
            "dispersion_entropy_upper_conditional_score",
            "dispersion_role_js_conditional_score",
        ),
        "mechanisms_combined": False,
        "position_bin_width": int(position_bin_width),
        "minimum_reference_sources": int(minimum_reference_sources),
        "labels_read": False,
    }
