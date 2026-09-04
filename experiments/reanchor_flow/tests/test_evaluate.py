from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reanchor_flow.claims import FORCED_CHUNK, NATURAL_BOUNDARY
from experiments.reanchor_flow.capture import CAPTURE_SCHEMA
from experiments.reanchor_flow.evaluate import json_ready
from experiments.reanchor_flow.hypotheses import drift_report, matched_boundary_effect
from experiments.reanchor_flow.signals import log_lift, validate_artifact_identity
from experiments.reanchor_flow.events import (
    aligned_change,
    boundary_events,
    event_features,
    onset_pairs,
    positive_onsets,
    validate_coordinates,
)
from experiments.reanchor_flow.metrics import (
    cluster_group_contrast,
    cluster_summary,
    metric,
    metric_with_cluster_ci,
)
def test_metric_uses_higher_score_as_hallucination_risk():
    result = metric([0, 0, 1, 1], [0.0, 0.2, 0.8, 1.0])
    assert result["auroc"] == 1.0
    assert result["average_precision"] == 1.0
    assert result["ap_lift"] == 2.0


def test_single_source_never_gets_a_bootstrap_interval():
    summary = cluster_summary([1.0, 2.0], ["only", "only"], repeats=20, seed=3)
    assert summary["ci95"] == [None, None]
    result = metric_with_cluster_ci(
        [0, 1], [0.0, 1.0], ["only", "only"], repeats=20, seed=3
    )
    assert result["auroc_ci95"] == [None, None]


def test_group_contrast_uses_within_source_pairs_only():
    result = cluster_group_contrast(
        [1.0, 3.0, 10.0, 2.0, 5.0],
        [False, True, True, False, True],
        ["a", "a", "only-positive", "b", "b"],
        repeats=20,
        seed=3,
    )
    assert result["paired_sources"] == 2
    assert result["difference"] == 2.5


def test_boundary_match_removes_by_index_with_array_payloads():
    clean = [
        {
            "sample_id": "sample",
            "source_id": "source",
            "preceding_token_id": token,
            "boundary": {"center": center, "evidence_entry": value},
            "array_payload": np.array([center]),
        }
        for center, token, value in ((8, 1, 0.0), (12, 2, 0.5))
    ]
    hallucinated = [
        {
            "sample_id": "sample",
            "source_id": "source",
            "preceding_token_id": 2,
            "boundary": {"center": 14, "evidence_entry": -0.5},
        }
    ]
    result = matched_boundary_effect(
        clean, hallucinated, "evidence_entry", bootstrap=0, seed=1
    )
    assert result["matched_pairs"] == 1
    assert result["source_mean"] == -1.0


def test_coordinates_require_prediction_to_equal_query_plus_one():
    result = {
        "prediction_position": np.array([3, 4]),
        "query_position": np.array([2, 3]),
        "target_token_id": np.array([13, 14]),
        "claim_start": np.array([3]),
        "claim_stop": np.array([5]),
    }
    assert validate_coordinates(result, np.array([10, 11, 12, 13, 14]), 3, "x") == 2
    result["query_position"] = np.array([3, 4])
    with pytest.raises(ValueError, match="q=p-1"):
        validate_coordinates(result, np.array([10, 11, 12, 13, 14]), 3, "x")


def test_positive_onset_is_not_replaced_by_sentence_start():
    np.testing.assert_array_equal(
        positive_onsets(np.array([False, False, True, True, False, True])),
        np.array([2, 5]),
    )


def test_aligned_curve_requires_one_common_complete_window():
    series = np.arange(10, dtype=float)
    assert aligned_change(series, 2, -3, 2) is None
    change = aligned_change(series, 4, -3, 2)
    np.testing.assert_allclose(change, np.array([-1, 0, 1, 2, 3, 4]))


def event_row():
    count = 30
    label = np.zeros(count, dtype=bool)
    label[12:14] = True
    evidence = np.linspace(0.1, 0.8, count)
    return {
        "source_id": "source",
        "sample_id": "sample",
        "response_start": 100,
        "claim_start": np.array([105]),
        "claim_stop": np.array([125]),
        "claim_boundary_kind": np.array([NATURAL_BOUNDARY]),
        "label": label,
        "target_token_id": np.full(count, 7),
        "evidence_specificity": evidence,
        "history_enrichment": 1.0 - evidence,
        "functional_log_lift_trace": None,
    }


def test_late_hallucination_does_not_label_the_sentence_boundary():
    events = boundary_events(
        event_row(), pre=5, post=3, curve_low=-5, curve_high=10
    )
    assert len(events) == 1
    assert events[0]["late_onset"]
    assert not events[0]["onset_near_boundary"]


def test_exact_boundary_onset_is_not_counted_as_near():
    row = event_row()
    row["label"][:] = False
    row["label"][5:7] = True
    events = boundary_events(
        row, pre=5, post=3, curve_low=-5, curve_high=10
    )
    assert events[0]["onset_at_boundary"]
    assert not events[0]["onset_near_boundary"]
    assert not events[0]["late_onset"]


def test_hallucination_onset_gets_a_same_response_token_match():
    pairs = onset_pairs(
        event_row(), pre=5, post=3, curve_low=-5, curve_high=10
    )
    assert len(pairs) == 1
    assert pairs[0]["positive"]["center"] == 12
    assert pairs[0]["token_matched"]
    assert pairs[0]["control"]["center"] != 5


def test_primary_scalar_event_does_not_require_complete_plot_window():
    row = event_row()
    event = event_features(
        row, 15, pre=5, post=3, curve_low=-5, curve_high=20
    )
    assert event is not None
    assert event["evidence_curve"] is None


def test_evidence_entry_is_not_dropped_by_missing_secondary_history():
    row = event_row()
    row["history_enrichment"][0] = np.nan
    event = event_features(
        row, 5, pre=5, post=3, curve_low=-5, curve_high=10
    )
    assert event is not None
    assert np.isfinite(event["evidence_entry"])
    assert np.isnan(event["history_entry_release"])


def test_forced_length_chunk_is_not_a_sentence_boundary_event():
    row = event_row()
    row["claim_boundary_kind"] = np.array([FORCED_CHUNK])
    assert not boundary_events(
        row, pre=5, post=3, curve_low=-5, curve_high=10
    )


def test_availability_null_removes_uniform_source_pool_drift():
    layers, events = 2, 12
    null = np.zeros((layers, events, 3), dtype=float)
    for event in range(events):
        counts = np.array([2.0, 3.0, float(event)])
        null[:, event] = counts / counts.sum()
    lift = log_lift(null, null)
    mean = np.nanmean(lift, axis=0)
    row = {
        "source_id": "s",
        "evidence_enrichment": mean[:, 0],
        "other_prompt_enrichment": mean[:, 1],
        "history_enrichment": mean[:, 2],
        "evidence_specificity": mean[:, 0] - mean[:, 1],
        "raw_evidence_share": null.mean(axis=0)[:, 0],
        "raw_history_share": null.mean(axis=0)[:, 2],
    }
    report = drift_report([row], bootstrap=0, seed=1)
    assert report["evidence_enrichment_change"]["source_mean"] == 0.0
    assert report["history_enrichment_change"]["source_mean"] == 0.0
    assert report["descriptive_raw_share_change"]["history"]["source_mean"] > 0


def test_report_values_are_strict_json_compatible():
    assert json_ready({"x": float("nan"), "y": np.float64(1.0)}) == {
        "x": None,
        "y": 1.0,
    }


def test_artifact_identity_mismatch_is_rejected():
    sample = SimpleNamespace(
        sample_id="sample",
        source_id="source",
        generator_model="model",
        observer_model="/models/model",
    )
    result = {
        "capture_schema": CAPTURE_SCHEMA,
        "sample_id": "wrong",
        "source_id": "source",
        "task_type": "QA",
        "model_id": "model",
        "observer_model": "/models/model",
        "generator_model": "model",
        "cached_observer_model": "/models/model",
        "dtype": "bfloat16",
        "query_chunk": 32,
    }
    entry = {
        "sample_id": "sample",
        "source_id": "source",
        "task_type": "QA",
        "model_id": "model",
        "observer_model": "/models/model",
        "generator_model": "model",
        "cached_observer_model": "/models/model",
        "dtype": "bfloat16",
        "query_chunk": 32,
    }
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        validate_artifact_identity(
            entry,
            result,
            sample,
            "QA",
            {"model_id": "model", "model": "/models/model", "dtype": "bfloat16"},
        )


def test_artifact_identity_accepts_three_way_metadata_match():
    sample = SimpleNamespace(
        sample_id="sample",
        source_id="source",
        generator_model="model",
        observer_model="/models/model",
    )
    values = {
        "capture_schema": CAPTURE_SCHEMA,
        "sample_id": "sample",
        "source_id": "source",
        "task_type": "QA",
        "model_id": "model",
        "observer_model": "/models/model",
        "generator_model": "model",
        "cached_observer_model": "/models/model",
        "dtype": "bfloat16",
        "query_chunk": 32,
    }
    entry = {name: value for name, value in values.items() if name != "capture_schema"}
    validate_artifact_identity(
        entry,
        values,
        sample,
        "QA",
        {"model_id": "model", "model": "/models/model", "dtype": "bfloat16"},
    )
