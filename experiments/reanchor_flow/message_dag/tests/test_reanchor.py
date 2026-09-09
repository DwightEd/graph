import numpy as np
import pytest

from experiments.reanchor_flow.message_dag.events import row_change
from experiments.reanchor_flow.message_dag.reanchor import (
    ReanchorConfig,
    ReanchorProfiler,
)


def test_attention_change_records_focal_and_diffuse_remote_redistribution():
    special = np.zeros(10, dtype=bool)
    previous = np.zeros((2, 10), dtype=np.float32)
    previous[:, 8] = 1
    current = np.zeros_like(previous)
    current[0, [1, 8]] = [0.6, 0.4]
    current[1, [1, 3, 8]] = [0.3, 0.3, 0.4]

    change = row_change(current, previous, query=9, special=special, window=2)

    np.testing.assert_allclose(change["remote_positive_gain"], [0.6, 0.6])
    np.testing.assert_allclose(change["remote_gain_focality"], [1.0, 0.5])
    np.testing.assert_allclose(change["remote_gain_effective_sources"], [1.0, 2.0])
    np.testing.assert_allclose(change["remote_gain_distance"], [8.0, 7.0])


def synthetic_scan():
    layers, heads, rows = 2, 4, 8
    shape = (layers, heads, rows)
    event = np.zeros(shape, dtype=bool)
    gain = np.zeros(shape, dtype=np.float32)
    focality = np.full(shape, np.nan, dtype=np.float32)
    effective = np.full(shape, np.nan, dtype=np.float32)
    distance = np.full(shape, np.nan, dtype=np.float32)
    peak = np.full(shape, -1, dtype=np.int32)

    def set_heads(row, layer, count, focus, peaks):
        event[layer, :count, row] = True
        gain[layer, :count, row] = 1
        focality[layer, :count, row] = focus
        effective[layer, :count, row] = 1 / focus
        distance[layer, :count, row] = np.arange(count) + 6
        peak[layer, :count, row] = peaks

    set_heads(1, 0, 3, 0.25, [0, 1, 2])
    set_heads(2, 1, 3, 0.90, [0, 0, 0])
    set_heads(3, 0, 3, 0.90, [0, 1, 2])
    set_heads(4, 1, 1, 0.90, [0])
    set_heads(5, 0, 1, 0.25, [0])
    special = np.zeros(10, dtype=bool)
    special[8] = True
    evidence = np.zeros(10, dtype=bool)
    evidence[0] = True
    return {
        "event": event,
        "remote_positive_gain": gain,
        "remote_gain_focality": focality,
        "remote_gain_effective_sources": effective,
        "remote_gain_distance": distance,
        "peak_source": peak,
        "token_ids": np.arange(10),
        "token_text": np.asarray([f"t{position}" for position in range(10)]),
        "row_position": np.arange(1, 9),
        "response_start": np.array(2),
        "special_mask": special,
        "evidence_mask": evidence,
        "labels_used": np.array(False),
    }


def test_profiler_distinguishes_head_breadth_focus_and_source_agreement():
    profile = ReanchorProfiler(ReanchorConfig()).run(synthetic_scan())

    assert profile["reanchor_type"].tolist() == [
        "excluded",
        "broad_diffuse",
        "broad_convergent",
        "broad_diverse",
        "sparse_focal",
        "sparse_diffuse",
        "none",
        "excluded",
    ]
    assert profile["is_reanchor"].tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert profile["active_head_fraction"][1] == 0.75
    assert profile["head_focality"][2] == 0.90
    assert profile["focal_head_fraction"][2] == 1.0
    assert profile["source_agreement"][2] == 1.0
    assert profile["source_agreement"][3] == 1 / 3
    assert profile["peak_evidence_fraction"][2] == 1.0
    assert profile["query_token_text"][2] == "t3"
    assert profile["dominant_source_position"][2] == 0
    assert profile["dominant_source_token_text"][2] == "t0"
    assert profile["labels_used"] is False


def test_profiler_uses_head_votes_not_gain_magnitude_for_majority_type():
    scan = synthetic_scan()
    row = 1
    scan["event"][0, :, row] = True
    scan["remote_positive_gain"][0, :, row] = [1, 1, 1, 10]
    scan["remote_gain_focality"][0, :, row] = [0.9, 0.9, 0.9, 0.1]
    scan["remote_gain_effective_sources"][0, :, row] = [1, 1, 1, 10]
    scan["remote_gain_distance"][0, :, row] = [8, 8, 8, 6]
    scan["peak_source"][0, :, row] = [0, 0, 0, 1]

    profile = ReanchorProfiler().run(scan)

    assert profile["head_focality"][row] < 0.5
    assert profile["focal_head_fraction"][row] == 0.75
    assert profile["source_agreement"][row] == 0.75
    assert profile["gain_source_agreement"][row] == 10 / 13
    assert profile["dominant_source_position"][row] == 0
    assert profile["reanchor_type"][row] == "broad_convergent"


def test_profiler_rejects_special_tokens_as_reanchor_sources():
    scan = synthetic_scan()
    scan["special_mask"][0] = True

    with pytest.raises(ValueError, match="ordinary peak sources"):
        ReanchorProfiler().run(scan)


def record(source, reanchor_type, labels, predictor_logprob=None):
    rows = np.arange(len(reanchor_type))
    token_count = len(rows) + 1
    eligible = np.zeros(len(rows), dtype=bool)
    eligible[1:] = True
    kinds = np.asarray(reanchor_type)
    is_reanchor = ~np.isin(kinds, ("none", "excluded"))
    metric = np.where(is_reanchor, 0.75, np.nan)
    return {
        "group": "test/QA",
        "source": source,
        "profile": {
            "row_position": rows,
            "eligible": eligible,
            "reanchor_type": kinds,
            "is_reanchor": is_reanchor,
            "active_head_fraction": metric,
            "head_focality": metric,
            "source_agreement": metric,
            "mean_distance": np.where(is_reanchor, 8.0, np.nan),
            "peak_prompt_fraction": metric,
            "peak_history_fraction": np.where(is_reanchor, 0.25, np.nan),
        },
        "labels": np.asarray(labels),
        "response_start": 1,
        "special_mask": np.zeros(token_count, dtype=bool),
        "predictor_logprob": (
            np.full(len(rows), -1.0)
            if predictor_logprob is None
            else np.asarray(predictor_logprob, dtype=float)
        ),
    }


def test_reanchor_incidence_uses_all_tokens_then_equal_weights_sources():
    from experiments.reanchor_flow.message_dag.reanchor_report import summarize_reanchor

    source_one = record(
        "s1",
        ["excluded", *("broad_convergent" for _ in range(10))],
        np.ones(11, dtype=int),
    )
    source_two = record("s2", ["excluded", "none"], np.ones(2, dtype=int))

    report = summarize_reanchor([source_one, source_two], bootstrap=0)
    incidence = report["groups"]["test/QA"]["incidence"]["any_reanchor"]["H"]
    overall = report["groups"]["test/QA"]["incidence"]["any_reanchor"]["all"]

    assert incidence["mean"] == 0.5
    assert incidence["sources"] == 2
    assert incidence["tokens"] == 11
    assert incidence["anchors"] == 10
    assert overall["mean"] == 0.5


def test_overall_incidence_keeps_ordinary_reanchor_before_special_target():
    from experiments.reanchor_flow.message_dag.reanchor_report import summarize_reanchor

    sample = record("s1", ["excluded", "sparse_focal"], [0, 0])
    sample["special_mask"][2] = True

    report = summarize_reanchor([sample], bootstrap=0)
    incidence = report["groups"]["test/QA"]["incidence"]["any_reanchor"]

    assert incidence["all"]["tokens"] == 1
    assert incidence["all"]["anchors"] == 1
    assert incidence["H"]["tokens"] == 0
    assert incidence["N"]["tokens"] == 0


def test_reanchor_types_report_separate_fixed_horizon_outcomes():
    from experiments.reanchor_flow.message_dag.reanchor_report import summarize_reanchor

    sample = record(
        "s1",
        ["excluded", "broad_convergent", "none", "broad_diffuse", "none"],
        [0, 1, 0, 0, 0],
        [np.nan, -3.0, -2.0, -1.0, -4.0],
    )

    report = summarize_reanchor([sample], bootstrap=0)
    downstream = report["groups"]["test/QA"]["downstream"]
    morphology = report["groups"]["test/QA"]["morphology"]

    assert downstream["broad_convergent"]["1"]["hallucination_rate"]["mean"] == 1
    assert (
        downstream["broad_convergent"]["1"]["hallucination_rate_minus_none"][
            "mean"
        ]
        == 1
    )
    assert (
        downstream["broad_convergent"]["1"]["hallucination_rate_minus_none"][
            "sources"
        ]
        == 1
    )
    assert downstream["broad_convergent"]["1"]["negative_logprob"]["mean"] == 3
    assert downstream["broad_diffuse"]["1"]["hallucination_rate"]["mean"] == 0
    assert downstream["broad_diffuse"]["1"]["negative_logprob"]["mean"] == 1
    assert morphology["broad_convergent"]["active_head_fraction"]["mean"] == 0.75
    assert morphology["broad_convergent"]["mean_distance"]["mean"] == 8


def test_downstream_horizons_weight_anchors_before_sources_not_available_tokens():
    from experiments.reanchor_flow.message_dag.reanchor_report import summarize_reanchor

    sample = record(
        "s1",
        [
            "excluded",
            "broad_convergent",
            "none",
            "none",
            "broad_convergent",
            "none",
        ],
        [0, 0, 1, 1, 1, 0],
    )

    report = summarize_reanchor([sample], bootstrap=0)
    outcome = report["groups"]["test/QA"]["downstream"]["broad_convergent"][
        "2-4"
    ]["hallucination_rate"]

    assert outcome["mean"] == 0.5
    assert outcome["anchors"] == 2
    assert outcome["target_tokens"] == 4
