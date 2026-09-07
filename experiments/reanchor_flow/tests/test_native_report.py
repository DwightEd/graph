"""Source weighting, all-mode inference, closure, and diagnostic plot contracts."""

import json

import numpy as np

from experiments.reanchor_flow.native_report import (
    NativeCohort,
    _bh_adjust,
    _source_inference,
    native_transition_examples,
    plot_native_mode_writes,
    render_native_sample,
)


def _sample(patterns, *, scale=1, layers=2, heads=3):
    patterns = np.asarray(patterns)
    n = len(patterns)
    head = np.arange(1, layers * heads + 1).reshape(layers, heads, 1) * np.ones(n) * scale
    mlp = np.ones((layers, n)) * -2 * scale
    attention = head.sum(1)
    stage = np.concatenate((np.zeros((1, n)), np.cumsum(attention + mlp, axis=0)))
    post = stage[:-1] + attention
    trace = {
        "head_margin": head, "mlp_margin": mlp, "attention_margin": attention,
        "stage_margin": stage, "post_attention_margin": post,
        "final_margin": stage[-1], "row_position": np.arange(n) + 5,
        "response_start": np.asarray(6), "token_ids": np.arange(n + 6),
        "edge_margin": head[..., None] * 0.75, "omitted_margin": head * 0.25,
        "edge_source_position": np.zeros((*head.shape, 1), dtype=int),
    }
    prediction = {
        "pattern_id": patterns, "coords": np.column_stack((np.arange(n), -np.arange(n))),
        "distance": np.ones(n), "transition": np.arange(n),
    }
    return trace, prediction


def _add(cohort, sample, source, patterns, labels, *, scale=1, task="QA"):
    trace, prediction = _sample(patterns, scale=scale)
    # All four rows stay in one pre-defined log2 position stratum.
    cohort.add(sample, source, task, np.arange(len(patterns)) + 7, np.asarray(labels), prediction, trace)


def test_source_weighting_and_unknowns_do_not_become_labels():
    cohort = NativeCohort(3, bootstrap=40)
    _add(cohort, "a", "source-a", [0, 1, 0, 0], [0, 1, -1, -1], scale=1)
    _add(cohort, "b", "source-a", [0, 1, 0, 0], [0, 1, -1, -1], scale=3)
    _add(cohort, "c", "source-b", [1, 0, 0, 0], [0, 1, -1, -1], scale=10)
    report = cohort.report()
    task = report["tasks"]["QA"]
    assert task["paired_sources"] == 2
    assert task["paired_samples"] == 3
    assert task["patterns"][0]["position_matched_h_minus_n"] == 0
    assert task["patterns"][0]["tokens_unknown"] == 6
    # Mode zero is in every sample. Sample means 1,3 first form source mean 2;
    # source means 2 and 10 then receive equal weight: head L0H0 = 6.
    assert task["patterns"][0]["signed_means"]["head_margin"][0][0] == 6
    assert task["patterns"][2]["containing_sources"] == 0
    assert task["patterns"][2]["position_matched_h_minus_n"] == 0
    assert task["patterns"][2]["signflip_p"] == 1
    assert task["patterns"][2]["bh_q"] == 1
    assert task["patterns"][2]["signed_means"]["head_margin"] is None
    json.dumps(report, allow_nan=False)


def test_no_mixed_label_stratum_reports_no_evidence_including_absent_modes():
    cohort = NativeCohort(4, bootstrap=20)
    trace, prediction = _sample([0, 1, 0, 0])
    # Both labels exist, but never inside the same position stratum.
    cohort.add("a", "s", "QA", [0, 1, 2, 3], [0, 1, 1, 0], prediction, trace)
    report = cohort.report()["tasks"]["QA"]
    assert report["paired_sources"] == 0
    assert len(report["patterns"]) == 4
    for mode in report["patterns"]:
        assert mode["position_matched_h_minus_n"] is None
        assert mode["signflip_p"] is None
        assert mode["bh_q"] is None


def test_single_source_does_not_produce_inferential_interval():
    cohort = NativeCohort(2, bootstrap=20)
    _add(cohort, "a", "s", [0, 1], [0, 1])
    mode = cohort.report()["tasks"]["QA"]["patterns"][1]
    assert mode["position_matched_h_minus_n"] == 1
    assert mode["ci_lower"] is None
    assert mode["signflip_p"] is None


def test_signflip_is_two_sided_and_keeps_zero_modes_in_bh_family():
    values = np.tile([0.4, -0.4, 0], (6, 1))
    mean, low, high, p = _source_inference(values, seed=11, bootstrap=40)
    np.testing.assert_allclose(mean, [0.4, -0.4, 0])
    np.testing.assert_allclose(low, mean)
    np.testing.assert_allclose(high, mean)
    np.testing.assert_allclose(p, [2 / 64, 2 / 64, 1])
    np.testing.assert_allclose(_bh_adjust(p), [3 / 64, 3 / 64, 1])


def test_scalar_closure_reports_mismatch_without_semantic_claim():
    cohort = NativeCohort(2, bootstrap=20)
    trace, prediction = _sample([0, 1])
    trace["omitted_margin"] = trace["omitted_margin"].copy()
    trace["omitted_margin"][0, 0, 0] += 0.25
    cohort.add("a", "s", "QA", [7, 8], [0, 1], prediction, trace)
    errors = cohort.report()["numerical_closure"]["raw_max_abs"]
    assert errors["stored_plus_omitted_edges_to_head"] == 0.25
    assert errors["mlp_residual_update"] == 0
    assert errors["final_stage_to_margin"] == 0


def test_same_source_is_pooled_across_tasks_before_inference():
    cohort = NativeCohort(2, bootstrap=20)
    _add(cohort, "a", "same-id", [0, 1], [0, 1], task="QA")
    _add(cohort, "b", "same-id", [1, 0], [0, 1], task="Summary")
    report = cohort.report()
    assert report["tasks"]["ALL"]["sources"] == 1
    assert report["tasks"]["ALL"]["patterns"][0]["position_matched_h_minus_n"] == 0
    assert report["tasks"]["ALL"]["patterns"][0]["signflip_p"] is None


def test_recorded_rounding_and_bias_close_without_hiding_raw_discrepancy():
    cohort = NativeCohort(2, bootstrap=20)
    trace, prediction = _sample([0, 1])
    trace["attention_margin"] = trace["attention_margin"] + 0.25
    trace["head_remainder_margin"] = np.full((2, 2), 0.25)
    trace["attention_add_remainder_margin"] = np.full((2, 2), -0.25)
    trace["readout_bias"] = np.ones(2)
    trace["readout_remainder"] = np.full(2, 0.125)
    trace["final_margin"] = trace["final_margin"] + 1.125
    cohort.add("a", "s", "QA", [7, 8], [0, 1], prediction, trace)
    result = cohort.report()["numerical_closure"]
    assert result["raw_max_abs"]["head_sum_to_attention"] == 0.25
    assert result["raw_max_abs"]["final_stage_to_margin"] == 1.125
    assert max(result["remainder_corrected_max_abs"].values()) == 0


def test_report_uses_labels_only_for_association_not_mode_explanation():
    reports = []
    for labels in ([0, 1], [1, 0]):
        cohort = NativeCohort(2, bootstrap=20)
        _add(cohort, "a", "s", [0, 1], labels)
        reports.append(cohort.report()["tasks"]["QA"]["patterns"])
    assert reports[0][0]["signed_means"] == reports[1][0]["signed_means"]
    assert reports[0][0]["position_matched_h_minus_n"] == -reports[1][0]["position_matched_h_minus_n"]


def test_transitions_respect_gaps_and_source_weighting():
    cohort = NativeCohort(2, bootstrap=20)
    trace, prediction = _sample([0, 1, 0, 1])
    prediction["has_previous"] = np.asarray([False, True, False, True])
    cohort.add("a", "source-a", "QA", np.arange(4), [0, 1, 0, 1], prediction, trace)
    _add(cohort, "b", "source-a", [0, 1], [0, 1])
    _add(cohort, "c", "source-b", [1, 0], [0, 1])
    task = cohort.report()["tasks"]["QA"]
    assert task["transition_counts"] == [[0, 3], [1, 0]]
    assert task["transition_pairs"] == 4
    assert task["transition_sources"] == 2
    np.testing.assert_allclose(task["source_balanced_transition_frequency"], [[0, 0.5], [0.5, 0]])


def test_transition_examples_preserve_signed_vectors_and_exclude_gaps():
    trace, prediction = _sample([0, 1, 0, 1, 0, 1])
    trace["head_sketch"] = np.zeros((2, 3, 6, 2))
    trace["mlp_sketch"] = np.zeros((2, 6, 2))
    trace["head_sketch"][1, 2, 5] = [-3, 4]
    trace["mlp_sketch"][0, 5] = [2, -2]
    prediction["has_previous"] = np.asarray([False, True, True, False, True, True])
    prediction["transition"] = np.asarray([0, 1, 2, 20, 4, 10])
    examples = native_transition_examples(trace, prediction, separation=3)
    assert [item["response_index"] for item in examples] == [5, 2]
    assert examples[0]["before_pattern_id"] == 0
    assert examples[0]["after_pattern_id"] == 1
    assert examples[0]["largest_head_sketch_change"] == {
        "layer": 1, "head": 2, "norm": 5, "signed_delta": [-3, 4],
    }
    assert examples[0]["largest_mlp_sketch_change"]["signed_delta"] == [2, -2]


def test_render_native_sample_retains_heads_and_uses_label_free_focus(tmp_path):
    trace, prediction = _sample(np.arange(150) % 3)
    trace["token_text"] = np.asarray([f"token-{i}" for i in trace["token_ids"]])
    result_a = render_native_sample(tmp_path / "native_a.png", trace, prediction, labels=np.zeros(150))
    result_b = render_native_sample(tmp_path / "native_b.png", trace, prediction, labels=np.ones(150))
    assert result_a == result_b
    assert result_a["focus_response_index"] == 149
    assert len(result_a["displayed_layer_head"]) == 6
    assert result_a["edges"][0]["layer"] == 1
    assert result_a["edges"][0]["source_token"] == "token-0"
    assert result_a["focus_token_text"] == "token-155"
    assert (tmp_path / "native_a.png").stat().st_size > 1000
    cohort = NativeCohort(3, bootstrap=20)
    _add(cohort, "a", "s", [0, 1], [0, 1])
    cohort.plot(tmp_path / "cohort.png")
    assert (tmp_path / "cohort.png").exists()
    plot_native_mode_writes(tmp_path / "mode_writes.png", cohort.report())
    assert (tmp_path / "mode_writes.png").stat().st_size > 1000
