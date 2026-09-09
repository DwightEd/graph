from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from experiments.reanchor_flow import subset_report


def _artifact_arrays() -> dict[str, object]:
    transport = np.zeros((2, 2, 1, 4), dtype=np.float32)
    transport[:, :, 0, 0] = [[0.3, 0.2], [0.4, 0.1]]
    action = np.zeros_like(transport)
    action[:, :, 0, 0] = [[0.4, -0.1], [0.3, 0.2]]
    action[:, :, 0, 2] = [[0.1, 0.1], [0.4, -0.1]]
    integration = np.zeros_like(transport)
    integration[:, :, 0, 3] = [[0.3, -0.1], [0.2, 0.1]]
    reanchor_score = np.asarray([[[0.2], [0.4]], [[0.1], [0.3]]], dtype=np.float32)
    bucket_action = np.zeros((2, 2, 1, 4), dtype=np.float32)
    bucket_action[0, 1, 0, 0] = 0.55
    return {
        "subset_audit_schema": subset_report.AUDIT_SCHEMA,
        "dataset_sample_id": "sample-1",
        "task_type": "QA",
        "response_start": 4,
        "prediction_position": 5,
        "query_position": 4,
        "selected_root_evaluated": True,
        "corridor_evaluated": True,
        "carrier_evaluated_count": 0,
        "full_chain_evaluated": False,
        "selected_root_confirmed": True,
        "corridor_confirmed": True,
        "corridor_restoration_valid": True,
        "carrier_any_confirmed": False,
        "full_chain_confirmed": False,
        "root_value_effect": 1.0,
        "selected_root_value_necessity": 0.9,
        "selected_root_causal_score": 0.7,
        "corridor_necessity": 0.8,
        "corridor_conditional_rescue": 0.7,
        "corridor_mediated_rescue": 0.6,
        "route_row_position": np.asarray([4], dtype=np.int32),
        "route_row_total": np.ones((2, 2, 1), dtype=np.float32),
        "route_row_retained": np.full((2, 2, 1), 0.75, dtype=np.float32),
        "route_head_transport": transport,
        "route_head_action": action,
        "route_head_integration": integration,
        "route_stage_position": np.asarray([4], dtype=np.int32),
        "route_state_continuity": np.asarray([[0.5], [0.8]], dtype=np.float32),
        "route_cross_head_vector_coherence": np.asarray(
            [[0.6], [0.8]], dtype=np.float32
        ),
        "route_cross_head_functional_agreement": np.asarray(
            [[0.5], [0.9]], dtype=np.float32
        ),
        "route_module_functional_agreement": np.asarray(
            [[0.7], [0.9]], dtype=np.float32
        ),
        "route_module_vector_cosine": np.asarray([[0.2], [-0.3]], dtype=np.float32),
        "target_reanchor_selection_recorded": True,
        "target_reanchor_policy": "reanchor",
        "target_reanchor_has_event": True,
        "target_reanchor_is_center": True,
        "target_reanchor_fallback": False,
        "target_reanchor_center_position": 4,
        "target_reanchor_window_offset": 0,
        "target_reanchor_layer": 0,
        "target_reanchor_head": 1,
        "target_reanchor_source_kind": "prompt_evidence",
        "target_reanchor_source_position": 1,
        "target_reanchor_source_unit_id": 1,
        "target_reanchor_score": 0.4,
        "target_reanchor_support": 2,
        "reanchor_score": reanchor_score,
        "reanchor_bucket_name": np.asarray(
            ["prompt_evidence", "other_prompt", "remote_response", "recent_local"]
        ),
        "reanchor_bucket_downstream_action": bucket_action,
        # The earlier candidate has a larger score on purpose.  Evaluation
        # must use only the dense row at this artifact's own query.
        "reanchor_candidate_position": np.asarray([3, 4], dtype=np.int32),
        "reanchor_candidate_score": np.asarray([0.99, 0.4], dtype=np.float32),
        "reanchor_candidate_current_target_match": np.asarray(
            [False, True], dtype=bool
        ),
        "reanchor_candidate_anchor_downstream_action": np.asarray(
            [1000.0, 0.5], dtype=np.float32
        ),
    }


def complete_capture(tmp_path):
    dataset_root = tmp_path / "cache"
    dataset_root.mkdir()
    output = tmp_path / "output"
    result = output / "audits" / "sample-1.npz"
    result.parent.mkdir(parents=True)
    np.savez_compressed(result, **_artifact_arrays())
    manifest = {
        "subset_manifest_schema": subset_report.MANIFEST_SCHEMA,
        "config": {"split": "test", "dataset_root": str(dataset_root.resolve())},
        "selection": [{"sample_id": "sample-1", "task_type": "QA"}],
        "analysis_complete": True,
        "labels_used_for_capture": False,
        "samples": {"sample-1": {"targets": [4]}},
        "audits": {
            "sample-1:q4": {
                "result": "audits/sample-1.npz",
                "sample_id": "sample-1",
                "query_position": 4,
                "positive_token_id": 9,
                "negative_token_id": 8,
            }
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset_root, output


def test_subset_evaluation_joins_labels_after_capture(tmp_path, monkeypatch) -> None:
    dataset_root, output = complete_capture(tmp_path)
    capture_loaded = False
    original_capture_rows = subset_report._capture_rows

    def capture_rows(*args):
        nonlocal capture_loaded
        rows = original_capture_rows(*args)
        capture_loaded = True
        return rows

    class Labels:
        @staticmethod
        def load(sample_ids):
            assert capture_loaded
            assert sample_ids == ("sample-1",)
            return {"sample-1": np.asarray([0, 1])}

    monkeypatch.setattr(subset_report, "_capture_rows", capture_rows)
    report = subset_report.evaluate_subset_split(
        dataset_root,
        output,
        label_source=Labels(),
    )

    hallucinated = report["groups"]["QA"]["hallucinated"]
    assert hallucinated["targets"] == 1
    assert hallucinated["confirmation_rate"]["corridor_confirmed"] == 1.0
    assert report["targets"][0]["route_evidence_origin_action"] == pytest.approx(0.68)
    assert report["targets"][0]["route_response_origin_action"] == pytest.approx(0.38)
    assert report["targets"][0]["route_origin_competition"] == pytest.approx(
        (0.38 - 0.68) / 1.7
    )
    assert report["targets"][0]["temporal_switch_score"] == pytest.approx(0.4)
    assert report["targets"][0]["target_reanchor_is_center"] is True
    assert report["targets"][0]["temporal_switch_source_kind"] == "prompt_evidence"
    assert report["targets"][0]["evidence_adoption"] == pytest.approx(0.55)
    assert report["targets"][0]["route_origin_competition_evaluated"] is True
    assert report["targets"][0]["route_query_retained_fraction"] == 0.75
    assert report["targets"][0]["route_query_unobserved_fraction"] == 0.25
    assert report["targets"][0]["selected_root_exact_bottleneck"] == 0.6
    metric = report["groups"]["QA"]["raw_axis_evaluation"]["route_origin_competition"]
    assert metric["evaluated_targets"] == 1
    assert metric["auroc"] is None
    assert metric["auprc"] is None
    assert set(report["groups"]["QA"]["raw_axis_evaluation"]) == {
        "route_origin_competition",
        "temporal_switch_score",
        "evidence_adoption",
    }
    assert report["axis_role"]["route_origin_competition"] == "static baseline"
    assert "direction-neutral" in report["axis_role"]["temporal_switch_score"]
    stratified = report["groups"]["QA"]["temporal_switch_by_source_kind"]
    assert stratified["prompt_evidence"]["evaluated_targets"] == 1
    assert stratified["other_prompt"]["evaluated_targets"] == 0
    assert stratified["remote_response"]["evaluated_targets"] == 0

    text = (output / "mechanism_evaluation.json").read_text(encoding="utf-8")
    assert "NaN" not in text
    assert json.loads(text)["groups"]["QA"]["clean"]["targets"] == 0


def test_raw_axis_evaluation_keeps_registered_axes_separate() -> None:
    rows = [
        {
            "hallucination_label": 0,
            "route_origin_competition": -0.4,
            "temporal_switch_score": 0.8,
            "evidence_adoption": 0.5,
        },
        {
            "hallucination_label": 1,
            "route_origin_competition": 0.7,
            "temporal_switch_score": 0.1,
            "evidence_adoption": -0.2,
        },
    ]
    metrics = subset_report.raw_axis_evaluation(rows)
    assert set(metrics) == {
        "route_origin_competition",
        "temporal_switch_score",
        "evidence_adoption",
    }
    assert metrics["route_origin_competition"]["auroc"] == 1.0
    assert metrics["route_origin_competition"]["auprc"] == 1.0
    assert metrics["evidence_adoption"]["auroc"] == 1.0
    assert metrics["evidence_adoption"]["auprc"] == 1.0
    temporal = metrics["temporal_switch_score"]
    assert temporal["evaluated_targets"] == 2
    assert temporal["hallucination_direction"] == (
        "neutral_raw_higher_reporting_convention"
    )
    assert temporal["auroc"] == 0.0
    assert temporal["negated_auroc"] == 1.0
    assert temporal["negated_auprc"] == 1.0


def test_mechanism_axes_use_signed_head_agreement_not_head_average() -> None:
    artifact = _artifact_arrays()
    action = np.asarray(artifact["route_head_action"]).copy()
    action[:, :, 0, 2] = [[1.0, -1.0], [1.0, -1.0]]
    artifact["route_head_action"] = action
    axes = subset_report.mechanism_axes(artifact)
    assert axes["route_response_origin_absolute_budget"] == 4.0
    assert axes["route_response_origin_signed_sum"] == 0.0
    assert axes["route_response_origin_head_agreement"] == 0.0
    assert axes["route_response_origin_action"] == 0.0


def test_zero_action_budget_is_not_an_evaluated_route_axis() -> None:
    artifact = _artifact_arrays()
    artifact["route_head_action"] = np.zeros_like(artifact["route_head_action"])
    axes = subset_report.mechanism_axes(artifact)
    assert axes["route_origin_competition_evaluated"] is False
    assert math.isnan(axes["route_origin_competition"])
    # Adoption comes from the unpruned prompt bucket, not the sparse route budget.
    assert axes["evidence_adoption_evaluated"] is True
    assert axes["evidence_adoption"] == pytest.approx(0.55)
    metric = subset_report.raw_axis_evaluation([{**axes, "hallucination_label": 1}])[
        "route_origin_competition"
    ]
    assert metric["total_targets"] == 1
    assert metric["evaluated_targets"] == 0


def test_temporal_axis_requires_the_selected_event_center() -> None:
    artifact = _artifact_arrays()
    artifact["target_reanchor_is_center"] = False
    artifact["target_reanchor_center_position"] = 3
    artifact["target_reanchor_window_offset"] = 1
    axes = subset_report.mechanism_axes(artifact)
    assert axes["target_reanchor_has_event"] is True
    assert axes["target_reanchor_is_center"] is False
    assert axes["temporal_switch_evaluated"] is False
    assert math.isnan(axes["temporal_switch_score"])
    assert axes["evidence_adoption_evaluated"] is False
    assert math.isnan(axes["evidence_adoption"])


def test_temporal_axis_excludes_no_event_fallback() -> None:
    artifact = _artifact_arrays()
    artifact.update(
        {
            "target_reanchor_has_event": False,
            "target_reanchor_is_center": False,
            "target_reanchor_fallback": True,
            "target_reanchor_center_position": -1,
            "target_reanchor_layer": -1,
            "target_reanchor_head": -1,
            "target_reanchor_source_kind": "none",
            "target_reanchor_score": 0.0,
        }
    )
    axes = subset_report.mechanism_axes(artifact)
    assert axes["target_reanchor_fallback"] is True
    assert axes["temporal_switch_evaluated"] is False
    assert math.isnan(axes["temporal_switch_score"])
    assert axes["evidence_adoption_evaluated"] is False


@pytest.mark.parametrize("recomputed", [0.4001, 0.0])
def test_temporal_axis_preserves_frozen_score_under_prefix_drift(recomputed) -> None:
    artifact = _artifact_arrays()
    artifact["reanchor_score"][0, 1, 0] = recomputed
    axes = subset_report.mechanism_axes(artifact)
    assert axes["temporal_switch_score"] == 0.4
    assert axes["temporal_switch_recomputed_score"] == pytest.approx(recomputed)
    assert axes["temporal_switch_score_delta"] == pytest.approx(
        recomputed - 0.4, abs=np.finfo(np.float32).eps
    )
    assert axes["temporal_switch_evaluated"] is True
    assert axes["evidence_adoption"] == pytest.approx(0.55)


def test_temporal_axis_rejects_mislabeled_current_target_candidate() -> None:
    artifact = _artifact_arrays()
    artifact["reanchor_candidate_position"] = np.asarray([3], dtype=np.int32)
    artifact["reanchor_candidate_current_target_match"] = np.asarray([True])
    with pytest.raises(ValueError, match="another position"):
        subset_report.mechanism_axes(artifact)


def test_adoption_is_defined_only_for_prompt_evidence_centers() -> None:
    artifact = _artifact_arrays()
    artifact["target_reanchor_source_kind"] = "remote_response"
    axes = subset_report.mechanism_axes(artifact)
    assert axes["temporal_switch_evaluated"] is True
    assert axes["temporal_switch_source_kind"] == "remote_response"
    assert axes["evidence_adoption_evaluated"] is False
    assert math.isnan(axes["evidence_adoption"])


def test_route_origin_axis_does_not_mix_selected_source_continuity() -> None:
    artifact = _artifact_arrays()
    baseline = subset_report.mechanism_axes(artifact)["route_origin_competition"]
    artifact["route_state_continuity"] = np.asarray([[-1.0], [-1.0]], dtype=np.float32)
    changed = subset_report.mechanism_axes(artifact)["route_origin_competition"]
    assert changed == baseline


def test_unevaluated_exact_effect_is_missing_not_zero() -> None:
    artifact = _artifact_arrays()
    artifact["corridor_evaluated"] = False
    axes = subset_report.mechanism_axes(artifact)
    assert math.isnan(axes["selected_root_exact_bottleneck"])
    assert axes["selected_root_exact_evaluated"] is False


def test_evaluated_but_unconfirmed_exact_effect_stays_separate() -> None:
    artifact = _artifact_arrays()
    artifact["corridor_confirmed"] = False
    axes = subset_report.mechanism_axes(artifact)
    assert axes["selected_root_exact_bottleneck"] == pytest.approx(0.6)
    assert axes["selected_root_exact_evaluated"] is True


def test_json_rows_encode_missing_exact_value_as_null() -> None:
    rows = subset_report._json_rows([{"selected_root_exact_bottleneck": float("nan")}])
    assert rows == [{"selected_root_exact_bottleneck": None}]


def test_confirmation_rates_use_only_evaluated_targets() -> None:
    not_run = {
        "selected_root_evaluated": False,
        "corridor_evaluated": False,
        "carrier_evaluated": False,
        "full_chain_evaluated": False,
        **{name: False for name in subset_report.CONFIRMATION_FIELDS},
    }
    rates, evaluated = subset_report._confirmation_rates([not_run])
    assert all(value is None for value in rates.values())
    assert all(value == 0 for value in evaluated.values())

    confirmed = {
        **not_run,
        "selected_root_evaluated": True,
        "selected_root_confirmed": True,
    }
    rates, evaluated = subset_report._confirmation_rates([not_run, confirmed])
    assert rates["selected_root_confirmed"] == 1.0
    assert evaluated["selected_root_confirmed"] == 1
    assert rates["corridor_confirmed"] is None


def test_subset_evaluation_refuses_incomplete_capture_before_labels(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "subset_manifest_schema": subset_report.MANIFEST_SCHEMA,
                "analysis_complete": False,
                "labels_used_for_capture": False,
            }
        ),
        encoding="utf-8",
    )

    class RejectLabels:
        @staticmethod
        def load(*_args):
            raise AssertionError("labels opened before capture completion")

    with pytest.raises(ValueError, match="incomplete"):
        subset_report.evaluate_subset_split(
            tmp_path / "cache",
            output,
            label_source=RejectLabels(),
        )


def test_subset_evaluation_requires_capture_dataset_root(tmp_path, monkeypatch) -> None:
    _dataset_root, output = complete_capture(tmp_path)
    other_dataset_root = tmp_path / "other-cache"
    other_dataset_root.mkdir()

    def reject_capture(*_args, **_kwargs):
        raise AssertionError("artifacts loaded before dataset identity check")

    monkeypatch.setattr(subset_report, "_capture_rows", reject_capture)
    with pytest.raises(ValueError, match="dataset_root differs from capture"):
        subset_report.evaluate_subset_split(
            other_dataset_root,
            output,
            label_source=object(),
        )
