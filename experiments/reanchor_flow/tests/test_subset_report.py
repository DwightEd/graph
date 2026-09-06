from __future__ import annotations

import json

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
    return {
        "subset_audit_schema": 2,
        "dataset_sample_id": "sample-1",
        "task_type": "QA",
        "response_start": 4,
        "prediction_position": 5,
        "query_position": 4,
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
    }


def complete_capture(tmp_path):
    dataset_root = tmp_path / "cache"
    dataset_root.mkdir()
    output = tmp_path / "output"
    result = output / "audits" / "sample-1.npz"
    result.parent.mkdir(parents=True)
    np.savez_compressed(result, **_artifact_arrays())
    manifest = {
        "subset_manifest_schema": 2,
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

    class Sample:
        def release_attention(self):
            pass

    class Labels:
        @staticmethod
        def response_labels(_sample):
            return torch.tensor([0, 1])

    class Dataset:
        def __getitem__(self, sample_id):
            assert sample_id == "sample-1"
            return Sample()

        @staticmethod
        def prepare_evaluation_labels(sample_ids):
            assert sample_ids == ["sample-1"]
            return Labels()

    def open_dataset(*_args, **kwargs):
        assert capture_loaded
        assert kwargs["retain_embedded_labels"] is True
        return Dataset()

    monkeypatch.setattr(subset_report, "_capture_rows", capture_rows)
    monkeypatch.setattr(subset_report, "open_research_dataset", open_dataset)
    report = subset_report.evaluate_subset_split(dataset_root, output)

    hallucinated = report["groups"]["QA"]["hallucinated"]
    assert hallucinated["targets"] == 1
    assert hallucinated["confirmation_rate"]["corridor_confirmed"] == 1.0
    assert report["targets"][0]["native_source_mediated_observed_margin"] == pytest.approx(
        0.6
    )
    assert report["targets"][0][
        "response_origin_supporting_action_candidate"
    ] == pytest.approx(
        0.38
    )

    text = (output / "mechanism_evaluation.json").read_text(encoding="utf-8")
    assert "NaN" not in text
    assert json.loads(text)["groups"]["QA"]["clean"]["targets"] == 0


def test_raw_axis_evaluation_keeps_registered_axes_separate() -> None:
    rows = [
        {
            "hallucination_label": 0,
            "native_source_mediated_observed_margin": 1.2,
            "response_origin_supporting_action_candidate": 0.1,
        },
        {
            "hallucination_label": 1,
            "native_source_mediated_observed_margin": 0.1,
            "response_origin_supporting_action_candidate": 1.3,
        },
    ]
    metrics = subset_report.raw_axis_evaluation(rows)
    assert set(metrics) == set(subset_report.AXIS_DIRECTION)
    assert all(metric["auroc"] == 1.0 for metric in metrics.values())


def test_mechanism_axes_use_signed_head_agreement_not_head_average() -> None:
    artifact = _artifact_arrays()
    action = np.asarray(artifact["route_head_action"]).copy()
    action[:, :, 0, 2] = [[1.0, -1.0], [1.0, -1.0]]
    artifact["route_head_action"] = action
    axes = subset_report.mechanism_axes(artifact)
    assert axes["response_origin_action_absolute_budget"] == 4.0
    assert axes["response_origin_action_signed_sum"] == 0.0
    assert axes["response_origin_functional_agreement"] == 0.0
    assert axes["response_origin_supporting_action_candidate"] == 0.0


def test_response_action_axis_does_not_mix_selected_source_continuity() -> None:
    artifact = _artifact_arrays()
    baseline = subset_report.mechanism_axes(artifact)[
        "response_origin_supporting_action_candidate"
    ]
    artifact["route_state_continuity"] = np.asarray(
        [[-1.0], [-1.0]], dtype=np.float32
    )
    changed = subset_report.mechanism_axes(artifact)[
        "response_origin_supporting_action_candidate"
    ]
    assert changed == baseline


def test_unconfirmed_exact_effect_remains_diagnosable_but_is_not_accepted() -> None:
    artifact = _artifact_arrays()
    artifact["corridor_confirmed"] = False
    axes = subset_report.mechanism_axes(artifact)
    assert axes["native_source_exact_bottleneck_ungated"] == pytest.approx(0.6)
    assert axes["native_source_support_gate"] is False
    assert axes["native_source_mediated_observed_margin"] == 0.0


def test_subset_evaluation_refuses_incomplete_capture_before_labels(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "subset_manifest_schema": 2,
                "analysis_complete": False,
                "labels_used_for_capture": False,
            }
        ),
        encoding="utf-8",
    )

    def reject_labels(*_args, **_kwargs):
        raise AssertionError("labels opened before capture completion")

    monkeypatch.setattr(subset_report, "open_research_dataset", reject_labels)
    with pytest.raises(ValueError, match="incomplete"):
        subset_report.evaluate_subset_split(tmp_path / "cache", output)


def test_subset_evaluation_requires_capture_dataset_root(
    tmp_path, monkeypatch
) -> None:
    _dataset_root, output = complete_capture(tmp_path)
    other_dataset_root = tmp_path / "other-cache"
    other_dataset_root.mkdir()

    def reject_capture(*_args, **_kwargs):
        raise AssertionError("artifacts loaded before dataset identity check")

    def reject_labels(*_args, **_kwargs):
        raise AssertionError("labels opened before dataset identity check")

    monkeypatch.setattr(subset_report, "_capture_rows", reject_capture)
    monkeypatch.setattr(subset_report, "open_research_dataset", reject_labels)
    with pytest.raises(ValueError, match="dataset_root differs from capture"):
        subset_report.evaluate_subset_split(other_dataset_root, output)
