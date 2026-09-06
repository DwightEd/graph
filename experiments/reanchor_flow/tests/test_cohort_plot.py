from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.reanchor_flow.cohort_plot import (
    BUCKETS,
    GROUPS,
    HeadCohort,
    plot_cohort_group,
    scan_observation,
    summarize_cohort,
)


def _scan(sample_id="s1"):
    transport = np.zeros((1, 2, 4, 4), dtype=np.float32)
    # Heads have opposite routing preferences; averaging them would erase the signal.
    transport[0, 0, :, 0] = [0.2, 0.8, 0.4, 0.6]
    transport[0, 0, :, 3] = 1 - transport[0, 0, :, 0]
    transport[0, 1, :, 0] = 1 - transport[0, 0, :, 0]
    transport[0, 1, :, 3] = transport[0, 0, :, 0]
    score = np.array([[[0, 0.6, 0, 0.2], [0, 0, 0.2, 0]]])
    return {
        "dataset_sample_id": sample_id,
        "task_type": "QA",
        "response_start": 10,
        "route_row_position": np.array([9, 10, 11, 12]),
        "reanchor_bucket_name": np.array(BUCKETS),
        "reanchor_bucket_transport": transport,
        "reanchor_score": score,
    }


def test_scan_uses_next_token_labels_and_keeps_heads_separate():
    means, coverage = scan_observation(_scan(), np.array([0, 1, 0, 1]))
    assert means.shape == (2, 5, 1, 2)
    np.testing.assert_allclose(means[0, 0, 0], [0.3, 0.7], atol=1e-7)
    np.testing.assert_allclose(means[1, 0, 0], [0.7, 0.3], atol=1e-7)
    # q=9 has no previous captured row and is excluded from switch-rate denominator.
    np.testing.assert_allclose(means[0, 4, 0], [0, 1])
    np.testing.assert_allclose(means[1, 4, 0], [1, 0])
    assert coverage["full_response_covered"] is True
    assert coverage["scanned_response_tokens"] == 4
    assert coverage["mixed_scanned_sample"] is True


def test_head_cohort_pairs_within_sample_instead_of_between_group_populations():
    moments = HeadCohort(("evidence",))
    moments.add(np.array([[[[0.2, 0.8]]], [[[0.8, 0.2]]]]))
    moments.add(np.array([[[[1.0, 1.0]]], [[[np.nan, np.nan]]]]))
    report = moments.summary()["metrics"]["evidence"]
    np.testing.assert_allclose(report[GROUPS[0]]["mean"], [[0.6, 0.9]])
    np.testing.assert_allclose(report[GROUPS[1]]["mean"], [[0.8, 0.2]])
    paired = report["within_sample_hallucinated_minus_nonhallucinated"]
    np.testing.assert_allclose(paired["mean"], [[0.6, -0.6]])
    assert paired["sample_count"] == [[1, 1]]
    assert report[GROUPS[0]]["sample_count"] == [[2, 2]]


def test_zero_transport_and_unannotated_rows_are_missing_not_clean():
    scan = _scan()
    scan["reanchor_bucket_transport"][0, 1] = 0
    means, coverage = scan_observation(scan, np.array([0, 1, -1, 1, 0]))
    assert np.isnan(means[:, :, 0, 1]).all()
    assert coverage["unannotated_scanned_tokens"] == 1
    assert coverage["scanned_nonhallucinated_tokens"] == 1
    assert coverage["full_response_tokens"] == 5
    assert coverage["annotated_response_tokens"] == 4
    assert not coverage["full_response_covered"]


def test_gaps_are_not_interpreted_as_adjacent_temporal_switches():
    scan = _scan()
    scan["route_row_position"] = np.array([9, 11, 12, 13])
    scan["reanchor_score"][:] = 1
    means, _ = scan_observation(scan, np.array([0, 0, 0, 1, 1]))
    assert np.isnan(means[0, 4]).all()
    np.testing.assert_allclose(means[1, 4], [[1, 1]])


def _save_cohort(tmp_path, n=3):
    samples = {}
    selection = []
    labels = {}
    for index in range(n):
        sample_id = f"s{index}"
        file_name = f"{sample_id}.npz"
        np.savez_compressed(tmp_path / file_name, **_scan(sample_id))
        samples[sample_id] = {"scan": file_name, "task_type": "QA"}
        selection.append({"sample_id": sample_id, "task_type": "QA"})
        labels[sample_id] = np.array([0, 1, 0, 1])
    return {"samples": samples, "selection": selection, "audits": {}}, labels


def test_cohort_full_scan_coverage_does_not_depend_on_selected_targets(tmp_path):
    manifest, labels = _save_cohort(tmp_path)
    report = summarize_cohort(tmp_path, manifest, labels, target_rows=[])
    group = report["groups"]["QA"]
    assert group["coverage"]["scanned_response_tokens"] == 12
    assert group["coverage"]["mixed_scanned_samples"] == 3
    assert group["coverage"]["response_token_coverage"] == 1
    assert group["selected_samples"] == 3
    assert group["functional"]["selected_targets"] == dict.fromkeys(GROUPS, 0)
    assert report["samples_without_full_scan"] == 0
    saved = json.loads((tmp_path / "cohort_summary.json").read_text())
    assert saved["groups"]["ALL"]["structure"]["sample_count"] == 3
    assert saved["groups"]["QA"]["structure"]["metrics"]["prompt_evidence"]


def test_missing_full_scan_is_not_filled_using_selected_target_prefix(tmp_path):
    manifest = {
        "samples": {"s1": {"task_type": "QA"}},
        "selection": [{"sample_id": "s1", "task_type": "QA"}],
        "audits": {},
    }
    report = summarize_cohort(
        tmp_path, manifest, {"s1": np.array([0, 1])}, target_rows=[]
    )
    assert report["samples_without_full_scan"] == 1
    assert report["groups"]["ALL"]["coverage"]["scanned_response_tokens"] == 0
    assert report["groups"]["ALL"]["coverage"]["response_token_coverage"] is None


def test_structural_plot_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    manifest, labels = _save_cohort(tmp_path)
    report = summarize_cohort(tmp_path, manifest, labels, target_rows=[])
    paths = plot_cohort_group(tmp_path, "QA", report["groups"]["QA"])
    assert paths == ["cohort_QA.png"]
    assert (tmp_path / paths[0]).stat().st_size > 10_000


def test_functional_report_uses_own_query_and_fixed_cut_without_confirmation(tmp_path):
    from experiments.reanchor_flow.tests.test_subset_report import _artifact_arrays

    manifest, labels = _save_cohort(tmp_path)
    target_rows = []
    for sample_id in labels:
        for label in (0, 1):
            artifact = _artifact_arrays()
            artifact["dataset_sample_id"] = sample_id
            artifact["query_position"] = 10 + label
            artifact["route_row_position"] = np.array([9, 10 + label])
            artifact["route_stage_position"] = np.array([10 + label])
            action = np.full((2, 2, 2, 4), 999.0)
            action[:, :, 1, :] = 1 + label
            action[0, 1, 1, 0] = -2 * (1 + label)
            artifact["reanchor_bucket_downstream_action"] = action
            artifact["root_value_effect"] = 0.2 + label
            artifact["selected_root_evaluated"] = False
            file_name = f"{sample_id}_q{10 + label}.npz"
            np.savez_compressed(tmp_path / file_name, **artifact)
            manifest["audits"][file_name] = {"result": file_name}
            target_rows.append(
                {
                    "sample_id": sample_id,
                    "query_position": 10 + label,
                    "hallucination_label": label,
                }
            )
    report = summarize_cohort(tmp_path, manifest, labels, target_rows=target_rows)
    functional = report["groups"]["QA"]["functional"]
    assert functional["selected_targets"] == dict.fromkeys(GROUPS, 3)
    action = functional["bucket_immediate_action"]["metrics"]["prompt_evidence"]
    np.testing.assert_allclose(action[GROUPS[0]]["mean"], [[1, -2], [1, 1]])
    np.testing.assert_allclose(action[GROUPS[1]]["mean"], [[2, -4], [2, 2]])
    assert functional["root_value_effect_sample_means"][GROUPS[0]] == [0.2] * 3
    assert functional["root_value_effect_sample_means"][GROUPS[1]] == [1.2] * 3
    paths = plot_cohort_group(tmp_path, "QA", report["groups"]["QA"])
    assert paths == ["cohort_QA.png", "cohort_functional_QA.png"]
    assert all((tmp_path / path).stat().st_size > 10_000 for path in paths)


def test_production_evaluation_of_scan_only_run_joins_all_samples(
    tmp_path, monkeypatch
):
    import torch

    from experiments.reanchor_flow import subset_report

    output = tmp_path / "output"
    output.mkdir()
    dataset_root = tmp_path / "cache"
    dataset_root.mkdir()
    manifest, labels = _save_cohort(output)
    manifest.update(
        subset_manifest_schema=subset_report.MANIFEST_SCHEMA,
        analysis_complete=True,
        analysis_scope="structure_only",
        labels_used_for_capture=False,
        config={"dataset_root": str(dataset_root.resolve())},
    )
    (output / subset_report.MANIFEST_NAME).write_text(json.dumps(manifest))

    class Sample:
        def __init__(self, sample_id):
            self.sample_id = sample_id

        def release_attention(self):
            pass

    class LabelStore:
        def response_labels(self, sample):
            return torch.from_numpy(labels[sample.sample_id])

    class Dataset:
        def __getitem__(self, sample_id):
            return Sample(sample_id)

        def prepare_evaluation_labels(self, sample_ids):
            assert set(sample_ids) == set(labels)
            return LabelStore()

    monkeypatch.setattr(
        subset_report, "open_research_dataset", lambda *a, **k: Dataset()
    )
    report = subset_report.evaluate_subset_split(dataset_root, output, plot=True)
    assert report["analysis_scope"] == "structure_only"
    assert report["targets"] == []
    assert set(report["groups"]) == {"ALL", "QA"}
    for axis in report["groups"]["QA"]["raw_axis_evaluation"].values():
        assert axis["auroc"] is None
        assert axis["auprc"] is None
    assert report["full_scan_coverage"]["QA"]["scanned_response_tokens"] == 12
    assert set(report["cohort_plots"]) == {"cohort_ALL.png", "cohort_QA.png"}
    assert all((output / name).exists() for name in report["cohort_plots"])
