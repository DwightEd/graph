import json

import numpy as np
import pytest
import torch

from experiments.constraint_routing_rhythm import evaluate
from experiments.constraint_routing_rhythm.artifacts import save_result


class FakeAttention:
    def __init__(self, response_idx, response_count):
        self.response_idx = response_idx
        self.token_ids = torch.cat(
            (
                torch.zeros(response_idx, dtype=torch.long),
                100 + torch.arange(response_count),
            )
        )


class FakeSample:
    def __init__(
        self,
        sample_id,
        source_id,
        task_type,
        response_idx,
        labels,
        generator_model="fixture-generator",
        observer_model="fixture-model",
    ):
        self.sample_id = sample_id
        self.source_id = source_id
        self.task_type = task_type
        self._attention = FakeAttention(response_idx, len(labels))
        self.labels = np.asarray(labels)
        self.generator_model = generator_model
        self.observer_model = observer_model
        self.released = 0

    def attention(self):
        return self._attention

    def release_attention(self):
        self.released += 1


class FakeLabels:
    def response_labels(self, sample):
        return sample.labels


class FakeDataset:
    def __init__(self, samples):
        self.samples = {sample.sample_id: sample for sample in samples}
        self.prepared = None

    def __contains__(self, sample_id):
        return sample_id in self.samples

    def __getitem__(self, sample_id):
        return self.samples[sample_id]

    def prepare_evaluation_labels(self, sample_ids):
        self.prepared = list(sample_ids)
        return FakeLabels()


def write_result(
    root,
    sample_id,
    source_id,
    task_type,
    response_start,
    score,
    margin,
    *,
    valid=None,
    audit=None,
):
    count = len(score)
    prediction = response_start + np.arange(count)
    audit = {} if audit is None else audit

    def audit_array(name):
        return np.asarray(audit.get(name, np.full(count, np.nan)), dtype=np.float32)

    save_result(
        root / f"{sample_id}.npz",
        {
            "sample_id": np.asarray(sample_id),
            "source_id": np.asarray(source_id),
            "task_type": np.asarray(task_type),
            "model_id": np.asarray("fixture-model"),
            "generator_model": np.asarray("fixture-generator"),
            "observer_model": np.asarray("fixture-model"),
            "query_position": prediction - 1,
            "prediction_position": prediction,
            "target_token_id": 100 + np.arange(count),
            "constraint_deficit": np.asarray(score, dtype=np.float32),
            "baseline_margin": np.asarray(margin, dtype=np.float32),
            "baseline_target_logprob": -np.asarray(margin, dtype=np.float32),
            "baseline_entropy": np.asarray(margin, dtype=np.float32) / 10,
            "functional_reach": np.linspace(0.1, 0.9, count, dtype=np.float32),
            "relay_capacity": np.linspace(0.9, 0.1, count, dtype=np.float32),
            "valid": np.isfinite(score) if valid is None else np.asarray(valid),
            "evidence_tokens": np.asarray(3, dtype=np.int32),
            "control_audited": np.asarray(bool(audit)),
            "matched_control_available": np.asarray(
                "matched_non_evidence_cut_delta" in audit
            ),
            "relay_audited": np.asarray("relay_interaction" in audit),
            "direct_response_cut_delta": audit_array("direct_response_cut_delta"),
            "matched_non_evidence_cut_delta": audit_array(
                "matched_non_evidence_cut_delta"
            ),
            "upstream_cut_delta": audit_array("upstream_cut_delta"),
            "downstream_cut_delta": audit_array("downstream_cut_delta"),
            "joint_cut_delta": audit_array("joint_cut_delta"),
            "relay_interaction": audit_array("relay_interaction"),
        },
    )


def write_manifest(result_root, dataset_root, samples, *, complete=True):
    entries = []
    for sample in samples:
        path = result_root / f"{sample.sample_id}.npz"
        with np.load(path, allow_pickle=False) as stored:
            events = len(stored["prediction_position"])
            generator_model = str(stored["generator_model"].item())
            observer_model = str(stored["observer_model"].item())
            audited = bool(stored["control_audited"].item())
            evidence_tokens = int(stored["evidence_tokens"].item())
        entries.append(
            {
                "sample_id": sample.sample_id,
                "source_id": sample.source_id,
                "task_type": sample.task_type,
                "generator_model": generator_model,
                "observer_model": observer_model,
                "result": path.relative_to(result_root.parent).as_posix(),
                "events": events,
                "full_response_events": len(sample.labels),
                "evidence_positions": list(range(evidence_tokens)),
                "audit_requested": audited,
                "plot_requested": False,
                "complete": True,
            }
        )
    (result_root.parent / evaluate.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "config": {
                    "model_id": "fixture-model",
                    "dataset_root": str(dataset_root.resolve()),
                    "smoke": any(
                        entry["events"] < entry["full_response_events"]
                        for entry in entries
                    ),
                    "max_events": (
                        max(entry["events"] for entry in entries)
                        if any(
                            entry["events"] < entry["full_response_events"]
                            for entry in entries
                        )
                        else None
                    ),
                    "audit_limit": 1
                    if any(entry["audit_requested"] for entry in entries)
                    else 0,
                },
                "analysis_complete": complete,
                "selected_samples": {
                    task: sum(
                        evaluate.canonical_task_type(sample.task_type) == task
                        for sample in samples
                    )
                    for task in evaluate.TASK_TYPES
                },
                "audit_source_ids": {
                    task: [
                        sample.source_id
                        for sample, entry in zip(samples, entries, strict=True)
                        if evaluate.canonical_task_type(sample.task_type) == task
                        and entry["audit_requested"]
                    ]
                    for task in evaluate.TASK_TYPES
                },
                "audit_sample_ids": {
                    task: [
                        sample.sample_id
                        for sample, entry in zip(samples, entries, strict=True)
                        if evaluate.canonical_task_type(sample.task_type) == task
                        and entry["audit_requested"]
                    ]
                    for task in evaluate.TASK_TYPES
                },
                "samples": entries,
            }
        ),
        encoding="utf-8",
    )


def test_binary_metrics_handle_ties_without_score_fitting():
    label = np.asarray([1, 0, 1, 0])

    assert evaluate.binary_auroc(label, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert evaluate.average_precision(label, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert evaluate.binary_auroc(label, [0.9, 0.1, 0.8, 0.2]) == 1.0
    assert evaluate.average_precision(label, [0.9, 0.1, 0.8, 0.2]) == 1.0
    assert evaluate.binary_auroc(np.ones(3), np.arange(3)) is None


def test_source_cluster_bootstrap_is_reproducible():
    label = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.asarray([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    source = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"])
    first = evaluate.bootstrap_metrics(label, score, source, repeats=40, seed=19)
    second = evaluate.bootstrap_metrics(label, score, source, repeats=40, seed=19)

    assert first == second
    # Draws containing only one of the two class-pure sources are invalid;
    # this count therefore also verifies that complete sources were sampled.
    assert first["bootstrap_replicates"] == 23
    assert first["bootstrap_reliable"] is False
    assert first["auroc_ci95"] == [None, None]
    assert first["average_precision_ci95"] == [None, None]


def test_source_cluster_bootstrap_refuses_a_single_source_ci():
    result = evaluate.bootstrap_metrics(
        [0, 1], [0.1, 0.9], ["only", "only"], repeats=20, seed=3
    )

    assert result["bootstrap_replicates"] == 0
    assert result["bootstrap_reliable"] is False
    assert result["auroc_ci95"] == [None, None]


def test_evaluate_reports_one_score_by_task_and_controls(tmp_path, monkeypatch):
    result_root = tmp_path / "results"
    result_root.mkdir()
    write_result(
        result_root,
        "qa-1",
        "source-qa",
        "QA",
        5,
        [0.1, 0.9, 0.95, 0.8],
        [3.0, 0.2, 2.0, 0.4],
        valid=[True, True, False, True],
        audit={
            "direct_response_cut_delta": [0.0, 0.4, 0.2, 0.3],
            "matched_non_evidence_cut_delta": [0.0, 0.2, 0.1, 0.2],
            "upstream_cut_delta": [0.1, 0.2, 0.3, 0.4],
            "downstream_cut_delta": [0.2, 0.3, 0.4, 0.5],
            "joint_cut_delta": [0.4, 0.7, 0.9, 1.1],
            "relay_interaction": [0.1, 0.2, 0.2, 0.2],
        },
    )
    write_result(
        result_root,
        "summary-1",
        "source-summary",
        "Summary",
        8,
        [0.2, 0.7],
        [2.0, 0.5],
        audit={
            "direct_response_cut_delta": [0.1, 0.4],
            "matched_non_evidence_cut_delta": [0.0, 0.2],
        },
    )
    samples = [
        FakeSample("qa-1", "source-qa", "QA", 5, [0, 1, 0, 1]),
        FakeSample("summary-1", "source-summary", "Summary", 8, [0, 1]),
    ]
    dataset = FakeDataset(samples)
    write_manifest(result_root, tmp_path / "dataset", samples)

    def open_dataset(path, *, device, retain_embedded_labels):
        assert path == tmp_path / "dataset"
        assert device == "cpu"
        assert retain_embedded_labels is True
        return dataset

    monkeypatch.setattr(evaluate, "open_research_dataset", open_dataset)
    reports = evaluate.evaluate_results(
        result_root,
        tmp_path / "dataset",
        tmp_path / "reports",
        bootstrap=12,
        seed=7,
    )

    assert set(reports) == {"QA", "Summary"}
    qa = reports["QA"]
    assert qa["primary_score"] == "constraint_deficit"
    assert qa["valid_tokens"] == 3
    assert qa["valid_coverage"] == 0.75
    assert qa["constraint_deficit"]["auroc"] == 1.0
    assert qa["constraint_deficit"]["average_precision"] == 1.0
    assert qa["model_id"] == "fixture-model"
    assert qa["response_scope"] == "full"
    assert qa["intended_scope"]["smoke"] is False
    assert qa["model_roles"]["intervention_model"] == "fixture-model"
    assert qa["model_roles"]["response_generator_models"] == ["fixture-generator"]
    assert set(qa["controls"]) == {
        "absolute_response_position",
        "relative_response_position",
        "response_length",
        "evidence_tokens",
        "negative_baseline_margin",
        "negative_baseline_target_logprob",
        "baseline_entropy",
    }
    assert set(qa["diagnostic_correlation_with_primary"]) == set(qa["controls"]) | {
        "functional_reach",
        "relay_capacity",
    }
    dissociation = qa["route_control_dissociation"]
    assert dissociation["finite_route_tokens"] == 3
    assert dissociation["high_route_quantile"] == 0.75
    assert dissociation["high_route_tokens"] == 1
    assert dissociation["primary_within_high_route"]["tokens"] == 1
    audit = qa["audit_diagnostics"]
    assert audit["coverage"]["scheduled_samples"] == 1
    assert audit["coverage"]["control_audited_samples"] == 1
    assert audit["evidence_vs_matched_non_evidence"]["paired_contrast"][
        "mean"
    ] == pytest.approx((0.1 + 0.7 + 0.6) / 3)
    assert audit["total_vs_direct_response"]["paired_contrast"]["tokens"] == 3
    assert audit["relay"]["relay_interaction"]["tokens"] == 3
    assert dataset.prepared == []
    assert all(sample.released == 1 for sample in samples)

    saved = json.loads((tmp_path / "reports" / "qa" / "report.json").read_text())
    assert saved == qa


def test_evaluate_rejects_nonprefix_response_alignment_and_releases(
    tmp_path, monkeypatch
):
    result_root = tmp_path / "results"
    result_root.mkdir()
    write_result(
        result_root,
        "sample",
        "source",
        "Data2txt",
        4,
        [0.2, 0.3],
        [1.0, 1.0],
    )
    sample = FakeSample("sample", "source", "Data2txt", 5, [0, 1, 0])
    dataset = FakeDataset([sample])
    write_manifest(result_root, tmp_path / "dataset", [sample])
    monkeypatch.setattr(
        evaluate, "open_research_dataset", lambda *args, **kwargs: dataset
    )

    with pytest.raises(ValueError, match="contiguous response prefix"):
        evaluate.evaluate_results(
            result_root,
            tmp_path / "dataset",
            tmp_path / "reports",
            bootstrap=0,
        )

    assert sample.released == 1


def test_evaluate_accepts_a_smoke_prefix(tmp_path, monkeypatch):
    result_root = tmp_path / "results"
    result_root.mkdir()
    write_result(
        result_root,
        "sample",
        "source",
        "QA",
        5,
        [0.1, 0.9],
        [1.0, 0.5],
    )
    sample = FakeSample("sample", "source", "QA", 5, [0, 1, 1])
    dataset = FakeDataset([sample])
    write_manifest(result_root, tmp_path / "dataset", [sample])
    monkeypatch.setattr(
        evaluate, "open_research_dataset", lambda *args, **kwargs: dataset
    )

    reports = evaluate.evaluate_results(
        result_root,
        tmp_path / "dataset",
        tmp_path / "reports",
        bootstrap=0,
    )

    assert reports["QA"]["tokens"] == 2
    assert reports["QA"]["hallucinated_tokens"] == 1
    assert reports["QA"]["response_scope"] == "prefix"
    assert reports["QA"]["full_response_tokens"] == 3
    assert reports["QA"]["analyzed_response_coverage"] == pytest.approx(2 / 3)
    assert sample.released == 1
