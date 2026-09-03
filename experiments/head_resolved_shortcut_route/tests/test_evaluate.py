import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.head_resolved_shortcut_route import evaluate

CONTROL_NAMES = (
    "absolute_response_position",
    "relative_response_position",
    "response_length",
    "observer_target_surprisal",
)


def _assert_bootstrap_metadata(metric, *, requested: int, sources: int) -> None:
    assert metric["bootstrap_requested"] == requested
    assert 0 <= metric["bootstrap_successful"] <= requested
    source_count = metric.get("valid_sources", metric.get("sources"))
    assert source_count == sources
    assert 0 <= metric["positive_sources"] <= sources
    assert 0 <= metric["negative_sources"] <= sources


def _artifact(
    *,
    offset: float = 0.0,
    targets: tuple[int, ...] = (21, 22, 23),
) -> SimpleNamespace:
    response_start = 3
    count = len(targets)
    source_token_id = torch.tensor([10, 11, 12, *targets[:-1]])
    query = torch.arange(response_start - 1, response_start - 1 + count)
    prediction = query + 1
    support_valid = torch.ones(count, 2, dtype=torch.bool)
    takeover_valid = support_valid.clone()
    takeover_valid[0] = False
    base = torch.linspace(0.1, 0.9, count) + offset
    axes = SimpleNamespace(
        carrier_drift=torch.stack((base, base + 0.05), dim=1),
        carrier_drift_defined=support_valid,
        prompt_source_dispersion=torch.stack((1.0 - base, base + 0.1), dim=1),
        prompt_source_dispersion_defined=support_valid,
        response_born_takeover=torch.stack((base, base + 0.2), dim=1),
        response_born_takeover_defined=takeover_valid,
    )
    return SimpleNamespace(
        response_start=response_start,
        source_token_id=source_token_id,
        top_k=64,
        cover_mass=0.95,
        events=SimpleNamespace(
            query_position=query,
            prediction_position=prediction,
            target_token_id=torch.tensor(targets),
        ),
        readout=SimpleNamespace(
            target_logprob=-(torch.linspace(0.25, 1.25, count) + offset)
        ),
        axes=axes,
    )


def _manifest(samples: int, *, observer: str = "observer", dataset_path: Path) -> dict:
    return {
        "schema": evaluate.SCHEMA,
        "version": evaluate.VERSION,
        "artifact_schema": evaluate.ARTIFACT_SCHEMA,
        "dataset_identity": {"path": str(dataset_path.resolve())},
        "source_identity": {"path": "source_info.jsonl"},
        "observer_identity": {"model": observer},
        "model_dtype": "torch.bfloat16",
        "top_k": 64,
        "cover_mass": 0.95,
        "task_types": ["QA", "Summary", "Data2txt"],
        "index": "index.jsonl",
        "samples": samples,
        "complete": True,
        "labels_used": False,
        "dataset_candidates": samples,
    }


def _write_collection(
    root: Path,
    rows: list[dict],
    *,
    observer: str = "observer",
    split_root: Path | None = None,
) -> None:
    samples = root / "samples"
    samples.mkdir(parents=True)
    saved_rows = []
    for row in rows:
        path = samples / row["path"]
        path.write_bytes(str(row["sample_id"]).encode())
        saved_rows.append(
            {
                **row,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            _manifest(
                len(saved_rows),
                observer=observer,
                dataset_path=split_root or root.parent / "split",
            )
        ),
        encoding="utf-8",
    )
    (root / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in saved_rows), encoding="utf-8"
    )


def _row(sample: str, *, task: str = "Summary", events: int = 3) -> dict:
    return {
        "sample_id": sample,
        "source_id": f"source-{sample}",
        "task_type": task,
        "generator_model": "generator",
        "path": f"{sample}.npz",
        "bytes": 0,
        "events": events,
        "response_start": 3,
    }


def _patch_artifacts(monkeypatch, artifacts: dict[str, SimpleNamespace], events=None):
    def load(path):
        sample = Path(path).stem
        return artifacts[sample]

    def validate(artifact):
        if events is not None:
            events.append(("validate", int(artifact.events.target_token_id[-1])))

    monkeypatch.setattr(evaluate, "load_route_artifact", load)
    monkeypatch.setattr(evaluate, "validate_artifact", validate)


class _Sample:
    task_type = "Summary"
    generator_model = "generator"

    def __init__(self, sample_id: str, tokens=(10, 11, 12, 21, 22, 23)):
        self.source_id = f"source-{sample_id}"
        self._tokens = torch.tensor(tokens)
        self.released = False

    def attention(self):
        return SimpleNamespace(token_ids=self._tokens, response_idx=3)

    def release_attention(self):
        self.released = True


class _Prepared:
    def response_labels(self, _sample):
        return torch.tensor([False, False, True])


class _Dataset:
    def __init__(self, samples):
        self.samples = samples

    def prepare_evaluation_labels(self, sample_ids):
        assert sample_ids == list(self.samples)
        return _Prepared()

    def __getitem__(self, sample_id):
        return self.samples[sample_id]


def test_evaluate_freezes_all_artifacts_before_labels_and_keeps_tasks_separate(
    tmp_path, monkeypatch
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_collection(first, [_row("a")], split_root=tmp_path / "split-a")
    _write_collection(second, [_row("b")], split_root=tmp_path / "split-b")
    order = []
    _patch_artifacts(
        monkeypatch,
        {"a": _artifact(), "b": _artifact(offset=0.01)},
        order,
    )
    original_write = evaluate._write_npz

    def write(path, arrays):
        if path.name == "frozen_axes.npz":
            assert "label" not in arrays
            order.append(("freeze", len(arrays["target_token_id"])))
        original_write(path, arrays)

    monkeypatch.setattr(evaluate, "_write_npz", write)

    datasets = {
        "split-a": _Dataset({"a": _Sample("a")}),
        "split-b": _Dataset({"b": _Sample("b")}),
    }

    def open_dataset(path, **kwargs):
        assert kwargs["device"] == "cpu"
        assert kwargs["verify_hashes"] is True
        phase = "labels" if kwargs["retain_embedded_labels"] else "canonical"
        order.append((phase, Path(path).name))
        return datasets[Path(path).name]

    monkeypatch.setattr(evaluate, "open_research_dataset", open_dataset)
    output = tmp_path / "result" / "report.json"
    report = evaluate.evaluate_all(
        inputs=[(first, tmp_path / "split-a"), (second, tmp_path / "split-b")],
        task_type="Summary",
        output=output,
        bootstrap=20,
        seed=3,
    )

    assert order == [
        ("validate", 23),
        ("validate", 23),
        ("canonical", "split-a"),
        ("canonical", "split-b"),
        ("freeze", 6),
        ("labels", "split-a"),
        ("labels", "split-b"),
    ]
    assert report["task_type"] == "Summary"
    assert report["physical_shards"] == 2
    assert report["capture_complete"] is True
    assert report["collection_status"] == "complete"
    assert report["analysis_status"] == "preregistered_association_evaluation"
    assert report["claims_boundary"].startswith("Association report only")
    assert report["tokens"] == 6
    assert report["hallucinated_tokens"] == 2
    assert report["detection_estimand"] == "token_micro"
    assert report["detection_bootstrap_unit"] == "source_id_cluster"
    assert set(report["detection"]) == set(evaluate.SCORE_ORDER)
    assert all(metric["auroc"] == 1.0 for metric in report["detection"].values())
    assert set(report["control_detection"]) == set(CONTROL_NAMES)
    support_audit = report["support_group_audit"]
    assert set(support_audit["metrics"]) == set(evaluate.SCORE_ORDER)
    for metric in support_audit["metrics"].values():
        assert {"raw", "position_matched", "position_surprisal_matched"} <= set(metric)
    common = report["common_validity_sensitivity"]
    assert common["valid_tokens"] == 4
    assert set(common["detection"]) == set(evaluate.SCORE_ORDER)
    assert set(report["by_generator_model"]) == {"generator"}
    generator = report["by_generator_model"]["generator"]
    assert set(generator["detection"]) == set(evaluate.SCORE_ORDER)
    for metric in report["detection"].values():
        _assert_bootstrap_metadata(metric, requested=20, sources=2)
    for metric in report["control_detection"].values():
        _assert_bootstrap_metadata(metric, requested=20, sources=2)
    _assert_bootstrap_metadata(
        report["support_group_audit"]["metrics"][evaluate.SCORE_ORDER[0]]["raw"],
        requested=20,
        sources=2,
    )
    assert all("auroc" not in metric for metric in report["veto_audit"].values())
    assert output.is_file()
    json.loads(output.read_text(encoding="utf-8"))

    with np.load(output.with_name("frozen_axes.npz"), allow_pickle=False) as frozen:
        assert "label" not in frozen
        np.testing.assert_array_equal(
            frozen["query_position"] + 1, frozen["prediction_position"]
        )
        np.testing.assert_array_equal(
            frozen["response_index"],
            frozen["prediction_position"] - frozen["response_start"],
        )
        assert frozen["target_token_id"].tolist() == [21, 22, 23] * 2
        assert frozen["canonical_source_ptr"].tolist() == [0, 5, 10]
        np.testing.assert_allclose(
            frozen["relative_response_position"],
            np.tile(np.asarray([1 / 6, 1 / 2, 5 / 6]), 2),
        )
        np.testing.assert_allclose(
            frozen["observer_target_surprisal"],
            [0.25, 0.75, 1.25, 0.26, 0.76, 1.26],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            frozen["negative_prompt_source_dispersion_support"][:3],
            [-0.9, -0.5, -0.1],
            atol=1e-7,
        )
    with np.load(output.with_name("token_results.npz"), allow_pickle=False) as result:
        assert result["label"].tolist() == [False, False, True] * 2


def test_artifact_misalignment_fails_before_label_dataset_is_opened(
    tmp_path, monkeypatch
):
    root = tmp_path / "state"
    _write_collection(root, [_row("a", events=4)])
    order = []
    _patch_artifacts(monkeypatch, {"a": _artifact()}, order)

    def forbidden(*_args, **_kwargs):
        pytest.fail("labels opened before all artifact contracts passed")

    monkeypatch.setattr(evaluate, "open_research_dataset", forbidden)
    output = tmp_path / "result" / "report.json"
    with pytest.raises(ValueError, match="event count"):
        evaluate.evaluate_all(
            inputs=[(root, tmp_path / "split")],
            task_type="Summary",
            output=output,
            bootstrap=0,
        )
    assert order == [("validate", 23)]
    assert not output.with_name("frozen_axes.npz").exists()


def test_q_to_p_alignment_is_checked_even_after_artifact_loader(monkeypatch):
    artifact = _artifact()
    artifact.events.prediction_position[1] += 1
    monkeypatch.setattr(evaluate, "validate_artifact", lambda _artifact: None)
    record = {
        "events": 3,
        "response_start": 3,
        "manifest_top_k": 64,
        "manifest_cover_mass": 0.95,
    }

    with pytest.raises(ValueError, match="prediction positions"):
        evaluate._validate_record_artifact(record, artifact)


def test_dataset_tokens_and_targets_are_rebound_to_the_frozen_identity(
    tmp_path, monkeypatch
):
    root = tmp_path / "state"
    _write_collection(root, [_row("a")])
    _patch_artifacts(monkeypatch, {"a": _artifact()})
    sample = _Sample("a", tokens=(10, 99, 12, 21, 22, 23))
    label_api_calls = []

    class Dataset(_Dataset):
        def prepare_evaluation_labels(self, sample_ids):
            label_api_calls.append(sample_ids)
            return super().prepare_evaluation_labels(sample_ids)

    monkeypatch.setattr(
        evaluate,
        "open_research_dataset",
        lambda *_args, **_kwargs: Dataset({"a": sample}),
    )
    output = tmp_path / "result" / "report.json"
    with pytest.raises(ValueError, match="token sequence changed"):
        evaluate.evaluate_all(
            inputs=[(root, tmp_path / "split")],
            task_type="Summary",
            output=output,
            bootstrap=0,
        )
    assert sample.released
    assert label_api_calls == []
    assert not output.with_name("frozen_axes.npz").exists()


def test_final_target_is_checked_even_when_the_teacher_forcing_source_matches(
    tmp_path, monkeypatch
):
    root = tmp_path / "state"
    _write_collection(root, [_row("a")])
    _patch_artifacts(monkeypatch, {"a": _artifact()})
    sample = _Sample("a", tokens=(10, 11, 12, 21, 22, 99))
    label_api_calls = []

    class Dataset(_Dataset):
        def prepare_evaluation_labels(self, sample_ids):
            label_api_calls.append(sample_ids)
            return super().prepare_evaluation_labels(sample_ids)

    monkeypatch.setattr(
        evaluate,
        "open_research_dataset",
        lambda *_args, **_kwargs: Dataset({"a": sample}),
    )

    with pytest.raises(ValueError, match="target token IDs changed"):
        evaluate.evaluate_all(
            inputs=[(root, tmp_path / "split")],
            task_type="Summary",
            output=tmp_path / "result" / "report.json",
            bootstrap=0,
        )
    assert sample.released
    assert label_api_calls == []


def test_label_rows_use_prediction_position_minus_response_start(monkeypatch, tmp_path):
    frozen = {
        "sample_id": np.repeat("a", 2),
        "record_index": np.zeros(2, dtype=np.int32),
        "response_index": np.asarray([0, 2], dtype=np.int32),
        "query_position": np.asarray([2, 4], dtype=np.int64),
        "prediction_position": np.asarray([3, 5], dtype=np.int64),
        "target_token_id": np.asarray([21, 23], dtype=np.int64),
        "record_sample_id": np.asarray(["a"]),
        "record_source_id": np.asarray(["source-a"]),
        "record_task_type": np.asarray(["Summary"]),
        "record_generator_model": np.asarray(["generator"]),
        "record_response_start": np.asarray([3], dtype=np.int32),
        "canonical_source_ptr": np.asarray([0, 5], dtype=np.int64),
        "canonical_source_token_id": np.asarray([10, 11, 12, 21, 22]),
    }
    records = [
        {
            "sample_id": "a",
            "physical_shard": 0,
            "split_root": tmp_path,
        }
    ]

    class Prepared:
        def response_labels(self, _sample):
            return torch.tensor([False, True, True])

    class Dataset(_Dataset):
        def prepare_evaluation_labels(self, _ids):
            return Prepared()

    sample = _Sample("a")
    monkeypatch.setattr(
        evaluate,
        "open_research_dataset",
        lambda *_args, **_kwargs: Dataset({"a": sample}),
    )

    assert evaluate._load_labels(records, frozen).tolist() == [False, True]
    assert sample.released


def test_fixed_directions_and_axis_specific_masks_are_not_label_flipped():
    label = np.asarray([False, True, False, True])
    arrays = {
        "source_id": np.asarray(["a", "a", "b", "b"]),
        "carrier_drift_support": np.asarray([1.0, 0.0, 1.0, 0.0]),
        "carrier_drift_support__valid": np.ones(4, dtype=bool),
        "negative_prompt_source_dispersion_support": np.asarray([0.0, 1.0, 0.0, 1.0]),
        "negative_prompt_source_dispersion_support__valid": np.asarray(
            [True, True, False, False]
        ),
        "response_born_takeover_support": np.asarray([0.0, 1.0, 0.0, 1.0]),
        "response_born_takeover_support__valid": np.ones(4, dtype=bool),
    }
    result = evaluate.detection_summary(label, arrays, bootstrap=20, seed=11)

    assert result["carrier_drift_support"]["auroc"] == 0.0
    assert result["negative_prompt_source_dispersion_support"]["auroc"] == 1.0
    assert result["negative_prompt_source_dispersion_support"]["valid_tokens"] == 2
    assert result["response_born_takeover_support"]["auroc"] == 1.0
    _assert_bootstrap_metadata(
        result["response_born_takeover_support"], requested=20, sources=2
    )


def test_sparse_class_clusters_do_not_emit_a_conditioned_bootstrap_interval():
    label = np.asarray([False, True])
    arrays = {"source_id": np.asarray(["correct-only", "hallucinated-only"])}
    for name in evaluate.SCORE_ORDER:
        arrays[name] = np.asarray([0.0, 1.0])
        arrays[f"{name}__valid"] = np.ones(2, dtype=bool)

    metric = evaluate.detection_summary(label, arrays, bootstrap=100, seed=17)[
        evaluate.SCORE_ORDER[0]
    ]

    assert metric["bootstrap_successful"] < 90
    assert metric["bootstrap_ci_reliable"] is False
    assert metric["auroc_ci95"] == [None, None]
    assert metric["average_precision_ci95"] == [None, None]


def test_veto_is_only_a_raw_group_audit():
    label = np.asarray([False, True, False, True])
    arrays = {"source_id": np.asarray(["a", "a", "b", "b"])}
    for name in evaluate.VETO_ORDER:
        arrays[name] = np.asarray([0.0, 2.0, 1.0, 5.0])
        arrays[f"{name}__valid"] = np.ones(4, dtype=bool)

    result = evaluate.veto_audit(label, arrays, bootstrap=10, seed=2)

    for metric in result.values():
        assert metric["hallucinated_minus_correct"] == 3.0
        _assert_bootstrap_metadata(metric, requested=10, sources=2)
        assert "auroc" not in metric
        assert "average_precision" not in metric
        assert metric["role"].startswith("raw_posthoc")


def test_pool_filters_task_and_rejects_mixed_observer_identity(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_collection(first, [_row("a"), _row("qa", task="QA")])
    _write_collection(second, [_row("b")], observer="different")

    records, _ = evaluate._pool_records([(first, tmp_path / "split")], "Summary")
    assert [record["sample_id"] for record in records] == ["a"]

    with pytest.raises(ValueError, match="different scientific identity"):
        evaluate._pool_records(
            [(first, tmp_path / "split"), (second, tmp_path / "split")],
            "Summary",
        )


def test_partial_collection_requires_explicit_nonformal_mode(tmp_path):
    root = tmp_path / "state"
    _write_collection(root, [_row("a")])
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="allow_partial=True"):
        evaluate._pool_records([(root, tmp_path / "split")], "Summary")
    records, manifests = evaluate._pool_records(
        [(root, tmp_path / "split")], "Summary", allow_partial=True
    )
    assert [record["sample_id"] for record in records] == ["a"]
    assert manifests[0]["complete"] is False


def test_complete_manifest_must_cover_every_dataset_candidate(tmp_path):
    root = tmp_path / "state"
    _write_collection(root, [_row("a")])
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_candidates"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not cover every dataset candidate"):
        evaluate._pool_records([(root, tmp_path / "split")], "Summary")


def test_pool_rejects_an_artifact_with_a_changed_digest(tmp_path):
    root = tmp_path / "state"
    _write_collection(root, [_row("a")])
    (root / "samples" / "a.npz").write_bytes(b"b")

    with pytest.raises(ValueError, match="digest changed"):
        evaluate._pool_records([(root, tmp_path / "split")], "Summary")


def test_pool_rejects_a_dataset_manifest_changed_after_collection(tmp_path):
    root = tmp_path / "state"
    split = tmp_path / "split"
    split.mkdir()
    cache_manifest = split / "manifest.json"
    cache_manifest.write_text('{"version":1}', encoding="utf-8")
    _write_collection(root, [_row("a")], split_root=split)
    collection_manifest_path = root / "manifest.json"
    collection_manifest = json.loads(
        collection_manifest_path.read_text(encoding="utf-8")
    )
    collection_manifest["dataset_identity"]["manifest_file"] = {
        "size": cache_manifest.stat().st_size,
        "sha256": hashlib.sha256(cache_manifest.read_bytes()).hexdigest(),
    }
    collection_manifest_path.write_text(
        json.dumps(collection_manifest), encoding="utf-8"
    )

    evaluate._pool_records([(root, split)], "Summary")
    cache_manifest.write_text('{"version":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="dataset manifest changed"):
        evaluate._pool_records([(root, split)], "Summary")


def test_report_is_task_specific_and_does_not_promote_veto():
    label = np.asarray([False, True])
    arrays = {
        "label": label,
        "sample_id": np.repeat("sample", 2),
        "source_id": np.repeat("source", 2),
        "generator_model": np.repeat("generator", 2),
        "response_index": np.asarray([0, 1]),
        "absolute_response_position": np.asarray([0.0, 1.0]),
        "relative_response_position": np.asarray([0.25, 0.75]),
        "response_length": np.repeat(2, 2),
        "observer_target_surprisal": np.asarray([0.2, 0.9]),
    }
    for name in evaluate.SCORE_ORDER:
        arrays[name] = np.asarray([0.0, 1.0])
        arrays[f"{name}__valid"] = np.ones(2, dtype=bool)
    for name in evaluate.VETO_ORDER:
        arrays[name] = np.asarray([1.0, 0.0])
        arrays[f"{name}__valid"] = np.ones(2, dtype=bool)

    report = evaluate.build_report(
        task_type="Data2txt", arrays=arrays, bootstrap=0, seed=1
    )

    assert report["task_type"] == "Data2txt"
    assert report["schema"] == evaluate.REPORT_SCHEMA
    assert tuple(report["detection"]) == evaluate.SCORE_ORDER
    assert set(report["control_detection"]) == set(CONTROL_NAMES)
    for metric in report["support_group_audit"]["metrics"].values():
        assert {"raw", "position_matched", "position_surprisal_matched"} <= set(metric)
    assert "common_validity_sensitivity" in report
    assert set(report["by_generator_model"]) == {"generator"}
    assert set(report["veto_audit"]) == set(evaluate.VETO_ORDER)
    assert report["labels_used_during"].endswith("after_frozen_axes")
    assert all("auroc" not in value for value in report["veto_audit"].values())


def test_position_matched_support_differences_use_each_axis_validity_mask():
    label = np.asarray([False, True, False, True])
    response_index = np.arange(4)
    arrays = {
        "label": label,
        "sample_id": np.repeat("sample", 4),
        "source_id": np.repeat("source", 4),
        "generator_model": np.repeat("generator", 4),
        "response_index": response_index,
        "absolute_response_position": response_index.astype(np.float64),
        "relative_response_position": (response_index + 0.5) / 20,
        "response_length": np.repeat(20, 4),
        "observer_target_surprisal": np.ones(4),
        "carrier_drift_support": np.asarray([0.0, 2.0, 100.0, 300.0]),
        "carrier_drift_support__valid": np.asarray([True, True, False, False]),
        "negative_prompt_source_dispersion_support": np.asarray(
            [100.0, 300.0, 1.0, 5.0]
        ),
        "negative_prompt_source_dispersion_support__valid": np.asarray(
            [False, False, True, True]
        ),
        "response_born_takeover_support": np.asarray([0.0, 1.0, 1.0, 4.0]),
        "response_born_takeover_support__valid": np.ones(4, dtype=bool),
    }
    for name in evaluate.VETO_ORDER:
        arrays[name] = np.zeros(4)
        arrays[f"{name}__valid"] = np.ones(4, dtype=bool)

    report = evaluate.build_report(
        task_type="Summary", arrays=arrays, bootstrap=0, seed=7
    )
    matched = {
        name: metric["position_matched"]
        for name, metric in report["support_group_audit"]["metrics"].items()
    }

    assert matched["carrier_drift_support"]["valid_tokens"] == 2
    assert matched["carrier_drift_support"]["hallucinated_minus_correct"] == 2.0
    assert matched["negative_prompt_source_dispersion_support"]["valid_tokens"] == 2
    assert (
        matched["negative_prompt_source_dispersion_support"][
            "hallucinated_minus_correct"
        ]
        == 4.0
    )
