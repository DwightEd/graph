"""重现六字段 CSR 的元数据缺失；不将标注或原回答内容加入训练输入。"""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.unsupervised_token_graph.offline_span.data import (
    load_observations, load_samples,
)
from experiments.unsupervised_token_graph.offline_span.metadata import (
    find_response_file, merge_metadata, metadata_fields, read_metadata,
)
from experiments.unsupervised_token_graph.offline_span.run import main

from experiments.unsupervised_token_graph.offline_span.tests.test_pipeline import (
    arguments, cache_fields, make_dataset, save_cache,
)


CORE_FIELDS = (
    "token_ids", "response_idx", "attention_diagonal",
    "response_row_ptr", "response_column_indices", "response_values",
)


def make_bare_dataset(root, source_format="jsonl"):
    cache = root / "attention" / "llama31_8b"
    response_file = make_dataset(cache)
    dataset = root / "dataset"
    dataset.mkdir()
    response_file.replace(dataset / "response.jsonl")

    sources = []
    for index in range(12):
        sources.append({"source_id": f"source_{index}", "task_type": "QA"})
        split = "train" if index < 8 else "test"
        path = cache / split / f"attention_{index}.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in CORE_FIELDS}
        np.savez(path, **arrays)

    if source_format == "jsonl":
        content = "\n".join(json.dumps(row) for row in sources)
        source_file = dataset / "source_info.jsonl"
    elif source_format == "dict":
        content = json.dumps({row["source_id"]: {"task_type": "QA"} for row in sources})
        source_file = dataset / "source_info.json"
    else:
        content = json.dumps(sources)
        source_file = dataset / "source_info.json"
    source_file.write_text(content)
    return cache, dataset


@pytest.mark.parametrize("source_format", ["jsonl", "list", "dict"])
def test_original_dataset_fills_only_grouping_metadata(tmp_path, source_format):
    cache, dataset = make_bare_dataset(tmp_path, source_format)
    samples, index = load_samples(cache / "train", split="train")
    assert len(samples) == 8
    assert all(sample.source_id and sample.task == "QA" for sample in samples)
    assert all(sample.generator == "fixture" for sample in samples)
    assert all(sample.metadata_file == str(dataset / "response.jsonl") for sample in samples)

    # 不凭“文件名相同”生成 offset 或认证回答文本完全一致。
    assert all(not len(sample.offsets) and not sample.response_sha256 for sample in samples)
    observations = load_observations(samples[0], index)
    assert observations["node_features"].shape == (16, 0)


def test_explicit_dataset_and_filters_use_enriched_identity(tmp_path):
    cache, dataset = make_bare_dataset(tmp_path)
    samples, _ = load_samples(
        cache / "test", split="test", tasks=("QA",), generators=("fixture",),
        dataset=dataset,
    )
    assert len(samples) == 4
    assert find_response_file(cache / "test", dataset) == dataset / "response.jsonl"
    assert find_response_file(cache / "test", dataset / "response.jsonl") == dataset / "response.jsonl"


def test_metadata_projection_never_accesses_labels_or_response():
    class IdentityOnly(dict):
        def __getitem__(self, key):
            if key not in {"id", "source_id", "model", "split"}:
                raise AssertionError(key)
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key not in {"task_type", "task"}:
                raise AssertionError(key)
            return super().get(key, default)

    raw = IdentityOnly(
        id=12, source_id=7, split="train", model="generator",
        response="DO NOT USE", labels=[{"start": 0, "end": 1}],
    )
    metadata = metadata_fields(raw, {"7": "QA"})
    assert metadata == {
        "id": "12", "source_id": "7", "split": "train",
        "generator": "generator", "task": "QA",
    }


def test_changed_labels_cannot_change_loaded_samples(tmp_path):
    cache, dataset = make_bare_dataset(tmp_path)
    before, _ = load_samples(cache / "train", split="train")
    response_file = dataset / "response.jsonl"
    rows = [json.loads(line) for line in response_file.read_text().splitlines()]
    for row in rows:
        row["labels"] = {"malformed_label_value": True}
        row["offsets"] = [[999, 1000]]
        row["response_sha256"] = "NOT_A_VERIFICATION"
    response_file.write_text("\n".join(json.dumps(row) for row in rows))
    after, _ = load_samples(cache / "train", split="train")
    assert [sample.metadata() for sample in before] == [sample.metadata() for sample in after]
    for first, second in zip(before, after):
        np.testing.assert_array_equal(first.token_ids, second.token_ids)
        np.testing.assert_array_equal(first.offsets, second.offsets)


def test_cache_and_dataset_source_disagreement_is_not_overwritten():
    records = {"1": {
        "id": "1", "source_id": "original", "split": "train",
        "generator": "fixture", "task": "QA",
    }}
    with pytest.raises(ValueError, match="source_id"):
        merge_metadata({"source_id": "different"}, "1", records)


def test_official_split_conflict_stops_before_training(tmp_path):
    cache, dataset = make_bare_dataset(tmp_path)
    response_file = dataset / "response.jsonl"
    rows = [json.loads(line) for line in response_file.read_text().splitlines()]
    rows[0]["split"] = "test"
    response_file.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="split"):
        load_samples(cache / "train", split="train")


def test_missing_task_is_reported_and_prepare_does_not_train(tmp_path, capsys):
    cache, dataset = make_bare_dataset(tmp_path)
    (dataset / "source_info.jsonl").unlink()
    args = arguments(cache, tmp_path / "output")
    main(args + ["--phase", "inspect"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[0]["missing_source_ids"] == 0
    assert rows[0]["unknown_tasks"] == 8
    assert rows[0]["ready_to_fit_metadata"] is False
    with pytest.raises(ValueError, match="缺少来源"):
        main(args + ["--phase", "prepare"])
    assert not (tmp_path / "output" / "model").exists()


def test_bare_cache_metadata_prepare_fit_score_and_inspect(tmp_path, capsys):
    cache, dataset = make_bare_dataset(tmp_path)
    output = tmp_path / "output"
    args = arguments(cache, output)

    main(args + ["--phase", "inspect"])
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(row["ready_to_fit_metadata"] for row in reports)
    assert reports[0]["unknown_tasks"] == 0
    assert reports[0]["uncovered_response_positions"] == [0]
    assert reports[0]["hidden_fields_in_selected_archive"] == []
    assert reports[0]["observation_scope"].endswith("not the whole split")
    assert not output.exists()

    for phase in ("prepare", "fit", "score"):
        main(args + ["--phase", phase])
    freeze = json.loads((output / "predictions" / "prediction_freeze.json").read_text())
    assert freeze["complete"] and freeze["labels_used"] is False
    assert len(freeze["records"]) == 4
    assert all(row["source_id"] and row["task"] == "QA" for row in freeze["records"])

    sample_path = output / "predictions" / freeze["records"][0]["file"]
    with np.load(sample_path) as archive:
        assert "offsets" not in archive.files
        assert "labels" not in archive.files


def test_complete_cache_does_not_open_dataset(tmp_path, monkeypatch):
    cache = tmp_path / "train"
    cache.mkdir()
    save_cache(cache / "attention_1.npz", "canonical")
    forbidden = tmp_path / "response.jsonl"
    forbidden.write_text("NOT JSON")
    original_open = Path.open

    def checked_open(path, *args, **kwargs):
        if path == forbidden:
            raise AssertionError("complete identity does not need the dataset")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked_open)
    samples, _ = load_samples(cache, split="train")
    assert samples[0].source_id == "source_1"
    assert samples[0].metadata_file == ""


def test_tokens_inspect_reports_hidden_presence_without_loading(tmp_path, monkeypatch):
    path = tmp_path / "attention_1.npz"
    fields = cache_fields()
    fields["hidden_states"] = np.ones((16, 8), np.float32)
    np.savez(path, **fields)
    samples, index = load_samples(path)

    original_get = np.lib.npyio.NpzFile.__getitem__

    def checked_get(archive, key):
        if key == "hidden_states":
            raise AssertionError("tokens mode must not load hidden")
        return original_get(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", checked_get)
    observations = load_observations(samples[0], index, feature_mode="tokens")
    assert observations["hidden_fields"] == ["hidden_states"]
    assert observations["node_features"].shape == (16, 0)


def test_duplicate_metadata_ids_are_rejected(tmp_path):
    response_file = tmp_path / "response.jsonl"
    row = {"id": "1", "source_id": "s", "split": "train", "model": "m", "task": "QA"}
    response_file.write_text(json.dumps(row) + "\n" + json.dumps(row))
    with pytest.raises(ValueError, match="重复"):
        read_metadata(response_file)


def test_unmatched_response_id_is_not_a_fabricated_source(tmp_path):
    cache, dataset = make_bare_dataset(tmp_path)
    response_file = dataset / "response.jsonl"
    rows = [json.loads(line) for line in response_file.read_text().splitlines()]
    response_file.write_text("\n".join(json.dumps(row) for row in rows if row["id"] != "0"))
    samples, _ = load_samples(cache / "train", split="train")
    unresolved = [sample for sample in samples if sample.response_id == "0"][0]
    assert unresolved.source_id == ""
    assert unresolved.task == "unknown"
