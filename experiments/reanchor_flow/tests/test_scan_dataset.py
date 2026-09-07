from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from experiments.reanchor_flow.scan_dataset import (
    BUCKET_NAMES,
    ScanDataset,
    ScanLabelStore,
)


def _capture(tmp_path, *, overrides=None, records=None):
    root = tmp_path / "test"
    root.mkdir(exist_ok=True)
    records = records or {"one": ("source-a", "QA")}
    entries = {}
    for sample_id, (source_id, task) in records.items():
        arrays = {
            "sample_scan_schema": 1,
            "dataset_sample_id": sample_id,
            "source_id": source_id,
            "task_type": task,
            "labels_used_for_capture": False,
            "response_start": 3,
            "sequence_length": 7,
            "full_response_tokens": 4,
            "processed_response_tokens": 4,
            "local_window": 2,
            "route_row_position": np.arange(2, 6),
            "reanchor_bucket_name": np.asarray(BUCKET_NAMES),
            "token_ids": np.arange(10, 17),
            "reanchor_bucket_transport": np.arange(96).reshape(2, 3, 4, 4),
            "reanchor_bucket_attention": np.full((2, 3, 4, 4), 0.25),
            "reanchor_bucket_source_position": np.zeros((2, 3, 4, 4), dtype=int),
            "reanchor_bucket_source_unit_id": np.ones((2, 3, 4, 4), dtype=int),
            "reanchor_score": np.zeros((2, 3, 4)),
        }
        arrays.update(overrides or {})
        np.savez_compressed(root / f"{sample_id}.npz", **arrays)
        entries[sample_id] = {
            "source_id": source_id,
            "task_type": task,
            "scan": f"{sample_id}.npz",
        }
    manifest = {
        "subset_manifest_schema": 3,
        "analysis_complete": True,
        "labels_used_for_capture": False,
        "config": {"split": "test", "dataset_root": str(tmp_path / "cache")},
        "samples": entries,
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    return root


def test_manifest_does_not_load_scans_or_labels(tmp_path, monkeypatch):
    root = _capture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("manifest inspection must not read scan arrays or labels")

    monkeypatch.setattr(np, "load", forbidden)
    dataset_module = ModuleType("research_dataset")
    dataset_module.open_research_dataset = forbidden
    monkeypatch.setitem(sys.modules, "research_dataset", dataset_module)
    scans = ScanDataset(root)
    assert scans.sample_ids == ("one",)
    assert scans.records[0].source_id == "source-a"


def test_load_keeps_heads_and_decompresses_only_requested_members(
    tmp_path, monkeypatch
):
    root = _capture(tmp_path)
    accessed = []
    original = np.lib.npyio.NpzFile.__getitem__

    def tracked(self, name):
        accessed.append(name)
        return original(self, name)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", tracked)
    sample = ScanDataset(root).load("one", fields=("reanchor_bucket_transport",))
    assert sample["reanchor_bucket_transport"].shape == (2, 3, 4, 4)
    assert sample["reanchor_bucket_transport"][1, 2, 3, 3] == 95
    np.testing.assert_array_equal(sample.row_position, [2, 3, 4, 5])
    np.testing.assert_array_equal(sample.response_index, [0, 1, 2, 3])
    assert "reanchor_score" not in accessed
    assert "reanchor_bucket_attention" not in accessed
    assert "token_ids" not in accessed


def test_metadata_only_load_and_source_balanced_weights(tmp_path):
    root = _capture(
        tmp_path,
        records={
            "one": ("source-a", "QA"),
            "two": ("source-a", "QA"),
            "three": ("source-b", "QA"),
            "four": ("source-c", "Summary"),
        },
    )
    scans = ScanDataset(root, tasks=("QA",))
    assert len(scans) == 3
    assert scans.sample_weights() == {"one": 0.5, "two": 0.5, "three": 1.0}
    assert scans.sample_weights(by_source=False) == dict.fromkeys(scans.sample_ids, 1.0)
    sample = scans.load("one", fields=())
    assert not sample.arrays
    assert sample.rows == 4


def test_select_is_metadata_only_and_recomputes_source_weights(tmp_path, monkeypatch):
    root = _capture(
        tmp_path,
        records={
            "one": ("source-a", "QA"),
            "two": ("source-a", "QA"),
            "three": ("source-b", "QA"),
        },
    )
    scans = ScanDataset(root)

    def forbidden(*args, **kwargs):
        raise AssertionError("selection must not read files")

    monkeypatch.setattr(type(root), "read_text", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    subset = scans.select(["three", "one"])
    assert subset.sample_ids == ("three", "one")
    assert subset.root is scans.root
    assert subset.config is scans.config
    assert subset.split == scans.split
    assert subset.records[1] is scans.records[0]
    assert subset.sample_weights() == {"three": 1.0, "one": 1.0}
    assert len(scans) == 3
    with pytest.raises(ValueError, match="not found"):
        scans.select(["unknown"])


@pytest.mark.parametrize(
    ("overrides", "fields", "message"),
    [
        ({"source_id": "wrong-source"}, (), "source_id"),
        ({"dataset_sample_id": "wrong-sample"}, (), "dataset_sample_id"),
        ({"task_type": "Summary"}, (), "task_type"),
        ({"labels_used_for_capture": True}, (), "label firewall"),
        ({"route_row_position": [2, 3, 5, 6]}, (), "contiguous"),
        ({"route_row_position": [3, 4, 5, 6]}, (), "contiguous"),
        ({"processed_response_tokens": 3}, (), "coverage"),
        (
            {"reanchor_bucket_transport": np.zeros((2, 3, 3, 4))},
            ("reanchor_bucket_transport",),
            "axes",
        ),
        (
            {"reanchor_bucket_transport": np.full((2, 3, 4, 4), np.nan)},
            ("reanchor_bucket_transport",),
            "finite",
        ),
        (
            {"reanchor_bucket_source_position": np.full((2, 3, 4, 4), 7)},
            ("reanchor_bucket_source_position",),
            "future",
        ),
    ],
)
def test_invalid_scan_identity_and_causality(tmp_path, overrides, fields, message):
    root = _capture(tmp_path, overrides=overrides)
    with pytest.raises(ValueError, match=message):
        ScanDataset(root).load("one", fields=fields)


def test_full_response_length_never_normalizes_causal_position(tmp_path):
    root = _capture(tmp_path)
    original = ScanDataset(root).load("one")
    _capture(tmp_path, overrides={"full_response_tokens": 100})
    longer_future = ScanDataset(root).load("one")
    assert longer_future.metadata["full_response_tokens"] == 100
    np.testing.assert_array_equal(original.response_index, longer_future.response_index)
    np.testing.assert_array_equal(
        original["reanchor_bucket_transport"],
        longer_future["reanchor_bucket_transport"],
    )
    assert "full_response_tokens" not in longer_future.arrays


def _fake_label_dataset(monkeypatch, labels, *, source_id="source-a", split="test"):
    calls = []

    class Tensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(labels)

    class Dataset:
        def __init__(self):
            self.manifest = {"split": split}

        def metadata(self, sample_id):
            calls.append(("metadata", sample_id))
            return {"source_id": source_id}

        def prepare_evaluation_labels(self, sample_ids):
            calls.append(("prepare", sample_ids))
            return SimpleNamespace(response_labels=lambda sample: Tensor())

        def __getitem__(self, sample_id):
            return SimpleNamespace(release_attention=lambda: calls.append("release"))

    def open_dataset(root, **kwargs):
        calls.append(("open", kwargs))
        return Dataset()

    dataset_module = ModuleType("research_dataset")
    dataset_module.open_research_dataset = open_dataset
    monkeypatch.setitem(sys.modules, "research_dataset", dataset_module)
    return calls


def test_labels_are_posthoc_predictor_q_plus_one_with_unannotated_preserved(
    tmp_path, monkeypatch
):
    calls = _fake_label_dataset(monkeypatch, [0, 1, -100, 2])
    scans = ScanDataset(_capture(tmp_path))
    sample = scans.load("one", fields=())
    labels = ScanLabelStore(scans)
    assert calls == []
    result = labels.load(sample)
    np.testing.assert_array_equal(result, [0, 1, -1, -1])
    assert calls[0] == ("open", {"device": "cpu", "retain_embedded_labels": True})
    assert ("prepare", ["one"]) in calls
    assert calls[-1] == "release"


@pytest.mark.parametrize(
    ("source", "split", "labels", "message"),
    [
        ("wrong-source", "test", [0, 1, 0, 0], "source_id"),
        ("source-a", "train", [0, 1, 0, 0], "split"),
        ("source-a", "test", [0, 1, 0], "length"),
    ],
)
def test_label_join_validates_identity(
    tmp_path, monkeypatch, source, split, labels, message
):
    _fake_label_dataset(monkeypatch, labels, source_id=source, split=split)
    scans = ScanDataset(_capture(tmp_path))
    with pytest.raises(ValueError, match=message):
        ScanLabelStore(scans).load(scans.load("one", fields=()))
