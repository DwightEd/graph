from __future__ import annotations

import json

import pytest
import torch

import formal_cache
from cache import sha256
from formal_cache import (
    FORMAL_CACHE_SCHEMA,
    FORMAL_TENSOR_FIELDS,
    formal_fingerprint,
    load_formal_sample,
    read_formal_sample_metadata,
)
from research_dataset import FormalResearchDataset

from experiments.reanchor_flow import subset
from experiments.reanchor_flow.subset_data import (
    SampleRecord,
    inspect_records,
    select_records,
)


def _spec(*, split: str = "test") -> dict:
    return {
        "attention_cache_schema": FORMAL_CACHE_SCHEMA,
        "split": split,
        "cache_dtype": "torch.float16",
        "attention_floor": 0.01,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "model_path": "/models/tiny-llama",
    }


def _payload(
    spec: dict,
    *,
    task_type=None,
    labels=None,
    malformed_attention: bool = False,
) -> dict:
    value = {
        "attention_cache_schema": FORMAL_CACHE_SCHEMA,
        "attention_cache_fingerprint": formal_fingerprint(spec),
        "response_id": "one",
        "source_id": "source-one",
        "split": spec["split"],
        "cache_dtype": spec["cache_dtype"],
        "num_attention_layers": 1,
        "num_attention_heads": 1,
        "quality": "good",
        "was_truncated": False,
        "response_idx": 2,
        "token_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "attention_diagonal": torch.ones((1, 1, 3), dtype=torch.float16),
        "response_row_ptr": torch.tensor([0, 0], dtype=torch.long),
        "response_column_indices": torch.empty(0, dtype=torch.long),
        "response_values": torch.empty(0, dtype=torch.float16),
        "attention_floor": spec["attention_floor"],
        "y_token": torch.tensor([0, 0, 1]) if labels is None else labels,
    }
    if task_type is not None:
        value["task_type"] = task_type
    if malformed_attention:
        value["token_ids"] = torch.tensor([float("nan")])
        value["attention_diagonal"] = torch.tensor([float("nan")])
        value["response_row_ptr"] = torch.tensor([99])
    return value


def _formal_split(
    tmp_path,
    *,
    task_type=None,
    labels=None,
    malformed_attention: bool = False,
):
    root = tmp_path / "test"
    root.mkdir()
    spec = _spec()
    path = root / "attention_one.pt"
    torch.save(
        _payload(
            spec,
            task_type=task_type,
            labels=labels,
            malformed_attention=malformed_attention,
        ),
        path,
    )
    digest = sha256(path)
    manifest = {
        "state": "complete",
        "cache_file_names": [path.name],
        "matched_samples": 1,
        "cache_files": 1,
        "cache_files_sha256": {path.name: digest},
        "attention_cache_spec": spec,
        "attention_cache_fingerprint": formal_fingerprint(spec),
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, path, digest, spec


def test_formal_metadata_reader_is_mmap_whitelisted_and_tensor_blind(
    tmp_path, monkeypatch
) -> None:
    root, path, digest, spec = _formal_split(
        tmp_path,
        labels=torch.tensor([7]),
        malformed_attention=True,
    )
    del root
    real_load = torch.load
    calls = []

    def observe_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_load(*args, **kwargs)

    monkeypatch.setattr(formal_cache.torch, "load", observe_load)
    metadata = read_formal_sample_metadata(
        path,
        digest,
        split="test",
        spec=spec,
        verify_hash=False,
    )
    assert calls == [{"map_location": "cpu", "weights_only": True, "mmap": True}]
    assert metadata["response_id"] == "one"
    assert not FORMAL_TENSOR_FIELDS.intersection(metadata)


def test_formal_attention_load_can_leave_invalid_embedded_labels_sealed(
    tmp_path,
) -> None:
    _root, path, digest, spec = _formal_split(
        tmp_path,
        labels=torch.tensor([9]),
    )
    sample, labels, metadata = load_formal_sample(
        path,
        digest,
        split="test",
        spec=spec,
        verify_hash=False,
        retain_labels=False,
    )
    assert sample.sample_id == "one"
    assert labels is None
    assert "y_token" not in metadata
    with pytest.raises(ValueError, match="formal y_token"):
        load_formal_sample(
            path,
            digest,
            split="test",
            spec=spec,
            verify_hash=False,
            retain_labels=True,
        )


def test_label_retaining_load_keeps_legacy_torch_load_signature(
    tmp_path, monkeypatch
) -> None:
    _root, path, digest, spec = _formal_split(tmp_path)
    real_load = torch.load
    calls = []

    def observe_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_load(*args, **kwargs)

    monkeypatch.setattr(formal_cache.torch, "load", observe_load)
    _sample, labels, _payload_value = load_formal_sample(
        path,
        digest,
        split="test",
        spec=spec,
        verify_hash=False,
    )
    assert labels.tolist() == [0, 0, 1]
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_formal_dataset_metadata_and_source_info_fill_task_without_full_load(
    tmp_path, monkeypatch
) -> None:
    root, _path, _digest, _specification = _formal_split(tmp_path)
    dataset = FormalResearchDataset(root)

    def full_load_forbidden(*_args, **_kwargs):
        raise AssertionError("metadata access loaded attention or labels")

    monkeypatch.setattr(formal_cache, "load_formal_sample", full_load_forbidden)
    records = inspect_records(
        dataset,
        sample_ids=("one",),
        source_info={"source-one": {"task_type": "Summary"}},
    )
    assert records[0].task_type == "Summary"
    assert records[0].source_id == "source-one"
    assert dataset._label_cache == {}


def test_formal_sample_metadata_preserves_label_retaining_legacy_behavior(
    tmp_path,
) -> None:
    root, _path, _digest, _specification = _formal_split(tmp_path)
    dataset = FormalResearchDataset(root, retain_labels=True)
    assert dataset["one"].source_id == "source-one"
    assert dataset._label_cache["one"].tolist() == [1]


def test_inspect_records_reads_only_explicit_metadata_ids() -> None:
    class Dataset:
        sample_ids = ["one", "two", "three"]

        def __init__(self):
            self.read = []

        def metadata(self, sample_id):
            self.read.append(sample_id)
            return {
                "source_id": f"source-{sample_id}",
                "task_type": None,
                "generator_model": "generator",
            }

    dataset = Dataset()
    records = inspect_records(
        dataset,
        sample_ids=("two",),
        source_info={"source-two": {"task_type": "QA"}},
    )
    assert dataset.read == ["two"]
    assert [record.sample_id for record in records] == ["two"]
    assert records[0].task_type == "QA"


def test_source_info_task_must_match_formal_metadata() -> None:
    class Dataset:
        sample_ids = ["one"]

        @staticmethod
        def metadata(_sample_id):
            return {
                "source_id": "source-one",
                "task_type": "QA",
                "generator_model": "",
            }

    with pytest.raises(ValueError, match="disagree on task type"):
        inspect_records(
            Dataset(),
            source_info={"source-one": {"task_type": "Summary"}},
        )


def test_automatic_selection_rejects_every_underfilled_requested_task() -> None:
    rows = (
        SampleRecord("q1", "source-q", "QA", "generator"),
        SampleRecord("q2", "source-q2", "QA", "generator"),
        SampleRecord("s1", "source-s", "Summary", "generator"),
    )
    with pytest.raises(ValueError, match="task Summary has 1 available samples; 2"):
        select_records(
            rows,
            tasks=("QA", "Summary"),
            samples_per_task=2,
            seed=7,
        )


def test_subset_capture_rejects_a_mismatched_dataset_split(
    tmp_path, monkeypatch
) -> None:
    dataset = type(
        "Dataset",
        (),
        {"manifest": {"split": "train"}, "spec": {}, "sample_ids": []},
    )()
    monkeypatch.setattr(subset, "open_research_dataset", lambda *_a, **_k: dataset)
    with pytest.raises(ValueError, match="differs from requested split"):
        subset.run_subset_split(
            None,
            None,
            tmp_path / "cache",
            tmp_path / "source.jsonl",
            tmp_path / "output",
            split="test",
            model_path=tmp_path / "model",
            model_dtype="float32",
        )
