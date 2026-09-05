from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.reanchor_flow.run import parser, validate_args
from experiments.reanchor_flow.subset_data import (
    SampleRecord,
    inspect_records,
    select_records,
)
from experiments.reanchor_flow.tests.etcc_helpers import paired_world, tiny_model


class FakeSample:
    def __init__(self, record: SampleRecord) -> None:
        self.record = record
        self.released = 0

    @property
    def source_id(self):
        return self.record.source_id

    @property
    def task_type(self):
        return self.record.task_type

    @property
    def generator_model(self):
        return self.record.generator_model

    def release_attention(self):
        self.released += 1


class FakeDataset:
    def __init__(self, records) -> None:
        self.sample_ids = [record.sample_id for record in records]
        self.samples = {record.sample_id: FakeSample(record) for record in records}

    def __getitem__(self, sample_id):
        return self.samples[sample_id]

    def labels(self):
        raise AssertionError("capture opened labels")

    def prepare_evaluation_labels(self, *_):
        raise AssertionError("capture opened labels")


def records():
    return (
        SampleRecord("q1", "source-a", "QA", "g"),
        SampleRecord("q2", "source-a", "QA", "g"),
        SampleRecord("q3", "source-b", "QA", "g"),
        SampleRecord("s1", "source-c", "Summary", "g"),
        SampleRecord("d1", "source-d", "Data2txt", "g"),
    )


def test_subset_selection_is_source_diverse_deterministic_and_label_free() -> None:
    dataset = FakeDataset(records())
    inspected = inspect_records(dataset)
    first = select_records(
        inspected,
        tasks=("QA",),
        samples_per_task=2,
        seed=19,
    )
    second = select_records(
        inspected,
        tasks=("QA",),
        samples_per_task=2,
        seed=19,
    )
    assert first == second
    assert len({record.source_id for record in first}) == 2
    assert all(sample.released == 1 for sample in dataset.samples.values())


def test_explicit_subset_preserves_requested_order() -> None:
    selected = select_records(
        records(),
        tasks=("QA", "Summary"),
        samples_per_task=1,
        seed=0,
        sample_ids=("s1", "q2"),
    )
    assert [record.sample_id for record in selected] == ["s1", "q2"]


def test_absolute_cache_model_identity_does_not_fall_back_to_basename(
    tmp_path,
) -> None:
    from experiments.reanchor_flow.subset import _model_matches

    requested = tmp_path / "requested" / "same-name"
    exact = SimpleNamespace(spec={"model_path": str(requested)})
    another = SimpleNamespace(
        spec={"model_path": str(tmp_path / "another" / "same-name")}
    )
    legacy = SimpleNamespace(spec={"model_path": "same-name"})
    assert _model_matches(exact, requested)
    assert not _model_matches(another, requested)
    assert _model_matches(legacy, requested)


@pytest.mark.parametrize("directory", ["worlds", "audits"])
def test_new_manifest_rejects_orphan_mechanism_artifacts(
    tmp_path, directory: str
) -> None:
    from experiments.reanchor_flow.subset import open_manifest

    output = tmp_path / "output"
    orphan = output / directory / "nested" / "orphan.npz"
    orphan.parent.mkdir(parents=True)
    np.savez_compressed(orphan, value=1)
    with pytest.raises(ValueError, match="no manifest"):
        open_manifest(
            output / "run_manifest.json",
            {"configuration": "new"},
            (records()[0],),
        )


def test_subset_cli_needs_no_pair_and_keeps_corridor_contract() -> None:
    command_parser = parser()
    subset = command_parser.parse_args(["subset"])
    assert subset.flow_signal == "message"
    assert subset.samples_per_task == 1
    assert subset.targets_per_sample == 1
    assert subset.carrier_scope == "response"
    assert not hasattr(subset, "pair")
    with pytest.raises(SystemExit):
        command_parser.parse_args(["corridor"])


def test_subset_argument_validation() -> None:
    command_parser = parser()
    args = command_parser.parse_args(["subset", "--samples-per-task", "0"])
    with pytest.raises(ValueError, match="samples-per-task"):
        validate_args(args)
    args = command_parser.parse_args(["subset", "--split", "all", "--sample-id", "one"])
    with pytest.raises(ValueError, match="concrete"):
        validate_args(args)


def test_subset_pipeline_resumes_valid_compact_target(tmp_path, monkeypatch) -> None:
    from experiments.reanchor_flow import subset, subset_data

    pair = paired_world()

    class AuditSample(FakeSample):
        def attention(self):
            return SimpleNamespace(
                token_ids=pair.clean_token_ids,
                response_idx=pair.response_start,
            )

    class AuditDataset(FakeDataset):
        spec = {}
        manifest = {"split": "test"}

        def __init__(self):
            record = SampleRecord("q1", "source-a", "QA", "generator")
            self.sample_ids = [record.sample_id]
            self.samples = {record.sample_id: AuditSample(record)}

    dataset = AuditDataset()

    def open_dataset(*_args, **kwargs):
        assert kwargs["retain_embedded_labels"] is False
        return dataset

    monkeypatch.setattr(subset, "open_research_dataset", open_dataset)
    monkeypatch.setattr(
        subset,
        "load_source_info",
        lambda _path: {
            "source-a": {
                "task_type": "QA",
                "prompt": "unused",
                "source_info": {"passages": "unused"},
            }
        },
    )
    monkeypatch.setattr(
        subset_data,
        "build_source_units",
        lambda *_args, **_kwargs: pair.units,
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}\n", encoding="utf-8")
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    model = tiny_model()
    tokenizer = SimpleNamespace(name_or_path="tiny-llama")
    options = dict(
        split="test",
        model_path=tmp_path / "tiny-llama",
        model_dtype="float32",
        tasks=("QA",),
        samples_per_task=1,
        targets_per_sample=1,
        target_policy="evenly-spaced",
        max_response_tokens=None,
        signal="message",
        carrier_scope="all",
        coverage=1.0,
        query_chunk=2,
        root_screen_limit=0,
        carrier_limit=1,
        saved_edges=4,
    )
    first = subset.run_subset_split(model, tokenizer, cache, source, output, **options)
    assert dataset.verify_hashes is True
    assert first == {
        "samples": 1,
        "targets": 1,
        "resumed": 0,
        "confirmed": first["confirmed"],
    }
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_complete"]
    assert manifest["labels_used_for_capture"] is False
    assert len(manifest["config_sha256"]) == 64
    audit_entry = next(iter(manifest["audits"].values()))
    assert len(audit_entry["sha256"]) == 64
    result_path = output / audit_entry["result"]
    with np.load(result_path, allow_pickle=False) as stored:
        assert int(stored["edge_saved_count"]) <= 4

    monkeypatch.setattr(
        subset,
        "audit_native_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("resume recomputed a completed audit")
        ),
    )
    second = subset.run_subset_split(model, tokenizer, cache, source, output, **options)
    assert second["targets"] == 1
    assert second["resumed"] == 1

    original_artifact = result_path.read_bytes()
    with np.load(result_path, allow_pickle=False) as stored:
        changed_artifact = {
            name: np.array(stored[name], copy=True) for name in stored.files
        }
    changed_artifact["corridor_confirmed"] = np.asarray(
        not bool(changed_artifact["corridor_confirmed"])
    )
    np.savez_compressed(result_path, **changed_artifact)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        subset.run_subset_split(model, tokenizer, cache, source, output, **options)
    result_path.write_bytes(original_artifact)
    subset.run_subset_split(model, tokenizer, cache, source, output, **options)

    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    world_path = output / manifest["samples"]["q1"]["world"]
    with np.load(world_path, allow_pickle=False) as stored:
        changed_world = {
            name: np.array(stored[name], copy=True) for name in stored.files
        }
    changed_world["token_ids"][0] += 1
    np.savez_compressed(world_path, **changed_world)
    with pytest.raises(ValueError, match="frozen sample"):
        subset.run_subset_split(model, tokenizer, cache, source, output, **options)
