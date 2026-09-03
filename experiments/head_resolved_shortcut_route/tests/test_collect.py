import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
collect_module = pytest.importorskip("experiments.head_resolved_shortcut_route.collect")


class FakeAttention:
    def __init__(self, token: int):
        self.token_ids = torch.tensor([1, token, 63])
        self.response_idx = 2


class FakeSample:
    def __init__(self, sample_id: str, task_type: str, token: int):
        self.sample_id = sample_id
        self.task_type = task_type
        self.source_id = f"source-{token}"
        self.generator_model = "generator"
        self._attention = FakeAttention(token)
        self.attention_calls = 0
        self.release_calls = 0

    def attention(self):
        self.attention_calls += 1
        return self._attention

    def release_attention(self):
        self.release_calls += 1


def install_collection_fakes(
    tmp_path, monkeypatch, *, save=None, interleaved_task=False
):
    samples = {
        "summary": FakeSample("summary", "Summary", 10),
        "qa-1": FakeSample("qa-1", "QA", 11),
        "data2txt": FakeSample("data2txt", "Data2txt", 12),
        "qa-2": FakeSample("qa-2", "qa", 13),
    }
    if interleaved_task:
        samples = {
            "summary": samples["summary"],
            "summary-2": FakeSample("summary-2", "Summary", 14),
            **{name: sample for name, sample in samples.items() if name != "summary"},
        }

    class FakeDataset:
        def __init__(self):
            self.sample_ids = list(samples)
            self.manifest = {"split": "train", "cache_schema": "ragtruth-v1"}

        def __getitem__(self, sample_id):
            return samples[sample_id]

    dataset_calls = []

    def open_dataset(path, **kwargs):
        dataset_calls.append((path, kwargs))
        return FakeDataset()

    capture_calls = []

    class FakeObserver:
        def capture(self, token_ids, response_start, evidence_mask):
            assert response_start == 2
            assert evidence_mask.dtype == torch.bool
            assert evidence_mask.shape == (2,)
            capture_calls.append(int(token_ids[1]))
            return SimpleNamespace(token_ids=token_ids, response_start=response_start)

    build_calls = []

    def build_artifact(captured, *, top_k, cover_mass):
        build_calls.append((int(captured.token_ids[1]), top_k, cover_mass))
        return SimpleNamespace(
            response_start=captured.response_start,
            events=SimpleNamespace(query_position=torch.tensor([1])),
        )

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    monkeypatch.setattr(collect_module, "open_research_dataset", open_dataset)
    monkeypatch.setattr(
        collect_module,
        "load_source_info",
        lambda _path: {sample.source_id: {} for sample in samples.values()},
    )
    monkeypatch.setattr(
        collect_module,
        "build_evidence_mask",
        lambda *_args: np.zeros(2, dtype=bool),
    )
    monkeypatch.setattr(
        collect_module.NativeRouteObserver,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeObserver(),
    )
    monkeypatch.setattr(collect_module, "build_route_artifact", build_artifact)
    monkeypatch.setattr(
        collect_module,
        "save_route_artifact",
        save or (lambda path, _artifact: Path(path).write_bytes(b"route-artifact")),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    source_info = tmp_path / "source_info.jsonl"
    source_info.write_text('{"source_id":"fixture"}\n', encoding="utf-8")
    model = tmp_path / "observer"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"llama"}\n', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"frozen weights")
    return SimpleNamespace(
        samples=samples,
        dataset_calls=dataset_calls,
        capture_calls=capture_calls,
        build_calls=build_calls,
        source_info=source_info,
        model=model,
    )


def test_capture_split_is_label_free_head_route_collection_and_resumes(
    tmp_path, monkeypatch
):
    fixture = install_collection_fakes(tmp_path, monkeypatch)
    output = tmp_path / "shortcut-routes"
    common = {
        "split_root": tmp_path / "train-cache",
        "source_info": fixture.source_info,
        "model_path": fixture.model,
        "output_root": output,
        "device": "cpu",
        "dtype": torch.float32,
        "top_k": 8,
        "cover_mass": 0.8,
    }

    first = collect_module.capture_split(**common, limit=1)

    assert first == json.loads((output / "manifest.json").read_text())
    assert set(first) == {
        "schema",
        "version",
        "artifact_schema",
        "dataset_identity",
        "source_identity",
        "observer_identity",
        "model_dtype",
        "top_k",
        "cover_mass",
        "task_types",
        "index",
        "dataset_candidates",
        "samples",
        "complete",
        "labels_used",
    }
    assert first["schema"] == collect_module.SCHEMA
    assert first["version"] == collect_module.VERSION == 1
    assert first["artifact_schema"] == collect_module.ARTIFACT_SCHEMA
    assert first["model_dtype"] == "torch.float32"
    assert first["top_k"] == 8
    assert first["cover_mass"] == 0.8
    assert first["samples"] == 3
    assert first["complete"] is False
    assert first["labels_used"] is False
    assert first["dataset_identity"]["manifest"]["split"] == "train"
    assert first["source_identity"]["path"] == str(fixture.source_info.resolve())
    assert first["observer_identity"]["path"] == str(fixture.model.resolve())
    assert fixture.dataset_calls == [
        (
            common["split_root"],
            {
                "device": "cpu",
                "verify_hashes": True,
                "retain_embedded_labels": False,
            },
        )
    ]
    assert fixture.capture_calls == [10, 11, 12]
    assert fixture.build_calls == [
        (10, 8, 0.8),
        (11, 8, 0.8),
        (12, 8, 0.8),
    ]

    falsely_complete = {**first, "complete": True}
    (output / "manifest.json").write_text(
        json.dumps(falsely_complete), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not cover the canonical dataset"):
        collect_module.capture_split(**common, limit=1)
    (output / "manifest.json").write_text(json.dumps(first), encoding="utf-8")

    rows = collect_module.load_index(output)
    assert [row["sample_id"] for row in rows] == ["summary", "qa-1", "data2txt"]
    assert [row["task_type"] for row in rows] == ["Summary", "QA", "Data2txt"]
    assert set(rows[0]) == {
        "sample_id",
        "source_id",
        "task_type",
        "generator_model",
        "path",
        "bytes",
        "sha256",
        "events",
        "response_start",
    }
    assert rows[0]["events"] == 1
    assert rows[0]["response_start"] == 2
    assert rows[0]["path"] == "summary.npz"
    assert len(rows[0]["sha256"]) == 64
    assert (output / "samples" / rows[0]["path"]).read_bytes() == b"route-artifact"
    assert not list(output.rglob("*.tmp"))
    assert not list(output.rglob("*.tmp.npz"))

    second = collect_module.capture_split(**common)

    assert second["samples"] == 4
    assert second["complete"] is True
    assert fixture.capture_calls == [10, 11, 12, 13]
    assert fixture.samples["summary"].attention_calls == 1
    assert fixture.samples["qa-1"].attention_calls == 1
    assert fixture.samples["qa-2"].release_calls == 1
    assert [row["sample_id"] for row in collect_module.load_index(output)] == [
        "summary",
        "qa-1",
        "data2txt",
        "qa-2",
    ]

    complete = collect_module.capture_split(**common, limit=1)
    assert complete == second
    assert fixture.capture_calls == [10, 11, 12, 13]

    with pytest.raises(ValueError, match="changed identity"):
        collect_module.capture_split(**{**common, "cover_mass": 0.9})
    assert fixture.capture_calls == [10, 11, 12, 13]

    tampered = {**second, "labels_used": True}
    (output / "manifest.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="changed identity: labels_used"):
        collect_module.capture_split(**common)


def test_atomic_artifact_failure_leaves_no_committed_row_and_can_resume(
    tmp_path, monkeypatch
):
    saves = []

    def flaky_save(path, _artifact):
        path = Path(path)
        path.write_bytes(b"partial" if not saves else b"complete")
        saves.append(path)
        if len(saves) == 1:
            raise RuntimeError("interrupted save")

    fixture = install_collection_fakes(tmp_path, monkeypatch, save=flaky_save)
    output = tmp_path / "routes"
    common = {
        "split_root": tmp_path / "train-cache",
        "source_info": fixture.source_info,
        "model_path": fixture.model,
        "output_root": output,
        "device": "cpu",
        "dtype": torch.float32,
        "limit": 1,
    }

    with pytest.raises(RuntimeError, match="interrupted save"):
        collect_module.capture_split(**common)

    assert not (output / "index.jsonl").exists()
    assert not (output / "samples" / "summary.npz").exists()
    assert not list(output.rglob("*.tmp.npz"))
    assert fixture.samples["summary"].release_calls == 1

    manifest = collect_module.capture_split(**common)

    assert manifest["samples"] == 3
    assert len(collect_module.load_index(output)) == 3
    assert (output / "samples" / "summary.npz").read_bytes() == b"complete"
    assert not list(output.rglob("*.tmp.npz"))
    assert fixture.samples["summary"].release_calls == 2


def test_partial_resume_keeps_the_journal_in_canonical_dataset_order(
    tmp_path, monkeypatch
):
    fixture = install_collection_fakes(tmp_path, monkeypatch, interleaved_task=True)
    output = tmp_path / "routes"
    common = {
        "split_root": tmp_path / "train-cache",
        "source_info": fixture.source_info,
        "model_path": fixture.model,
        "output_root": output,
        "device": "cpu",
        "dtype": torch.float32,
    }

    partial = collect_module.capture_split(**common, limit=1)
    assert partial["complete"] is False
    assert [row["sample_id"] for row in collect_module.load_index(output)] == [
        "summary",
        "qa-1",
        "data2txt",
    ]

    complete = collect_module.capture_split(**common)
    assert complete["complete"] is True
    assert [row["sample_id"] for row in collect_module.load_index(output)] == [
        "summary",
        "summary-2",
        "qa-1",
        "data2txt",
        "qa-2",
    ]


def test_resume_rejects_duplicate_missing_or_changed_artifacts(tmp_path):
    root = tmp_path / "state"
    samples = root / "samples"
    samples.mkdir(parents=True)
    artifact = samples / "first.npz"
    artifact.write_bytes(b"artifact")
    row = {
        "sample_id": "first",
        "path": "first.npz",
        "bytes": artifact.stat().st_size,
        "sha256": collect_module._sha256(artifact),
    }

    collect_module._validate_resume_index(root, [row])
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        collect_module._validate_resume_index(root, [row, row])
    with pytest.raises(ValueError, match="path does not match"):
        collect_module._validate_resume_index(root, [{**row, "path": "../first.npz"}])
    with pytest.raises(ValueError, match="missing sample artifact"):
        collect_module._validate_resume_index(
            root,
            [{"sample_id": "missing", "path": "missing.npz", "bytes": 1}],
        )
    with pytest.raises(ValueError, match="size-changed"):
        collect_module._validate_resume_index(root, [{**row, "bytes": 1}])
    artifact.write_bytes(b"Artifact")
    with pytest.raises(ValueError, match="digest-changed"):
        collect_module._validate_resume_index(root, [row])


def test_capture_all_uses_shortcut_route_directory_and_forwards_fixed_knobs(
    tmp_path, monkeypatch
):
    calls = []

    def capture_split(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(collect_module, "capture_split", capture_split)
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    result = collect_module.capture_all(
        split_roots=(cache / "train", cache / "test"),
        source_info=tmp_path / "source_info.jsonl",
        model_path=tmp_path / "model",
        output_root=output,
        device="cpu",
        dtype=torch.float32,
        limit=2,
        top_k=13,
        cover_mass=0.97,
    )

    expected = [
        output / collect_module.STATE_DIRECTORY / "train",
        output / collect_module.STATE_DIRECTORY / "test",
    ]
    pairs = list(zip(expected, (cache / "train", cache / "test"), strict=True))
    assert collect_module.STATE_DIRECTORY == "shortcut_route_state"
    assert [call["output_root"] for call in calls] == expected
    assert all(call["top_k"] == 13 for call in calls)
    assert all(call["cover_mass"] == 0.97 for call in calls)
    assert result == {task: pairs for task in collect_module.TASK_TYPES}


def test_public_capture_api_has_no_retired_branch_or_fake_chunk_controls():
    parameters = inspect.signature(collect_module.capture_split).parameters
    assert "predictor_chunk" not in parameters
    assert "logit_chunk" not in parameters
    assert "route_cover_mass" not in parameters
    source = Path(collect_module.__file__).read_text(encoding="utf-8")
    assert "FunctionalTraceReplay" not in source
    assert "BRANCH_NAMES" not in source
    assert "REGISTER_NAMES" not in source
    assert "score_inputs" not in source


def test_input_stamps_are_readable_and_detect_changes(tmp_path):
    source = tmp_path / "source_info.jsonl"
    source.write_text('{"source_id":"same"}\n', encoding="utf-8")
    first = collect_module.file_stamp(source)
    assert first["path"] == str(source.resolve())
    assert first["size"] == source.stat().st_size
    assert len(first["sha256"]) == 64

    model = tmp_path / "observer"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"frozen weights")
    before = collect_module.model_stamp(model)
    assert len(before["sha256"]) == 64
    assert before["files"][0][:2] == ["model.safetensors", 14]
    assert len(before["files"][0][2]) == 64
    weights.write_bytes(b"changed weights")
    assert collect_module.model_stamp(model) != before
