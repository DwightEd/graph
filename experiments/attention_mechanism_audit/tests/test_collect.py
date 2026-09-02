import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
collect_module = pytest.importorskip("experiments.attention_mechanism_audit.collect")


SCORE_INPUTS = (
    "full_logprob",
    "full_margin",
    "no_evidence_logprob",
    "no_evidence_margin",
    "no_history_logprob",
    "no_history_margin",
    "no_evidence_history_logprob",
    "no_evidence_history_margin",
)


def _trace(response_tokens: int = 1):
    return {
        "attention_role_mass": torch.zeros(2, response_tokens, 3, 4),
        "edge_role_mass": torch.zeros(2, response_tokens, 3, 4),
        "head_role_write_norm": torch.zeros(2, response_tokens, 3, 4),
        "role_head_coherence": torch.zeros(2, response_tokens, 4),
        "route_source_index": torch.zeros(
            2, response_tokens, 3, 2, 2, dtype=torch.int32
        ),
        "route_source_magnitude": torch.zeros(2, response_tokens, 3, 2, 2),
        "route_source_remainder": torch.zeros(2, response_tokens, 3, 2),
        "route_source_cover_size": torch.zeros(
            2, response_tokens, 3, 2, dtype=torch.int32
        ),
    }


def _capture(token_ids, evidence_mask=None):
    if evidence_mask is None:
        evidence_mask = torch.zeros(2, dtype=torch.bool)
    return {
        "token_ids": token_ids.clone(),
        "response_start": 2,
        "evidence_mask": torch.as_tensor(evidence_mask, dtype=torch.bool).clone(),
        "peak_cuda_reserved_bytes": int(token_ids[1]),
        "trace": _trace(),
        "score_inputs": {
            name: torch.tensor([-float(index + 1)])
            for index, name in enumerate(SCORE_INPUTS)
        },
    }


def test_capture_split_captures_all_tasks_and_resumes_without_replaying_samples(
    tmp_path, monkeypatch
):
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

    samples = {
        "summary": FakeSample("summary", "Summary", 10),
        "qa-1": FakeSample("qa-1", "QA", 11),
        "data2txt": FakeSample("data2txt", "Data2txt", 12),
        "qa-2": FakeSample("qa-2", "qa", 13),
    }

    class FakeDataset:
        def __init__(self):
            self.sample_ids = list(samples)
            self.manifest = {"split": "train"}

        def __getitem__(self, sample_id):
            return samples[sample_id]

    capture_calls = []

    class FakeReplay:
        def capture(
            self,
            token_ids,
            response_start,
            evidence_mask,
            *,
            predictor_chunk,
            top_k,
            logit_chunk,
            route_cover_mass,
        ):
            assert evidence_mask.dtype == torch.bool
            assert evidence_mask.shape == (2,)
            assert predictor_chunk == 128
            assert top_k == 8
            assert logit_chunk == 64
            assert route_cover_mass == 0.8
            capture_calls.append(int(token_ids[1]))
            return _capture(token_ids, evidence_mask)

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    monkeypatch.setattr(
        collect_module,
        "open_research_dataset",
        lambda *_args, **_kwargs: FakeDataset(),
    )
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
        collect_module.FunctionalTraceReplay,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeReplay(),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    output = tmp_path / "mechanism-state"
    common = {
        "split_root": tmp_path / "train-cache",
        "source_info": tmp_path / "source_info.jsonl",
        "model_path": tmp_path / "model",
        "output_root": output,
        "device": "cpu",
        "dtype": torch.float32,
    }
    first = collect_module.capture_split(**common, limit=1)

    assert first["samples"] == 3
    assert first["resumed_samples"] == 0
    assert first["new_samples"] == 3
    assert first["selected_samples_seen"] == 3
    assert first["eligible_samples"] is None
    assert first["complete"] is False
    assert first["version"] == collect_module.VERSION == 7
    assert first["labels_used"] is False
    assert first["task_types"] == ["QA", "Summary", "Data2txt"]
    assert first["capture_spec"] == {
        "branches": ["full", "no_evidence", "no_history", "no_evidence_history"],
        "pathway_contrasts": ["evidence", "history", "interaction"],
        "source_roles": [
            "evidence",
            "other_prompt",
            "response_history",
            "predictor_self",
        ],
        "route_roles": ["evidence", "response_history"],
        "route_cover_mass": 0.8,
        "top_k": 8,
    }
    assert "intervention_batch" not in first
    assert capture_calls == [10, 11, 12]
    first_rows = [
        json.loads(line)
        for line in (output / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_id"] for row in first_rows] == [
        "summary",
        "qa-1",
        "data2txt",
    ]
    assert [row["task_type"] for row in first_rows] == [
        "Summary",
        "QA",
        "Data2txt",
    ]
    assert first_rows[0]["prompt_tokens"] == 2
    assert first_rows[0]["evidence_tokens"] == 0
    assert first_rows[0]["response_tokens"] == 1
    assert first_rows[0]["target_response_sha256"] == (
        "c713558bfa07123bfbdeec433f51384cc214f60e3f4484222f750e8bf1454652"
    )
    assert first_rows[0]["artifact_contract"]["version"] == 7
    assert first_rows[0]["token_ids_sha256"]
    assert first_rows[0]["evidence_mask_sha256"]
    saved = torch.load(output / "samples" / "summary.pt", weights_only=True)
    assert set(saved) == {
        "artifact_contract",
        "token_ids",
        "response_start",
        "evidence_mask",
        "trace",
        "score_inputs",
    }
    assert saved["evidence_mask"].dtype == torch.bool
    assert saved["evidence_mask"].shape == (2,)
    assert tuple(saved["score_inputs"]) == SCORE_INPUTS
    assert set(saved["trace"]) == set(_trace())

    second = collect_module.capture_split(**common)

    assert second["samples"] == 4
    assert second["resumed_samples"] == 3
    assert second["new_samples"] == 1
    assert second["selected_samples_seen"] == 4
    assert second["eligible_samples"] == 4
    assert second["complete"] is True
    assert capture_calls == [10, 11, 12, 13]
    assert samples["summary"].attention_calls == 1
    assert samples["qa-1"].attention_calls == 1
    final_rows = [
        json.loads(line)
        for line in (output / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_id"] for row in final_rows] == [
        "summary",
        "qa-1",
        "data2txt",
        "qa-2",
    ]
    assert [row["task_type"] for row in final_rows] == [
        "Summary",
        "QA",
        "Data2txt",
        "QA",
    ]
    assert len({row["sample_id"] for row in final_rows}) == len(final_rows)

    third = collect_module.capture_split(**common, limit=1)

    assert third["complete"] is True
    assert third["eligible_samples"] == 4
    assert third["selected_samples_seen"] == 3
    assert capture_calls == [10, 11, 12, 13]

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    draft = json.loads(json.dumps(manifest))
    del draft["capture_spec"]["route_roles"]
    manifest_path.write_text(json.dumps(draft), encoding="utf-8")
    with pytest.raises(ValueError, match="changed identity"):
        collect_module.capture_split(**common)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="changed identity"):
        collect_module.capture_split(**common, route_cover_mass=0.9)
    assert capture_calls == [10, 11, 12, 13]

    manifest_path.unlink()
    with pytest.raises(ValueError):
        collect_module.capture_split(**common)
    assert capture_calls == [10, 11, 12, 13]

    manifest_path.write_text(
        json.dumps(
            {"schema": collect_module.SCHEMA, "version": collect_module.VERSION - 1}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="changed identity"):
        collect_module.capture_split(**common)


def test_input_identity_is_stable_after_relocation(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    source_content = '{"source_id": "same"}\n'
    for index, root in enumerate((first, second)):
        root.mkdir()
        (root / "source_info.jsonl").write_text(source_content, encoding="utf-8")
        model = root / f"observer-{index}"
        model.mkdir()
        (model / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"same frozen weights")

    assert collect_module._file_identity(
        first / "source_info.jsonl"
    ) == collect_module._file_identity(second / "source_info.jsonl")
    first_identity = collect_module._model_identity(first / "observer-0")
    second_identity = collect_module._model_identity(second / "observer-1")
    assert first_identity == second_identity

    (second / "observer-1" / "model.safetensors").write_bytes(b"different frozen wt")
    assert len(b"different frozen wt") == len(b"same frozen weights")
    assert collect_module._model_identity(second / "observer-1") != first_identity


def test_capture_all_uses_its_formal_state_directory_without_touching_old_states(
    tmp_path, monkeypatch
):
    output = tmp_path / "output"
    old_manifest = output / "mechanism_state" / "train" / "manifest.json"
    old_manifest.parent.mkdir(parents=True)
    old_contents = b'{"schema":"old-mechanism-state","version":5}\n'
    old_manifest.write_bytes(old_contents)
    cache = tmp_path / "cache"
    calls = []

    def capture_split(**kwargs):
        calls.append(kwargs["output_root"])

    monkeypatch.setattr(collect_module, "capture_split", capture_split)
    result = collect_module.capture_all(
        split_roots=(cache / "train", cache / "test"),
        source_info=tmp_path / "source_info.jsonl",
        model_path=tmp_path / "model",
        output_root=output,
    )

    expected = [
        output / collect_module.STATE_DIRECTORY / "train",
        output / collect_module.STATE_DIRECTORY / "test",
    ]
    pairs = list(zip(expected, (cache / "train", cache / "test"), strict=True))
    assert calls == expected
    assert result == {task: pairs for task in collect_module.TASK_TYPES}
    assert old_manifest.read_bytes() == old_contents


def test_save_rejects_response_misaligned_mechanism_trace(tmp_path):
    capture = _capture(torch.tensor([1, 2, 3]))
    capture["trace"]["head_source_entropy"] = torch.zeros(2, 2, 3)

    with pytest.raises(ValueError, match="not response-token aligned"):
        collect_module._save(tmp_path / "sample.pt", capture)

    assert not (tmp_path / "sample.pt").exists()


def test_save_rejects_prompt_misaligned_evidence_mask(tmp_path):
    capture = _capture(torch.tensor([1, 2, 3]))
    capture["evidence_mask"] = torch.zeros(3, dtype=torch.bool)

    with pytest.raises(ValueError, match="evidence_mask is not prompt-token aligned"):
        collect_module._save(tmp_path / "sample.pt", capture)

    assert not (tmp_path / "sample.pt").exists()


def test_resume_index_requires_unique_ids_and_existing_artifacts(tmp_path):
    root = tmp_path / "state"
    samples = root / "samples"
    samples.mkdir(parents=True)
    (samples / "first.pt").write_bytes(b"artifact")
    row = {"sample_id": "first", "path": "first.pt"}

    collect_module._validate_resume_index(root, [row])
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        collect_module._validate_resume_index(root, [row, row])
    with pytest.raises(ValueError, match="missing sample artifact"):
        collect_module._validate_resume_index(
            root, [{"sample_id": "missing", "path": "missing.pt"}]
        )
    with pytest.raises(ValueError, match="size-changed"):
        collect_module._validate_resume_index(
            root,
            [{"sample_id": "first", "path": "first.pt", "bytes": 1}],
        )
