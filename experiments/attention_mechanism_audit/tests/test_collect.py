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
    layers, heads, registers, roles, top_k = 2, 3, 2, 4, 2
    route = (layers, response_tokens, registers, top_k)
    register = (layers, response_tokens, registers)
    role = (layers, response_tokens, heads, registers, roles)
    trace = {
        "register_route_source_index": torch.full(route, -1, dtype=torch.int32),
        "register_route_head_index": torch.full(route, -1, dtype=torch.int16),
        "register_route_magnitude": torch.zeros(route),
        "register_route_contribution": torch.zeros(route),
        "register_route_root_contribution": torch.zeros(route),
        "register_route_carrier_contribution": torch.zeros(route),
        "register_route_gate_contribution": torch.zeros(route),
        "register_route_remainder_magnitude": torch.zeros(register),
        "register_route_remainder_contribution": torch.zeros(register),
        "register_route_remainder_root_contribution": torch.zeros(register),
        "register_route_remainder_carrier_contribution": torch.zeros(register),
        "register_route_remainder_gate_contribution": torch.zeros(register),
        "register_route_cover_size": torch.zeros(register, dtype=torch.int32),
        "register_role_mass": torch.zeros(role),
        "register_role_contribution": torch.zeros(role),
        "register_role_root_contribution": torch.zeros(role),
        "register_role_carrier_contribution": torch.zeros(role),
        "register_role_gate_contribution": torch.zeros(role),
        "register_role_effective_routes": torch.zeros(
            layers, response_tokens, registers, roles
        ),
        "register_norm": torch.zeros(layers, response_tokens, registers, 4),
        "register_mlp_alignment": torch.zeros(register),
        "register_conservation_error": torch.zeros(register),
        "register_attention_edge_error": torch.zeros(register),
        "register_step_gram": torch.zeros(layers, response_tokens, registers, layers),
        "interaction_norm": torch.zeros(layers, response_tokens, 4),
        "final_register_norm": torch.zeros(1, response_tokens, registers),
    }
    for family in ("attention", "edge"):
        trace[f"prompt_{family}_effective_sources"] = torch.zeros(
            layers, response_tokens
        )
        trace[f"prompt_{family}_effective_rank"] = torch.zeros(layers, response_tokens)
        trace[f"prompt_{family}_anchor_index"] = torch.full(
            (layers, response_tokens, heads), -1, dtype=torch.int32
        )
    return trace


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

    output = tmp_path / "dual-register-state"
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps({"schema": "old", "version": 7, "samples": 0}),
        encoding="utf-8",
    )
    common = {
        "split_root": tmp_path / "train-cache",
        "source_info": tmp_path / "source_info.jsonl",
        "model_path": tmp_path / "model",
        "output_root": output,
        "device": "cpu",
        "dtype": torch.float32,
        "predictor_chunk": 128,
        "top_k": 8,
    }
    first = collect_module.capture_split(**common, limit=1)

    assert first["samples"] == 3
    assert first["resumed_samples"] == 0
    assert first["new_samples"] == 3
    assert first["selected_samples_seen"] == 3
    assert first["eligible_samples"] is None
    assert first["complete"] is False
    assert first["version"] == collect_module.VERSION == 8
    assert first["labels_used"] is False
    assert first["task_types"] == ["QA", "Summary", "Data2txt"]
    assert first["capture_spec"] == {
        "branches": ["full", "no_evidence", "no_history", "no_evidence_history"],
        "registers": ["evidence_adoption", "autonomous_history"],
        "register_stages": [
            "input_state",
            "attention_write",
            "mlp_write",
            "output_state",
        ],
        "source_roles": [
            "evidence",
            "other_prompt",
            "response_history",
            "predictor_self",
        ],
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
    assert first_rows[0]["target_token_ids"] == [63]
    assert first_rows[0]["artifact_contract"]["version"] == 8
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
    del draft["capture_spec"]["registers"]
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


def test_input_stamps_are_readable_and_detect_changes(tmp_path):
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

    source_stamp = collect_module.file_stamp(first / "source_info.jsonl")
    assert source_stamp["path"].endswith("source_info.jsonl")
    assert source_stamp["size"] == len(source_content.encode())
    first_identity = collect_module.model_stamp(first / "observer-0")
    second_identity = collect_module.model_stamp(second / "observer-1")
    assert first_identity["path"] != second_identity["path"]
    assert [entry[:2] for entry in first_identity["files"]] == [
        entry[:2] for entry in second_identity["files"]
    ]

    (second / "observer-1" / "model.safetensors").write_bytes(b"changed weights")
    assert collect_module.model_stamp(second / "observer-1") != second_identity


def test_capture_all_uses_its_formal_state_directory_without_touching_old_states(
    tmp_path, monkeypatch
):
    assert collect_module.STATE_DIRECTORY == "dual_register_state"
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
