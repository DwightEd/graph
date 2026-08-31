import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

import experiments.attention_mechanism_audit.audit as audit_module
from experiments.attention_mechanism_audit.audit import mechanism_effects


def test_mechanism_effects_keep_only_the_score_inputs():
    scores = {
        "full": {"target_logprob": torch.tensor([-1.0, -2.0])},
        "evidence_removed": {"target_logprob": torch.tensor([-3.0, -4.0])},
        "response_removed": {"target_logprob": torch.tensor([-2.0, -5.0])},
        "evidence_response_removed": {
            "target_logprob": torch.tensor([-4.0, -7.0]),
            "target_margin": torch.tensor([0.5, -0.5]),
        },
    }
    scores["full"]["target_margin"] = torch.tensor([0.25, -0.25])

    effects = mechanism_effects(scores)

    torch.testing.assert_close(
        effects["evidence_message_effect"], torch.tensor([2.0, 2.0])
    )
    torch.testing.assert_close(
        effects["response_message_effect"], torch.tensor([1.0, 3.0])
    )
    assert set(effects) == {
        "evidence_message_effect",
        "response_message_effect",
        "evidence_response_removed_margin",
        "full_margin",
    }
    assert effects["evidence_response_removed_margin"] is scores[
        "evidence_response_removed"
    ]["target_margin"]
    assert effects["full_margin"] is scores["full"]["target_margin"]


def test_capture_split_resumes_from_the_journal_without_replaying_samples(
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
        "qa-2": FakeSample("qa-2", "qa", 12),
        "qa-3": FakeSample("qa-3", "QA", 13),
    }

    class FakeDataset:
        sample_ids = list(samples)
        manifest = {"split": "train"}

        def __getitem__(self, sample_id):
            return samples[sample_id]

    capture_calls = []

    class FakeReplay:
        checkpoint = "frozen-observer"

        def capture(self, token_ids, response_start, _roles, **kwargs):
            capture_calls.append((int(token_ids[1]), kwargs["retain_raw"]))
            target = token_ids[response_start:].clone()
            return {
                "token_ids": token_ids.clone(),
                "target_ids": target,
                "peak_cuda_reserved_bytes": int(token_ids[1]),
                "scores": {
                    "full": {
                        "target_logprob": torch.tensor([-1.0]),
                        "target_margin": torch.tensor([0.5]),
                    },
                    "evidence_removed": {
                        "target_logprob": torch.tensor([-2.0])
                    },
                    "response_removed": {
                        "target_logprob": torch.tensor([-3.0])
                    },
                    "evidence_response_removed": {
                        "target_logprob": torch.tensor([-4.0]),
                        "target_margin": torch.tensor([-0.5]),
                    },
                },
            }

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    monkeypatch.setattr(
        audit_module, "open_research_dataset", lambda *_args, **_kwargs: FakeDataset()
    )
    monkeypatch.setattr(
        audit_module,
        "load_source_info",
        lambda _path: {sample.source_id: {} for sample in samples.values()},
    )
    monkeypatch.setattr(
        audit_module,
        "build_prompt_role_ids",
        lambda *_args: np.zeros(2, dtype=np.int8),
    )
    monkeypatch.setattr(
        audit_module.FunctionalTraceReplay,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeReplay(),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    output = tmp_path / "traces"
    common = {
        "split_root": tmp_path / "train-cache",
        "source_info": tmp_path / "source_info.jsonl",
        "model_path": tmp_path / "model",
        "output_root": output,
        "device": "cpu",
        "dtype": torch.float32,
    }
    first = audit_module.capture_split(**common, limit=2)

    assert first["samples"] == 2
    assert first["resumed_samples"] == 0
    assert first["new_samples"] == 2
    assert first["selected_qa_seen"] == 2
    assert first["eligible_qa"] is None
    assert first["complete"] is False
    assert capture_calls == [(11, False), (12, False)]
    first_rows = [
        json.loads(line)
        for line in (output / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_id"] for row in first_rows] == ["qa-1", "qa-2"]

    second = audit_module.capture_split(**common)

    assert second["samples"] == 3
    assert second["resumed_samples"] == 2
    assert second["new_samples"] == 1
    assert second["selected_qa_seen"] == 3
    assert second["eligible_qa"] == 3
    assert second["complete"] is True
    assert capture_calls == [(11, False), (12, False), (13, False)]
    assert samples["qa-1"].attention_calls == 1
    assert samples["qa-2"].attention_calls == 1
    final_rows = [
        json.loads(line)
        for line in (output / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_id"] for row in final_rows] == ["qa-1", "qa-2", "qa-3"]
    assert len({row["sample_id"] for row in final_rows}) == len(final_rows)

    third = audit_module.capture_split(**common, limit=1)

    assert third["complete"] is True
    assert third["eligible_qa"] == 3
    assert third["selected_qa_seen"] == 1
    assert capture_calls == [(11, False), (12, False), (13, False)]
