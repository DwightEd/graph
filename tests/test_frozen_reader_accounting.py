"""Reader failures and framing must remain visible after artifact publication."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import GenerationConfig

from route_graph.audit_output import merge_response
from route_graph.frozen_reader import FrozenReader


class Tokenizer:
    eos_token_id = 0

    def __init__(self, raw):
        self.raw = raw

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, generated, **kwargs):
        return self.raw


def reader(tmp_path, raw, *, limit=8192):
    model = SimpleNamespace(
        training=False, generation_config=GenerationConfig(),
        dtype=torch.float32, device=torch.device("cpu"),
        generate=lambda **kwargs: torch.tensor([[1, 2, 3, 4, 5]]),
    )
    return FrozenReader(
        model, Tokenizer(raw), tmp_path,
        {"model": "fixture", "tokenizer": "fixture", "code_sha256": "fixture"},
        context_limit=limit,
    )


def test_raw_framing_survives_reader_and_cache_hit(tmp_path):
    raw = ' \n {"answer": 1}] \t'
    r = reader(tmp_path, raw)
    assert r.ask("instruction", {}) == {"answer": 1}
    assert r.ask("instruction", {}) == {"answer": 1}
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    framing = saved["json_framing"]
    assert saved["raw_output"] == raw
    assert framing["raw_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    left, right = framing["raw_root_span"]
    assert framing["raw_prefix"] + raw[left:right] + framing["raw_suffix"] == raw
    assert r.calls == 1
    assert r.outcomes == {"returned_requests": 2, "cache_hits": 1, "json_recovered": 2}


@pytest.mark.parametrize(
    "raw,limit,budget,failure",
    [('{}]', 8192, 3, 'generation_limit'),
     ('{', 8192, 10, 'invalid_json'),
     ('{}]', 3, 10, 'context_limit')],
)
def test_failures_never_count_as_successful_framing(tmp_path, raw, limit, budget, failure):
    r = reader(tmp_path, raw, limit=limit)
    for _ in range(2):
        assert r.ask("instruction", {}, max_new_tokens=budget)["reader_error"] == failure
    assert r.outcomes == {
        "returned_requests": 2, "cache_hits": 1, "failure_" + failure: 2,
    }


def test_final_artifact_includes_a_and_c_requests_even_without_certificates():
    row = {"id": "1", "source_id": "s", "task": "QA", "generator": "g",
           "response_sha256": "x", "response": "uncovered"}
    a = {"questions": [], "words": [{"start": 0, "end": 9,
         "coverage_state": "coverage_unknown"}], "risk_claims_total": 0,
         "risk_claims_with_valid_contrast": 0,
         "reader_outcomes": {"returned_requests": 3, "json_recovered": 1,
                             "failure_invalid_json": 2}}
    c = {"q": {"reader_outcomes": {"returned_requests": 2, "cache_hits": 1,
                                    "json_strict": 1, "failure_generation_limit": 1}}}
    result = merge_response(row, a, {}, c, {})
    assert result["reader_request_accounting"] == {
        "returned_requests": 5, "json_recovered": 1, "failure_invalid_json": 2,
        "cache_hits": 1, "json_strict": 1, "failure_generation_limit": 1,
    }
    assert result["words"][0]["semantic_abstain"]
