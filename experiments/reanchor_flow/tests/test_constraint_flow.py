"""Behavioral gates for exact accounting, native Llama and path semantics."""
import re

import numpy as np
import pytest
import torch

from experiments.reanchor_flow.constraint_flow import (
    ConstraintPair, capture_constraint_pair, confirm_paths, select_constraint_paths, symmetric_av,
)


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(17)
    return LlamaForCausalLM(LlamaConfig(vocab_size=512, hidden_size=32, intermediate_size=48,
                            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)).eval()


def pair():
    return ConstraintPair(np.array([[1,4,7,9,12,15,18,20], [1,5,7,9,12,15,18,20]]), 3, 24, 25)


def test_symmetric_split_distinguishes_content_change_from_route_change():
    a0 = torch.tensor([[[.7, .3]]], dtype=torch.float64)
    a1 = torch.tensor([[[.2, .8]]], dtype=torch.float64)
    v0 = torch.tensor([[[1., -2.], [2., 3.]]], dtype=torch.float64)
    v1 = v0 + torch.tensor([[[3., 4.], [-2., -1.]]], dtype=torch.float64)
    content, routing = symmetric_av(a0, a1, v0, v1)
    torch.testing.assert_close(content + routing, a1 @ v1 - a0 @ v0)
    assert torch.count_nonzero(symmetric_av(a0, a0, v0, v1)[1]) == 0
    assert torch.count_nonzero(symmetric_av(a0, a1, v0, v0)[0]) == 0
    # A large vector change can have zero support on a particular readout.
    orthogonal = torch.tensor([1., 0.], dtype=torch.float64)
    c, _ = symmetric_av(a0, a0, v0, v0 + torch.tensor([0., 10.], dtype=torch.float64))
    assert c.norm() > 0 and float((c * orthogonal).sum()) == 0


def test_pair_target_and_shared_prefix_contract():
    original = pair()
    bad = original.token_ids.copy()
    bad[1, -1] += 1
    with pytest.raises(ValueError, match="identical"):
        ConstraintPair(bad, 3, 24, 25).check()
    with pytest.raises(ValueError, match="differ"):
        ConstraintPair(original.token_ids, 3, 24, 24).check()


def test_native_hf_logits_complete_ledger_and_chunk_invariance():
    model, spec = tiny_model(), pair()
    trace = capture_constraint_pair(model, spec, query_chunk=2)
    with torch.inference_mode():
        logits = model(torch.as_tensor(spec.token_ids), use_cache=False).logits[:, -1]
    expected = (logits[:, spec.positive_token_id] - logits[:, spec.negative_token_id]).numpy()
    np.testing.assert_allclose(trace["margin_pair"], expected, atol=2e-6)
    np.testing.assert_allclose(trace["ledger_terms"].sum(), expected[1]-expected[0], atol=2e-6)
    assert trace["head_native_code"].shape[:2] == (3, 4)
    assert trace["root_value_pair"].shape[2] == 2  # shared KV, not averaged query heads
    np.testing.assert_array_equal(trace["query_to_kv"], [0,0,1,1])
    np.testing.assert_allclose(trace["source_sum_rounding"], 0, atol=2e-6)
    np.testing.assert_allclose(trace["residual_add_rounding"], 0, atol=2e-6)
    other = capture_constraint_pair(model, spec, query_chunk=3)
    for field in ("margin_pair", "head_content_code", "head_routing_code", "residual_pair", "source_content_projection"):
        np.testing.assert_allclose(trace[field], other[field], atol=2e-6)
    reverse = capture_constraint_pair(model, ConstraintPair(spec.token_ids, 3, 25, 24), query_chunk=2)
    for field in ("ledger_terms", "source_content_projection", "source_routing_projection", "mlp_projection"):
        np.testing.assert_allclose(trace[field], -reverse[field], atol=2e-6)


def test_attention_without_values_does_not_become_constraint_flow():
    model = tiny_model()
    for layer in model.model.layers:
        layer.self_attn.v_proj.weight.data.zero_()
    trace = capture_constraint_pair(model, pair(), query_chunk=2)
    assert np.max(trace["target_attention_pair"]) > 0
    np.testing.assert_allclose(trace["source_content_projection"], 0)
    np.testing.assert_allclose(trace["source_routing_projection"], 0)
    assert abs(float(trace["target_margin_delta"])) < 1e-7
    assert len(select_constraint_paths(model, trace)[0]) == 0
    measured = confirm_paths(model, trace, np.array([[1,0,0,4,2,1,7]]), query_chunk=2)
    np.testing.assert_allclose(measured["path_effect"], 0, atol=1e-7)


def test_bfloat16_exposes_rounding_and_identical_pair_has_zero_response():
    model = tiny_model().to(torch.bfloat16)
    spec = pair()
    trace = capture_constraint_pair(model, spec, query_chunk=2)
    assert np.isfinite(trace["ledger_terms"]).all()
    np.testing.assert_allclose(trace["ledger_terms"].sum() + trace["ledger_rounding"], trace["target_margin_delta"])
    # Report finite precision; do not hide it in a semantic component.
    assert "av_rounding" in trace and "final_norm_rounding" in trace
    identical = ConstraintPair(np.repeat(spec.token_ids[:1], 2, axis=0), 3, 24, 25)
    null = capture_constraint_pair(model, identical, query_chunk=2)
    np.testing.assert_array_equal(null["ledger_terms"], np.zeros(5))
    assert float(null["target_margin_delta"]) == 0
    paths, _ = select_constraint_paths(model, null)
    assert paths.shape == (0, 7)


def test_path_confirmation_has_identity_limit_and_enforces_layer_order():
    model = tiny_model()
    trace = capture_constraint_pair(model, pair(), query_chunk=2)
    paths, roles = select_constraint_paths(model, trace)
    assert "waad" not in trace and 1 <= len(paths) <= 2
    assert roles[0] == "candidate"
    if len(paths) == 2:
        np.testing.assert_array_equal(paths[0, [0,1,2,4,5,6]], paths[1, [0,1,2,4,5,6]])
        assert abs(paths[0, 3]-paths[1, 3]) == 1
    measured = confirm_paths(model, trace, paths, query_chunk=2)
    assert np.isfinite(measured["path_effect"]).all()
    # Replacing an incoming edge by itself must not invent downstream influence.
    trace["root_attention_pair"][:, 1] = trace["root_attention_pair"][:, 0]
    trace["root_value_pair"][:, 1] = trace["root_value_pair"][:, 0]
    identity = confirm_paths(model, trace, paths, query_chunk=2)
    np.testing.assert_allclose(identity["path_effect"], 0, atol=1e-7)
    np.testing.assert_allclose(identity["path_value_delta"], 0, atol=1e-7)
    invalid = paths.copy()
    invalid[0, 4] = invalid[0, 1]
    with pytest.raises(ValueError, match="deeper"):
        confirm_paths(model, trace, invalid)


class WordTokenizer:
    def __init__(self):
        self.vocabulary = {}

    def encode(self, text, add_special_tokens=False):
        result = []
        for word in re.findall(r"\w+|[^\w\s]", text):
            result.append(self.vocabulary.setdefault(word, len(self.vocabulary)+2))
        return result

    def apply_chat_template(self, messages, **kwargs):
        return [1] + self.encode(messages[0]["content"]) + [1]

    def decode(self, ids):
        reverse = {v:k for k,v in self.vocabulary.items()}
        return " ".join(reverse.get(i, "<s>") for i in ids)


def test_cli_freezes_control_paths_resumes_and_analyzes_without_model(tmp_path, monkeypatch):
    import transformers
    from experiments.reanchor_flow.constraint_flow_run import parser, run

    model, tokenizer = tiny_model(), WordTokenizer()
    calls = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    def load(*args, **kwargs):
        calls.append(1)
        return model
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load)
    args = parser().parse_args(["--model", str(tmp_path/"model"), "--device", "cpu", "--dtype", "float32",
        "--synthetic-sources", "1", "--confirm-sources", "1", "--query-chunk", "3", "--bootstrap", "5",
        "--no-plot", "--output", str(tmp_path/"output")])
    report = run(args)
    assert len(report["pairs"]) == 4 and len(calls) == 1
    assert report["detection_metrics_run"] is False
    frozen = report["pairs"][0]["paths"]
    for row in report["pairs"]:
        assert row["paths"] == frozen
        with np.load(args.output / row["path"], allow_pickle=False) as trace:
            assert not any("label" in key for key in trace.files)
            assert int(trace["query"]) == len(trace["token_ids"][0])-1
            assert trace["valid_query_mask"].all()
    again = run(args)
    assert len(calls) == 1 and again["pairs"] == report["pairs"]
    def forbidden(*args, **kwargs):
        raise AssertionError("offline analysis cannot load model/tokenizer")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", forbidden)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", forbidden)
    args.phase = "analyze"
    assert run(args)["pairs"] == report["pairs"]


def test_synthetic_sources_are_distinct_and_controls_preserve_baseline():
    from experiments.reanchor_flow.constraint_flow_run import synthetic_records, encode_pair

    records = synthetic_records(12)
    assert len({r["condition0"] for r in records}) == 12
    tokenizer = WordTokenizer()
    for i in range(0, len(records), 4):
        encoded = [encode_pair(r, tokenizer) for r in records[i:i+4]]
        for control in encoded[1:]:
            np.testing.assert_array_equal(control.token_ids[0], encoded[0].token_ids[0])
            assert control.positive_token_id == encoded[0].positive_token_id
        rename = records[i+1]
        # The renamed queried object still has the baseline color, while a plain
        # query edit asks for the other color. No accidental 'Answer' -> 'Bnswer'.
        entity = re.search(r"What color is object ([AB])", rename["condition1"])[1]
        assert f"Object {entity} is {rename['negative']}." in rename["condition1"]
        assert "Answer from the records." in rename["condition1"]
    records[0]["prefix_validated"] = False
    with pytest.raises(ValueError, match="semantic"):
        encode_pair(records[0], tokenizer)
