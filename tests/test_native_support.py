"""Algorithm invariants and actual tiny-model execution, not natural-data results."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from state_audit.attribution import _projection_gram
from state_audit.model.adapter import ModelAdapter
from state_audit.model.replay import attention_backend, prefill_cache
from state_audit.native_forward import iter_forward_traces
from state_audit.native_ledger import finish_ledger, readout_direction
from state_audit.native_trace import layer_messages, native_hooks, source_masks
from state_audit.storage import read_arrays, read_json, write_json
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.native_support.evaluate import evaluate
from experiments.native_support.inputs import load_responses
from experiments.native_support.pipeline import capture_response, score_response
from experiments.native_support.score import (
    NUMERIC_TERMS,
    head_fingerprint,
    prompt_focus,
    support_step,
)


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=40, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    return ModelAdapter(LlamaForCausalLM(config))


def ledger(prompt_write=2.0, history_write=8.0, ffn=0.0):
    edges = np.array([[[prompt_write, history_write]]])
    result = {name: np.asarray(0.0) for name in NUMERIC_TERMS}
    result.update(
        edge_logit_write=edges, mlp_score=np.array([ffn]), initial_score=np.asarray(0.0),
        group_ids=np.array([0, 1]), attention=np.array([[[0.7, 0.3]]]),
        entropy=np.asarray(0.5), surprisal=np.asarray(0.2),
        logit_gap=np.asarray(prompt_write + history_write + ffn), ledger_error=np.asarray(0.0),
    )
    return result


def test_graph_reuses_supported_and_opposed_history_with_same_rule():
    arrays = ledger()
    fingerprint = head_fingerprint(arrays["edge_logit_write"], arrays["group_ids"])
    negative, _, weights = support_step(arrays, 1, [-0.8], [fingerprint])
    positive, _, _ = support_step(arrays, 1, [0.8], [fingerprint])
    np.testing.assert_allclose(weights, [0.8], atol=1e-7)
    assert negative["risk"] == pytest.approx(0.44)
    assert positive["risk"] == pytest.approx(-0.84)
    assert negative["risk"] == pytest.approx(negative["direct_risk"] + negative["inherited_risk"])


def test_new_prompt_support_can_overcome_inherited_risk():
    arrays = ledger(prompt_write=20)
    fingerprint = head_fingerprint(arrays["edge_logit_write"], arrays["group_ids"])
    metrics, _, _ = support_step(arrays, 1, [-0.8], [fingerprint])
    assert metrics["risk"] < 0


def test_opposing_history_and_changed_heads_do_not_copy_risk():
    arrays = ledger(history_write=-8)
    fingerprint = head_fingerprint(arrays["edge_logit_write"], arrays["group_ids"])
    metrics, _, weights = support_step(arrays, 1, [-1.0], [fingerprint])
    assert weights[0] == 0 and metrics["inherited_risk"] == 0
    arrays = ledger()
    opposite = np.zeros_like(fingerprint)
    opposite[-1] = 1  # special-negative channel is disjoint from this observation.
    metrics, _, weights = support_step(arrays, 1, [-1.0], [opposite])
    assert weights[0] == 0


def test_budget_scaling_and_ffn_sign_do_not_invent_error_labels():
    arrays = ledger(ffn=10)
    fingerprint = head_fingerprint(arrays["edge_logit_write"], arrays["group_ids"])
    baseline, _, _ = support_step(arrays, 1, [-0.4], [fingerprint])
    arrays["mlp_score"] *= -1
    reversed_ffn, _, _ = support_step(arrays, 1, [-0.4], [fingerprint])
    assert baseline["risk"] == reversed_ffn["risk"]
    arrays["edge_logit_write"] *= 100
    arrays["mlp_score"] *= 100
    scaled, _, _ = support_step(arrays, 1, [-0.4], [fingerprint])
    assert baseline["risk"] == pytest.approx(scaled["risk"])


def test_focus_can_move_between_two_prompt_regions():
    earlier = np.array([[[0.4, 0.4, 0.05, 0.05]]])
    current = np.array([[[0.05, 0.05, 0.4, 0.4, 0.1]]])
    focus = prompt_focus(current, 4, [earlier], width=2)
    assert focus["focus_start"].item() == 2
    assert focus["focus_gain"].item() == pytest.approx(0.7)
    first = prompt_focus(current, 4, [], width=2)
    assert np.isnan(first["focus_gain"]).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_forward_matches_actual_logits_without_backward(model, dtype):
    model.native.to(dtype)
    tokens = list(range(1, 16))
    with patch("torch.autograd.grad", side_effect=AssertionError("backward forbidden")):
        arrays = next(iter_forward_traces(model, tokens, 7, [2], [1], prefill_chunk_size=3))
    with torch.no_grad():
        hidden = model.forward(tokens[:9])[-1]
        logits = model.native.lm_head(hidden).float().numpy()
    candidates = [tokens[9], int(arrays["competitor_id"])]
    np.testing.assert_allclose(arrays["candidate_logits"], logits[candidates], atol=2e-3)
    assert abs(float(arrays["ledger_error"])) < 3e-6
    assert arrays["attention"].shape == (3, 4, 9)
    assert arrays["edge_value_energy"].shape == (3, 4, 9)
    assert np.isfinite(arrays["edge_value_energy"]).all()
    assert "edge_margin_sensitivity" not in arrays
    assert arrays["observed_id"].item() != arrays["competitor_id"].item()
    for module in model.native.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks
    assert all(parameter.grad is None for parameter in model.native.parameters())


def test_queries_reuse_cache_and_saved_rows_exclude_future_keys(model):
    tokens = list(range(1, 18))
    calls = []
    original = model.native.model.forward

    def observe(*args, **kwargs):
        calls.extend(kwargs["input_ids"][0].tolist())
        return original(*args, **kwargs)

    with patch.object(model.native.model, "forward", side_effect=observe):
        before = list(iter_forward_traces(model, tokens, 7, [0, 1, 2], [1], prefill_chunk_size=3))
    assert calls == tokens[:9]
    altered = tokens[:10] + [30] * 7
    after = list(iter_forward_traces(model, altered, 7, [0, 1, 2], [1]))
    for left, right in zip(before, after):
        np.testing.assert_allclose(left["edge_logit_write"], right["edge_logit_write"], atol=1e-6)


def audit_reference(model, tokens, prompt, target, groups):
    """Original single-query teaching ledger is independent of the new GPU path."""
    query = prompt + target - 1
    with torch.no_grad(), attention_backend(model, "eager"):
        cache = prefill_cache(model, tokens[:query], 3)
        with native_hooks(model, with_grad=False) as records:
            hidden = model.native.model(
                input_ids=model.input_ids([tokens[query]]), past_key_values=cache,
                use_cache=True, return_dict=True,
            ).last_hidden_state[0, -1]
            logits = model.native.lm_head(hidden).float()
            observed = tokens[query + 1]
            top = logits.topk(2).indices.tolist()
            candidates = [observed, top[1] if top[0] == observed else top[0]]
            direction, difference, scale = readout_direction(model, records[2]["residual_after"][-1], candidates)
            grams = {layer: _projection_gram(model, layer) for layer in range(3)}
            masks = source_masks(groups, 3, hidden.device)
            arrays = layer_messages(model, records, cache, direction, grams, masks)
            return finish_ledger(model, records, arrays, hidden, logits, candidates, direction, difference, scale)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunk", [1, 3, 8])
def test_batch_matches_original_audit_ledger_and_graph(model, dtype, chunk):
    model.native.to(dtype)
    tokens = list(range(1, 16))
    batched = list(iter_forward_traces(model, tokens, 7, range(8), [1, 11], query_chunk_size=chunk))
    support, fingerprints, expected_support, expected_fingerprints = [], [], [], []
    tolerance = 2e-3 if dtype == torch.bfloat16 else 1e-6
    for target, arrays in enumerate(batched):
        reference = audit_reference(model, tokens, 7, target, arrays["group_ids"])
        for name in set(arrays) & set(reference):
            np.testing.assert_allclose(arrays[name], reference[name], atol=tolerance, rtol=0.01,
                                       err_msg=f"{name} at target {target}, chunk {chunk}")
        reference.update({name: arrays[name] for name in ("group_ids", "entropy", "surprisal")})
        metrics, fingerprint, _ = support_step(arrays, 7, support, fingerprints)
        expected, expected_fingerprint, _ = support_step(reference, 7, expected_support, expected_fingerprints)
        assert metrics["risk"] == pytest.approx(expected["risk"], abs=tolerance)
        support.append(metrics["support"])
        fingerprints.append(fingerprint)
        expected_support.append(expected["support"])
        expected_fingerprints.append(expected_fingerprint)
        assert abs(float(arrays["ledger_error"])) < 3e-6
        assert arrays["attention"].shape[-1] == 7 + target


def test_block_uses_fewer_forwards_and_cannot_read_future_within_block(model):
    tokens = list(range(1, 16))
    calls = []
    original = model.native.model.forward

    def observe(*args, **kwargs):
        calls.append((model.native.config._attn_implementation, kwargs["input_ids"].shape[-1]))
        return original(*args, **kwargs)

    with patch.object(model.native.model, "forward", side_effect=observe):
        before = list(iter_forward_traces(model, tokens, 7, range(8), [1],
                                         prefill_chunk_size=3, query_chunk_size=3))
    assert calls == [("sdpa", 3), ("sdpa", 3), ("eager", 3), ("eager", 3), ("eager", 2)]
    altered = tokens[:8] + [30] * 7
    after = list(iter_forward_traces(model, altered, 7, range(8), [1], query_chunk_size=3))
    for name in ("candidate_logits", "edge_logit_write", "edge_value_energy", "mlp_score"):
        np.testing.assert_allclose(before[0][name], after[0][name], atol=1e-6)
    sparse = list(iter_forward_traces(model, tokens, 7, [0, 1, 4, 5, 6], [1], query_chunk_size=3))
    for arrays in sparse:
        np.testing.assert_allclose(arrays["edge_value_energy"], before[int(arrays["target"])]["edge_value_energy"], atol=1e-6)


def test_hook_failure_restores_model_without_parameter_edits(model):
    with (
        patch.object(model.native.lm_head, "forward", side_effect=RuntimeError("readout")),
        pytest.raises(RuntimeError, match="readout"),
    ):
        next(iter_forward_traces(model, list(range(1, 16)), 7, [0], [1]))
    for module in model.native.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks
    assert all(parameter.requires_grad for parameter in model.native.parameters())


class Tokenizer:
    all_special_ids = (1,)

    def decode(self, tokens):
        return ''.join(str(token) for token in tokens)


def response():
    tokens = list(range(1, 14))
    return {"id": "sample", "source_id": "source", "token_ids": tokens, "prompt_length": 7,
                "token_text": [str(token) for token in tokens]}


def test_capture_resume_score_and_evaluate_are_separate(model, tmp_path):
    item = response()
    directory = tmp_path / "responses/0000"
    capture_response(model, Tokenizer(), item, directory, 3, query_chunk_size=1, compress_cache=True)
    first = score_response(item, directory)
    retained = (directory / "token_000000.npz").read_bytes()
    (directory / "token_000002.npz").unlink()
    capture_response(model, Tokenizer(), item, directory, 3)
    assert (directory / "token_000000.npz").read_bytes() == retained
    timing = read_json(directory / "capture_timing.json")
    assert timing["captured_tokens"] == 1 and timing["reused_tokens"] == 5
    assert timing["compressed"] is False
    second = score_response(item, directory)
    np.testing.assert_allclose([r["risk"] for r in first], [r["risk"] for r in second], atol=1e-7)
    saved = read_arrays(directory / "scores.npz")
    assert np.max(np.abs(saved["risk"])) <= 1
    for target in range(len(first)):
        start, stop = saved["history_indptr"][target:target + 2]
        assert (saved["history_indices"][start:stop] < target).all()
        assert saved["history_weights"][start:stop].sum() <= 1 + 1e-7
    settings = {"responses": [item]}
    write_json(tmp_path / "settings.json", settings)
    assert len(second) == 6
    annotations = tmp_path / "annotations.json"
    write_json(annotations, {"sample": {"token_ids": item["token_ids"][7:], "labels": [0, 1, 1, 0, 1, 0]}})
    result = evaluate(tmp_path, annotations)
    assert result["methods"]["support_graph"]["span_onset_vs_normal"]["positives"] == 2
    assert result["methods"]["support_graph"]["first_error_vs_normal"]["positives"] == 1


def test_extra_labels_roles_and_candidates_do_not_enter_detector_input(tmp_path):
    item = response()
    path = tmp_path / "input.json"
    write_json(path, {"model": "tiny", "responses": [item]})
    before = load_responses(path)
    item.update(labels=[1] * 6, roles={"correct": [2]}, candidates=[[3], [4]], side="wrong")
    write_json(path, {"model": "tiny", "responses": [item]})
    assert before == load_responses(path)
