import copy

import numpy as np
import pytest
import torch

from state_audit.analysis.roles import attention_from_keys
from state_audit.capture import CaptureSpec, capture_hooks, capture_sample
from state_audit.dataset import load_examples
from state_audit.demo import build_demo
from state_audit.experiments.interventions import factorial_interaction, score_targets
from state_audit.generation import GenerationOptions, make_answer
from state_audit.intervention import intervene
from state_audit.model import load_model
from state_audit.operations import Delete, Inject, Replace, Steer, Target
from state_audit.state import ModelState
from state_audit.storage import read_arrays, read_json
from state_audit.validation import check_trace


def test_native_alignment_gqa_and_readout(tiny_run):
    model, _, root, *_ = tiny_run
    directory = root / "samples" / "000000"
    answer = read_json(directory / "answer.json")
    prompt = answer["prompt_length"]
    readout = ModelState.open(directory / "trace").readout
    np.testing.assert_array_equal(
        readout["queries"], np.arange(prompt - 1, len(answer["token_ids"]) - 1)
    )
    ids = model.input_ids(answer["token_ids"])
    with torch.inference_mode():
        logits = model.native(ids[:, :-1], use_cache=False).logits[0, prompt - 1 :].float()
    logps = logits.log_softmax(-1)
    expected = logps.gather(-1, ids[0, prompt:, None]).squeeze(-1).numpy()
    np.testing.assert_allclose(readout["target_logp"], expected, atol=1e-6)
    np.testing.assert_allclose(
        readout["logit_entropy"], -(logps.exp() * logps).sum(-1).numpy(), atol=1e-6
    )
    check_trace(root, 0)
    for layer in ModelState.open(directory / "trace").iter_layers():
        assert layer["query"].shape[0] == 4 and layer["key"].shape[0] == 2
        for target, query in enumerate(layer.positions):
            assert np.all(layer["attention"][:, target, query + 1 :] == 0)


def test_future_changes_do_not_change_earlier_states(tiny_run, tmp_path):
    model, _, root, *_ = tiny_run
    original = root / "samples" / "000000"
    answer = copy.deepcopy(read_json(original / "answer.json"))
    target = 4
    answer["token_ids"][answer["prompt_length"] + target] = answer["token_ids"][0]
    changed = tmp_path / "changed"
    capture_sample(model, answer, changed, CaptureSpec(layers=(0, 1)))
    for layer in (0, 1):
        old = ModelState.open(original / "trace").layer(layer)
        new = ModelState.open(changed).layer(layer)
        np.testing.assert_allclose(
            old["residual_after"][: target + 1], new["residual_after"][: target + 1], atol=1e-7
        )


def test_attention_deletion_equals_source_message_subtraction_for_three_heads(tiny_run):
    model, _, root, *_ = tiny_run
    directory = root / "samples" / "000000"
    answer = read_json(directory / "answer.json")
    trace = ModelState.open(directory / "trace").layer(0)
    target = 3
    query = int(trace.positions[target])
    heads = (0, 1, 3)
    sources = (2, 4, 6)
    expanded = np.repeat(trace["value"], 2, axis=0)
    messages = (trace["attention"][:, target, sources, None] * expanded[:, sources]).sum(1)
    edge_target = Target("attention", (0,), positions=(query,), heads=heads, keys=sources)
    head_target = Target("head_readout", (0,), positions=(query,), heads=heads)
    cut = score_targets(model, answer, [target], [Delete(edge_target)])
    subtract = score_targets(model, answer, [target], [Inject(head_target, -messages[list(heads)])])
    np.testing.assert_allclose(cut["target_logp"], subtract["target_logp"], atol=1e-6)
    baseline = score_targets(model, answer, [target])
    sham = score_targets(model, answer, [target], [Delete(edge_target, strength=0)])
    assert sham == baseline
    assert abs(cut["target_logp"][0] - baseline["target_logp"][0]) > 1e-7


def test_replace_steer_and_capture_observe_actual_changed_states(tiny_run, tmp_path):
    model, _, root, *_ = tiny_run
    directory = root / "samples" / "000000"
    answer = read_json(directory / "answer.json")
    base = ModelState.open(directory / "trace").layer(0)
    query = int(base.positions[2])
    target = Target("residual_after", (0,), positions=(query,))
    baseline = score_targets(model, answer, [2])
    sham = score_targets(model, answer, [2], [Replace(target, base.at("residual_after", [query]))])
    assert baseline == sham
    direction = np.ones(32)
    spec = CaptureSpec(layers=(0,), representations=("residual_after",))
    with intervene(model, [Steer(target, direction, amount=0.5)]):
        capture_sample(model, answer, tmp_path / "changed", spec)
    changed = ModelState.open(tmp_path / "changed").layer(0)["residual_after"]
    expected = base["residual_after"].copy()
    expected[2] += 0.5 / np.sqrt(32)
    np.testing.assert_allclose(changed, expected, atol=1e-7)
    assert set(ModelState.open(tmp_path / "changed").layer(0).tensors) == {"residual_after"}


def test_hooks_and_attention_forward_are_restored_after_failure(tiny_run):
    model, _, root, *_ = tiny_run
    answer = read_json(root / "samples" / "000000" / "answer.json")
    attention = model.layers[0].self_attn
    original = attention.forward
    future = Target("attention", (0,), positions=(0,), keys=(1,))
    with pytest.raises(ValueError, match="masked edge"):
        score_targets(model, answer, [0], [Inject(future, 1.0)])
    assert attention.forward == original
    with pytest.raises(RuntimeError, match="interrupted"):
        with capture_hooks(model, CaptureSpec(layers=(0,)), np.array([0]), root / "unused"):
            raise RuntimeError("interrupted")
    for module in model.native.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_factorial_groups_are_not_limited_to_two_heads(tiny_run):
    model, _, root, *_ = tiny_run
    answer = read_json(root / "samples" / "000000" / "answer.json")
    query = answer["prompt_length"] + 2
    group_a = [Delete(Target("head_readout", (0,), positions=(query,), heads=(0, 1, 2)))]
    group_b = [Delete(Target("mlp_write", (1,), positions=(query,)))]
    result = factorial_interaction(model, answer, [3], group_a, group_b)
    logps = {name: row["target_logp"][0] for name, row in result["conditions"].items()}
    expected = (
        result["baseline"]["target_logp"][0] - logps["a_only"] - logps["b_only"] + logps["both"]
    )
    assert result["interaction"][0] == pytest.approx(expected)


def test_mistral_sliding_mask_is_preserved(tmp_path):
    data, checkpoint = build_demo(tmp_path / "fixture", "mistral")
    model, tokenizer = load_model(str(checkpoint), "main", "cpu", "float32")
    model.native.config.sliding_window = 8
    options = GenerationOptions(mode="replay", template="chat", max_length=256)
    answer = make_answer(model, tokenizer, load_examples(data)[0], options, seed=0)
    capture_sample(model, answer, tmp_path / "trace", CaptureSpec(layers=(0,)))
    trace = read_arrays(tmp_path / "trace" / "layer_000.npz")
    np.testing.assert_allclose(
        attention_from_keys(trace, trace["key"]), trace["attention"], atol=1e-6
    )
    for token, query in enumerate(trace["queries"]):
        assert np.all(trace["attention"][:, token, : query - 7] == 0)
    target = Target("attention", (0,), positions=(int(trace["queries"][0]),), keys=(0,))
    with pytest.raises(ValueError, match="masked edge"):
        score_targets(model, answer, [0], [Inject(target, 0.1)])


def test_all_scope_keeps_prompt_states_and_response_readout_alignment(tiny_run, tmp_path):
    model, _, root, *_ = tiny_run
    directory = root / "samples" / "000000"
    answer = read_json(directory / "answer.json")
    baseline = ModelState.open(directory / "trace")
    capture_sample(model, answer, tmp_path / "all", CaptureSpec(layers=(0,), scope="all"))
    state = ModelState.open(tmp_path / "all")
    assert state.layer(0).positions[0] == 0
    assert state.at("embedding", [0]).shape == (1, 32)
    np.testing.assert_array_equal(state.readout["target_logp"], baseline.readout["target_logp"])
    response_view = state.layer(0).select(baseline.layer(0).positions)
    for name in baseline.layer(0).tensors:
        np.testing.assert_allclose(response_view[name], baseline.layer(0)[name], atol=1e-7)
