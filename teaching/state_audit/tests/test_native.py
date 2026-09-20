import copy

import numpy as np
import pytest
import torch

from state_audit.capture import capture_hooks, capture_sample
from state_audit.datasets import load_examples
from state_audit.demo import build_demo
from state_audit.generation import make_answer
from state_audit.intervention import four_worlds, interaction_result
from state_audit.models import input_tensor, load_model
from state_audit.roles import attention_from_keys
from state_audit.storage import read_arrays, read_json


def test_native_alignment_gqa_and_readout(tiny_run):
    model, _, root, *_ = tiny_run
    directory = root / "samples" / "000000"
    answer = read_json(directory / "answer.json")
    prompt = answer["prompt_length"]
    readout = read_arrays(directory / "trace" / "readout.npz")
    np.testing.assert_array_equal(
        readout["queries"], np.arange(prompt - 1, len(answer["token_ids"]) - 1)
    )
    ids = input_tensor(model, answer["token_ids"])
    with torch.inference_mode():
        logits = model(ids[:, :-1], use_cache=False).logits[0, prompt - 1 :].float()
    logps = logits.log_softmax(-1)
    expected = logps.gather(-1, ids[0, prompt:, None]).squeeze(-1).numpy()
    np.testing.assert_allclose(readout["target_logp"], expected, atol=1e-6)
    entropy = -(logps.exp() * logps).sum(-1).numpy()
    np.testing.assert_allclose(readout["logit_entropy"], entropy, atol=1e-6)
    for layer in (0, 1):
        trace = read_arrays(directory / "trace" / f"layer_{layer:03d}.npz")
        assert trace["query"].shape[0] == 4 and trace["key"].shape[0] == 2
        np.testing.assert_allclose(
            attention_from_keys(trace, trace["key"]), trace["attention"], atol=1e-6
        )
        for target, query in enumerate(trace["queries"]):
            assert np.all(trace["attention"][:, target, query + 1 :] == 0)


def test_future_changes_do_not_change_earlier_states(tiny_run, tmp_path):
    model, _, root, *_ = tiny_run
    original = root / "samples" / "000000"
    answer = copy.deepcopy(read_json(original / "answer.json"))
    target = 4
    answer["token_ids"][answer["prompt_length"] + target] = answer["token_ids"][0]
    changed = tmp_path / "changed"
    capture_sample(model, answer, changed, [0, 1])
    for layer in (0, 1):
        old = read_arrays(original / "trace" / f"layer_{layer:03d}.npz")
        new = read_arrays(changed / f"layer_{layer:03d}.npz")
        np.testing.assert_allclose(
            old["residual_after"][: target + 1], new["residual_after"][: target + 1], atol=1e-7
        )


def test_four_worlds_sham_and_empty_history(tiny_run):
    model, _, root, *_ = tiny_run
    empty = four_worlds(model, root, 0, 0, 0, 0, 1, "history")
    assert len(set(empty["worlds"].values())) == 1
    assert empty["interaction"] == 0
    result = four_worlds(model, root, 0, 0, 3, 0, 1, "evidence")
    assert result["worlds"]["full"] == result["sham_logp"]
    assert abs(result["worlds"]["full"] - result["worlds"]["without_both"]) > 1e-7
    assert result["interaction"] == pytest.approx(
        result["effect_a_with_b"] - result["effect_a_without_b"]
    )


def test_hooks_are_removed_on_failure(tiny_run, tmp_path):
    model, *_ = tiny_run
    attention = model.model.layers[0].self_attn
    with pytest.raises(RuntimeError, match="interrupted"):
        with capture_hooks(model, [0], np.array([0]), tmp_path):
            raise RuntimeError("interrupted")
    assert not attention._forward_hooks
    assert not attention._forward_pre_hooks
    assert not attention.o_proj._forward_pre_hooks
    for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
        assert not projection._forward_hooks


def test_conditional_effect_uses_all_four_worlds():
    result = interaction_result(dict(full=4, without_a=3, without_b=1, without_both=3))
    assert result == dict(effect_a_with_b=1, effect_a_without_b=-2, interaction=3)


def test_mistral_sliding_mask_is_preserved(tmp_path):
    data, checkpoint = build_demo(tmp_path / "fixture", "mistral")
    model, tokenizer = load_model(str(checkpoint), "main", "cpu", "float32")
    model.config.sliding_window = 8
    options = dict(mode="replay", template="chat", max_length=256)
    answer = make_answer(model, tokenizer, load_examples(data)[0], options, seed=0)
    capture_sample(model, answer, tmp_path / "trace", [0])
    trace = read_arrays(tmp_path / "trace" / "layer_000.npz")
    np.testing.assert_allclose(
        attention_from_keys(trace, trace["key"]), trace["attention"], atol=1e-6
    )
    for token, query in enumerate(trace["queries"]):
        assert np.all(trace["attention"][:, token, : query - 7] == 0)
        assert np.all(trace["attention_bias"][token, : query - 7] < -1e10)
