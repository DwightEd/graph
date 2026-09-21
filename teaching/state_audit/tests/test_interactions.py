import numpy as np
import pytest
import torch
from state_audit.analysis.interactions import adaptation_effects, factorial_effects
from state_audit.capture import capture_targets
from state_audit.experiments.contrasts import score_contrast
from state_audit.experiments.messages import MessageSite, measure_messages, restore_messages
from state_audit.intervention import intervene
from state_audit.operations import Delete, Inject, Target
from state_audit.storage import read_json


def contrast(tiny_run):
    model, _, root, *_ = tiny_run
    answer = read_json(root / "samples/000000/answer.json")
    prefix = answer["token_ids"][: answer["prompt_length"] + 3]
    candidates = [answer["response_ids"][3:5], answer["response_ids"][5:8]]
    return model, prefix, candidates


def test_contrast_matches_native_teacher_forcing_and_keeps_lengths(tiny_run):
    model, prefix, candidates = contrast(tiny_run)
    result = score_contrast(model, prefix, candidates)
    for index, ids in enumerate(candidates):
        with torch.inference_mode():
            tokens = model.input_ids(prefix + ids)
            logits = model.native(tokens[:, :-1], use_cache=False).logits[0, len(prefix) - 1 :]
            logp = logits.float().log_softmax(-1)
            expected = logp.gather(-1, model.input_ids(ids)[0, :, None]).squeeze(-1)
        np.testing.assert_allclose(result["token_logp"][index], expected.numpy(), atol=1e-6)
    assert result["candidate_lengths"] == [2, 3]
    assert result["sum_margin"] == pytest.approx(
        sum(result["token_logp"][0]) - sum(result["token_logp"][1])
    )


def test_selected_capture_observes_intervention_and_preserves_layer_identity(tiny_run):
    model, prefix, _ = contrast(tiny_run)
    target = Target("head_readout", (0, 1), positions=(len(prefix) - 1,), heads=(0, 3))
    with torch.inference_mode(), intervene(model, [Delete(target)]):
        with capture_targets(model, {"heads": target}) as states:
            model.forward(prefix)
    assert set(states["heads"]) == {0, 1}
    for values in states["heads"].values():
        assert values.shape == (1, 2, 8)
        assert np.all(values == 0)


def test_source_message_subtraction_equals_edge_deletion_for_gqa(tiny_run):
    model, prefix, candidates = contrast(tiny_run)
    site = MessageSite("route", 0, (0, 1, 3), (len(prefix) - 1,), (2, 4, 6))
    baseline = measure_messages(model, prefix, candidates, [site])
    cut = measure_messages(model, prefix, candidates, [site], [site.deletion()])
    subtract = Inject(site.target, -baseline[1][site.name]["readout"])
    patched = measure_messages(model, prefix, candidates, [site], [subtract])
    for left, right in zip(cut[0]["token_logp"], patched[0]["token_logp"]):
        np.testing.assert_allclose(left, right, atol=1e-6)
    assert np.all(cut[1]["route"]["mass"] == 0)
    expected = baseline[1]["route"]["total_readout"] - baseline[1]["route"]["readout"]
    np.testing.assert_allclose(patched[1]["route"]["total_readout"], expected, atol=1e-7)


def test_restoration_recomputes_later_source_messages(tiny_run):
    model, prefix, candidates = contrast(tiny_run)
    sites = [
        MessageSite(
            f"layer{layer}", layer, (0, 1, 3), (len(prefix) - 1,), tuple(range(len(prefix)))
        )
        for layer in (0, 1)
    ]

    def evaluate(operations):
        return measure_messages(model, prefix, candidates, sites, operations)

    baseline = evaluate([])
    operations = [site.deletion() for site in sites]
    deleted = evaluate(operations)
    restored = restore_messages(evaluate, sites, baseline[1], operations, deleted)
    for left, right in zip(baseline[0]["token_logp"], restored[0]["token_logp"]):
        np.testing.assert_allclose(left, right, atol=2e-6)
    for site in sites:
        np.testing.assert_allclose(
            restored[1][site.name]["total_readout"],
            baseline[1][site.name]["total_readout"],
            atol=1e-6,
        )


def test_all_head_projection_reconstructs_native_write_without_bias(tiny_run):
    model, prefix, _ = contrast(tiny_run)
    targets = {
        "heads": Target("head_readout", (0,), positions=(len(prefix) - 1,)),
        "write": Target("attention_write", (0,), positions=(len(prefix) - 1,)),
    }
    with torch.inference_mode(), capture_targets(model, targets) as state:
        model.forward(prefix)
        projected = model.project_heads(0, state["heads"][0], range(4)).sum(1).numpy()
    projection = model.layers[0].self_attn.o_proj
    if projection.bias is not None:
        projected += projection.bias.detach().numpy()
    np.testing.assert_allclose(projected, state["write"][0], atol=1e-7)


def test_effect_signs_and_compensation_do_not_use_boolean_negation():
    result = factorial_effects({"11": 0.2, "01": -0.5, "10": 0.9, "00": -0.3})
    assert result["a_with_b"] == pytest.approx(0.7)
    assert result["a_without_b"] == pytest.approx(1.2)
    assert result["interaction"] == pytest.approx(-0.5)
    adaptation = adaptation_effects(1.0, 0.7, 0.2)
    assert adaptation["adaptation"] == pytest.approx(0.5)
    assert adaptation["absolute_effect_reduction"] == pytest.approx(0.5)


def test_empty_source_is_observed_zero_not_an_invented_message(tiny_run):
    model, prefix, candidates = contrast(tiny_run)
    site = MessageSite("empty", 0, (0, 3), (len(prefix) - 1,), ())
    baseline = measure_messages(model, prefix, candidates, [site])
    assert np.all(baseline[1]["empty"]["mass"] == 0)
    assert np.all(baseline[1]["empty"]["write"] == 0)
    removed = measure_messages(model, prefix, candidates, [site], [site.deletion()])
    assert removed[0] == baseline[0]


def test_bfloat16_source_delete_restore_matches_native_exactly(tiny_run):
    model, prefix, candidates = contrast(tiny_run)
    model.native.to(torch.bfloat16)
    with torch.no_grad():
        for layer in model.layers:
            layer.self_attn.v_proj.weight.mul_(16)
    sites = [
        MessageSite(f"route{layer}", layer, (0, 3), (len(prefix) - 1,), (2, 4, 6))
        for layer in (0, 1)
    ]

    def evaluate(operations):
        return measure_messages(model, prefix, candidates, sites, operations)

    baseline = evaluate([])
    operations = [site.deletion() for site in sites]
    restored = restore_messages(evaluate, sites, baseline[1], operations)
    assert restored[0] == baseline[0]
    for site in sites:
        np.testing.assert_array_equal(
            restored[1][site.name]["total_readout"], baseline[1][site.name]["total_readout"]
        )
