"""Native gradient fidelity and causal intervention semantics, using real Llama."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.path_conflict.native import Intervention, forward
from experiments.path_conflict.operators import grouped_values
from experiments.path_conflict.scoring import evaluate_candidates
from experiments.path_conflict.target_gradients import edge_scores, trace_target
from experiments.path_conflict.transport import freeze_json, run_panel
from experiments.path_conflict.transport_plan import select_units, select_message_pairs, causal_order
from experiments.path_conflict.transport_report import report_transport
from experiments.path_conflict.transport_trials import message_action


@pytest.fixture
def native_model():
    torch.manual_seed(18)
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=41, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, initializer_range=.1)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval().requires_grad_(False)
    probe = dict(prefix_ids=[1, 3, 5, 2, 6], candidates=[[7, 8], [9, 10, 11]],
                 record_writes=False, groups={name: np.array(values, dtype=int) for name, values in {
                     "scope": [0], "supported_value": [1], "value_source": [2],
                     "other_prompt": [], "query_self": [4],
                     "recent_history": [3], "remote_history": []}.items()})
    return model, probe


def full_tape_scores(model, probe):
    """Independent whole-model backward, with all layer activations retained."""
    captures = [{} for layer in model.model.layers]
    handles = []
    for index, layer in enumerate(model.model.layers):
        def heads(module, args, index=index):
            captures[index]["heads"] = args[0]
        def values(module, args, output, index=index):
            captures[index]["values"] = output.detach()
        def attention(module, args, output, index=index):
            captures[index]["attention"] = output[1].detach()[0]
        handles.extend([layer.self_attn.o_proj.register_forward_pre_hook(heads),
                        layer.self_attn.v_proj.register_forward_hook(values),
                        layer.self_attn.register_forward_hook(attention)])
    ids = torch.tensor([probe["prefix_ids"]])
    embeddings = model.model.embed_tokens(ids).detach().requires_grad_(True)
    try:
        result = model.model(inputs_embeds=embeddings, use_cache=False, output_attentions=True)
        direction = model.lm_head.weight[7] - model.lm_head.weight[9]
        margin = result.last_hidden_state[0, -1] @ direction
        gradients = torch.autograd.grad(margin, [row["heads"] for row in captures])
    finally:
        for handle in handles:
            handle.remove()
    scores = []
    for row, gradient in zip(captures, gradients):
        values = grouped_values(row["values"], 4, 2)[0]
        gradient = gradient[0].view(5, 4, 4).transpose(0, 1)
        scores.append(edge_scores(row["attention"], values, gradient).numpy())
    return np.stack(scores), float(margin.detach())


def test_layer_replay_matches_complete_qkv_mlp_backward(native_model):
    model, probe = native_model
    expected, margin = full_tape_scores(model, probe)
    edges, arrays, metadata = trace_target(model, probe, progress=False, objective="next_margin")
    np.testing.assert_allclose(arrays["target_sources"], expected[:, :, -1], atol=1e-6)
    np.testing.assert_allclose(arrays["node_signed"], expected.sum(-1), atol=1e-6)
    assert metadata["next_margin"] == pytest.approx(margin, abs=1e-6)
    assert max(row["replay_max_error"] for row in metadata["diagnostics"]) == 0
    assert (edges.source <= edges.receiver).all()
    assert np.count_nonzero(expected[:-1, :, :-1]) > 0  # Earlier receivers really participate.


def test_final_gate_derivative_matches_finite_native_change(native_model):
    model, probe = native_model
    scores, _ = full_tape_scores(model, probe)
    for layer, head, receiver, source in [(0, 1, 2, 1), (1, 2, 4, 3), (2, 3, 4, 4)]:
        unit = dict(layer=layer, head=head, receiver=receiver, source=source)
        epsilon = .002
        plus, _ = evaluate_candidates(model, probe, (message_action(unit, -epsilon),), sequence=False)
        minus, _ = evaluate_candidates(model, probe, (message_action(unit, epsilon),), sequence=False)
        measured = (plus["next_margin"] - minus["next_margin"]) / (2 * epsilon)
        assert measured == pytest.approx(scores[layer, head, receiver, source], abs=8e-5, rel=.02)


def test_capture_does_not_retain_all_layers_attention(native_model):
    model, probe = native_model
    outputs = []
    def collect(module, args, kwargs, output):
        outputs.append((kwargs["output_attentions"], output.attentions))
    handle = model.model.register_forward_hook(collect, with_kwargs=True)
    try:
        trace_target(model, probe, progress=False)
        evaluate_candidates(model, probe)
    finally:
        handle.remove()
    assert len(outputs) == 6
    assert all(requested is False and attention is None for requested, attention in outputs)


def test_complete_candidate_gradient_uses_tail_but_cuts_only_prefix(native_model):
    model, probe = native_model
    # Identical first tokens make first-token discovery completely uninformative.
    probe = dict(probe, candidates=[[7, 8], [7, 10, 11]])
    edges, arrays, metadata = trace_target(model, probe, progress=False)
    baseline, _ = evaluate_candidates(model, probe)
    assert metadata["next_margin"] == 0
    assert metadata["objective_value"] == pytest.approx(baseline["sequence_margin"], abs=1e-6)
    assert (edges.receiver < len(probe["prefix_ids"])).all()
    unit = edges.iloc[edges.final_linear_support.abs().argmax()].to_dict()
    epsilon = .002
    plus, _ = evaluate_candidates(model, probe, (message_action(unit, -epsilon),))
    minus, _ = evaluate_candidates(model, probe, (message_action(unit, epsilon),))
    measured = (plus["sequence_margin"] - minus["sequence_margin"]) / (2 * epsilon)
    assert measured == pytest.approx(unit["final_linear_support"], abs=1e-4, rel=.02)
    assert abs(unit["final_linear_support"]) > 1e-3
    assert arrays["node_signed"].shape == (3, 4, 5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_low_precision_trace_is_exportable(native_model, dtype, tmp_path):
    model, probe = native_model
    model.to(dtype=dtype)
    edges, arrays, metadata = trace_target(model, probe, progress=False)
    baseline, _ = evaluate_candidates(model, probe)
    np.savez_compressed(tmp_path / "trace.npz", **arrays)
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert metadata["objective_value"] == pytest.approx(baseline["sequence_margin"], abs=2e-6)
    assert max(row["replay_max_error"] for row in metadata["diagnostics"]) == 0
    assert not edges.empty


def test_specific_past_receiver_and_same_world_replacement(native_model):
    model, probe = native_model
    unit = dict(unit="past", layer=0, head=1, receiver=2, source=1)
    probe["capture_units"] = [unit]
    baseline, run = evaluate_candidates(model, probe)
    message = run.message_writes["past"]
    action = message_action(unit, operation="replace", replacement=message)
    sham, _ = evaluate_candidates(model, probe, (action,))
    for metric in ("next_margin", "sequence_margin"):
        assert sham[metric] == pytest.approx(baseline[metric], abs=2e-6)
    action = Intervention(0, ("mlp",), queries=(2,), operation="replace",
                          replacement=run.message_writes["mlp_past"])
    sham, _ = evaluate_candidates(model, probe, (action,))
    assert sham["sequence_margin"] == pytest.approx(baseline["sequence_margin"], abs=2e-6)
    _, changed = forward(model, probe["prefix_ids"], probe, (message_action(unit),))
    assert changed.changes[0]["receiver_positions"] == [2]
    assert changed.changes[0]["source_positions"] == [1]


def test_restoring_causally_earlier_receiver_cannot_repair_later_cut(native_model):
    model, probe = native_model
    early = dict(unit="early", layer=1, head=1, receiver=1, source=0)
    later_position = dict(unit="cut", layer=0, head=2, receiver=4, source=1)
    probe["capture_units"] = [early]
    _, run = evaluate_candidates(model, probe)
    cut = message_action(later_position)
    restore = message_action(early, operation="replace", replacement=run.message_writes["early"])
    removed, _ = evaluate_candidates(model, probe, (cut,))
    restored, _ = evaluate_candidates(model, probe, (cut, restore))
    assert restored["next_margin"] == pytest.approx(removed["next_margin"], abs=2e-6)
    assert causal_order(early, later_position) is None


def test_selection_ignores_roles_and_pairs_opposing_different_sources(native_model):
    model, probe = native_model
    edges, _, _ = trace_target(model, probe, progress=False)
    units = select_units(edges, max_units=4, control_units=1, seed=3)
    relabeled = edges.assign(source_role="wrong", gold=1, side="unsupported")
    repeated = select_units(relabeled, 4, 1, 3)
    assert [row["unit"] for row in units] == [row["unit"] for row in repeated]
    manual = [dict(unit="a", layer=0, head=0, receiver=4, source=0,
                   final_linear_support=3., selection="signed_gradient"),
              dict(unit="b", layer=2, head=1, receiver=4, source=4,
                   final_linear_support=-2., selection="signed_gradient")]
    pair, = select_message_pairs(manual)
    assert pair["kind"] == "opposed"
    assert {pair["left"], pair["right"]} == {"a", "b"}


def test_real_panel_full_sequence_four_worlds_and_resume(native_model, tmp_path, monkeypatch):
    model, probe = native_model
    tokenizer = SimpleNamespace(decode=lambda ids: "token_" + str(ids[0]))
    args = SimpleNamespace(output=str(tmp_path), max_units=3, control_units=1,
                           max_pairs=2, edge_top_k=2, seed=0, doses=[.25, 1.],
                           objective="sequence_margin")
    identity = dict(case_id="tiny", source_id="s", side="supported", panel="natural")
    run_panel(model, tokenizer, probe, identity, args)
    directory = tmp_path / "panels" / "tiny_supported_natural"
    pairs = pd.read_csv(directory / "interactions.csv")
    np.testing.assert_allclose(pairs.interaction,
                               pairs.full - pairs.without_left - pairs.without_right + pairs.without_both)
    assert set(pairs.metric) == {"next_margin", "sequence_margin"}
    effects = pd.read_csv(directory / "interventions.csv")
    assert effects.correct_token_logps.str.contains(",").all()
    mediation = pd.read_csv(directory / "mediation.csv")
    assert not mediation.empty
    assert set(mediation.site) == {"message", "mlp"}
    assert mediation.sham_delta.abs().max() < 3e-6
    before = pairs.to_dict("records")
    def forbid_forward(*args, **kwargs):
        raise AssertionError("resume unexpectedly ran the model")
    monkeypatch.setattr(model, "forward", forbid_forward)
    monkeypatch.setattr(model.model, "forward", forbid_forward)
    run_panel(model, tokenizer, probe, identity, args)
    assert pd.read_csv(directory / "interactions.csv").to_dict("records") == before
    freeze_json(tmp_path / "transport_config.json", {
        "protocol": "target_transport_v3", "objective": "sequence_margin"})
    report_transport(tmp_path)
    assert (tmp_path / "flow_review.tar.gz").is_file()
    changed = dict(probe, prefix_ids=[1, 3, 5, 2, 8])
    with pytest.raises(ValueError, match="inputs/settings changed"):
        run_panel(model, tokenizer, changed, identity, args)
