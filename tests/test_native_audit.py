from dataclasses import replace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from route_graph.native_audit import NativeGate, native_forward


def model():
    torch.manual_seed(41)
    config = LlamaConfig(
        vocab_size=20,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


def clean_hooks(llm):
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in llm.modules())


def test_native_shams_preserve_all_logits_and_joint_gates_recompute_causally():
    llm, ids = model(), [1, 2, 3, 4, 5, 6, 7]
    base, _, _ = native_forward(llm, ids)
    gates = [
        NativeGate(0, (4, 5), "route", 1, (1, 2, 3), (1,), (2,)),
        NativeGate(0, (4, 5), "mlp", 1),
        NativeGate(1, (4, 5), "content", 1, (1, 2, 3), (1,)),
    ]
    sham, _, _ = native_forward(llm, ids, gates)
    assert torch.equal(sham, base)
    changed, _, _ = native_forward(llm, ids, [replace(g, strength=0) for g in gates])
    assert torch.equal(changed[:4], base[:4])
    assert not torch.equal(changed[4:], base[4:])
    clean_hooks(llm)


def test_native_mlp_matches_direct_branch_hook():
    llm, ids = model(), [1, 2, 3, 4, 5, 6]

    def direct_gate(module, args, output):
        result = output.clone()
        result[0, [3, 4]] *= 0.5
        return result

    handle = llm.model.layers[0].mlp.register_forward_hook(direct_gate)
    try:
        with torch.no_grad():
            expected = llm(torch.tensor([ids])).logits[0]
    finally:
        handle.remove()
    actual, _, _ = native_forward(llm, ids, [NativeGate(0, (3, 4), "mlp", 0.5)])
    assert torch.equal(actual, expected)
    clean_hooks(llm)


def test_branch_matched_input_donor_sham_and_selected_message_replacement():
    llm, ids = model(), [1, 2, 3, 4, 5, 6]
    captures = [(0, (1, 2)), (1, (1, 2))]
    base, native_values, _ = native_forward(llm, ids, capture=captures)
    donor_sham, same_values, _ = native_forward(
        llm, ids, input_scale=((1,), 1), capture=captures
    )
    assert torch.equal(base, donor_sham)
    assert all(
        torch.equal(native_values[i].values, same_values[i].values)
        for i in native_values
    )
    gates = [
        NativeGate(
            i,
            (3, 4),
            "donor",
            keys=(1, 2),
            selected=(1, 2),
            donor=same_values[i],
            expected_input_scale=((1,), 1),
        )
        for i in (0, 1)
    ]
    recipient_sham, _, _ = native_forward(llm, ids, gates)
    assert torch.equal(recipient_sham, base)
    _, changed_values, _ = native_forward(
        llm, ids, input_scale=((1,), 0), capture=captures
    )
    recipient, _, _ = native_forward(
        llm,
        ids,
        [
            replace(g, donor=changed_values[g.layer], expected_input_scale=((1,), 0))
            for g in gates
        ],
    )
    assert torch.equal(recipient[:3], base[:3])
    assert not torch.equal(recipient[3:], base[3:])
    clean_hooks(llm)


def test_forward_exception_removes_owned_hooks():
    llm, ids = model(), [1, 2, 3, 4, 5, 6]
    bad = NativeGate(0, (3,), "donor", keys=(1,), selected=(1,), donor=torch.zeros(9))
    with pytest.raises(ValueError, match="donor provenance"):
        native_forward(llm, ids, [bad], capture=[(1, (1,))])
    clean_hooks(llm)
    with pytest.raises(ValueError, match="at most one"):
        native_forward(llm, ids, [bad, bad])
    invalid_route = NativeGate(0, (3,), "route", 0, (1, 2), (1, 2))
    with pytest.raises(ValueError, match="no key"):
        native_forward(llm, ids, [invalid_route], capture=[(1, (1,))])
    clean_hooks(llm)


def test_reject_donor_wrong_branch_keys_model_or_type():
    llm, ids = model(), [1, 2, 3, 4, 5, 6]
    _, values, _ = native_forward(llm, ids, capture=[(0, (1, 2))])
    gate = NativeGate(0, (3, 4), "donor", keys=(1, 2), selected=(1,), donor=values[0])
    for wrong in (
        replace(values[0], input_ids=tuple(ids + [7])),
        replace(values[0], keys=(2, 1)),
        replace(values[0], model_identity=id(llm) + 1),
        replace(values[0], values=values[0].values.long()),
        replace(values[0], values="not a tensor"),
        replace(values[0], values=torch.zeros_like(values[0].values)),
        replace(values[0], input_scale=((1,), 0.5)),
    ):
        with pytest.raises(ValueError, match="donor provenance"):
            native_forward(llm, ids, [replace(gate, donor=wrong)])
        clean_hooks(llm)
    values[0].values.zero_()
    with pytest.raises(ValueError, match="donor provenance"):
        native_forward(llm, ids, [gate])
    clean_hooks(llm)


def test_mixed_visibility_history_domain_is_causal_but_all_future_gate_is_invalid():
    llm, ids = model(), [1, 2, 3, 4, 5, 6]
    base, _, _ = native_forward(llm, ids)
    gate = NativeGate(0, (2, 4), "content", 0, (1, 3), (3,))
    actual, _, _ = native_forward(llm, ids, [gate])
    assert torch.equal(actual[:4], base[:4])
    assert not torch.equal(actual[4:], base[4:])
    with pytest.raises(ValueError, match="visible"):
        native_forward(llm, ids, [replace(gate, queries=(2,))])
    clean_hooks(llm)
