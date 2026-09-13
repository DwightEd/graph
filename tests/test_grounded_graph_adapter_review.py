"""CPU contracts for the grounded graph adapter, not semantic-owner claims."""

import math

import pytest
import torch
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from next_iteration.grounded_graph_adapter import (
    GroundedGraphAdapter,
    pointer_loss,
    token_loss,
)


def _adapter():
    torch.manual_seed(19)
    return GroundedGraphAdapter(
        input_dim=4, adapter_dim=3, node_types=3, edge_types=2, message_steps=1
    )


def _graph_inputs():
    torch.manual_seed(23)
    return {
        "source_states": torch.randn(4, 4),
        "query_states": torch.randn(2, 4),
        "node_types": torch.tensor([0, 1, 2, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        "edge_types": torch.tensor([0, 1, 0], dtype=torch.long),
        "graph_ids": torch.tensor([5, 5, 5, 5], dtype=torch.long),
        "query_graph_ids": torch.tensor([5, 5], dtype=torch.long),
        "source_available": torch.tensor([True, True, True, True]),
    }


def test_node_permutation_is_equivariant_and_edges_change_a_nonzero_residual():
    adapter, values = _adapter(), _graph_inputs()
    with torch.no_grad():
        adapter.output.weight.normal_()
        adapter.gate.weight.normal_()
        adapter.gate.bias.zero_()

    original = adapter(**values)
    without_edges = adapter(**values, use_edges=False)
    assert not torch.allclose(original["hidden"], without_edges["hidden"])

    # new node j is old node permutation[j]; inverse maps old endpoints to new.
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    permuted = {**values}
    for key in ("source_states", "node_types", "graph_ids", "source_available"):
        permuted[key] = values[key][permutation]
    permuted["edge_index"] = inverse[values["edge_index"]]
    got = adapter(**permuted)

    assert torch.allclose(got["hidden"], original["hidden"], atol=2e-6, rtol=2e-6)
    assert torch.allclose(got["pointer"][:, inverse], original["pointer"], atol=2e-6, rtol=2e-6)


def test_sample_membership_masks_attention_and_rejects_cross_sample_edges():
    adapter = _adapter()
    values = _graph_inputs()
    values.update({
        "graph_ids": torch.tensor([5, 5, 7, 7], dtype=torch.long),
        "query_graph_ids": torch.tensor([5, 7], dtype=torch.long),
        "edge_index": torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        "edge_types": torch.tensor([0, 1], dtype=torch.long),
    })
    first = adapter(**values)
    assert torch.equal(first["eligible"], torch.tensor([[True, True, False, False], [False, False, True, True]]))
    assert torch.equal(first["pointer"] * ~first["eligible"], torch.zeros_like(first["pointer"]))

    changed = {**values, "source_states": values["source_states"].clone()}
    changed["source_states"][2:] += 1_000
    second = adapter(**changed)
    assert torch.allclose(first["pointer"][0], second["pointer"][0])
    assert torch.allclose(first["node_states"][:2], second["node_states"][:2])

    invalid = {**values, "edge_index": torch.tensor([[0], [2]], dtype=torch.long), "edge_types": torch.tensor([0])}
    with pytest.raises(ValueError, match="crosses independent samples"):
        adapter(**invalid)


def test_bfloat16_tiny_llama_features_are_explicitly_promoted_and_preserve_first_step_logits():
    torch.manual_seed(29)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=31, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
    )).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        states = model.model(ids, use_cache=False, return_dict=True).last_hidden_state[0]
        expected_logits = model(ids, use_cache=False).logits[0, 2:-1]

    adapter = GroundedGraphAdapter(input_dim=16, adapter_dim=4, node_types=2, edge_types=1, message_steps=1)
    output = adapter(
        states[:2], states[2:-1], torch.tensor([0, 1]),
        torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long),
        torch.tensor([0, 0]), torch.tensor([0, 0]),
    )
    assert output["hidden"].dtype == torch.float32
    restored_logits = F.linear(output["hidden"].to(model.lm_head.weight.dtype), model.lm_head.weight)
    assert torch.equal(restored_logits, expected_logits)

    targets = ids[0, 3:]
    loss = token_loss(output["hidden"], targets, model.lm_head.weight, chunk_size=1)
    exact = F.cross_entropy(expected_logits.float(), targets)
    assert torch.allclose(loss, exact, atol=1e-7, rtol=1e-7)
    loss.backward()
    assert model.lm_head.weight.grad is None
    assert adapter.output.weight.grad is not None and torch.isfinite(adapter.output.weight.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in adapter.parameters())


def test_checkpointed_token_loss_matches_uncheckpointed_value_and_hidden_gradient():
    torch.manual_seed(31)
    weight = torch.randn(23, 7).requires_grad_(False)
    target = torch.tensor([1, 7, 3, 11, 4, 20], dtype=torch.long)
    checkpointed_hidden = torch.randn(6, 7, requires_grad=True)
    direct_hidden = checkpointed_hidden.detach().clone().requires_grad_(True)

    actual = token_loss(checkpointed_hidden, target, weight, chunk_size=2)
    direct = F.cross_entropy(F.linear(direct_hidden, weight), target)
    assert torch.allclose(actual, direct, atol=1e-7, rtol=1e-7)
    actual.backward()
    direct.backward()
    assert torch.allclose(checkpointed_hidden.grad, direct_hidden.grad, atol=1e-7, rtol=1e-7)
    assert weight.grad is None


def test_pointer_loss_is_or_over_multi_positive_occurrences_and_rejects_cross_graph_positive():
    logits = torch.tensor([[0.2, -0.3, 1.7]])
    output = {"pointer_logits": logits, "eligible": torch.tensor([[True, True, True]]),
              "hidden": torch.zeros(1, 2, requires_grad=True)}
    positives = torch.tensor([[True, True, False]])
    actual = pointer_loss(output, positives, torch.tensor([True]))
    expected = torch.logsumexp(logits, -1) - torch.logsumexp(logits[:, :2], -1)
    assert math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-7)

    output["eligible"] = torch.tensor([[True, False, True]])
    with pytest.raises(ValueError, match="escapes its sample or is unavailable"):
        pointer_loss(output, positives, torch.tensor([True]))
