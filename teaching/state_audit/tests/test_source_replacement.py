"""Regression for separately rounded source removal/restoration and query-local donors."""

import pytest
import torch
from state_audit.operations import ReplaceSource, Target
from state_audit.operations.messages import replace_source_readouts


def tensors():
    random = torch.Generator().manual_seed(31)
    weights = torch.randn(1, 4, 24, 24, generator=random).softmax(-1).bfloat16()
    values = (torch.randn(1, 4, 24, 128, generator=random) * 8).bfloat16()
    return weights, values


def reference(weights, values, target):
    selected = weights[0][target.indices(weights[0])].clone()
    value_target = Target("value", target.layers, target.keys, target.heads)
    payload = values[0][value_target.indices(values[0])].clone()
    return ReplaceSource(target, selected, payload)


def test_large_bfloat16_cut_restore_is_exact_where_old_addition_fails():
    weights, values = tensors()
    target = Target("attention", (0,), (15, 20), (0, 3), (2, 4, 6, 8))
    operation = reference(weights, values, target)
    cut = weights.clone()
    cut[0][target.indices(cut[0])] = 0
    baseline = weights @ values
    restored = replace_source_readouts(cut, values, [operation])
    assert torch.equal(restored, baseline)

    # The old float32 source reconstruction then BF16 add is not an inverse.
    readout = Target("value", (0,), target.positions, target.heads)
    indices = readout.indices(baseline[0])
    source = (operation.weights.float() @ operation.values.float()).bfloat16()
    old = (cut @ values)[0][indices] + source
    assert not torch.equal(old, baseline[0][indices])


@pytest.mark.parametrize("source_count", [0, 1, 3, 5])
def test_donor_replaces_only_selected_sources_and_queries(source_count):
    weights, values = tensors()
    target = Target("attention", (0,), (15, 20), (1,), (2, 4, 6))
    donor_weights = torch.full((1, 2, source_count), 0.125, dtype=torch.bfloat16)
    donor_values = torch.full((1, source_count, 128), 4.0, dtype=torch.bfloat16)
    operation = ReplaceSource(target, donor_weights, donor_values)
    restored = replace_source_readouts(weights, values, [operation])
    original = weights @ values
    other = torch.ones(original.shape[:-1], dtype=torch.bool)
    other[0, 1, [15, 20]] = False
    assert torch.equal(restored[other], original[other])
    for query in target.positions:
        complement = weights[0, 1, query].float().clone()
        complement[list(target.keys)] = 0
        expected = complement @ values[0, 1].float() + source_count * 0.125 * 4
        assert torch.equal(restored[0, 1, query], expected.bfloat16())


def test_disjoint_sources_at_same_head_query_compose_without_overwriting():
    weights, values = tensors()
    targets = [Target("attention", (0,), (15,), (1,), keys) for keys in ((2, 4), (6, 8))]
    operations = [reference(weights, values, target) for target in targets]
    cut = weights.clone()
    for target in targets:
        cut[0][target.indices(cut[0])] = 0
    assert torch.equal(replace_source_readouts(cut, values, operations), weights @ values)
