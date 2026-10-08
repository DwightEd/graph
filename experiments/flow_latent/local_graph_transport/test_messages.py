"""CPU algebra/structure tests; no native-model or detection-effect claims."""
import pytest
import torch
from torch.nn import functional as F

from .messages import (
    aggregate_attention_write, aggregate_partitions, decompose_messages,
    partition_senders, project_edges, state_conservation_error,
)


def sample_worlds():
    generator = torch.Generator().manual_seed(73)
    attention_plus = torch.randn(3, 2, 5, generator=generator, dtype=torch.float64).softmax(-1)
    attention_minus = torch.randn(3, 2, 5, generator=generator, dtype=torch.float64).softmax(-1)
    values_plus = torch.randn(5, 2, 4, generator=generator, dtype=torch.float64)
    values_minus = torch.randn(5, 2, 4, generator=generator, dtype=torch.float64)
    output_weight = torch.randn(7, 8, generator=generator, dtype=torch.float64)
    return attention_plus, attention_minus, values_plus, values_minus, output_weight


def test_content_and_routing_reconstruct_each_native_edge_and_total_write():
    attention_plus, attention_minus, values_plus, values_minus, weight = sample_worlds()
    terms = decompose_messages(attention_plus, attention_minus, values_plus, values_minus)
    expected_edges = (attention_plus[..., None] * values_plus.permute(1, 0, 2)[None] -
                      attention_minus[..., None] * values_minus.permute(1, 0, 2)[None])
    torch.testing.assert_close(terms['content'] + terms['routing'], expected_edges,
                               rtol=1e-12, atol=1e-12)
    native_plus = torch.einsum('rhs,shd->rhd', attention_plus, values_plus)
    native_minus = torch.einsum('rhs,shd->rhd', attention_minus, values_minus)
    expected_write = F.linear(native_plus.flatten(1), weight) - F.linear(native_minus.flatten(1), weight)
    actual_write = aggregate_attention_write(terms['content'] + terms['routing'], weight)
    torch.testing.assert_close(actual_write, expected_write, rtol=1e-12, atol=1e-12)


def test_unchanged_values_can_have_nonzero_routing_difference():
    attention_plus, attention_minus, values_plus, _, _ = sample_worlds()
    terms = decompose_messages(attention_plus, attention_minus, values_plus, values_plus)
    torch.testing.assert_close(terms['content'], torch.zeros_like(terms['content']), rtol=0, atol=0)
    assert torch.count_nonzero(terms['routing']) > 0
    expected = (attention_plus - attention_minus)[..., None] * values_plus.permute(1, 0, 2)[None]
    torch.testing.assert_close(terms['routing'], expected, rtol=0, atol=0)


def test_unchanged_routes_can_transport_nonzero_value_difference():
    attention_plus, _, values_plus, values_minus, _ = sample_worlds()
    terms = decompose_messages(attention_plus, attention_plus, values_plus, values_minus)
    torch.testing.assert_close(terms['routing'], torch.zeros_like(terms['routing']), rtol=0, atol=0)
    expected = attention_plus[..., None] * (values_plus - values_minus).permute(1, 0, 2)[None]
    torch.testing.assert_close(terms['content'], expected, rtol=0, atol=0)
    assert torch.count_nonzero(terms['content']) > 0


def test_identical_worlds_have_zero_terms_and_world_swap_reverses_signed_terms():
    attention_plus, attention_minus, values_plus, values_minus, _ = sample_worlds()
    identical = decompose_messages(attention_plus, attention_plus, values_plus, values_plus)
    for term in identical.values():
        assert torch.count_nonzero(term) == 0
    forward = decompose_messages(attention_plus, attention_minus, values_plus, values_minus)
    reversed_worlds = decompose_messages(attention_minus, attention_plus, values_minus, values_plus)
    for name in forward:
        torch.testing.assert_close(reversed_worlds[name], -forward[name], rtol=0, atol=0)
        assert (forward[name] < 0).any()


def test_projected_edge_sum_equals_native_projection_without_head_or_sender_loss():
    attention_plus, attention_minus, values_plus, values_minus, weight = sample_worlds()
    terms = decompose_messages(attention_plus, attention_minus, values_plus, values_minus)
    edges = terms['content'] + terms['routing']
    projected = project_edges(edges, weight)
    assert projected.shape == (3, 2, 5, 7)
    torch.testing.assert_close(projected.sum(dim=(1, 2)), aggregate_attention_write(edges, weight),
                               rtol=1e-12, atol=1e-12)


def test_nonidentity_output_transport_is_not_attention_times_sender_hidden_difference():
    # Even with identity V projection, W_O rotates the transported direction.
    sender_hidden_delta = torch.tensor([1., 0.], dtype=torch.float64)
    attention = torch.ones(1, 1, 1, dtype=torch.float64)
    values_plus = sender_hidden_delta.reshape(1, 1, 2)
    values_minus = torch.zeros_like(values_plus)
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    terms = decompose_messages(attention, attention, values_plus, values_minus)
    actual = aggregate_attention_write(terms['content'], rotation)
    naive = attention[0, 0, 0] * sender_hidden_delta
    torch.testing.assert_close(actual[0], torch.tensor([0., 1.], dtype=torch.float64), rtol=0, atol=0)
    assert not torch.allclose(actual[0], naive)


def test_partition_is_complete_exclusive_and_excludes_future_including_source_self():
    source = torch.tensor([True, True, False, False, False, False, False, False, False])
    receivers = torch.tensor([1, 3, 5, 7])
    groups = partition_senders(source, answer_start=4, receiver_positions=receivers, local_radius=2)
    membership = torch.stack(list(groups.values())).sum(dim=0)
    causal = torch.arange(len(source))[None] <= receivers[:, None]
    torch.testing.assert_close(membership, causal.long(), rtol=0, atol=0)
    assert groups['self'][0, 1] and not groups['source'][0, 1]
    assert groups['self'][1, 3] and not groups['other_prompt'][1, 3]
    assert groups['local'][3, 5] and groups['local'][3, 6]
    assert groups['remote'][3, 4]
    assert not groups['local'][3, 7]
    assert groups['source'][3, 0] and groups['source'][3, 1]
    assert groups['other_prompt'][3, 2] and groups['other_prompt'][3, 3]
    assert not membership[:, 8].any()


def test_partition_aggregation_preserves_remote_and_every_receiver_without_renormalizing():
    source = torch.tensor([True, False, False, False, False, False, False, False])
    receivers = torch.tensor([3, 5, 7])
    groups = partition_senders(source, 3, receivers, local_radius=1)
    generator = torch.Generator().manual_seed(91)
    edges = torch.randn(3, 2, 8, 4, generator=generator, dtype=torch.float64)
    causal = torch.arange(8)[None] <= receivers[:, None]
    group_messages = aggregate_partitions(edges, groups)
    expected = (edges * causal[:, None, :, None]).sum(dim=2)
    torch.testing.assert_close(torch.stack(list(group_messages.values())).sum(dim=0), expected,
                               rtol=1e-12, atol=1e-12)
    assert group_messages['remote'][2].abs().sum() > 0
    assert not torch.allclose(expected[0], expected[1])


def test_receivers_keep_distinct_edge_coordinates_instead_of_a_span_mean():
    attention = torch.tensor([[[1., 0.]], [[0., 1.]]], dtype=torch.float64)
    values_plus = torch.tensor([[[2., 3.]], [[-4., 5.]]], dtype=torch.float64)
    values_minus = torch.zeros_like(values_plus)
    terms = decompose_messages(attention, attention, values_plus, values_minus)
    write = aggregate_attention_write(terms['content'], torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(write, values_plus[:, 0], rtol=0, atol=0)
    assert not torch.allclose(write[0], write[1])


@pytest.mark.parametrize('scale', [0., 1., 100.])
def test_state_conservation_holds_for_arbitrary_worlds_so_is_not_a_detector(scale):
    generator = torch.Generator().manual_seed(42)
    before_plus = torch.randn(3, 7, generator=generator, dtype=torch.float64) * scale
    before_minus = torch.randn(3, 7, generator=generator, dtype=torch.float64) * scale
    attention_plus = torch.randn(3, 7, generator=generator, dtype=torch.float64) * scale
    attention_minus = torch.randn(3, 7, generator=generator, dtype=torch.float64) * scale
    mlp_plus = torch.randn(3, 7, generator=generator, dtype=torch.float64) * scale
    mlp_minus = torch.randn(3, 7, generator=generator, dtype=torch.float64) * scale
    after_plus = before_plus + attention_plus + mlp_plus
    after_minus = before_minus + attention_minus + mlp_minus
    error = state_conservation_error(before_plus - before_minus, attention_plus - attention_minus,
                                     mlp_plus - mlp_minus, after_plus - after_minus)
    torch.testing.assert_close(error, torch.zeros_like(error), rtol=0, atol=1e-12)


def test_state_conservation_reports_a_missing_write_as_error_at_its_own_node():
    zero = torch.zeros(3, 7, dtype=torch.float64)
    incomplete_after = zero.clone()
    incomplete_after[1, 4] = 2.5
    error = state_conservation_error(zero, zero, zero, incomplete_after)
    torch.testing.assert_close(error, incomplete_after, rtol=0, atol=0)
    assert torch.count_nonzero(error) == 1
