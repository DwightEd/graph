"""Coordinate preservation, paired routing, and matched local-reader controls."""
import pytest
import torch

from .readout import LocalTransportReader, edge_factors, graph_inputs, rewire_local_weights


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reader_inputs(rows=10, width=8):
    generator = torch.Generator().manual_seed(73)
    node_fields = torch.randn(rows, 6, 8, generator=generator, dtype=torch.float64)
    boundary_fields = torch.randn(rows, 10, 8, generator=generator, dtype=torch.float64)
    value_fields = torch.randn(rows, 2, 8, generator=generator, dtype=torch.float64)
    indices = torch.arange(rows)[:, None] - torch.arange(1, width + 1)[None]
    valid = indices >= 0
    attention = torch.rand(2, rows, 2, width, generator=generator, dtype=torch.float64)
    attention *= valid[None, :, None]
    scalars = torch.randn(rows, 8, generator=generator, dtype=torch.float64)
    return [node_fields, boundary_fields, value_fields, attention, indices, valid, scalars]


def tiny_reader():
    torch.manual_seed(42)
    return LocalTransportReader(model_dim=8, heads=2, head_dim=4, dropout=0.).double()


def test_full_edge_factors_reconstruct_native_minus_blocked_message():
    fields = torch.tensor([[[2., -3., 1., 4.]], [[5., 7., -2., 3.]]], dtype=torch.float64)
    attention = torch.tensor([[[[.2, .8]], [[.7, .3]]],
                              [[[.6, .4]], [[.1, .9]]]], dtype=torch.float64)
    indices = torch.tensor([[0, 1], [1, 0]])
    factors = edge_factors(fields, attention, indices)
    native, content, routing = factors.chunk(3, dim=-1)
    values = fields[indices]
    native_values, difference = values.chunk(2, dim=-1)
    blocked_values = native_values - difference
    native_expected = attention[0].transpose(1, 2)[..., None] * native_values
    blocked_expected = attention[1].transpose(1, 2)[..., None] * blocked_values
    torch.testing.assert_close(native, native_expected)
    torch.testing.assert_close(content + routing, native_expected - blocked_expected)
    assert (content < 0).any() and (routing < 0).any()


def test_rewire_preserves_both_worlds_heads_mass_validity_and_short_lags():
    inputs = reader_inputs()
    attention, _, valid = inputs[3:6]
    rewired = rewire_local_weights(attention, valid)
    torch.testing.assert_close(rewired.sum(-1), attention.sum(-1))
    torch.testing.assert_close(rewired[..., :2], attention[..., :2])
    assert not torch.equal(rewired, attention)
    assert not rewired.masked_select(~valid[None, :, None].expand_as(rewired)).any()
    torch.testing.assert_close(rewired[:, 9, :, 2:4], attention[:, 9, :, 2:4].flip(-1))
    torch.testing.assert_close(rewired[:, 9, :, 4:8], attention[:, 9, :, 4:8].flip(-1))


def test_rewire_does_not_exchange_valid_and_invalid_noncontiguous_edges():
    attention = torch.arange(16, dtype=torch.float64).reshape(2, 1, 1, 8)
    valid = torch.tensor([[True, True, False, True, True, False, True, False]])
    attention *= valid[None, :, None]
    actual = rewire_local_weights(attention, valid)
    torch.testing.assert_close(actual.sum(-1), attention.sum(-1))
    torch.testing.assert_close(actual[..., 3], attention[..., 3])
    torch.testing.assert_close(actual[..., 4], attention[..., 6])
    torch.testing.assert_close(actual[..., 6], attention[..., 4])
    assert not actual[..., [2, 5, 7]].any()


@pytest.mark.parametrize('variant', ['real', 'rewired', 'uniform', 'self'])
def test_graph_controls_preserve_per_world_head_mass_and_endpoint_input(variant):
    _, _, _, attention, indices, valid, _ = reader_inputs()
    original_indices = indices.clone()
    weights, senders = graph_inputs(attention, indices, valid, variant)
    torch.testing.assert_close(weights.sum(-1), attention.sum(-1))
    torch.testing.assert_close(indices, original_indices)
    assert not weights.masked_select(~valid[None, :, None].expand_as(weights)).any()
    if variant == 'self':
        expected = torch.arange(len(indices))[:, None].expand_as(indices)
        torch.testing.assert_close(senders, expected)
    else:
        torch.testing.assert_close(senders, indices.clamp_min(0))


@pytest.mark.parametrize('variant', ['real', 'rewired', 'uniform', 'self'])
def test_no_future_node_value_or_boundary_changes_earlier_outputs(variant):
    reader = tiny_reader().eval()
    inputs = reader_inputs()
    baseline = reader(*inputs, variant=variant)
    changed = [value.clone() for value in inputs]
    for index in (0, 1, 2, 6):
        changed[index][6:] *= 97
    result = reader(*changed, variant=variant)
    torch.testing.assert_close(result[:6], baseline[:6], rtol=0, atol=0)


def test_real_graph_uses_past_sender_values_beyond_current_node_and_scalars():
    reader = tiny_reader().eval()
    inputs = reader_inputs()
    baseline = reader(*inputs, variant='real')
    changed = [value.clone() for value in inputs]
    changed[2][0] += 10
    result = reader(*changed, variant='real')
    torch.testing.assert_close(result[0], baseline[0], rtol=0, atol=0)
    assert not torch.isclose(result[1], baseline[1], rtol=0, atol=1e-10)
    self_before = reader(*inputs, variant='self')
    self_after = reader(*changed, variant='self')
    torch.testing.assert_close(self_after[1:], self_before[1:], rtol=0, atol=0)


def test_invalid_edges_have_zero_message_despite_arbitrary_stored_attention():
    reader = tiny_reader().eval()
    inputs = reader_inputs()
    baseline = reader(*inputs)
    changed = [value.clone() for value in inputs]
    invalid = ~changed[5][None, :, None].expand_as(changed[3])
    changed[3][invalid] = 1e9
    result = reader(*changed)
    torch.testing.assert_close(result, baseline, rtol=0, atol=0)


def test_all_original_node_boundary_and_head_coordinates_receive_training_gradients():
    reader = tiny_reader()
    inputs = reader_inputs()
    for index in (0, 1, 2):
        inputs[index].requires_grad_(True)
    logits = reader(*inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.arange(len(logits)).remainder(2).double())
    loss.backward()
    for projection in (reader.node_projection, reader.boundary_projection):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad) == projection.weight.numel()
    assert torch.count_nonzero(reader.head_projection.grad) == reader.head_projection.numel()
    assert torch.count_nonzero(inputs[0].grad) == inputs[0].numel()
    assert torch.count_nonzero(inputs[1].grad) == inputs[1].numel()
    assert torch.count_nonzero(inputs[2].grad[:-1]) == inputs[2][:-1].numel()
    assert inputs[2].grad[-1].count_nonzero() == 0


def test_variants_share_parameter_count_and_return_distinct_per_node_embeddings():
    reader = tiny_reader().eval()
    inputs = reader_inputs()
    parameters = sum(parameter.numel() for parameter in reader.parameters())
    names = tuple(reader.state_dict())
    results = []
    for variant in ('real', 'rewired', 'uniform', 'self'):
        logits, embedding = reader(*inputs, variant=variant, return_embedding=True)
        assert logits.shape == (10,) and embedding.shape == (10, 136)
        assert sum(parameter.numel() for parameter in reader.parameters()) == parameters
        assert tuple(reader.state_dict()) == names
        assert not torch.equal(embedding[2], embedding[3])
        results.append(logits)
    assert not torch.equal(results[0], results[3])


def test_zero_local_mass_is_identical_for_all_matched_controls():
    reader = tiny_reader().eval()
    inputs = reader_inputs()
    inputs[3].zero_()
    native = reader(*inputs, variant='real')
    for variant in ('rewired', 'uniform', 'self'):
        torch.testing.assert_close(reader(*inputs, variant=variant), native, rtol=0, atol=0)
