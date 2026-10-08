"""Scientific invariants for first-only candidate/source controls."""
import torch

from .first_address import FirstAddressReader, first_choice_loss, reverse_valid_payload


def example():
    torch.manual_seed(2)
    return dict(node_fields=torch.randn(3, 6, 12), boundary_fields=torch.randn(3, 10, 12),
        query=torch.randn(3, 12), source_key=torch.randn(3, 5, 6),
        source_value=torch.randn(3, 5, 6), summed_payload=torch.randn(3, 6),
        source_valid=torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool),
        candidate_vectors=torch.randn(3, 2, 12))


def test_padding_cannot_contribute():
    model = FirstAddressReader(12, 6, 4, 5, dropout=0).eval()
    fields = example()
    original = model(**fields)
    fields['source_key'][~fields['source_valid']] = 10000
    fields['source_value'][~fields['source_valid']] = -10000
    changed, weights = model(**fields, return_attention=True)
    torch.testing.assert_close(original, changed)
    assert torch.count_nonzero(weights[~fields['source_valid']]) == 0


def test_payload_reverse_preserves_keys_and_padding():
    fields = example()
    original = fields['source_value']
    reversed_value = reverse_valid_payload(original, fields['source_valid'])
    for row in range(len(original)):
        count = int(fields['source_valid'][row].sum())
        torch.testing.assert_close(reversed_value[row, :count], original[row, :count].flip(0))
        torch.testing.assert_close(reversed_value[row, count:], original[row, count:])


def test_candidate_independent_address_and_shared_dropout():
    model = FirstAddressReader(12, 6, 4, 5, dropout=.5).train()
    fields = example()
    fields['candidate_vectors'][:, 1] = fields['candidate_vectors'][:, 0]
    risk, attention = model(**fields, return_attention=True)
    torch.testing.assert_close(risk[:, 0], risk[:, 1])
    fields['candidate_vectors'] = torch.randn_like(fields['candidate_vectors'])
    _, changed_attention = model(**fields, return_attention=True)
    torch.testing.assert_close(attention, changed_attention)


def test_every_parameter_operative_in_all_controls():
    fields = example()
    for variant in ('group_summed', 'address', 'payload_rewired'):
        model = FirstAddressReader(12, 6, 4, 5, dropout=0)
        first_choice_loss(model(**fields, variant=variant)).backward()
        assert all(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
                   for parameter in model.parameters())


def test_pair_objective_has_correct_risk_direction():
    better = torch.tensor([[-2., 2.]])
    reversed_risk = torch.tensor([[2., -2.]])
    assert first_choice_loss(better) < first_choice_loss(reversed_risk)


def test_cosine_address_is_invariant_to_query_key_projection_scale():
    fields = example()
    model = FirstAddressReader(12, 6, 4, 5, dropout=0, address_energy='cosine').eval()
    risks = {variant: model(**fields, variant=variant).detach() for variant in
             ('group_summed', 'address', 'payload_rewired')}
    with torch.no_grad():
        model.address_query.weight.mul_(100)
        model.address_key.weight.mul_(100)
    for variant, risk in risks.items():
        torch.testing.assert_close(risk, model(**fields, variant=variant), atol=1e-5, rtol=1e-5)


def test_cosine_address_padding_and_bounded_energy():
    fields = example()
    model = FirstAddressReader(12, 6, 4, 5, dropout=0,
        address_energy='cosine', address_temperature=8.).eval()
    original, weights = model(**fields, return_attention=True)
    fields['source_key'][~fields['source_valid']] = 10000
    fields['source_value'][~fields['source_valid']] = -10000
    changed = model(**fields)
    torch.testing.assert_close(original, changed)
    for row in range(len(weights)):
        valid_weights = weights[row, fields['source_valid'][row]]
        assert torch.log(valid_weights.max() / valid_weights.min()) <= 16.0001
