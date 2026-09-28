import numpy as np
import torch
from itertools import combinations, product

from .messages import projected_norm, group_messages, norm_route
from .head_null import identity_overlap


def test_native_factorization_equals_explicit_output_messages_and_gram():
    torch.manual_seed(42)
    values = torch.randn(4, 7, 3, dtype=torch.float64)
    weight = torch.randn(4, 12, 3, dtype=torch.float64)
    attention = torch.randn(4, 5, 7, dtype=torch.float64).softmax(-1)
    masks = (torch.arange(7) < 3, torch.arange(7) >= 5, (torch.arange(7) >= 3) & (torch.arange(7) < 5))
    output_values = values @ weight.transpose(1, 2)
    torch.testing.assert_close(projected_norm(values, weight), output_values.norm(dim=-1))
    factors, grams, mass, _ = group_messages(attention, values, weight, masks)
    for group, mask in enumerate(masks):
        expected = (attention * mask) @ output_values
        actual = factors[group] @ weight.transpose(1, 2)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(grams[group], expected.permute(1, 0, 2) @ expected.permute(1, 2, 0))
        assert torch.all(actual.norm(dim=-1) <= mass[group] + 1e-12)


def test_opposite_messages_reveal_cancellation_without_changing_attention():
    values = torch.tensor([[[1., 0.], [-1., 0.], [0., 1.]]])
    attention = torch.tensor([[[.4, .4, .2]]])
    weight = torch.eye(2)[None]
    masks = (torch.tensor([True, True, False]), torch.tensor([False, False, True]), torch.zeros(3, dtype=torch.bool))
    _, grams, masses, _ = group_messages(attention, values, weight, masks)
    net = grams.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt().permute(0, 2, 1)
    assert float(norm_route(masses[None])) < 0
    assert float(norm_route(net[None])) > 0
    assert float(net[0, 0, 0]) == 0


def test_conditional_head_overlap_matches_exhaustive_identity_null():
    first = np.array([[True, True, False, False], [True, False, False, False]])
    second = np.array([[True, True, False, False], [True, True, False, False]])
    result = identity_overlap(first, second)
    overlaps = []
    for choices in product(*(list(combinations(range(4), int(row.sum()))) for row in second)):
        shuffled = np.zeros_like(second)
        for layer, chosen in enumerate(choices):
            shuffled[layer, list(chosen)] = True
        overlaps.append(int((first & shuffled).sum()))
    assert result['observed'] == 3
    assert np.isclose(result['expected_under_within_layer_identity_shuffle'], np.mean(overlaps))
    assert np.isclose(result['upper_tail_probability'], np.mean(np.array(overlaps) >= 3))
