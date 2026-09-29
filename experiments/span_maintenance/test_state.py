"""Scientific invariants: causal coordinates, trivial locality and explicit weights."""
import numpy as np

from .state import FIELDS, maintain_layer, signed_readout
from .boundary import punctuation_boundaries, causal_punctuation_weights
from .payload import adjacent_change


def reading(mode, count=8):
    result = np.zeros((2, count, 2 + count - 1))
    result[:, 0, 0] = 1
    for target in range(1, count):
        result[:, target, 0] = .1
        key = 2 if mode == 'anchor' else 2 + target - 1
        result[:, target, key] = .9
    return result


def test_trivial_lag_is_not_endpoint_specific_evidence():
    measured, weights, _ = maintain_layer(reading('lag'), 2, list(range(8)))
    np.testing.assert_allclose(measured[:, 2:, FIELDS.index('lag_overlap')], 1)
    np.testing.assert_allclose(measured[..., FIELDS.index('distance_excess')], 0)
    np.testing.assert_allclose(weights.sum(-1), 1)


def test_anchor_and_lag_modes_remain_separate():
    anchor, _, _ = maintain_layer(reading('anchor'), 2, list(range(8)))
    lag, _, _ = maintain_layer(reading('lag'), 2, list(range(8)))
    np.testing.assert_allclose(anchor[:, 2:, FIELDS.index('anchor_overlap')], 1)
    assert np.all(anchor[:, 2:, FIELDS.index('lag_overlap')] < 1)
    assert np.all(lag[:, 2:, FIELDS.index('anchor_overlap')] < 1)


def test_prefix_consistency_and_no_future_weights():
    attention = reading('lag')
    full, weights, dependency = maintain_layer(attention, 2, list(range(8)))
    prefix, prefix_weights, prefix_dependency = maintain_layer(attention[:, :5, :6], 2, list(range(5)))
    np.testing.assert_allclose(full[:, :5], prefix)
    np.testing.assert_allclose(weights[:5, :5], prefix_weights)
    np.testing.assert_allclose(dependency[:5, :5], prefix_dependency)
    assert not np.any(np.triu(weights, 1))


def test_new_prompt_address_reduces_age_without_punctuation():
    attention = reading('lag')
    attention[:, 5] = 0
    attention[:, 5, 1] = .9
    attention[:, 5, 6] = .1
    measured, _, _ = maintain_layer(attention, 2, list(range(8)))
    assert np.all(measured[:, 5, FIELDS.index('age')] < measured[:, 4, FIELDS.index('age')])
    np.testing.assert_allclose(measured[:, 5, FIELDS.index('prompt_refresh')], .9)


def test_explicit_kernel_equals_online_weighted_statistic():
    measured, weights, _ = maintain_layer(reading('anchor'), 2, list(range(8)))
    values = np.arange(8) ** 2
    state = np.zeros(2)
    age = np.zeros(2)
    result = []
    for target, value in enumerate(values):
        retained = measured[:, target, FIELDS.index('continuation')] * age
        age = 1 + retained
        state = (retained * state + value) / age
        result.append(state.mean())
    np.testing.assert_allclose(weights @ values, result)


def test_signs_are_not_cancelled_and_common_head_permutation_is_equivariant():
    effect = np.array([[[3., -3., 0.], [2., -2., 0.]]])
    signed, _ = signed_readout(effect, 2)
    np.testing.assert_array_equal(signed[0, 0], [3, 3, 0, 0])
    attention = reading('lag')
    attention[1] = reading('anchor')[1]
    actual, weights, _ = maintain_layer(attention, 2, list(range(8)))
    permuted, swapped, _ = maintain_layer(attention[::-1], 2, list(range(8)))
    np.testing.assert_allclose(actual[::-1], permuted)
    np.testing.assert_allclose(weights, swapped)


def test_punctuation_hint_uses_past_only_and_preserves_stable_soft_regime():
    text = ['A', '.', ' ', 'Next', '\n', 'value', '.']
    boundary = punctuation_boundaries(text)
    np.testing.assert_array_equal(boundary, [False, False, True, False, False, True, False])
    np.testing.assert_array_equal(boundary[:4], punctuation_boundaries(text[:4]))
    cue = np.zeros(8, dtype=bool)
    cue[5] = True
    hard, _, _ = maintain_layer(reading('lag'), 2, list(range(8)), cue, 'hard')
    soft, _, _ = maintain_layer(reading('lag'), 2, list(range(8)), cue, 'soft')
    np.testing.assert_allclose(hard[:, 5, FIELDS.index('age')], 1)
    assert np.all(soft[:, 5, FIELDS.index('age')] > 1)
    weights = causal_punctuation_weights(cue)
    assert np.count_nonzero(weights[5]) == 1


def test_payload_magnitude_change_is_visible_and_zero_pair_is_unknown():
    measured = adjacent_change(np.array([[1., 0.], [2., 0.], [0., 0.], [0., 0.]]))
    np.testing.assert_allclose(measured[1:3], [1 / 3, 1])
    assert np.isnan(measured[0]) and np.isnan(measured[3])
