"""Scientific timing and grouped-weight contracts for the supervised runner."""
import numpy as np

from .data import scalar_features, source_loss_weights


def test_prechoice_cannot_read_actual_token_likelihood_or_total_length():
    pack = dict(context=np.arange(18, dtype=float).reshape(3, 6),
                observations=np.arange(33, dtype=float).reshape(3, 11))
    original = scalar_features(pack, 'prechoice')
    changed = {name: value.copy() for name, value in pack.items()}
    changed['observations'][:, :4] += 1000
    changed['context'][:, :3] += 1000
    changed['context'][:, 4] += 1000
    np.testing.assert_array_equal(original, scalar_features(changed, 'prechoice'))
    np.testing.assert_array_equal(original[:, :3], 0)
    np.testing.assert_array_equal(original[:, 6], pack['context'][:, 3])


def test_posttoken_raw_source_restores_token_deviation_instead_of_mean_broadcast():
    pack = dict(context=np.ones((3, 6)), observations=np.zeros((3, 11)))
    pack['observations'][:, 0] = [-1, 0, 2]
    pack['observations'][:, 1] = [2, 0, -1]
    actual = scalar_features(pack)
    np.testing.assert_array_equal(actual[:, 0], [0, 1, 3])
    np.testing.assert_array_equal(actual[:, 1], [3, 1, 0])


def test_source_loss_gives_equal_source_mass_and_no_development_weight():
    pack = dict(source_index=np.asarray([0, 0, 1, 1, 1, 1, 2, 2]),
                development=np.asarray([False] * 6 + [True] * 2))
    weights = source_loss_weights(pack)
    np.testing.assert_allclose(weights[:2].sum(), weights[2:6].sum())
    np.testing.assert_array_equal(weights[6:], 0)
    np.testing.assert_allclose(weights[:6].mean(), 1)
