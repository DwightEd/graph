import numpy as np

from experiments.unsupervised_token_graph.structured_compatibility.training import (
    causal_sticky,
    donor,
    estimate_rho,
    valid_positions,
)


def test_valid_positions_requires_current_and_previous_coverage():
    coverage = np.array([True, True, False, True, True])
    assert valid_positions(coverage).tolist() == [1, 4]


def test_donor_prefers_same_local_block():
    random = np.random.default_rng(3)
    candidates = np.array([8, 9, 10, 20])
    chosen = donor(9, candidates, random, block=8)
    assert chosen in (8, 10)


def test_sticky_is_causal_and_resets_at_missing_values():
    values = np.array([1., 3., np.nan, 5., 7.])
    result = causal_sticky(values, .5)
    np.testing.assert_allclose(
        result[[0, 1, 3, 4]],
        [1., 2., 5., 6.],
    )
    assert np.isnan(result[2])


def test_rho_uses_only_observed_score_sequences():
    sequence = np.arange(8, dtype=float)
    rho = estimate_rho([sequence])
    assert .9 <= rho <= .95
