"""Fixed-foil target isolation, exact-query causal cuts, shared edges, and cache controls."""


import numpy as np

from experiments.native_support.message_carriers.token_representation import aggregate_units


def test_position_weight_has_explicit_direction_and_beta_zero_recovers_uniform_mean():
    units = [dict(start=0, stop=3), dict(start=3, stop=4)]
    values = np.array([9., 0., 0., 7.])
    np.testing.assert_allclose(aggregate_units(values, units), [3, 3, 3, 7])
    assert aggregate_units(values, units, 1)[0] < 3
    assert aggregate_units(values, units, -1)[0] > 3
    assert aggregate_units(values, units, 1)[3] == 7
