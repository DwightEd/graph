"""CPU-only zero/submass regression tests; no model or natural labels."""

from types import SimpleNamespace

import numpy as np

from experiments.unsupervised_token_graph.head_roles.inputs import extract, route_features


def test_present_empty_ordinary_row_is_zero_and_does_not_remove_other_heads():
    sample = SimpleNamespace(response_length=3, prompt_length=2,
                             token_ids=np.array([99, 1, 2, 3, 4]), offsets=np.empty((0, 2), int))
    empty = SimpleNamespace(layer=0, head=0, queries=[2, 3], row=lambda _: (np.array([0]), np.array([1.])))
    normal = SimpleNamespace(layer=0, head=1, queries=[2, 3], row=lambda _: (np.array([1, 2]), np.array([.2, .3])))
    saved = extract(sample, [empty, normal], [99], 10)
    np.testing.assert_array_equal(saved["coverage"], [False, True, True])
    assert np.isnan(saved["observations"][0]).all()
    np.testing.assert_array_equal(saved["observations"][1:, 0, 0], 0.)
    np.testing.assert_allclose(saved["observations"][1:, 0, 1, -1], .5)


def test_weighted_entropy_and_routes_scale_with_mass_without_epsilon_floor():
    keys, weights = np.array([1, 3, 7, 8]), np.array([.1, .2, .1, .2])
    first = np.asarray(route_features(keys, weights, query=8, prompt_length=3, recent_window=2))
    second = np.asarray(route_features(keys, weights * 1e-12, query=8, prompt_length=3, recent_window=2))
    np.testing.assert_allclose(second, first * 1e-12, rtol=1e-12, atol=0)
    np.testing.assert_allclose(first[:4].sum(), first[-1])
    np.testing.assert_array_equal(route_features(np.array([], int), np.array([]), 8, 3, 2), np.zeros(7))
