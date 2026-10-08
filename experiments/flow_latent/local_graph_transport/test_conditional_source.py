import numpy as np

from .conditional_source import conditional_rank, fit_conditional_reference


def test_conditioning_removes_attention_shift_and_preserves_route_excess():
    attention = np.repeat(np.arange(16), 3)
    route = attention * 10. + np.tile([0., 1., 2.], 16)
    reference = fit_conditional_reference(route, attention, np.arange(48))
    rank = conditional_rank(reference, route, attention).reshape(16, 3)
    np.testing.assert_allclose(rank, np.tile([1/6, .5, 5/6], (16, 1)))


def test_duplicate_attention_edges_do_not_create_empty_bins():
    attention = np.array([0., 0., 1., 1.])
    route = np.array([0., 1., 10., 11.])
    reference = fit_conditional_reference(route, attention, np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(conditional_rank(reference, route, attention), [.25, .75, .25, .75])
