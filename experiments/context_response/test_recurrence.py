import numpy as np
from .recurrence import propagate, similarities


def test_paths_stop_at_bottleneck_and_do_not_invent_seeds():
    seed = np.array([.99, .1, .1, .1, .1])
    result, origin = propagate(seed, [np.array([.98, .97, 0., .99])], steps=3)
    np.testing.assert_allclose(result, [.99, .98, .97, .1, .1])
    np.testing.assert_array_equal(origin[:3], [0, 0, 0])
    unchanged, _ = propagate(seed, [np.zeros(4)])
    np.testing.assert_array_equal(unchanged, seed)


def test_causal_output_does_not_read_future_seed():
    seed = np.array([.1, .1, .99])
    causal, _ = propagate(seed, [np.ones(2)], causal=True)
    offline, _ = propagate(seed, [np.ones(2)], causal=False)
    np.testing.assert_array_equal(causal, seed)
    np.testing.assert_allclose(offline, [.99, .99, .99])


def test_head_identity_control_keeps_norm_but_breaks_pattern():
    z = np.zeros((12, 1024, 4))
    z[:, 19, 0] = 1.
    np.testing.assert_array_equal(similarities(z)[0], np.ones(11))
    assert similarities(z, shuffle=True)[0].sum()<2


def test_barrier_cannot_be_jumped_and_cycles_cannot_duplicate_seeds():
    from .recurrence_barrier import cut_edges, corroborate
    edges = [np.ones(3), np.ones(2)]
    cut = cut_edges(edges, np.array([.9, .01, .9]))
    np.testing.assert_array_equal(cut[0], [1, 0, 1])
    np.testing.assert_array_equal(cut[1], [0, 0])
    result, _ = corroborate(np.array([.99, .1, .1, .1]), edges)
    np.testing.assert_allclose(result, [.99, .1, .1, .1])
    result, _ = corroborate(np.array([.99, .98, .1, .1]), cut)
    np.testing.assert_allclose(result, [.99, .98, .1, .1])
