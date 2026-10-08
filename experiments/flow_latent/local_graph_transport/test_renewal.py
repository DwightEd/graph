import numpy as np
import torch

from .renewal import crossing_entropy_gate, entropy_events, gated_graph, renewal_field
from .smooth import normalize_edge_weights, smooth_field
from .unlabeled import local_graph_edges


def test_entropy_events_are_fixed_upper_half_cdf():
    np.testing.assert_allclose(entropy_events([0., .25, .5, .75, 1.]), [0., 0., 0., .5, 1.])


def test_zero_event_preserves_original_solver():
    weights = np.array([[0., 0.], [.3, 0.], [.2, .4], [.1, .2]])
    unary = np.array([.1, .7, .2, .9])
    source, target, raw = local_graph_edges(weights)
    original = smooth_field(torch.from_numpy(unary), source, target, raw)['solution'].numpy()
    renewed, _ = renewal_field(unary, weights, np.zeros(4))
    np.testing.assert_allclose(renewed, original, atol=1e-12)


def test_crossing_edges_cannot_skip_a_barrier():
    # Event at token2 attenuates every edge from before2 to2 or beyond2.
    source = np.array([0, 1, 0, 1, 2, 2])
    target = np.array([2, 2, 4, 4, 3, 4])
    gates = crossing_entropy_gate([0., 0., 1., 0., 0.], source, target)
    np.testing.assert_allclose(gates[:4], np.exp(-4.))
    np.testing.assert_array_equal(gates[4:], np.ones(2))


def test_gate_never_reads_a_future_event():
    source = np.array([0, 0, 1])
    target = np.array([1, 2, 2])
    first = crossing_entropy_gate([0., .3, .5, 0.], source, target)
    later = crossing_entropy_gate([0., .3, .5, 1.], source, target)
    np.testing.assert_array_equal(first, later)


def test_gates_preserve_incident_budget_without_weak_edge_amplification():
    weights = np.array([[0., 0.], [.01, 0.], [3., .02], [2., 4.]])
    source, target, raw = local_graph_edges(weights)
    original = normalize_edge_weights(source, target, raw, 4)
    source, target, gated = gated_graph(weights, [.1, .9, .5, .8])
    assert torch.all(gated <= original)
    assert torch.all(gated <= raw)
    np.testing.assert_array_equal(normalize_edge_weights(source, target, gated, 4).numpy(), gated.numpy())
    degree = torch.zeros(4, dtype=torch.float64)
    degree.index_add_(0, source, gated)
    degree.index_add_(0, target, gated)
    assert float(degree.max()) <= 1.


def test_single_token_empty_graph_keeps_unary():
    result, diagnostics = renewal_field(np.array([.7]), np.zeros((1, 8)), [.8])
    np.testing.assert_array_equal(result, [.7])
    assert diagnostics['max_gradient'] == 0.


def test_gated_solution_keeps_the_same_per_token_modification_bound():
    weights = np.array([[0., 0.], [.8, 0.], [.4, .4], [.2, .3]])
    unary = np.array([-10., 2., 20., -4.])
    result, diagnostics = renewal_field(unary, weights, [.1, 0., .8, .2])
    assert diagnostics['max_gradient'] <= 1e-8
    assert np.max(np.abs(result - unary)) <= .5 + 1e-8


def test_bounded_unary_matches_quadratic_laplacian_average():
    weights = np.array([[0., 0.], [.8, 0.], [.4, .4], [.2, .3]])
    unary = np.array([.1, .9, .8, .3])
    events = [.1, 0., .8, .2]
    source, target, gated = gated_graph(weights, events)
    laplacian = torch.zeros((4, 4), dtype=torch.float64)
    for sender, receiver, weight in zip(source, target, gated):
        laplacian[sender, sender] += weight
        laplacian[receiver, receiver] += weight
        laplacian[sender, receiver] -= weight
        laplacian[receiver, sender] -= weight
    linear_solution = torch.linalg.solve(torch.eye(4, dtype=torch.float64) + .5 * laplacian,
                                        torch.from_numpy(unary)).numpy()
    actual, _ = renewal_field(unary, weights, events)
    np.testing.assert_allclose(actual, linear_solution, atol=1e-8, rtol=0.)
    assert actual.min() >= unary.min() - 1e-8
    assert actual.max() <= unary.max() + 1e-8
