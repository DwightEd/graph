"""CPU optimizer and prefix-causality checks, not natural detection results."""
import pytest
import torch

from .smooth import (
    filter_prefix_field, huber_graph_objective, normalize_edge_weights, smooth_field,
)


def chain_edges(count):
    sources = torch.arange(count - 1)
    targets = sources + 1
    weights = torch.ones(count - 1, dtype=torch.float64)
    return sources, targets, weights


def test_zero_penalty_and_isolated_nodes_preserve_their_original_unaries():
    unary = torch.tensor([1., -2., 7.], dtype=torch.float64)
    sources = torch.tensor([0])
    targets = torch.tensor([1])
    weights = torch.tensor([4.], dtype=torch.float64)
    unpenalized = smooth_field(unary, sources, targets, weights, penalty=0)
    torch.testing.assert_close(unpenalized['solution'], unary, rtol=0, atol=0)
    assert unpenalized['max_actual_change'] == 0
    assert unpenalized['lambda_delta_bound'] == 0
    smoothed = smooth_field(unary, sources, targets, weights)
    assert smoothed['solution'][2] == unary[2]
    empty = torch.empty(0, dtype=torch.long)
    isolated = smooth_field(unary, empty, empty, torch.empty(0, dtype=torch.float64))
    torch.testing.assert_close(isolated['solution'], unary, rtol=0, atol=0)


def test_raw_degree_budget_bounds_every_normalized_incident_degree():
    sources = torch.tensor([0, 0, 0, 1, 2])
    targets = torch.tensor([1, 2, 3, 3, 3])
    raw = torch.tensor([2., 20., 200., 7., .03], dtype=torch.float64)
    normalized = normalize_edge_weights(sources, targets, raw, node_count=5)
    degrees = torch.zeros(5, dtype=torch.float64)
    degrees.index_add_(0, sources, normalized)
    degrees.index_add_(0, targets, normalized)
    assert (degrees <= 1 + 1e-15).all()
    assert degrees[4] == 0
    assert (normalized >= 0).all()
    raw_degree = torch.zeros(5, dtype=torch.float64)
    raw_degree.index_add_(0, sources, raw)
    raw_degree.index_add_(0, targets, raw)
    expected = raw / torch.maximum(raw_degree[sources], raw_degree[targets]).clamp_min(1.)
    torch.testing.assert_close(normalized, expected, rtol=0, atol=0)


def test_weak_single_edge_is_not_amplified_to_unit_weight():
    sources = torch.tensor([0])
    targets = torch.tensor([1])
    raw = torch.tensor([.001], dtype=torch.float64)
    normalized = normalize_edge_weights(sources, targets, raw, node_count=2)
    torch.testing.assert_close(normalized, raw, rtol=0, atol=0)
    unary = torch.tensor([0., 10.], dtype=torch.float64)
    result = smooth_field(unary, sources, targets, raw)
    assert result['max_actual_change'] <= .5 * .001 + 1e-12
    assert result['solution'][1] - result['solution'][0] >= 9.999 - 1e-12


@pytest.mark.parametrize('huber_delta', [1., .2])
def test_solution_is_stationary_and_respects_per_token_unary_change_bound(huber_delta):
    unary = torch.tensor([-.2, .3, -.1, 2., -.7], dtype=torch.float64)
    sources, targets, raw = chain_edges(len(unary))
    result = smooth_field(unary, sources, targets, raw, penalty=.5, huber_delta=huber_delta)
    weights = normalize_edge_weights(sources, targets, raw, len(unary))
    field = result['solution'].clone().requires_grad_(True)
    objective = huber_graph_objective(field, unary, sources, targets, weights,
                                      penalty=.5, huber_delta=huber_delta)
    independent_gradient = torch.autograd.grad(objective, field)[0]
    assert result['converged']
    assert result['max_gradient'] <= 1e-8
    assert independent_gradient.abs().max() <= 1e-8
    assert (field.detach() - unary).abs().max() <= .5 * huber_delta + 1e-12
    assert result['max_actual_change'] <= result['lambda_delta_bound'] + 1e-12
    assert result['objective'] == float(objective.detach())


def test_small_local_noise_is_denoised_without_equalizing_receivers():
    unary = torch.tensor([.1, -.1, .1, -.1], dtype=torch.float64)
    sources, targets, weights = chain_edges(len(unary))
    result = smooth_field(unary, sources, targets, weights)
    field = result['solution']
    assert field.square().sum() < unary.square().sum()
    assert (field[1:] - field[:-1]).square().sum() < (unary[1:] - unary[:-1]).square().sum()
    assert field.max() - field.min() > .01
    assert not torch.allclose(field, unary.mean().expand_as(unary))


def test_large_jump_is_retained_by_huber_gradient_cap():
    unary = torch.tensor([0., 0., 10., 10.], dtype=torch.float64)
    sources, targets, weights = chain_edges(len(unary))
    result = smooth_field(unary, sources, targets, weights)
    field = result['solution']
    assert field[2] - field[1] > 9
    assert (field - unary).abs().max() <= .5 + 1e-12
    assert not torch.allclose(field, unary.mean().expand_as(unary))


def test_topology_and_relative_weights_change_solution_without_span_broadcast():
    unary = torch.tensor([0., 1., 0., 1.], dtype=torch.float64)
    sources, targets, weights = chain_edges(len(unary))
    chain = smooth_field(unary, sources, targets, weights)['solution']
    star = smooth_field(unary, torch.tensor([0, 0, 0]), torch.tensor([1, 2, 3]), weights)['solution']
    weighted = smooth_field(unary, sources, targets,
                            torch.tensor([100., 1., 1.], dtype=torch.float64))['solution']
    assert not torch.allclose(chain, star)
    assert not torch.allclose(chain, weighted)
    for field in (chain, star, weighted):
        assert field.max() - field.min() > .1
        assert not torch.allclose(field, unary.mean().expand_as(unary))


def test_offline_future_can_change_early_tokens_but_prefix_output_cannot():
    unary = torch.tensor([0., 0., 10.], dtype=torch.float64)
    altered = torch.tensor([0., 0., -10.], dtype=torch.float64)
    sources, targets, weights = chain_edges(len(unary))
    offline = smooth_field(unary, sources, targets, weights)['solution']
    offline_altered = smooth_field(altered, sources, targets, weights)['solution']
    assert not torch.allclose(offline[:2], offline_altered[:2])
    online = filter_prefix_field(unary, sources, targets, weights)
    online_altered = filter_prefix_field(altered, sources, targets, weights)
    torch.testing.assert_close(online[:2], torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(online[:2], online_altered[:2], rtol=0, atol=0)
    assert online[-1] != online_altered[-1]
    assert online[0] != offline[0]


def test_unknown_future_weights_cannot_rescale_prefix_degrees_or_prior_outputs():
    unary = torch.tensor([0., 1., 10.], dtype=torch.float64)
    sources = torch.tensor([0, 1, 0])
    targets = torch.tensor([1, 2, 2])
    ordinary = torch.tensor([1., 1., 1.], dtype=torch.float64)
    future_changed = torch.tensor([1., 10000., 20000.], dtype=torch.float64)
    online = filter_prefix_field(unary, sources, targets, ordinary)
    changed = filter_prefix_field(unary, sources, targets, future_changed)
    torch.testing.assert_close(online[:2], changed[:2], rtol=0, atol=0)
    torch.testing.assert_close(online[1], torch.tensor(.75, dtype=torch.float64), rtol=0, atol=1e-8)
    # It also works when the caller has no future unaries, but knows future edge IDs.
    prefix_only = filter_prefix_field(unary[:2], sources, targets, future_changed)
    torch.testing.assert_close(prefix_only, online[:2], rtol=0, atol=0)
    full_budget = normalize_edge_weights(sources, targets, future_changed, len(unary))
    prefix_budget = normalize_edge_weights(sources[:1], targets[:1], future_changed[:1], 2)
    assert full_budget[0] < 1e-4
    assert prefix_budget[0] == 1


def test_failed_convergence_and_invalid_weight_sign_are_explicit():
    unary = torch.tensor([0., 10.], dtype=torch.float64)
    sources, targets, weights = chain_edges(len(unary))
    with pytest.raises(RuntimeError, match='did not converge after 0 iterations'):
        smooth_field(unary, sources, targets, weights, max_iterations=0)
    with pytest.raises(ValueError, match='nonnegative edge weights'):
        normalize_edge_weights(sources, targets, -weights, node_count=2)
    with pytest.raises(ValueError, match='strict-past sender'):
        filter_prefix_field(unary, targets, sources, weights)
