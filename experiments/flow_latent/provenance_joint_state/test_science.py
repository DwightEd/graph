"""Contracts affecting provenance, neighbor coupling, and EM interpretation."""
import numpy as np
import torch
from scipy.linalg import cholesky_banded
from transformers import LlamaConfig, LlamaForCausalLM

from .features import source_anchored_basis, matched_rewire
from .gaussian import (Parameters, covariance_block, fit, posterior,
                       residual_operator, selected_inverse)
from .measure import block_source_reads


def test_source_block_prevents_source_content_leaking_through_past_kv():
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32,
        intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64)).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = tokens.clone()
    changed[0, 1:3] = torch.tensor([21, 22])
    with torch.no_grad(), block_source_reads(model, [False, True, True, False], 6):
        first = model.model(tokens).last_hidden_state
        second = model.model(changed).last_hidden_state
    assert torch.allclose(first[:, 3:], second[:, 3:], atol=1e-7)
    with torch.no_grad():
        first = model.model(tokens).last_hidden_state
        second = model.model(changed).last_hidden_state
    assert not torch.allclose(first[:, 3:], second[:, 3:])


def test_selected_inverse_matches_dense_inverse_and_neighbor_update():
    generator = np.random.default_rng(42)
    observations = generator.normal(size=(7, 5))
    edges = generator.normal(size=(7, 3, 2)) * .1
    edges[0] = 0
    parameters = Parameters(np.array([[1, 0], [.2, .4], [.7, -.2], [0, .5], [-.3, .1]]),
        generator.normal(size=(2, 2, 2)) * .2, np.array([.3, .5]), np.ones(5), 1.)
    mean, covariance, _ = posterior(observations, edges, parameters)
    operator = residual_operator(edges, parameters.transition).toarray()
    loading = np.kron(np.eye(7), parameters.loading)
    precision = operator.T @ np.diag(np.tile(1 / parameters.innovation, 7)) @ operator + loading.T @ loading
    expected_covariance = np.linalg.inv(precision)
    expected_mean = expected_covariance @ loading.T @ observations.ravel()
    assert np.allclose(mean.ravel(), expected_mean, atol=1e-11)
    for target in range(7):
        for parent in range(max(0, target - 3), target + 1):
            assert np.allclose(covariance_block(covariance, target, parent, 2),
                expected_covariance[target * 2:target * 2 + 2, parent * 2:parent * 2 + 2], atol=1e-11)
    updated = observations.copy()
    updated[0, 0] += 5
    assert not np.allclose(posterior(updated, edges, parameters)[0][1], mean[1])


def test_joint_em_monotone_and_fixed_anchor():
    generator = np.random.default_rng(9)
    sequences = [(generator.normal(size=(20, 5)), generator.normal(size=(20, 3, 2)) * .1)
                 for _ in range(3)]
    loading = generator.normal(size=(5, 2)) * .2
    loading[0] = [1, 0]
    initial = Parameters(loading, np.zeros((2, 2, 2)), np.ones(2), np.ones(5), 1.)
    parameters, diagnostics = fit(sequences, initial, iterations=12)
    assert np.all(np.diff(diagnostics['objective']) >= -1e-7)
    assert np.array_equal(parameters.loading[0], [1, 0])
    assert np.isfinite(parameters.transition).all()


def test_weighted_evidence_em_and_sparse_anchor_mask_are_consistent():
    generator = np.random.default_rng(17)
    sequences = [(generator.normal(size=(16, 5)), generator.normal(size=(16, 3, 2)) * .1)]
    loading = generator.normal(size=(5, 2)) * .2
    loading[0] = [1, 0]
    weight = np.array([1, .25, .25, .25, .25])
    initial = Parameters(loading, np.zeros((2, 2, 2)), np.ones(2), np.ones(5), 1.,
                         observation_weight=weight)
    parameters, diagnostic = fit(sequences, initial, iterations=10)
    assert np.all(np.diff(diagnostic['objective']) >= -1e-7)
    observations, edges = sequences[0]
    hidden = np.arange(16) % 5 == 2
    first = posterior(observations, edges, parameters, hide_anchor=hidden)[0]
    changed = observations.copy()
    changed[hidden, 0] += 100
    second = posterior(changed, edges, parameters, hide_anchor=hidden)[0]
    assert np.allclose(first, second, atol=1e-10)


def test_rewire_preserves_lag_group_and_head_sums_and_future_mask():
    generator = np.random.default_rng(42)
    signed = generator.normal(size=(12, 8, 32, 32))
    mass = generator.random(signed.shape)
    for target in range(12):
        signed[target, target:] = 0
        mass[target, target:] = 0
    record = dict(edge_signed=signed, edge_mass=mass)
    shuffled, shuffled_mass, diagnostic = matched_rewire(record, 42)
    for lower, upper in ((0, 1), (1, 2), (2, 4), (4, 8)):
        assert np.allclose(signed[:, lower:upper].sum(1), shuffled[:, lower:upper].sum(1))
        assert np.allclose(mass[:, lower:upper].sum(1), shuffled_mass[:, lower:upper].sum(1))
    for target in range(12):
        assert not shuffled[target, target:].any()
    assert diagnostic['valid_null']


def test_source_edge_basis_uses_observed_gaps_and_preserves_small_head_direction():
    generator = np.random.default_rng(42)
    signed = generator.normal(size=(40, 8, 32, 32))
    gap = generator.normal(size=40)
    for target in range(40):
        signed[target, target:] = 0
        for lag in range(1, min(target, 8) + 1):
            signed[target, lag - 1, 0, 0] = .001 * gap[target - lag]
    record = dict(source_gap=gap, edge_signed=signed)
    flat = signed.reshape(-1, 1024)
    basis = source_anchored_basis([record], flat, 0, 1)
    donor_prediction = (flat @ basis[:, 0]).reshape(40, 8)
    valid = np.arange(40)[:, None] >= np.arange(1, 9)
    actual = gap[(np.arange(40)[:, None] - np.arange(1, 9)).clip(min=0)]
    assert np.corrcoef(actual[valid], donor_prediction[valid])[0, 1] > .99
