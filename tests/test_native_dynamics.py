"""Mathematical checks, not natural hallucination-detection experiments."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from state_audit.analysis.ffn_transport import ffn_transport

from experiments.native_support.dynamics_core import (
    effect_source_moments,
    evidence_reads,
    future_attention,
    mode_posteriors,
    observe_state,
    predict_state,
    source_moments,
    transition_emissions,
)


def test_remote_reads_include_both_reanchor_peak_and_sustained_focus():
    attention = np.zeros((1, 2, 3, 20))
    attention[0, 0, :, 0] = [.05, .9, .9]
    attention[0, 1, :, 1] = [.8, .8, .8]
    masks = np.eye(20)[:2]
    reads = evidence_reads(attention, masks)
    np.testing.assert_allclose(reads["read_mass"][0, 0, :, 0], [.05, .9, .9])
    np.testing.assert_allclose(reads["read_change"][0, 0, 1:, 0], [.85, 0])
    assert reads["read_mass"][0, 1, 2, 1] == .8
    assert np.isnan(reads["read_change"][..., 0, :]).all()


def test_future_attention_preserves_heads_and_marks_end_censoring():
    attention = np.array([[[[1., 0., 0.], [.3, .7, 0.], [.1, .4, .5]]]])
    result = future_attention(attention)
    np.testing.assert_allclose(result["future_mean_attention"][0, 0, :2], [.2, .4])
    np.testing.assert_equal(result["future_query_count"], [2, 1, 0])
    assert np.isnan(result["future_mean_attention"][0, 0, -1])


def test_source_direction_moments_do_not_double_weight_message_magnitude():
    effects = np.array([[[3., 0.], [0., 1.]]])
    result = effect_source_moments(effects)
    np.testing.assert_allclose(result["effect_scale"][:, None] * result["mean"], effects.sum(1))
    np.testing.assert_allclose(result["net_effect"], effects.sum(1))


def test_head_specialization_is_not_within_head_uncertainty():
    effects = np.zeros((2, 2, 3))
    specialized = source_moments(np.eye(2), effects)
    diffuse = source_moments(np.ones((2, 2)), effects)
    assert specialized["within_entropy"] == 0
    assert specialized["between_head_js"] == pytest.approx(np.log(2))
    assert diffuse["within_entropy"] == pytest.approx(np.log(2))
    assert diffuse["between_head_js"] == 0


def test_equal_entropy_does_not_imply_equal_effect_disagreement():
    mass = np.ones((1, 2))
    agreeing = source_moments(mass, np.array([[[1., 0.], [1., 0.]]]))
    opposing = source_moments(mass, np.array([[[1., 0.], [-1., 0.]]]))
    np.testing.assert_equal(agreeing["entropy"], opposing["entropy"])
    np.testing.assert_equal(agreeing["covariance"], np.zeros((1, 2, 2)))
    assert opposing["covariance"][0, 0, 0] == 1


def test_absent_source_observation_is_missing_not_certain():
    result = source_moments(np.zeros((2, 3)), np.ones((2, 3, 4)))
    assert not result["valid"].any()
    assert np.isnan(result["mean"]).all()
    assert np.isnan(result["entropy"]).all()
    assert np.isnan(result["within_entropy"])


def test_input_uncertainty_and_gaussian_update_match_joint_conditioning():
    mean, covariance = predict_state(
        np.zeros(2), np.eye(2), np.eye(2), np.eye(2), np.array([1., 2.]),
        np.diag([2., 0.]), np.eye(2) * .1, np.zeros(2),
    )
    np.testing.assert_allclose(covariance, np.diag([3.1, 1.1]))
    readout, noise, value = np.array([[1., 2.]]), np.array([[.3]]), np.array([4.])
    updated, uncertainty, log_density = observe_state(mean, covariance, value, readout, noise)
    marginal = readout @ covariance @ readout.T + noise
    conditional = covariance @ readout.T @ np.linalg.inv(marginal)
    np.testing.assert_allclose(updated, mean + conditional @ (value - readout @ mean))
    np.testing.assert_allclose(uncertainty, covariance - conditional @ readout @ covariance)
    assert log_density == pytest.approx(multivariate_normal.logpdf(value, readout @ mean, marginal))


def test_smoothing_matches_enumeration_and_legitimately_uses_future():
    emission = np.array([[0., 0.], [0., 2.], [0., 3.], [0., 1.]])
    transition = np.array([[.9, .1], [.1, .9]])
    initial = np.array([.5, .5])
    result = mode_posteriors(emission, transition, initial)
    sequences = list(product(range(2), repeat=len(emission)))
    logs = []
    for sequence in sequences:
        log_joint = np.log(initial[sequence[0]])
        for position, state in enumerate(sequence):
            log_joint += emission[position, state]
            if position:
                log_joint += np.log(transition[sequence[position - 1], state])
        logs.append(log_joint)
    exact = np.zeros_like(emission)
    for sequence, weight in zip(sequences, np.exp(logs - logsumexp(logs))):
        for position, state in enumerate(sequence):
            exact[position, state] += weight
    np.testing.assert_allclose(result["smoothed"], exact)
    assert result["log_likelihood"] == pytest.approx(logsumexp(logs))
    assert result["smoothed"][0, 1] > result["filtered"][0, 1]
    changed = emission.copy()
    changed[2:] = changed[2:, ::-1]
    after = mode_posteriors(changed, transition, initial)
    np.testing.assert_equal(after["filtered"][:2], result["filtered"][:2])
    assert not np.allclose(after["smoothed"][:2], result["smoothed"][:2])


def test_persistent_mode_can_exit_without_hard_span_boundary():
    emission = np.array([[0., 5.]] * 6 + [[5., 0.]] * 6)
    result = mode_posteriors(emission, np.array([[.9, .1], [.1, .9]]), np.array([.5, .5]))
    assert (result["smoothed"][:5, 1] > .9).all()
    assert (result["smoothed"][7:, 1] < .1).all()


def test_source_update_and_history_persistence_have_distinct_likelihoods():
    history = np.ones((10, 2))
    controls = np.zeros((10, 2))
    controls[0] = [1., -1.]
    controls[-1] = [-1., 1.]
    observed = history.copy()
    observed[-1] += controls[-1]
    emission = transition_emissions(
        observed, history, controls, np.zeros((10, 2, 2)),
        np.array([np.eye(2), np.eye(2)]), np.array([np.eye(2), np.zeros((2, 2))]),
        np.zeros((2, 2)), np.array([np.eye(2), np.eye(2)]) * .02,
    )
    assert emission[0, 1] > emission[0, 0]
    np.testing.assert_allclose(emission[1:-1, 0], emission[1:-1, 1])
    assert emission[-1, 0] > emission[-1, 1]
    result = mode_posteriors(emission, np.array([[.98, .02], [.02, .98]]), np.array([.5, .5]))
    assert result["smoothed"][0, 1] > .99
    assert result["filtered"][3, 1] > .9
    assert result["smoothed"][-1, 1] < .01


def test_ffn_tangent_matches_autograd_and_small_directional_difference():
    random = torch.Generator().manual_seed(17)
    def linear(inputs, outputs):
        layer = torch.nn.Linear(inputs, outputs, bias=False, dtype=torch.float64)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(outputs, inputs, generator=random) / inputs**.5)
        return layer

    block = SimpleNamespace(
        post_attention_layernorm=SimpleNamespace(
            weight=torch.linspace(.5, 1.5, 5, dtype=torch.float64), variance_epsilon=1e-6,
        ),
        mlp=SimpleNamespace(gate_proj=linear(5, 9), up_proj=linear(5, 9), down_proj=linear(9, 5)),
    )
    residual = torch.randn(5, dtype=torch.float64, generator=random)
    directions = torch.randn(3, 5, dtype=torch.float64, generator=random)
    def native(value):
        normalized = block.post_attention_layernorm.weight * value / (value.square().mean() + 1e-6).sqrt()
        return value + block.mlp.down_proj(
            torch.nn.functional.silu(block.mlp.gate_proj(normalized)) * block.mlp.up_proj(normalized)
        )

    output, tangents = ffn_transport(block, residual, directions)
    jacobian = torch.autograd.functional.jacobian(native, residual)
    torch.testing.assert_close(output, native(residual))
    torch.testing.assert_close(tangents, directions @ jacobian.T)
    step = 1e-5
    for direction, tangent in zip(directions, tangents):
        difference = (native(residual + step * direction) - native(residual - step * direction)) / (2 * step)
        torch.testing.assert_close(tangent, difference, atol=1e-8, rtol=1e-7)


def test_reverse_write_can_preserve_decision_information_by_rotation():
    evidence = np.array([1., 0.])
    transform = np.array([[0., -1.], [1., 0.]])
    ff_write = transform @ evidence - evidence
    output_direction = np.array([0., 1.])
    assert ff_write @ evidence < 0
    assert output_direction @ transform @ evidence == 1


def test_input_to_output_transfer_is_invariant_to_latent_coordinates():
    transition = np.array([[.9, .2], [.1, .8]])
    injection = np.array([[1.], [.3]])
    readout = np.array([[2., -1.]])
    coordinates = np.array([[2., .5], [.2, 1.]])
    inverse = np.linalg.inv(coordinates)
    expected = readout @ transition @ transition @ injection
    changed = readout @ inverse @ np.linalg.matrix_power(coordinates @ transition @ inverse, 2) @ coordinates @ injection
    np.testing.assert_allclose(changed, expected)
