"""Scientific checks: exact inference, source weighting and state identification."""

from itertools import product

import numpy as np
from scipy.special import logsumexp

from .sequence import (SequenceModel, anchored_means, batch_sequences, constrained_means, fit_sequence,
                       forward_backward, lagged_inputs, ordering, viterbi)
from .sequence_data import answer_weights, features, fit_reference, select_reference, transform_answer


def exhaustive(emission, transition, initial):
    paths = list(product(range(len(initial)), repeat=len(emission)))
    scores = np.array([np.log(initial[path[0]]) + sum(emission[t, state] for t, state in enumerate(path))
                       + sum(transition[t, path[t - 1], path[t]] for t in range(1, len(path))) for path in paths])
    probability = np.exp(scores - logsumexp(scores))
    gamma = np.zeros_like(emission)
    xi = np.zeros_like(transition)
    for path, weight in zip(paths, probability):
        for time, state in enumerate(path):
            gamma[time, state] += weight
            if time:
                xi[time, path[time - 1], state] += weight
    return gamma, xi, logsumexp(scores), paths[scores.argmax()], scores.max()


def test_variable_transition_inference_and_independent_answer_ends():
    rng = np.random.default_rng(73)
    for states in (2, 4):
        emission = rng.normal(size=(3, 5, states))
        logits = rng.normal(size=(3, 5, states, states))
        transition = logits - logsumexp(logits, axis=-1, keepdims=True)
        initial = rng.dirichlet(np.ones(states))
        valid = np.arange(5)[None] < np.array([1, 3, 5])[:, None]
        gamma, xi, likelihood = forward_backward(emission, transition, initial, valid)
        for answer, length in enumerate((1, 3, 5)):
            expected = exhaustive(emission[answer, :length], transition[answer, :length], initial)
            np.testing.assert_allclose(gamma[answer, :length], expected[0], atol=1e-12)
            np.testing.assert_allclose(xi[answer, :length], expected[1], atol=1e-12)
            np.testing.assert_allclose(likelihood[answer], expected[2], atol=1e-12)
            path, score = viterbi(emission[answer, :length], transition[answer, :length], initial)
            np.testing.assert_array_equal(path, expected[3])
            np.testing.assert_allclose(score, expected[4], atol=1e-12)
            assert not gamma[answer, length:].any()
            assert not xi[answer, length:].any()


def test_joint_mahalanobis_projection_respects_both_state_axes():
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(7, 7))
    precision = np.linalg.inv(matrix.T @ matrix + np.eye(7))
    target = rng.normal(size=(4, 7))
    mass = np.array([.01, .4, .1, .49])
    means = constrained_means(target, mass, precision)
    assert np.min(ordering(4, 7) @ means.ravel()) >= .2 - 1e-8
    feasible = means.copy()
    assert np.sum(mass[:, None] * ((means - target) @ precision) * (means - target)) >= 0
    np.testing.assert_allclose(constrained_means(feasible, mass, precision), feasible, atol=1e-8)


def test_lagged_inputs_reset_and_never_use_current_observation():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(9, 7))
    previous = lagged_inputs(values)
    modified = values.copy()
    modified[5:] += 99
    np.testing.assert_array_equal(previous[:6], lagged_inputs(modified)[:6])
    batch = batch_sequences([values[:4], values[4:]])
    np.testing.assert_array_equal(batch[1][:, 0], [[1, 0, 0, 0, 0]] * 2)


def fake_answer(values, source='a', generator='g'):
    return dict(values=values, record=dict(tokens=len(values), source_id=source, generator=generator),
                pack=dict(target=np.arange(len(values))))


def test_source_balancing_and_fitted_nuisance_reference():
    rng = np.random.default_rng(42)
    answers = [fake_answer(rng.normal(size=(length, 7)), source)
               for length, source in ((12, 'a'), (27, 'a'), (40, 'b'))]
    weights = answer_weights(answers)
    np.testing.assert_allclose(weights * [12, 27, 40], [.25, .25, .5])
    reference = fit_reference(answers)
    residual = [transform_answer(row, reference) for row in answers]
    mean = sum(weight * value.sum(axis=0) for weight, value in zip(weights, residual))
    np.testing.assert_allclose(mean, 0., atol=1e-9)


def test_reference_excludes_entire_control_source_without_loading_gold():
    context, observed = np.zeros((12, 6)), np.zeros((12, 11))
    pack = dict(context=context, observations=observed, target=np.arange(12), token_id=np.arange(12),
                unit_index=np.zeros(12, dtype=int))
    rows = [dict(id=str(i), source_id=str(i // 2), partition='fit' if i < 6 else 'dev',
                 packed_start=i, packed_stop=i + 1, tokens=1) for i in range(12)]
    train, dev = select_reference(pack, dict(records=rows, source_cache='/tmp/cache'), {'0', '3'}, 2, 2)
    assert {row['record']['source_id'] for row in train} == {'1', '2'}
    assert {row['record']['source_id'] for row in dev} == {'4', '5'}


def test_feature_recovery_keeps_original_full_local_contrasts():
    rng = np.random.default_rng(42)
    context, observed = rng.normal(size=(10, 6)), rng.normal(size=(10, 11))
    values = features(dict(context=context, observations=observed))
    np.testing.assert_array_equal(values[:, 0], context[:, 1] + observed[:, 1])
    np.testing.assert_array_equal(values[:, 1], context[:, 0] + observed[:, 0])
    np.testing.assert_array_equal(values[:, 2:], observed[:, 2:7])


def test_unit_support_uses_fixed_source_observations_without_changing_other_channels():
    rng = np.random.default_rng(83)
    pack = dict(context=rng.normal(size=(10, 6)), observations=rng.normal(size=(10, 11)))
    unit = features(pack, unit_support=True)
    raw = features(pack)
    np.testing.assert_array_equal(unit[:, :2], pack['context'][:, [1, 0]])
    np.testing.assert_array_equal(unit[:, 2:], raw[:, 2:])


def test_constrained_fit_increases_objective_and_returns_complete_scores():
    rng = np.random.default_rng(73)
    sequences = [rng.normal(size=(length, 7)) for length in (17, 21, 28)]
    weights = answer_weights([fake_answer(value, str(i)) for i, value in enumerate(sequences)])
    model, history = fit_sequence(sequences, sequences, weights, weights, iterations=3, transition_steps=30)
    assert np.min(np.diff([row['objective'] for row in history])) >= -1e-7
    scored = model.score(sequences[0])
    np.testing.assert_allclose(scored['posterior'].sum(axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(scored['risk'], scored['posterior'][:, [1, 3]].sum(axis=1))
    assert np.isfinite(scored['risk']).all()
    assert scored['enter'][0] == scored['exit'][0] == 0


def test_physical_mean_order_does_not_guarantee_source_evidence_direction():
    covariance = np.eye(7)
    covariance[0, 6] = covariance[6, 0] = .8
    precision = np.linalg.inv(covariance)
    target = np.zeros((4, 7))
    target[[1, 3], 0] = .5
    target[[1, 3], 6] = 3.
    target[[2, 3], 4] = .5
    means = constrained_means(target, np.ones(4), precision)
    axis = np.array([.5, .5, 0, 0, 0, 0, 0])
    assert np.min(ordering(4, 7) @ means.ravel()) >= .2
    assert (means[1] - means[0]) @ precision @ axis < 0


def test_anchoring_preserves_likelihood_direction_for_arbitrary_full_covariance():
    rng = np.random.default_rng(83)
    for states in (2, 4):
        matrix = rng.normal(size=(7, 7))
        covariance = matrix.T @ matrix + .5 * np.eye(7)
        precision = np.linalg.inv(covariance)
        target = rng.normal(size=(states, 7))
        means = anchored_means(target, rng.dirichlet(np.ones(states)), precision)
        assert np.min(ordering(states, 7) @ means.ravel()) >= .2 - 1e-8
        for strong in range(0, states, 2):
            direction = (means[strong + 1] - means[strong]) @ precision
            assert direction[0] > 0
            np.testing.assert_allclose(direction[0], direction[1], atol=1e-8)
            np.testing.assert_allclose(direction[2:], 0., atol=1e-8)


def test_anchored_emission_risk_monotonic_with_fixed_neighbour_messages():
    rng = np.random.default_rng(83)
    matrix = rng.normal(size=(7, 7))
    covariance = matrix.T @ matrix + np.eye(7)
    means = anchored_means(rng.normal(size=(4, 7)), np.ones(4), np.linalg.inv(covariance))
    model = SequenceModel(means, covariance, np.zeros((4, 4, 5)), rng.dirichlet(np.ones(4)))
    values = rng.normal(size=(13, 7))
    emission, transition = model.potentials(values, lagged_inputs(values))
    valid = np.ones((1, len(values)), dtype=bool)
    before = forward_backward(emission[None], transition[None], model.initial, valid)[0]
    modified = values.copy()
    modified[5, :2] += .3
    changed = model.potentials(modified, lagged_inputs(values))[0]
    after = forward_backward(changed[None], transition[None], model.initial, valid)[0]
    assert after[0, 5, [1, 3]].sum() > before[0, 5, [1, 3]].sum()


def test_anchored_fit_optimizes_same_objective_within_restricted_family():
    rng = np.random.default_rng(83)
    sequences = [rng.normal(size=(length, 7)) for length in (17, 21, 28)]
    weights = answer_weights([fake_answer(value, str(i)) for i, value in enumerate(sequences)])
    _, history = fit_sequence(sequences, sequences, weights, weights, iterations=3, anchored=True)
    assert np.min(np.diff([row['objective'] for row in history])) >= -1e-7
