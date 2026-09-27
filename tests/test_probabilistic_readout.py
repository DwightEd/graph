"""Quadratic structure must represent conditional dependence, not label leakage."""

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.probabilistic_detection.model import ConditionalGaussian
from experiments.probabilistic_detection.readout import QuadraticReadout


def test_interaction_readout_recovers_zero_marginal_signal_on_held_out_rows():
    rng = np.random.default_rng(731)
    context = rng.normal(size=(5000, 6))
    observations = rng.normal(size=(5000, 11))
    labels = (observations[:, 0] * observations[:, 1] > 0).astype(int)
    train, test = slice(0, 3500), slice(3500, None)
    weights = np.ones(3500)
    transform = ConditionalGaussian().fit(context[train], observations[train], labels[train])
    scores = {}
    for design in ("linear", "squares", "interactions"):
        model = QuadraticReadout(transform, design)
        model.fit_matrix(model.matrix(context[train], observations[train]), labels[train], weights)
        scores[design] = roc_auc_score(labels[test], model.decision_function(context[test], observations[test]))
    assert scores["interactions"] > .97
    assert scores["linear"] < .6 and scores["squares"] < .6


def test_prediction_batch_composition_does_not_change_readout():
    rng = np.random.default_rng(41)
    context, observations = rng.normal(size=(300, 6)), rng.normal(size=(300, 11))
    labels = (observations[:, 0] > 0).astype(int)
    transform = ConditionalGaussian().fit(context, observations, labels)
    model = QuadraticReadout(transform, "conditioned")
    model.fit_matrix(model.matrix(context, observations), labels, np.ones(300))
    together = model.decision_function(context[:10], observations[:10])
    separately = [model.decision_function(context[i:i+1], observations[i:i+1])[0] for i in range(10)]
    np.testing.assert_allclose(together, separately, atol=1e-12)


def test_no_position_design_ignores_all_explicit_position_and_length_columns():
    rng = np.random.default_rng(62)
    context, observations = rng.normal(size=(100, 6)), rng.normal(size=(100, 11))
    transform = ConditionalGaussian().fit(context, observations, (observations[:, 0] > 0).astype(int))
    model = QuadraticReadout(transform, ablation="no_position")
    changed = context.copy()
    changed[:, 2:] += 100
    np.testing.assert_array_equal(model.matrix(context, observations), model.matrix(changed, observations))
