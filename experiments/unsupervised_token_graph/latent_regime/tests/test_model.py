from types import SimpleNamespace

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.unsupervised_token_graph.latent_regime.model import (
    apply_standardization,
    filtered_score,
    fit_best,
    orient_rare_state,
    RegimeModel,
    standardize,
)


def test_standardize_respects_sequence_weights():
    sequences = [
        np.zeros((10, 1, 1), dtype=np.float32),
        np.full((2, 1, 1), 10., dtype=np.float32),
    ]
    weights = np.array([.1, .5])
    center, _ = standardize(sequences, weights)
    assert np.isclose(center.item(), 5.)


def test_orient_rare_state_moves_lower_occupancy_to_state_one():
    model = RegimeModel(
        means=np.array([[[1.]], [[2.]]]),
        covariance=np.array([[[1.]]]),
        precision=np.array([[[1.]]]),
        logdet=np.array([0.]),
        initial=np.array([.2, .8]),
        transition=np.array([[.8, .2], [.1, .9]]),
        occupancy=np.array([.2, .8]),
        log_likelihood=0.,
    )
    oriented = orient_rare_state(model)
    assert oriented.occupancy[1] == .2
    assert oriented.means[1, 0, 0] == 1.


def test_sticky_hmm_recovers_synthetic_persistent_rare_state():
    random = np.random.default_rng(4)
    sequences = []
    labels = []

    transition = np.array([[.98, .02], [.12, .88]])
    for _ in range(24):
        state = 0
        states = []
        values = []
        for _ in range(50):
            state = random.choice(2, p=transition[state])
            states.append(state)
            mean = np.array([0., 0., 0., 0.])
            if state:
                mean = np.array([2.5, -2.0, 2.0, -2.5])
            values.append(
                random.normal(mean, .45).reshape(1, 4)
            )
        sequences.append(np.asarray(values, dtype=np.float32))
        labels.append(np.asarray(states))

    weights = np.full(len(sequences), 1 / 50)
    center, scale = standardize(sequences, weights)
    normalized = [
        apply_standardization(values, center, scale)
        for values in sequences
    ]
    args = SimpleNamespace(
        seed=7,
        starts=1,
        iterations=8,
        ridge=1e-2,
        sticky_prior=2.,
        tolerance=1e-4,
    )
    model = fit_best(normalized, weights, args)

    truth = np.concatenate(labels)
    score = np.concatenate([
        filtered_score(values, model)
        for values in normalized
    ])
    assert roc_auc_score(truth, score) > .9
