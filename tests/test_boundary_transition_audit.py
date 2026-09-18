import numpy as np
import pandas as pd

from experiments.charm_structure_audit.boundary_transition_audit import (
    sentence_start_flags,
    transition_labels,
    robust_reference,
    outlier_score,
    routing_entropy,
)


def test_surface_sentence_start_never_uses_gold():
    table = pd.DataFrame(dict(
        id=["a", "a", "a", "a", "b", "b"],
        text=["Hello", ".", " Next", " token", "One", "\nTwo"],
    ))
    flags = sentence_start_flags(table)
    assert flags.tolist() == [True, False, True, False, True, True]


def test_transition_labels_are_exact():
    table = pd.DataFrame(dict(
        previous_gold=[-1, 0, 0, 1, 1],
        gold=[0, 0, 1, 1, 0],
    ))
    assert transition_labels(table).tolist() == [
        "first", "normal", "onset", "continuation", "recovery"
    ]


def test_boundary_conditioned_reference_scores_own_center_low():
    features = np.array([
        [0., 0.], [1., 1.], [10., 10.], [11., 11.]
    ])
    boundary = np.array([False, False, True, True])
    refs = robust_reference(features, boundary)
    scores = outlier_score(features, boundary, refs)
    assert np.isfinite(scores).all()
    assert scores.max() < 2


def test_routing_entropy_is_high_for_uniform_heads():
    uniform = np.ones((1, 8))
    peaked = np.array([[1., 0., 0., 0., 1., 0., 0., 0.]])
    assert routing_entropy(uniform, heads=4)[0] > routing_entropy(peaked, heads=4)[0]
