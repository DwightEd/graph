import numpy as np

from experiments.evidence_route_state.detector import StickyRouteHMM


def known_hmm() -> StickyRouteHMM:
    """A named three-state model with enough temporal coupling to expose smoothing."""

    model = StickyRouteHMM()
    model.initial_ = np.array([0.8, 0.15, 0.05])
    model.transition_ = np.array(
        [
            [0.90, 0.09, 0.01],
            [0.05, 0.90, 0.05],
            [0.01, 0.09, 0.90],
        ]
    )
    model.means_ = np.array(
        [
            [0.10, 0.20],  # exploration: broad route
            [0.85, 0.10],  # grounded focus: narrow but evidence-rooted
            [0.85, 0.90],  # captured: narrow and unrooted
        ]
    )
    model.variances_ = np.full((3, 2), 0.04)
    return model


def test_filtered_posterior_never_reads_future_tokens():
    model = known_hmm()
    common_prefix = np.array([[0.1, 0.2], [0.8, 0.2], [0.9, 0.5]])
    grounded_future = np.vstack([common_prefix, [[0.9, 0.0], [0.8, 0.1]]])
    captured_future = np.vstack([common_prefix, [[0.9, 1.0], [0.8, 0.9]]])

    grounded_filtered = model.filtered_posterior(grounded_future)
    captured_filtered = model.filtered_posterior(captured_future)

    np.testing.assert_allclose(
        grounded_filtered[: len(common_prefix)],
        captured_filtered[: len(common_prefix)],
        atol=1e-12,
    )

    # Smoothing is deliberately different: this makes the prefix assertion a
    # real online-filtering test rather than an insensitive fixture.
    grounded_smoothed = model.smoothed_posterior(grounded_future)
    captured_smoothed = model.smoothed_posterior(captured_future)
    assert not np.allclose(
        grounded_smoothed[: len(common_prefix)],
        captured_smoothed[: len(common_prefix)],
    )


def test_independent_control_uses_only_the_current_observation():
    model = known_hmm()
    shared = np.array([0.85, 0.55])
    first = np.array([[0.1, 0.2], shared, [0.9, 0.9]])
    second = np.array([[0.9, 0.9], shared, [0.1, 0.2]])

    np.testing.assert_allclose(
        model.independent_posterior(first)[1],
        model.independent_posterior(second)[1],
        atol=1e-12,
    )
    assert not np.allclose(
        model.filtered_posterior(first)[1],
        model.filtered_posterior(second)[1],
    )


def test_expected_dwell_time_is_fixed_by_self_transition():
    model = known_hmm()
    np.testing.assert_allclose(model.expected_dwell_time(), 10.0)


def test_captured_score_requires_contraction_and_unrooted_takeover():
    model = known_hmm()
    observations = np.array(
        [
            [0.10, 0.90],  # takeover without contraction: exploration regime
            [0.90, 0.10],  # legitimate narrow focus
            [0.90, 0.90],  # narrow, unrooted route
        ]
    )
    scores = model.score(observations)

    assert scores[2] > scores[1]
    assert scores[2] > scores[0]


def test_fitted_state_names_are_fixed_by_route_geometry():
    sequences = []
    rng = np.random.default_rng(7)
    centers = np.array([[0.1, 0.2], [0.85, 0.1], [0.85, 0.9]])
    for center in centers:
        sequences.append(center + rng.normal(scale=0.025, size=(30, 2)))

    model = StickyRouteHMM(n_iter=40).fit(sequences)

    assert model.means_[model.exploration_state, 0] == np.min(model.means_[:, 0])
    remaining = [model.grounded_state, model.captured_state]
    assert model.means_[model.captured_state, 1] == np.max(model.means_[remaining, 1])
