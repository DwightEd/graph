"""Controls isolate cross-head products without changing marginals or using future tokens."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.unsupervised_token_graph.fixed_graph.reference import novelty_distance
from experiments.unsupervised_token_graph.head_geometry import cross_terms, geometry


def arguments(dimensions=0, signals=(0,)):
    return SimpleNamespace(
        signal_indices=list(signals),
        coordinates="raw",
        window=4,
        dimensions=dimensions,
        seed=17,
        ridge=0.1,
        layer_bands=False,
    )


def observations(seed=0):
    return np.random.default_rng(seed).random((17, 2, 3, 2))


def test_equal_marginals_and_current_can_have_different_cross_terms():
    same = np.array([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0]])
    opposite = np.array([[1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]])
    first = cross_terms.moment_parts((same.T @ same / 3)[None], same.mean(0)[None], 1)
    second = cross_terms.moment_parts((opposite.T @ opposite / 3)[None], opposite.mean(0)[None], 1)
    np.testing.assert_array_equal(same[-1], opposite[-1])
    np.testing.assert_array_equal(first["diagonal"], second["diagonal"])
    np.testing.assert_array_equal(first["persistence"], second["persistence"])
    np.testing.assert_allclose(first["cross"], -second["cross"])
    assert first["cross"][0, 0] > 0


def test_multiple_signals_keep_whole_head_blocks():
    mask = cross_terms.head_pair_mask(4, 2)
    row, column = np.triu_indices(4)
    assert set(zip(row[mask], column[mask])) == {(0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, 3)}
    assert set(zip(row[~mask], column[~mask])) == {(0, 2), (0, 3), (1, 2), (1, 3)}


def test_persistent_state_has_cross_terms_but_no_covariance():
    mean = np.array([[2.0, -3.0, 1.0]])
    moment = mean[:, :, None] * mean[:, None, :]
    parts = cross_terms.moment_parts(moment, mean, 1)
    np.testing.assert_array_equal(parts["cross"], parts["persistence"])
    np.testing.assert_array_equal(parts["covariance"], 0.0)
    assert np.any(parts["cross"] != 0)


def test_legacy_ablation_occurs_before_whitening():
    moment = np.array([[[2.0, 0.7], [0.7, 3.0]]])
    root = np.array([[1.0, 0.2], [0.2, 0.8]])
    parts = cross_terms.legacy_parts(moment, root, 1)
    expected = geometry.symmetric_vector(root @ np.diag([2.0, 3.0]) @ root - np.eye(2))
    np.testing.assert_allclose(parts["moment_diagonal"][0], expected)
    np.testing.assert_allclose(parts["moment"], parts["moment_diagonal"] + parts["moment_cross"])
    # Whitening may make the descriptor dense even when all original cross terms were removed.
    assert parts["moment_diagonal"][0, 1] != 0


@pytest.mark.parametrize("dimensions", [0, 8])
@pytest.mark.parametrize("signals", [(0,), (0, 1)])
def test_legacy_bridge_and_additive_decomposition(dimensions, signals):
    args = arguments(dimensions, signals)
    values = observations()
    model = geometry.fit_geometry(values, signals, args.ridge)
    coverage = np.ones(len(values), bool)
    old, _, _ = geometry.embed(values, coverage, [8, 9], model, args)
    new, _, _ = cross_terms.embed(values, coverage, [8, 9], model, args)
    for name in ("all__raw", "all__moment"):
        np.testing.assert_array_equal(new[name], old[name])
    diagonal_end = new["all__pair_diagonal"].shape[1]
    full = new["all__pair_full"][:, diagonal_end:]
    covariance = new["all__pair_covariance"][:, diagonal_end:]
    persistence = new["all__pair_persistence"][:, diagonal_end:]
    np.testing.assert_allclose(full, covariance + persistence, atol=2e-6)


def test_prefix_causality_and_gaps_reset_history():
    args = arguments(8)
    values = observations()
    coverage = np.ones(len(values), bool)
    coverage[6] = False
    values[6] = np.nan
    model = geometry.fit_coordinates(observations(1), [0], args.ridge)
    full, positions, counts = cross_terms.embed(values, coverage, [8, 9], model, args)
    prefix, _, _ = cross_terms.embed(values[:11], coverage[:11], [8, 9], model, args)
    for name in full:
        np.testing.assert_array_equal(full[name][positions < 11], prefix[name])
    index = int(np.flatnonzero(positions == 7)[0])
    assert counts[7] == 1
    np.testing.assert_array_equal(full["all__pair_full"][index], full["all__pair_full_w1"][index])
    for array in full.values():
        assert np.isfinite(array).all()


def test_instant_control_ignores_history_and_smoother_uses_only_past():
    args = arguments(8)
    values = observations()
    model = geometry.fit_coordinates(observations(1), [0], args.ridge)
    coverage = np.ones(len(values), bool)
    before, _, _ = cross_terms.embed(values, coverage, [8, 9], model, args)
    values[:-1] *= 5
    after, _, _ = cross_terms.embed(values, coverage, [8, 9], model, args)
    np.testing.assert_array_equal(before["all__pair_full_w1"][-1], after["all__pair_full_w1"][-1])
    assert not np.allclose(before["all__pair_full"][-1], after["all__pair_full"][-1])
    scores = {"all__pair_state": np.array([0.0, 2.0, np.nan, 4.0, 8.0, 100.0])}
    counts = geometry.window_counts(np.isfinite(scores["all__pair_state"]), 3)
    cross_terms.add_smooth_scores(scores, counts, 3)
    np.testing.assert_allclose(
        scores["all__pair_state_smooth"], [0.0, 1.0, np.nan, 4.0, 6.0, 112 / 3]
    )


def test_shared_metric_columns_and_reference_anchor_order():
    args = arguments(8)
    values = observations()
    model = geometry.fit_coordinates(values, [0], args.ridge)
    matrices, _, _ = cross_terms.embed(values, np.ones(len(values), bool), [8, 9], model, args)
    references = cross_terms.fit_references(matrices, 5)
    current_end = matrices["all__pair_state"].shape[1]
    diagonal_end = matrices["all__pair_diagonal"].shape[1]
    full = references["all__pair_full"]
    for name in ("pair_state", "pair_diagonal", "pair_cross"):
        reference = references["all__" + name]
        chosen = np.arange(len(reference["scale"]))
        if name == "pair_cross":
            chosen = np.r_[np.arange(current_end), np.arange(diagonal_end, len(full["scale"]))]
        for field in ("scale", "center"):
            np.testing.assert_array_equal(reference[field], full[field][chosen])
        np.testing.assert_array_equal(reference["bank"], full["bank"][:, chosen])
    for name in ("pair_covariance", "pair_persistence", "pair_full_w1"):
        np.testing.assert_array_equal(references["all__" + name]["scale"], full["scale"])
    for name in ("moment_diagonal", "moment_cross"):
        for field in ("center", "scale"):
            np.testing.assert_array_equal(
                references["all__" + name][field], references["all__moment"][field]
            )
    for name, matrix in matrices.items():
        assert np.isfinite(novelty_distance(matrix, references[name])).all()


def test_no_covered_tokens_produces_empty_embeddings():
    args = arguments(8)
    model = geometry.fit_coordinates(observations(), [0], args.ridge)
    matrices, positions, counts = cross_terms.embed(
        np.full((4, 2, 3, 2), np.nan), np.zeros(4, bool), [8, 9], model, args
    )
    assert len(positions) == 0
    assert counts.sum() == 0
    assert all(len(matrix) == 0 for matrix in matrices.values())
