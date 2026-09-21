from types import SimpleNamespace

import numpy as np
import pytest

from experiments.unsupervised_token_graph.head_geometry.geometry import (
    conditional_vectors, embed, fit_conditionals, fit_geometry, head_views, matrix_function, second_moments,
    symmetric_vector, window_counts,
)
from experiments.unsupervised_token_graph.head_geometry.inputs import channel_values, extract
from experiments.unsupervised_token_graph.fixed_graph.inputs import sample_channels
from experiments.unsupervised_token_graph.fixed_graph.tests.test_fixed_graph import write_cache
from experiments.unsupervised_token_graph.offline_span.data import load_samples


def settings():
    return SimpleNamespace(signal_indices=[0], window=4, ridge=.1,
                           dimensions=16, seed=17, layer_bands=True)


def observations(seed=1, tokens=18):
    return np.random.default_rng(seed).uniform(.01, .8, (tokens, 4, 3, 2))


def test_special_keys_removed_without_amplifying_submass_and_prediction_alignment():
    sample = SimpleNamespace(response_length=2, prompt_length=4,
                             token_ids=np.array([99, 1, 2, 3, 4, 5]))
    channel = SimpleNamespace(queries=[4], row=lambda _: (np.array([0, 1, 4]),
                                                         np.array([.9, .05, .05])))
    values, masses = channel_values(channel, sample, [99])
    assert np.isnan(values[0]).all()
    np.testing.assert_allclose(values[1], [.05, .05])
    np.testing.assert_allclose(masses[1], [1., .1])


def test_chat_control_added_tokens_are_excluded_without_a_named_special_role(monkeypatch):
    from transformers import AutoTokenizer
    from experiments.unsupervised_token_graph.head_geometry.inputs import special_ids
    tokenizer = SimpleNamespace(all_special_ids=[99], added_tokens_decoder={
        98: SimpleNamespace(special=True), 97: SimpleNamespace(special=False)})
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    args = SimpleNamespace(special_token_ids=None, tokenizer="unused")
    assert special_ids(args) == [98, 99]


def test_zero_is_observed_but_absent_query_stays_missing():
    sample = SimpleNamespace(response_length=3, prompt_length=2,
                             token_ids=np.array([99, 1, 2, 3, 4]), offsets=np.empty((0, 2), int))
    channel = SimpleNamespace(layer=0, head=0, queries=[2, 3], row=lambda row: (
        np.array([0]) if row == 0 else np.array([0, 3]),
        np.array([1.]) if row == 0 else np.array([1., 1e-12])))
    arrays = extract(sample, [channel], [99])
    np.testing.assert_array_equal(arrays["coverage"], [False, True, True])
    np.testing.assert_array_equal(arrays["head_observed"][:, 0, 0], [False, True, True])
    np.testing.assert_array_equal(arrays["conditional_defined"][:, 0, 0], [False, False, True])
    np.testing.assert_allclose(arrays["observations"][1, 0, 0], [0., 0.])
    assert arrays["observations"][2, 0, 0, 0] == pytest.approx(1e-12, rel=1e-5, abs=0)


def test_conditional_residual_agrees_with_leave_whole_head_out_regression():
    random = np.random.default_rng(4)
    train = random.normal(size=(200, 2, 6))
    train[:, :, 2:4] += train[:, :, :2]
    test = random.normal(size=(3, 2, 6))
    model = fit_conditionals(train, heads=3, ridge=.1)
    vectors = conditional_vectors(test, model)
    covariance = np.linalg.inv(model["precision"])
    centered = test - train.mean(axis=0)
    for layer in range(2):
        held, other = np.array([2, 3]), np.array([0, 1, 4, 5])
        cov = covariance[layer]
        regression = cov[np.ix_(held, other)] @ np.linalg.inv(cov[np.ix_(other, other)])
        residual = centered[:, layer, held] - centered[:, layer, other] @ regression.T
        conditional_cov = cov[np.ix_(held, held)] - regression @ cov[np.ix_(other, held)]
        energy = np.einsum("ni,ij,nj->n", residual, np.linalg.inv(conditional_cov), residual)
        np.testing.assert_allclose((vectors["conditional_energy"][:, layer, 1] ** 2).sum(-1), energy)


def test_conditional_energy_detects_broken_pair_with_equal_marginal_energy():
    # Same amplitudes; the sign relation changes, without any time history.
    train = np.array([[[-1., -1.]], [[1., 1.]], [[-2., -2.]], [[2., 2.]]])
    model = fit_conditionals(train, heads=2, ridge=.1)
    vectors = conditional_vectors(np.array([[[1., 1.]], [[1., -1.]]]), model)
    independent = (vectors["independent_energy"] ** 2).sum(axis=(1, 2, 3))
    conditional = (vectors["conditional_energy"] ** 2).sum(axis=(1, 2, 3))
    np.testing.assert_allclose(independent[0], independent[1])
    assert conditional[1] > conditional[0]


def test_current_conditional_energy_is_unchanged_by_window_length():
    args = settings()
    train, test = observations(1), observations(2)
    model = fit_geometry(train, [0], args.ridge)
    covered = np.ones(len(test), bool)
    first, _, counts = embed(test, covered, np.arange(4), model, args)
    assert counts[-1] == args.window
    args.window = 1
    second, _, _ = embed(test, covered, np.arange(4), model, args)
    for scope in ["all", "L0-0", "L1-1", "L2-2", "L3-3"]:
        np.testing.assert_array_equal(first[scope + "__conditional_energy"], second[scope + "__conditional_energy"])


@pytest.mark.parametrize("layout", ["attention", "data", "canonical"])
def test_real_cache_readers_produce_same_special_filtered_values(tmp_path, layout):
    paths = [tmp_path / "dense.npz", tmp_path / "other.npz"]
    outputs = []
    for path, encoding in zip(paths, ["attention", layout]):
        write_cache(path, layout=encoding)
        samples, index = load_samples(path)
        channels = sample_channels(samples[0], index, None, None)
        outputs.append(extract(samples[0], channels, [10]))
    np.testing.assert_allclose(outputs[0]["observations"], outputs[1]["observations"], equal_nan=True)
    assert not outputs[0]["coverage"][0]
    assert not outputs[0]["coverage"][5]


def test_same_average_different_heads_remain_distinct():
    values = np.array([[[[.1], [.7], [.4]]], [[[.7], [.1], [.4]]]])
    raw, contrast = head_views(values, [0])
    np.testing.assert_allclose(raw[0].mean(), raw[1].mean())
    assert not np.allclose(contrast[0], contrast[1])
    np.testing.assert_allclose(contrast.sum(axis=-1), 0, atol=1e-15)


def test_constant_signed_pattern_is_not_erased_as_rolling_covariance():
    values = np.tile([2., -2.], (6, 1))
    counts = window_counts(np.ones(6, bool), 4)
    moment = second_moments(values, np.array([5]), counts, 4, .1)[0]
    np.testing.assert_allclose(moment, [[4.1, -4.], [-4., 4.1]])
    assert np.linalg.eigvalsh(moment).min() == pytest.approx(.1)


def test_spd_log_known_diagonal_and_symmetric_vector_metric():
    matrix = np.diag([1., np.e ** 2])[None]
    np.testing.assert_allclose(matrix_function(matrix, np.log)[0], np.diag([0., 2.]))
    symmetric = np.array([[[1., 2.], [2., 3.]]])
    assert np.linalg.norm(symmetric_vector(symmetric)) == pytest.approx(np.linalg.norm(symmetric))


def test_diagonal_control_ignores_relations_with_equal_marginal_energy():
    positive = np.tile([1., 1.], (4, 1))
    negative = np.tile([1., -1.], (4, 1))
    counts = window_counts(np.ones(4, bool), 4)
    first = second_moments(positive, np.array([3]), counts, 4, .1)[0]
    second = second_moments(negative, np.array([3]), counts, 4, .1)[0]
    np.testing.assert_array_equal(first.diagonal(), second.diagonal())
    assert first[0, 1] == -second[0, 1]


def test_missing_rows_reset_windows_and_do_not_insert_zero_observations():
    covered = np.array([False, True, True, False, True, True])
    counts = window_counts(covered, 4)
    np.testing.assert_array_equal(counts, [0, 1, 2, 0, 1, 2])
    values = np.array([[np.nan], [90.], [90.], [np.nan], [2.], [3.]])
    result = second_moments(values, np.array([4, 5]), counts, 4, .1)
    np.testing.assert_allclose(result[:, 0, 0], [4.1, 6.6])


def test_prefix_replay_preserves_every_representation_and_layer_band():
    args = settings()
    train, test = observations(1), observations(2)
    model = fit_geometry(train, [0], args.ridge)
    covered = np.ones(len(test), bool)
    full, positions, _ = embed(test, covered, np.arange(4), model, args)
    prefix, _, _ = embed(test[:9], covered[:9], np.arange(4), model, args)
    assert "L1-1__log_moment" in full
    for name in full:
        np.testing.assert_allclose(full[name][:9], prefix[name], atol=1e-6)
    selected, chosen, _ = embed(test, covered, np.arange(4), model, args, [8, 3])
    np.testing.assert_array_equal(chosen, [8, 3])
    for name in full:
        np.testing.assert_allclose(selected[name], full[name][chosen], atol=1e-6)


def test_current_coordinate_keeps_sign_lost_by_second_moment():
    args = settings()
    train = observations(1)
    model = fit_geometry(train, [0], args.ridge)
    model["center"][:] = 0
    first = observations(2)
    second = 1 - first
    coverage = np.ones(len(first), bool)
    left, _, _ = embed(first, coverage, np.arange(4), model, args)
    right, _, _ = embed(second, coverage, np.arange(4), model, args)
    name = "all__log_moment"
    current_width = first.shape[1] * first.shape[2]
    np.testing.assert_allclose(left[name][:, current_width:], right[name][:, current_width:], atol=1e-5)
    assert not np.allclose(left[name][:, :current_width], right[name][:, :current_width])


def test_current_heads_are_uncompressed_and_exact_relation_mode_is_available():
    args = settings()
    train, test = observations(1), observations(2)
    model = fit_geometry(train, [0], args.ridge)
    args.dimensions = 0
    full, _, _ = embed(test, np.ones(len(test), bool), np.arange(4), model, args)
    assert full["all__contrast"].shape == (len(test), 4 * 3)
    assert full["all__log_moment"].shape == (len(test), 4 * 3 + 4 * 6)
