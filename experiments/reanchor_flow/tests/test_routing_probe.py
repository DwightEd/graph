from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from experiments.reanchor_flow.routing_probe import SupervisedRoutingProbe
from experiments.reanchor_flow.routing_transition import RoutingSequence


def _sequence(seed, rows=96):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(rows, 1, 2, 4))
    context = np.zeros((rows, 8))
    context[:, 0] = 1
    context[:, 1] = np.log1p(np.arange(rows)) / 5
    context[:, 2] = context[:, 1] ** 2
    return RoutingSequence(state, context, np.arange(rows)), (state[:, 0, 0, 0] > 0).astype(int)


def _fit():
    sequences = [_sequence(seed) for seed in range(16)]
    return SupervisedRoutingProbe().fit(lambda: iter(sequences))


def test_supervised_readout_can_identify_head_signal_beyond_position():
    probe = _fit()
    sequence, labels = _sequence(123, rows=500)
    scores = probe.score(sequence)
    assert roc_auc_score(labels, scores["supervised_routing"]) > 0.94
    assert roc_auc_score(labels, scores["supervised_position"]) < 0.6
    assert probe.fit_metadata["labels_used_for_fit"] is True
    assert probe.fit_metadata["head_averaging"] is False


def test_future_states_cannot_change_previous_scores():
    probe = _fit()
    sequence, _ = _sequence(99)
    short = replace(
        sequence, state=sequence.state[:17], context=sequence.context[:17],
        response_index=sequence.response_index[:17],
    )
    scores, short_scores = probe.score(sequence), probe.score(short)
    for name in probe.NAMES:
        np.testing.assert_allclose(scores[name][:17], short_scores[name], atol=1e-12)


def test_unknown_labels_are_excluded_from_scaling_and_fit():
    sequence, labels = _sequence(1)
    unknown, _ = _sequence(2)
    unknown = replace(unknown, state=unknown.state * 1e6)
    one = SupervisedRoutingProbe().fit(lambda: iter([(sequence, labels)]))
    two = SupervisedRoutingProbe().fit(lambda: iter([
        (sequence, labels), (unknown, np.full(len(unknown.state), -1)),
    ]))
    assert two.fit_metadata["fit_known_rows"] == len(labels)
    assert two.fit_metadata["fit_sample_count"] == 1
    for name in one.NAMES:
        np.testing.assert_array_equal(one.mean[name], two.mean[name])
        np.testing.assert_array_equal(one.coefficient[name], two.coefficient[name])


def test_head_channels_remain_distinct_and_gaps_have_no_fake_transition():
    sequence, _ = _sequence(2, rows=4)
    sequence = replace(sequence, response_index=np.array([0, 1, 4, 5]))
    features = SupervisedRoutingProbe._features(sequence, np.arange(4))
    current = sequence.state.reshape(4, -1)
    np.testing.assert_array_equal(features["supervised_routing"][:, :8], current)
    np.testing.assert_array_equal(features["supervised_routing"][[0, 2], 8:16], 0)
    np.testing.assert_array_equal(
        features["supervised_routing"][1, 8:16], current[1] - current[0]
    )
    np.testing.assert_array_equal(features["supervised_position"], sequence.context[:, 1:])


def test_fit_caps_rows_before_feature_building_and_replays_four_passes(monkeypatch):
    sequence, labels = _sequence(10, rows=900)
    calls, sizes = [], []
    original = SupervisedRoutingProbe._features

    def features(sequence, rows):
        sizes.append(len(rows))
        return original(sequence, rows)

    def factory():
        calls.append(1)
        yield sequence, labels

    monkeypatch.setattr(SupervisedRoutingProbe, "_features", staticmethod(features))
    probe = SupervisedRoutingProbe(max_rows_per_sample=37).fit(factory)
    assert len(calls) == 4
    assert sizes == [37] * 4
    assert probe.fit_metadata["fit_known_rows"] == 37


def test_numeric_roundtrip_preserves_every_score(tmp_path):
    probe = _fit()
    path = tmp_path / "probe.npz"
    probe.save(path)
    with np.load(path, allow_pickle=False) as arrays:
        assert all(arrays[key].dtype.kind != "O" for key in arrays.files)
    loaded = SupervisedRoutingProbe.load(path)
    sequence, _ = _sequence(23)
    for name, values in probe.score(sequence).items():
        np.testing.assert_array_equal(values, loaded.score(sequence)[name])
    assert loaded.fit_metadata == probe.fit_metadata


def test_constant_context_scaling_is_finite_and_source_balanced_without_warnings():
    batches = []
    for seed in range(8):
        sequence, labels = _sequence(seed, rows=63)
        sequence.context[:, 3] = np.log1p(10) / 8
        batches.append((replace(sequence, sample_weight=1 / (seed + 1)), labels))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        probe = SupervisedRoutingProbe().fit(lambda: iter(batches))
    assert probe.scale["supervised_position"][2] == 1.0
    # The scaler preserves the caller's source weights, independent of labels.
    values = np.concatenate([sequence.state.reshape(63, -1) for sequence, _ in batches])
    weights = np.concatenate([
        np.full(63, sequence.sample_weight / 63) for sequence, _ in batches
    ])
    expected_mean = np.average(values, axis=0, weights=weights)
    expected_var = np.average((values - expected_mean) ** 2, axis=0, weights=weights)
    np.testing.assert_allclose(probe.mean["supervised_routing"][:8], expected_mean, atol=1e-15)
    np.testing.assert_allclose(probe.scale["supervised_routing"][:8], np.sqrt(expected_var))
    assert all(np.isfinite(score).all() for score in probe.score(batches[0][0]).values())


def test_missing_class_reports_supervised_diagnostic_unavailable():
    sequence, _ = _sequence(1)
    with pytest.raises(ValueError, match="both known classes"):
        SupervisedRoutingProbe().fit(lambda: iter([(sequence, np.zeros(len(sequence.state)))]))
