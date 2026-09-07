from __future__ import annotations

from dataclasses import replace

import numpy as np

from experiments.reanchor_flow.routing_transition import (
    ILR_BASIS,
    RoutingSequence,
    RoutingTransitionModel,
)
from experiments.reanchor_flow.scan_dataset import ScanSample


def _scan(seed=0, rows=32, scale=1.0):
    rng = np.random.default_rng(seed)
    transport = scale * rng.lognormal(size=(1, 2, rows, 4))
    return ScanSample(
        "s",
        "source",
        "QA",
        11,
        np.arange(10, 10 + rows),
        {"reanchor_bucket_transport": transport},
        {"local_window": 3},
    )


def _ar_sequence(seed, rows=96):
    rng = np.random.default_rng(seed)
    state = np.empty((rows, 1, 2, 4))
    state[0] = rng.normal(size=(1, 2, 4))
    for row in range(1, rows):
        state[row] = 0.94 * state[row - 1] + rng.normal(scale=0.12, size=(1, 2, 4))
    context = np.zeros((rows, 8))
    context[:, 0] = 1
    return RoutingSequence(state, context, np.arange(rows))


def _fitted():
    sequences = [_ar_sequence(seed) for seed in range(12)]
    return RoutingTransitionModel().fit(lambda: iter(sequences))


def test_orthonormal_composition_and_absolute_transport_have_separate_axes():
    np.testing.assert_allclose(ILR_BASIS @ ILR_BASIS.T, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(ILR_BASIS.sum(axis=-1), 0, atol=1e-15)
    one = RoutingSequence.from_scan(_scan(scale=1))
    doubled = RoutingSequence.from_scan(_scan(scale=2))
    np.testing.assert_allclose(one.state[..., :3], doubled.state[..., :3], atol=1e-14)
    assert np.all(doubled.state[..., 3] > one.state[..., 3])


def test_context_encodes_causal_source_availability_and_no_future_length():
    scan = _scan(rows=10)
    sequence = RoutingSequence.from_scan(scan)
    np.testing.assert_array_equal(
        sequence.context[:, 6], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    # At predictor index 4 and local_window=3, earliest response distance=3.
    np.testing.assert_array_equal(
        sequence.context[:, 7], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    )
    changed = replace(
        scan, metadata={"local_window": 3, "full_response_tokens": 999999}
    )
    np.testing.assert_array_equal(
        sequence.context, RoutingSequence.from_scan(changed).context
    )


def test_future_extension_cannot_change_past_states_context_or_scores():
    scan = _scan(rows=50)
    prefix = replace(
        scan,
        row_position=scan.row_position[:17],
        arrays={
            "reanchor_bucket_transport": scan["reanchor_bucket_transport"][:, :, :17]
        },
    )
    whole = RoutingSequence.from_scan(scan)
    short = RoutingSequence.from_scan(prefix)
    np.testing.assert_array_equal(whole.state[:17], short.state)
    np.testing.assert_array_equal(whole.context[:17], short.context)
    model = RoutingTransitionModel().fit(lambda: iter([whole]))
    a, b = model.score(whole), model.score(short)
    np.testing.assert_allclose(a.joint[:17], b.joint, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(a.innovation[:17], b.innovation, rtol=1e-12, atol=1e-10)


def test_fit_is_two_pass_streaming_and_never_uses_correctness_metadata():
    calls = []
    scan = _scan()

    def factory():
        calls.append(1)
        for label in (0, 1):
            yield RoutingSequence.from_scan(
                replace(
                    scan, metadata={"local_window": 3, "hallucination_label": label}
                )
            )

    with_labels = RoutingTransitionModel().fit(factory)
    assert len(calls) == 2
    without_labels = RoutingTransitionModel().fit(
        lambda: iter([RoutingSequence.from_scan(scan)] * 2)
    )
    for name in ("current", "joint"):
        np.testing.assert_array_equal(
            with_labels.coefficients[name], without_labels.coefficients[name]
        )
        np.testing.assert_array_equal(
            with_labels.covariances[name], without_labels.covariances[name]
        )


def test_chain_rule_is_exact_and_heads_are_preserved():
    sequence, model = _ar_sequence(99), _fitted()
    score = model.score(sequence)
    assert score.per_head_joint.shape == (96, 1, 2)
    np.testing.assert_allclose(score.joint, score.per_head_joint.sum(axis=(1, 2)))
    np.testing.assert_allclose(
        score.joint[1:], score.previous[1:] + score.innovation[1:]
    )
    assert np.isnan(score.innovation[0])
    assert score.joint[0] == score.current[0]
    pair = np.concatenate((sequence.state[:-1], sequence.state[1:]), axis=-1)
    mean = np.einsum("tc,clhd->tlhd", sequence.context[1:], model.coefficients["joint"])
    residual = pair - mean
    covariance = model.covariances["joint"]
    energy = np.einsum(
        "tlhi,lhij,tlhj->tlh", residual, np.linalg.inv(covariance), residual
    )
    direct = 0.5 * (energy + np.linalg.slogdet(covariance)[1] + 8 * np.log(2 * np.pi))
    np.testing.assert_allclose(score.per_head_joint[1:], direct, rtol=1e-10, atol=1e-10)
    excess = model.head_excess(score)
    np.testing.assert_allclose(excess[1:], 0.5 * (energy - 8), atol=1e-10)
    residual_first = sequence.state[:1] - np.einsum(
        "tc,clhd->tlhd", sequence.context[:1], model.coefficients["current"]
    )
    energy_first = np.einsum(
        "tlhi,lhij,tlhj->tlh", residual_first, model.current_precision, residual_first
    )
    np.testing.assert_allclose(excess[:1], 0.5 * (energy_first - 4), atol=1e-10)


def test_temporal_disruption_detected_even_when_static_distribution_identical():
    sequence, model = _ar_sequence(42, rows=256), _fitted()
    # Reordering leaves the distribution of states exactly unchanged, but breaks
    # the learned conditional dependency. No correctness labels enter the model.
    permutation = np.random.default_rng(9).permutation(len(sequence.state))
    disrupted = replace(sequence, state=sequence.state[permutation])
    normal, anomaly = model.score(sequence), model.score(disrupted)
    np.testing.assert_allclose(
        np.sort(normal.current), np.sort(anomaly.current), atol=1e-12
    )
    assert np.median(anomaly.innovation[1:]) > np.median(normal.innovation[1:]) + 10
    assert np.median(anomaly.joint[1:]) > np.median(normal.joint[1:]) + 10


def test_saved_model_reproduces_every_score_and_uses_no_pickle(tmp_path):
    model, sequence = _fitted(), _ar_sequence(21)
    path = tmp_path / "model.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as stored:
        assert all(stored[key].dtype.kind != "O" for key in stored.files)
    restored = RoutingTransitionModel.load(path)
    original, loaded = model.score(sequence), restored.score(sequence)
    for name in original.__dataclass_fields__:
        np.testing.assert_allclose(getattr(original, name), getattr(loaded, name))


def test_duplicate_samples_with_half_weight_do_not_change_model():
    sequences = [_ar_sequence(1), _ar_sequence(2)]
    first = RoutingTransitionModel(max_rows_per_sample=32).fit(lambda: iter(sequences))
    duplicated = [replace(sequence, sample_weight=0.5) for sequence in sequences] * 2
    second = RoutingTransitionModel(max_rows_per_sample=32).fit(
        lambda: iter(duplicated)
    )
    for name in ("current", "joint"):
        np.testing.assert_allclose(
            first.coefficients[name], second.coefficients[name], atol=1e-12
        )
        np.testing.assert_allclose(
            first.covariances[name], second.covariances[name], atol=1e-12
        )


def test_missing_predecessor_is_not_interpreted_as_adjacent_transition():
    model, sequence = _fitted(), _ar_sequence(9, rows=8)
    sequence = replace(sequence, response_index=np.array([0, 1, 2, 4, 5, 6, 7, 8]))
    score = model.score(sequence)
    assert not score.has_previous[3]
    assert np.isnan(score.innovation[3])
    assert score.joint[3] == score.current[3]
