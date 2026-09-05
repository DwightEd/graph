import numpy as np

from experiments.reanchor_flow.detector import (
    RAW_FEATURES,
    ReanchorFailureDetector,
    SourceBalancedECDF,
    compose_scores,
    raw_features,
)


def detector_record(value, count=100):
    position = (np.arange(count) + 0.5) / count
    row = {
        "relative_position": position,
        "baseline_entropy": 1.0 + position,
        "baseline_target_logprob": -2.0 + position,
    }
    row.update(
        {name: np.full(count, value, dtype=np.float64) for name in RAW_FEATURES}
    )
    return row


def capture(evidence):
    evidence = np.asarray(evidence, dtype=np.float64)
    layer, head, count = evidence.shape
    shape = (layer, head, count)
    return {
        "head_evidence_transport_share": evidence,
        "head_history_transport_share": np.full(shape, 0.1),
        "head_route_change": np.full(shape, 0.1),
        "head_predictor_reuse": np.full(shape, 0.1),
        "head_emitted_token_anchor": np.full(shape, 0.1),
        "baseline_target_logprob": np.full(count, -1.0),
        "baseline_entropy": np.full(count, 1.0),
        "context_distribution_js": np.full(count, 0.1),
        "context_target_logprob_gain": np.full(count, 0.1),
        "context_adoption_margin": np.full(count, 0.1),
        "context_target_log_rank": np.zeros(count),
    }


def test_source_balancing_is_invariant_to_within_source_duplication():
    original = SourceBalancedECDF.fit(
        np.array([0.0, 1.0, 10.0, 11.0]),
        np.array(["a", "a", "b", "b"]),
    )
    duplicated = SourceBalancedECDF.fit(
        np.array([0.0, 1.0, 0.0, 1.0, 10.0, 11.0]),
        np.array(["a", "a", "a", "a", "b", "b"]),
    )
    query = np.array([1.0, 10.0, 12.0])
    assert np.allclose(original.score(query), duplicated.score(query))


def test_train_calibration_assigns_extreme_test_failures_high_scores():
    train = [detector_record(value) for value in (0.1, 0.2, 0.3)]
    detector = ReanchorFailureDetector.fit(train, ["a", "b", "c"])
    scored = detector.score(detector_record(2.0, count=20))
    assert np.all(scored.score["online_failure"] == 1.0)
    assert np.all(scored.score["offline_failure"] == 1.0)


def test_entry_deficit_preserves_head_local_route_change():
    evidence = np.full((3, 2, 7), 0.8)
    evidence[..., 4] = 0.1
    result = capture(evidence)
    result["head_route_change"][..., 4] = 1.0
    feature = raw_features(result, route_window=2)
    assert feature["evidence_entry_deficit"][4] > 0.6
    assert feature["evidence_entry_deficit"][5] == 0.0


def test_evidence_reentry_detects_route_recovery():
    evidence = np.full((3, 2, 7), 0.8)
    evidence[..., 4] = 0.1
    result = capture(evidence)
    result["head_route_change"][..., 4:6] = 1.0
    feature = raw_features(result, route_window=2)
    assert feature["evidence_entry_deficit"][4] > 0.6
    assert feature["evidence_reentry_strength"][5] > 0.3


def test_failure_state_persists_until_evidence_reentry():
    tail = {name: np.full(5, 0.1) for name in RAW_FEATURES}
    tail["adoption_deficit"] = np.array([0.1, 0.9, 0.1, 0.1, 0.1])
    tail["history_dominance"] = np.array([0.1, 0.1, 0.9, 0.9, 0.9])
    tail["evidence_reentry_strength"] = np.array([0.1, 0.1, 0.1, 0.95, 0.1])
    score = compose_scores(tail)
    assert score["onset_trigger"][1] == 0.9
    assert score["online_failure"][2] == 0.9
    assert score["online_failure"][3] == 0.1


def test_context_js_is_a_control_not_a_failure_gate():
    low = {name: np.full(3, 0.2) for name in RAW_FEATURES}
    high = {name: value.copy() for name, value in low.items()}
    high["context_distribution_js"][:] = 0.99
    assert np.array_equal(
        compose_scores(low)["online_failure"],
        compose_scores(high)["online_failure"],
    )


def test_future_reuse_changes_only_the_offline_score():
    tail = {name: np.full(1, 0.1) for name in RAW_FEATURES}
    tail.update(
        route_demand=np.full(1, 0.4),
        evidence_entry_deficit=np.full(1, 0.4),
        context_opposition=np.full(1, 0.9),
        context_distribution_js=np.full(1, 0.3),
        adoption_deficit=np.full(1, 0.4),
        context_target_log_rank=np.full(1, 0.4),
        late_evidence_route_loss=np.full(1, 0.95),
    )
    before = compose_scores(tail)
    tail["predictor_reuse"] = np.full(1, 0.95)
    tail["emitted_token_anchor"] = np.full(1, 0.95)
    after = compose_scores(tail)
    assert np.array_equal(before["online_failure"], after["online_failure"])
    assert after["offline_failure"][0] > before["offline_failure"][0]
