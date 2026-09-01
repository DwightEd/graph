import copy

import numpy as np

from experiments.attention_mechanism_audit.detect import (
    carrier_log_volume,
    crossfit_partitions,
    factorial_contrasts,
    score_records,
    source_fold_assignments,
)


def _artifact(seed: int, tokens: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    layers, heads = 2, 3
    trace = {}
    for family in ("attention", "edge"):
        trace[f"prompt_{family}_effective_sources"] = rng.uniform(
            1.5, 4.0, (layers, tokens)
        )
        trace[f"prompt_{family}_effective_rank"] = rng.uniform(
            1.0, 2.5, (layers, tokens)
        )
        trace[f"prompt_{family}_anchor_index"] = rng.integers(
            0, 5, (layers, tokens, heads)
        )
    full = rng.normal(-2, 0.2, tokens)
    return {
        "response_start": 5,
        "trace": trace,
        "score_inputs": {
            "full_logprob": full,
            "no_evidence_logprob": full - 1,
            "no_history_logprob": full - 2,
            "no_evidence_history_logprob": full - 4,
        },
    }


def _records(count=15):
    return [
        {
            "sample_id": f"sample-{i}",
            "source_id": f"source-{i}",
            "task_type": "QA",
            "artifact": _artifact(i),
        }
        for i in range(count)
    ]


def test_factorial_contrasts_match_symmetric_effect_equations():
    np.testing.assert_allclose(factorial_contrasts(_artifact(1)), [[1.5, 2.5, -1]] * 7)


def test_carrier_volume_is_product_of_source_head_and_temporal_support():
    artifact = _artifact(1, 2)
    for family in ("attention", "edge"):
        artifact["trace"][f"prompt_{family}_effective_sources"][:] = 2
        artifact["trace"][f"prompt_{family}_effective_rank"][:] = 3
        artifact["trace"][f"prompt_{family}_anchor_index"][:] = 0
    np.testing.assert_allclose(carrier_log_volume(artifact, "edge"), np.log(6))
    artifact["trace"]["prompt_edge_anchor_index"][:, 1, :] = 1
    assert np.all(carrier_log_volume(artifact, "edge")[1] > np.log(6))


def test_crossfit_is_source_disjoint_and_deterministic():
    sources = [f"s-{i}" for i in range(20)]
    assert source_fold_assignments(sources, folds=5, seed=4) == source_fold_assignments(
        list(reversed(sources)), folds=5, seed=4
    )
    for part in crossfit_partitions(sources, folds=5, seed=4):
        fit, calibration, test = map(
            set,
            (part["fit_sources"], part["calibration_sources"], part["test_sources"]),
        )
        assert not (fit & calibration or fit & test or calibration & test)


def test_scores_are_deterministic_label_sealed_and_direction_fixed():
    records = _records()
    first, metadata = score_records(records, seed=7)
    labeled = copy.deepcopy(records)
    for record in labeled:
        record["label"] = np.ones(7)
    second, repeated = score_records(labeled, seed=7)
    assert metadata == repeated
    assert metadata["crossfit_complete"]
    for sample in first:
        assert set(first[sample]) == {
            "functional_route_collapse",
            "attention_route_collapse",
            "confidence",
        }
        for name in first[sample]:
            np.testing.assert_array_equal(first[sample][name], second[sample][name])
        for name in ("functional_route_collapse", "attention_route_collapse"):
            assert np.all((first[sample][name] >= 0) & (first[sample][name] <= 1))


def test_fewer_prompt_carriers_always_increases_raw_mechanism_direction():
    records = _records()
    baseline, _ = score_records(records, seed=7)
    changed = copy.deepcopy(records)
    target = changed[0]
    target["artifact"]["trace"]["prompt_edge_effective_sources"][:] = 1
    target["artifact"]["trace"]["prompt_edge_effective_rank"][:] = 1
    target["artifact"]["trace"]["prompt_edge_anchor_index"][:] = 0
    rescored, _ = score_records(changed, seed=7)
    assert rescored[target["sample_id"]]["functional_route_collapse"].mean() >= (
        baseline[target["sample_id"]]["functional_route_collapse"].mean()
    )


def test_too_few_sources_exposes_no_fallback_detector():
    scores, metadata = score_records(_records(1))
    assert not metadata["mechanism_scores_available"]
    assert not scores["sample-0"]["functional_route_collapse"].any()
