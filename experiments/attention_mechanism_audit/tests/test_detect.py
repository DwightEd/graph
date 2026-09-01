import copy
import gc
import json

import numpy as np

from experiments.attention_mechanism_audit import detect
from experiments.attention_mechanism_audit.detect import (
    NUISANCE_NAMES,
    SCORE_NAMES,
    crossfit_partitions,
    factorial_contrasts,
    nuisance_design,
    raw_scores,
    score_records,
    source_fold_assignments,
)


def _artifact(seed: int, tokens: int = 7, prompt: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    layers, heads, roles = 2, 3, 4
    full = rng.normal(-2, 0.2, tokens)
    effective = rng.uniform(1.2, 8.0, (layers, tokens, roles))
    mass = rng.uniform(0.1, 2.0, (layers, tokens, heads, roles))
    effective[:, :2, 2] = 0
    mass[:, :2, :, 2] = 0
    evidence_mask = np.zeros(prompt, dtype=bool)
    evidence_mask[1 : max(2, prompt - 1)] = True
    return {
        "response_start": prompt,
        "evidence_mask": evidence_mask,
        "trace": {
            "edge_role_effective_routes": effective,
            "edge_role_mass": mass,
        },
        "score_inputs": {
            "full_logprob": full,
            "full_margin": rng.normal(1, 0.2, tokens),
            "no_evidence_logprob": full - rng.normal(0.5, 0.1, tokens),
            "no_history_logprob": full - rng.normal(0.8, 0.1, tokens),
            "no_evidence_history_logprob": full - rng.normal(1.4, 0.1, tokens),
        },
    }


def _records(count: int = 15) -> list[dict]:
    return [
        {
            "sample_id": f"sample-{index}",
            "source_id": f"source-{index}",
            "task_type": "QA",
            "artifact": _artifact(index, tokens=5 + index % 3, prompt=5 + index % 4),
        }
        for index in range(count)
    ]


def test_raw_endpoint_is_unsupported_strict_history_takeover():
    artifact = _artifact(1, tokens=2)
    inputs = artifact["score_inputs"]
    inputs["full_logprob"] = np.array([-2.0, -3.0])
    inputs["no_evidence_logprob"] = np.array([-1.0, -4.0])
    inputs["no_evidence_history_logprob"] = np.array([-5.0, -6.0])
    scores = raw_scores(artifact)
    np.testing.assert_allclose(scores["unsupported_history_takeover"], [5.0, 1.0])
    np.testing.assert_allclose(scores["evidence_bypass"], [1.0, -1.0])
    np.testing.assert_allclose(scores["confidence"], [2.0, 3.0])


def test_factorial_contrasts_match_symmetric_effect_equations():
    artifact = _artifact(1)
    full = artifact["score_inputs"]["full_logprob"]
    artifact["score_inputs"].update(
        no_evidence_logprob=full - 1,
        no_history_logprob=full - 2,
        no_evidence_history_logprob=full - 4,
    )
    np.testing.assert_allclose(factorial_contrasts(artifact), [[1.5, 2.5, -1]] * 7)


def test_nuisance_design_keeps_prompt_evidence_and_response_lengths_separate():
    artifact = _artifact(2, tokens=4, prompt=6)
    artifact["evidence_mask"][:] = [False, True, True, False, True, False]
    design = nuisance_design(artifact)
    assert tuple(NUISANCE_NAMES) == (
        "intercept",
        "relative_position",
        "relative_position_squared",
        "log_prompt_length",
        "log_evidence_length",
        "log_response_length",
    )
    np.testing.assert_allclose(design[:, 3], np.log1p(6))
    np.testing.assert_allclose(design[:, 4], np.log1p(3))
    np.testing.assert_allclose(design[:, 5], np.log1p(4))


def test_crossfit_is_source_disjoint_and_deterministic():
    sources = [f"s-{index}" for index in range(20)]
    assert source_fold_assignments(sources, folds=5, seed=4) == source_fold_assignments(
        list(reversed(sources)), folds=5, seed=4
    )
    for partition in crossfit_partitions(sources, folds=5, seed=4):
        fit, calibration, test = map(
            set,
            (
                partition["fit_sources"],
                partition["calibration_sources"],
                partition["test_sources"],
            ),
        )
        assert not (fit & calibration or fit & test or calibration & test)


def test_scores_are_deterministic_label_sealed_and_fixed_direction():
    records = _records()
    first, metadata = score_records(records, seed=7)
    labeled = copy.deepcopy(list(reversed(records)))
    for record in labeled:
        record["label"] = np.ones(
            len(record["artifact"]["score_inputs"]["full_logprob"])
        )
    second, repeated = score_records(labeled, seed=7)
    assert metadata == repeated
    assert metadata["crossfit_complete"]
    json.dumps(metadata, allow_nan=False)
    assert "position/length" in metadata["fit"]
    assert "out-of-fold percentile" in metadata["score_scale"]
    for sample_id in first:
        assert tuple(first[sample_id]) == (*SCORE_NAMES, "detection_valid")
        for name in SCORE_NAMES:
            np.testing.assert_array_equal(
                first[sample_id][name], second[sample_id][name]
            )
        np.testing.assert_array_equal(
            first[sample_id]["detection_valid"], second[sample_id]["detection_valid"]
        )
        assert not first[sample_id]["detection_valid"][:2].any()
        assert first[sample_id]["detection_valid"][2:].all()
        for name in SCORE_NAMES[:-1]:
            assert np.all((first[sample_id][name] >= 0) & (first[sample_id][name] <= 1))
            assert not first[sample_id][name][:2].any()


def test_path_artifacts_are_prepared_and_released_one_at_a_time(monkeypatch):
    state = {"alive": 0, "peak": 0, "calls": 0}

    class TrackingArtifact(dict):
        def __init__(self, value):
            super().__init__(value)
            state["alive"] += 1
            state["peak"] = max(state["peak"], state["alive"])

        def __del__(self):
            state["alive"] -= 1

    def load(record):
        state["calls"] += 1
        index = int(str(record["sample_id"]).split("-")[-1])
        return TrackingArtifact(
            _artifact(index, tokens=5 + index % 3, prompt=5 + index % 4)
        )

    monkeypatch.setattr(detect, "_load_artifact", load)
    records = [
        {
            "sample_id": f"sample-{index}",
            "source_id": f"source-{index}",
            "task_type": "QA",
            "path": f"sample-{index}.pt",
        }
        for index in range(15)
    ]
    scores, metadata = score_records(records)
    gc.collect()
    assert metadata["crossfit_complete"]
    assert len(scores) == len(records)
    assert state == {"alive": 0, "peak": 1, "calls": len(records)}


def test_too_few_sources_exposes_no_fallback_detector():
    scores, metadata = score_records(_records(1))
    assert not metadata["mechanism_scores_available"]
    for name in SCORE_NAMES[:-1]:
        assert not scores["sample-0"][name].any()
    assert scores["sample-0"]["confidence"].any()
    np.testing.assert_array_equal(
        scores["sample-0"]["detection_valid"], [False, False, True, True, True]
    )


def test_rank_deficient_nuisance_fit_is_unavailable():
    _scores, metadata = score_records(_records(3))
    assert not metadata["mechanism_scores_available"]
    assert "rank deficient" in metadata["reason"]
    assert metadata["partitions"][0]["nuisance"]


def test_noncomparable_history_contraction_gets_neutral_control_score():
    scores, metadata = score_records(_records(), seed=7)
    assert metadata["mechanism_scores_available"]
    assert metadata["score_coverage"]["history_route_contraction"]["tokens"] > 0
    for sample in scores.values():
        assert sample["detection_valid"][2]
        assert sample["history_route_contraction"][2] == 0.5


def test_all_short_responses_make_the_mechanism_unavailable():
    records = _records(5)
    for record in records:
        record["artifact"] = _artifact(1, tokens=2)
    scores, metadata = score_records(records)
    assert not metadata["mechanism_scores_available"]
    assert metadata["evaluated_tokens"] == 0
    assert all(not score["detection_valid"].any() for score in scores.values())


def test_short_response_source_is_excluded_without_affecting_valid_sources():
    records = _records(8)
    records.append(
        {
            "sample_id": "short",
            "source_id": "short-source",
            "task_type": "QA",
            "artifact": _artifact(99, tokens=2),
        }
    )
    scores, metadata = score_records(records)
    assert metadata["crossfit_complete"]
    assert "short-source" not in metadata["source_folds"]
    assert not scores["short"]["detection_valid"].any()
