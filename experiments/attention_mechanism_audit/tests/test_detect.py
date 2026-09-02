import copy
import gc

import numpy as np

from experiments.attention_mechanism_audit import detect
from experiments.attention_mechanism_audit.detect import (
    REGISTER_NAMES,
    SCORE_NAMES,
    factorial_contrasts,
    raw_scores,
    score_records,
)


def _artifact(tokens: int = 3, layers: int = 2) -> dict:
    full = -np.arange(1, tokens + 1, dtype=np.float64)
    gram = np.zeros((layers, tokens, len(REGISTER_NAMES), layers))
    for token in range(tokens):
        gram[:, token, 0, :] = np.eye(layers) * (token + 1)
        gram[:, token, 1, :] = np.eye(layers) * (4 * (token + 1))
    return {
        "trace": {"register_step_gram": gram},
        "score_inputs": {
            "full_logprob": full,
            "no_evidence_logprob": full + 1,
            "no_history_logprob": full - 2,
            "no_evidence_history_logprob": full - 4,
        },
    }


def _records(count: int = 3) -> list[dict]:
    return [
        {
            "sample_id": f"sample-{index}",
            "source_id": f"source-{index}",
            "task_type": "QA",
            "artifact": _artifact(tokens=2 + index),
        }
        for index in range(count)
    ]


def test_raw_scores_are_the_fixed_equations():
    artifact = _artifact(tokens=3)
    artifact["score_inputs"].update(
        full_logprob=np.array([-2.0, -3.0, -2.0]),
        no_evidence_logprob=np.array([-1.0, -4.0, -1.0]),
        no_history_logprob=np.array([-5.0, -2.0, -5.0]),
        no_evidence_history_logprob=np.array([-5.0, -6.0, -5.0]),
    )
    scores = raw_scores(artifact)
    np.testing.assert_allclose(scores["provenance_takeover"], [0.0, 0.0, np.log(4.0)])
    np.testing.assert_allclose(scores["evidence_bypass"], [1.0, -1.0, 1.0])
    np.testing.assert_allclose(scores["symmetric_route_capture"], [4.0, -2.0, 4.0])
    np.testing.assert_allclose(scores["unsupported_history_takeover"], [5.0, 1.0, 5.0])
    np.testing.assert_allclose(scores["confidence"], [2.0, 3.0, 2.0])


def test_provenance_score_uses_cross_layer_direction_persistence():
    artifact = _artifact(tokens=3)
    gram = artifact["trace"]["register_step_gram"]
    gram[:, 2, 0, :] = [[1.0, 0.9], [0.9, 1.0]]
    gram[:, 2, 1, :] = np.eye(2)
    score = raw_scores(artifact)["provenance_takeover"]
    np.testing.assert_allclose(score, [0.0, 0.0, np.log(1.0 / 1.9)])


def test_factorial_contrasts_match_symmetric_effect_equations():
    artifact = _artifact(tokens=3)
    full = artifact["score_inputs"]["full_logprob"]
    artifact["score_inputs"].update(
        no_evidence_logprob=full - 1,
        no_history_logprob=full - 2,
        no_evidence_history_logprob=full - 4,
    )
    np.testing.assert_allclose(factorial_contrasts(artifact), [[1.5, 2.5, -1]] * 3)


def test_scores_are_label_sealed_raw_and_have_explicit_validity():
    records = _records()
    first, metadata = score_records(records, seed=7)
    labeled = copy.deepcopy(list(reversed(records)))
    for record in labeled:
        record["label"] = np.ones(99, dtype=bool)
        record["artifact"]["label"] = np.zeros(99, dtype=bool)
    second, repeated = score_records(labeled, seed=7)

    assert metadata == repeated
    assert metadata["fit"] == metadata["calibration"] == "none"
    assert metadata["labels_used"] is False
    assert "raw fixed equations" in metadata["score_scale"]
    for sample_id, sample in first.items():
        for name in SCORE_NAMES:
            np.testing.assert_array_equal(sample[name], second[sample_id][name])
            np.testing.assert_array_equal(
                sample[f"{name}__valid"], second[sample_id][f"{name}__valid"]
            )
        assert sample["provenance_takeover"][0] == 0
        assert not sample["provenance_takeover__valid"][:2].any()
        assert sample["provenance_takeover__valid"][2:].all()
        np.testing.assert_array_equal(
            sample["detection_valid"],
            np.logical_and.reduce(
                tuple(sample[f"{name}__valid"] for name in SCORE_NAMES)
            ),
        )
        assert sample["evidence_bypass__valid"].all()
        assert sample["confidence__valid"].all()
        assert not sample["symmetric_route_capture__valid"][:2].any()
        assert not sample["unsupported_history_takeover__valid"][:2].any()


def test_path_artifacts_are_loaded_and_released_one_at_a_time(monkeypatch):
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
        return TrackingArtifact(_artifact(tokens=2 + int(record["sample_id"][-1])))

    monkeypatch.setattr(detect, "_load_artifact", load)
    records = [
        {
            "sample_id": f"sample-{index}",
            "source_id": f"source-{index}",
            "task_type": "QA",
            "path": f"sample-{index}.pt",
        }
        for index in range(5)
    ]
    scores, metadata = score_records(records)
    gc.collect()
    assert len(scores) == len(records)
    assert metadata["evaluated_tokens"] == sum(index for index in range(5))
    assert metadata["crossfit"] == "not_applicable"
    assert metadata["intrinsic_score_coverage"]["evidence_bypass"]["tokens"] == sum(
        2 + index for index in range(5)
    )
    assert metadata["comparison_coverage"]["tokens"] == sum(range(5))
    assert state == {"alive": 0, "peak": 1, "calls": len(records)}
