from dataclasses import replace

import numpy as np
import pytest

from experiments.attention_mechanism_audit.artifacts import (
    ALIGNMENT,
    COUNTERFACTUAL_VARIANTS,
    OBJECTIVE,
    MechanismArtifact,
    file_sha256,
    load_artifact,
    save_artifact,
)


def mechanism_table() -> MechanismArtifact:
    token_sample = np.asarray(["s1", "s1", "s2", "s2", "s2"])
    swap = np.asarray([True, True, False, False, False])
    availability = np.ones((5, len(COUNTERFACTUAL_VARIANTS)), dtype=np.bool_)
    availability[:, 4:] = swap[:, None]
    token_feature = np.asarray(
        [
            [np.nan, -0.1, -0.2],
            [0.3, -0.2, -0.3],
            [np.nan, np.nan, np.nan],
            [0.4, np.nan, np.nan],
            [0.5, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    answer_names = (
        "drift_functional_history_to_grounding_log_ratio__layer_mean__mean",
        "counterfactual_evidence_bypass__mean",
    )
    return MechanismArtifact(
        sample_id=np.asarray(["s1", "s2"]),
        source_id=np.asarray(["a", "b"]),
        task_type=np.asarray(["QA", "QA"]),
        generator_model=np.asarray(["g1", "g2"]),
        prompt_length=np.asarray([3, 4], dtype=np.int32),
        response_length=np.asarray([2, 3], dtype=np.int32),
        answer_feature_names=answer_names,
        answer_feature=np.asarray([[0.3, -0.15], [0.45, np.nan]], dtype=np.float32),
        token_sample_id=token_sample,
        token_source_id=np.asarray(["a", "a", "b", "b", "b"]),
        token_index=np.asarray([0, 1, 0, 1, 2], dtype=np.int32),
        token_response_length=np.asarray([2, 2, 3, 3, 3], dtype=np.int32),
        response_token_id=np.asarray([11, 12, 21, 22, 23], dtype=np.int64),
        predictor_position=np.asarray([2, 3, 3, 4, 5], dtype=np.int32),
        cached_query_index=np.asarray([-1, 0, -1, 0, 1], dtype=np.int32),
        cached_route_available=np.asarray([False, True, False, True, True]),
        counterfactual_variant_available=availability,
        token_feature_names=(
            "drift_functional_history_to_grounding_log_ratio__layer_mean",
            "counterfactual_evidence_bypass",
            "counterfactual_swapped_evidence_delta",
        ),
        token_feature=token_feature,
        metadata={
            "labels_used": False,
            "audit_scope": "selected_samples",
            "alignment": ALIGNMENT,
            "objective": OBJECTIVE,
            "dataset_manifest_sha256": "a" * 64,
            "role_index_sha256": "b" * 64,
            "source_info_sha256": "c" * 64,
            "model_fingerprint": "d" * 64,
            "tokenizer_fingerprint": "e" * 64,
            "swap_assignment_sha256": "f" * 64,
            "attention_binding_sha256": "1" * 64,
            "attribution_seed_assignment_sha256": "2" * 64,
            "implementation_sha256": "0" * 64,
            "counterfactual_variants": list(COUNTERFACTUAL_VARIANTS),
            "answer_feature_directions": {
                answer_names[0]: "high",
                answer_names[1]: "high",
            },
            "primary_answer_feature_names": [answer_names[0]],
            "onset_feature_names": [
                "drift_functional_history_to_grounding_log_ratio__layer_mean"
            ],
            "mechanism_observability": {},
            "observer_generator_audit": {},
            "cache_replay_attention_binding": {
                "verified_every_answer": True,
                "absolute_tolerance": 0.005,
                "retained_endpoints_compared": 10,
                "diagonal_endpoints_compared": 10,
                "retained_max_abs_error": 0.0,
                "diagonal_max_abs_error": 0.0,
                "known_mass_max_abs_error": 0.0,
            },
            "functional_attribution": {
                "jacobian_estimator": "iid Rademacher Hutchinson diagonal VJP",
                "gradient_probe_count": 4,
            },
            "runtime": {
                "attention_implementation": "eager",
                "torch": "test",
                "transformers": "test",
            },
        },
    ).validate()


def test_artifact_round_trip_is_byte_deterministic(tmp_path):
    table = mechanism_table()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    save_artifact(first, table)
    save_artifact(second, table)

    assert first.read_bytes() == second.read_bytes()
    assert file_sha256(first) == file_sha256(second)
    loaded = load_artifact(first)
    np.testing.assert_array_equal(loaded.cached_query_index, table.cached_query_index)
    np.testing.assert_array_equal(
        loaded.counterfactual_variant_available,
        table.counterfactual_variant_available,
    )
    assert loaded.metadata == table.metadata


def test_artifact_rejects_same_token_route_and_fabricated_missing_swap():
    table = mechanism_table()
    wrong_query = table.cached_query_index.copy()
    wrong_query[1] = 1
    with pytest.raises(ValueError, match="cached queries"):
        replace(table, cached_query_index=wrong_query).validate()

    fabricated = table.token_feature.copy()
    fabricated[2, 1] = 0.0
    with pytest.raises(ValueError, match="swaps must remain NaN"):
        replace(table, token_feature=fabricated).validate()


def test_artifact_rejects_unavailable_cached_route_values_and_extra_answers():
    table = mechanism_table()
    wrong_route = table.token_feature.copy()
    wrong_route[0, 0] = 0.0
    with pytest.raises(ValueError, match="cached-query mechanisms"):
        replace(table, token_feature=wrong_route).validate()

    with pytest.raises(ValueError, match="sample ID sets"):
        replace(
            table,
            sample_id=np.asarray(["s1", "s2", "extra"]),
            source_id=np.asarray(["a", "b", "c"]),
            task_type=np.asarray(["QA", "QA", "QA"]),
            generator_model=np.asarray(["g1", "g2", "g3"]),
            prompt_length=np.asarray([3, 4, 5]),
            response_length=np.asarray([2, 3, 1]),
            answer_feature=np.vstack((table.answer_feature, [[0.0, 0.0]])),
        ).validate()


def test_artifact_requires_frozen_directions_for_every_answer_feature():
    table = mechanism_table()
    metadata = dict(table.metadata)
    metadata["answer_feature_directions"] = {
        table.answer_feature_names[0]: "high"
    }
    with pytest.raises(ValueError, match="every answer feature"):
        replace(table, metadata=metadata).validate()
