from __future__ import annotations

import numpy as np

from experiments.attention_operator_validation.artifacts import (
    FeatureTable,
    implementation_sha256,
)
from experiments.attention_operator_validation.evaluate import frozen_evaluation_spec


FEATURE_NAMES = (
    "prompt_mass_mean",
    "prompt_code_effective_heads_mean",
    "identity_control",
    "operator_raw_control",
    "operator_normalized_control",
    "operator_permuted_control",
)


def make_table(*, manifest_sha256: str = "a" * 64) -> FeatureTable:
    directions, groups = frozen_evaluation_spec(FEATURE_NAMES)
    return FeatureTable(
        sample_id=np.asarray(("sample-a", "sample-b"), dtype=str),
        source_id=np.asarray(("source-a", "source-b"), dtype=str),
        task_type=np.asarray(("QA", "QA"), dtype=str),
        response_length=np.asarray((2, 3), dtype=np.int32),
        feature_names=FEATURE_NAMES,
        feature=np.arange(12, dtype=np.float32).reshape(2, 6),
        metadata={
            "labels_used": False,
            "audit_scope": "selected_samples",
            "implementation_sha256": implementation_sha256(),
            "dataset_manifest_sha256": manifest_sha256,
            "operator_sha256": "b" * 64,
            "data_root": "/frozen/test",
            "split": "test",
            "task": "QA",
            "feature_directions": directions,
            "probe_groups": groups,
        },
    )


__all__ = ["FEATURE_NAMES", "make_table"]
