import numpy as np

from experiments.non_neural_structure_audit.features import FEATURE_NAMES
from experiments.non_neural_structure_audit.reference import (
    fit_reference,
    standardize,
)


def test_reference_uses_unlabeled_task_position_rows_and_has_a_scale_floor():
    values = np.arange(4 * 2 * len(FEATURE_NAMES), dtype=np.float32).reshape(
        4, 2, len(FEATURE_NAMES)
    )
    rows = [("QA", 0, values[0]), ("QA", 0, values[1])]

    reference = fit_reference(rows, minimum_scale=0.1)
    standardized = standardize(
        values[:1], task="QA", buckets=np.asarray([0]), reference=reference
    )

    assert np.all(reference.scale >= 0.1)
    assert standardized.shape == values[:1].shape
