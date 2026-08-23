import numpy as np
import pytest

from experiments.non_neural_structure_audit.evaluation import (
    align_query_to_next_token,
    pre_onset_slope,
)


def test_post_token_query_is_aligned_to_the_next_generated_token():
    score = np.asarray([0.1, 0.8, 0.2], dtype=np.float32)
    labels = np.asarray([0, 0, 1], dtype=np.int8)
    eligible = np.asarray([True, True, True])

    aligned = align_query_to_next_token(score, labels, eligible)

    np.testing.assert_array_equal(aligned.token_index, [1, 2])
    np.testing.assert_array_equal(aligned.labels, [0, 1])
    np.testing.assert_allclose(aligned.score, [0.1, 0.8])


def test_pre_onset_slope_never_reads_the_error_token_query():
    score = np.asarray([0.0, 1.0, 2.0, 100.0], dtype=np.float32)
    labels = np.asarray([0, 0, 0, 1], dtype=np.int8)

    slope = pre_onset_slope(score, labels, np.ones(4, dtype=bool), window=3)

    assert slope == pytest.approx(1.0)
