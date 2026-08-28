import numpy as np
import pytest

from experiments.attention_mechanism_audit.alignment import (
    align_cached_queries,
    gather_chosen_logits,
    predecessor_alignment,
)


def test_response_predictor_positions_use_predecessor_query():
    alignment = predecessor_alignment([1, 2, 3, 11, 12, 13], 3)

    np.testing.assert_array_equal(alignment.predictor_position, [2, 3, 4])
    np.testing.assert_array_equal(alignment.target_position, [3, 4, 5])
    np.testing.assert_array_equal(alignment.target_token_id, [11, 12, 13])


def test_cached_response_queries_mark_first_response_token_unavailable():
    alignment = predecessor_alignment([1, 2, 3, 11, 12, 13], 3)

    np.testing.assert_array_equal(alignment.cached_query_index, [-1, 0, 1])
    np.testing.assert_array_equal(
        alignment.cached_route_available, [False, True, True]
    )
    cached = np.asarray([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    shifted = align_cached_queries(cached, alignment)
    assert np.isnan(shifted[0]).all()
    np.testing.assert_array_equal(shifted[1:], cached[:-1])


def test_chosen_score_cannot_use_same_token_logit():
    alignment = predecessor_alignment([1, 2, 3, 11, 12, 13], 3)
    logits = np.zeros((6, 16), dtype=np.float64)
    logits[2, 11], logits[3, 12], logits[4, 13] = 2, 4, 6
    logits[3, 11], logits[4, 12], logits[5, 13] = 20, 40, 60

    np.testing.assert_array_equal(gather_chosen_logits(logits, alignment), [2, 4, 6])


def test_alignment_rejects_a_partial_or_extra_response_query_cache():
    with pytest.raises(ValueError, match="must equal the response length"):
        predecessor_alignment([1, 2, 3, 11, 12, 13], 3, cached_query_count=2)


@pytest.mark.parametrize("response_start", [0, 3, 4])
def test_alignment_requires_nonempty_prompt_and_response(response_start):
    with pytest.raises(ValueError, match="split prompt and response"):
        predecessor_alignment([1, 2, 3], response_start)
