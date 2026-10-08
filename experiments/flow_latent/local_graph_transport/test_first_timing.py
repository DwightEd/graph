"""Candidate timing and source-program pairing invariants."""
import numpy as np
import pytest
from collections import defaultdict

from .first_timing import timing_records, timing_source_groups, timing_metrics
from .test_source_choices import programs
from .run_first_timing import append_compact


def source_programs():
    rows = programs()
    for row in rows:
        row['prompt_with_source'] = [100, 200]
    return rows


def test_prechoice_controls_have_same_past_and_postcandidate_uses_actual_token():
    records = timing_records(source_programs())
    assert [row['label'] for row in records] == [0, 1, 0, 1]
    assert [row['candidate_id'] for row in records] == [10, 20, 20, 10]
    assert [row['id'] for row in records] == ['s_original_0', 's_original_1', 's_swapped_1', 's_swapped_0']
    assert all(row['first_divergence'] == 1 for row in records)
    assert len(timing_source_groups(records)) == 1


def test_different_prechoice_history_is_rejected():
    rows = source_programs()
    rows[1]['answer_ids'][0] = 999
    with pytest.raises(ValueError, match='same past prefix'):
        timing_records(rows)


def test_timing_metrics_keep_candidate_and_source_swap_directions():
    records = timing_records(source_programs())
    scores = {row['id']: -1. if row['label'] == 0 else 1. for row in records}
    result = timing_metrics(records, scores)
    assert result['auroc'] == 1.
    assert result['candidate_ranking_rate'] == 1.
    assert result['source_swap']['direction_rate'] == 1.


def test_postcandidate_selects_next_node_and_the_actual_candidate_embedding():
    fields = dict(node_fields=np.arange(4)[:, None, None] * np.ones((4, 6, 8)),
        boundary_fields=np.zeros((4, 10, 8)), value_fields=np.zeros((4, 2, 8)),
        local_attention=np.zeros((2, 4, 2, 8)), indices=np.zeros((4, 8), dtype=np.int64),
        valid=np.zeros((4, 8), dtype=bool), scalars=np.zeros((4, 8)))
    base = dict(first_divergence=1, candidate_id=3)
    embedding = np.eye(8, dtype=np.float32)
    for timing, expected in (('prechoice', 1), ('postcandidate', 2)):
        parts, records = defaultdict(list), []
        append_compact(parts, records, base, fields, embedding, timing, 0, 0)
        assert records[0]['observer_row'] == expected
        assert np.all(parts['node_fields'][0] == expected)
        np.testing.assert_array_equal(parts['candidate_vectors'][0], embedding[[3]])
