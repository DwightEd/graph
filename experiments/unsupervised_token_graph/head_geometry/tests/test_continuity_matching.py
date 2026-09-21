"""Matched controls preserve span identity, common coverage and lexical constraints."""

import numpy as np

from experiments.unsupervised_token_graph.head_geometry.continuity_matching import (
    match_spans,
)


def block_with_spans(gold, length=24, pieces=None):
    pieces = pieces or [f"word{index} " for index in range(length)]
    ends = np.cumsum([len(piece) for piece in pieces])
    offsets = np.column_stack([np.r_[0, ends[:-1]], ends])
    error = np.zeros(length, dtype=bool)
    for start, end in gold:
        error[start:end] = True
    return {
        "record": {"id": "answer", "task": "QA", "source_id": "source", "generator": "model"},
        "text": "".join(pieces), "offsets": offsets, "token_ids": np.arange(length),
        "gold": gold, "views": {"all_error": (error, np.ones(length, bool))},
        "scores": {"raw": np.arange(length, dtype=float)},
    }


def test_matching_uses_no_scores_and_preserves_original_gold():
    block = block_with_spans([(8, 11), (16, 18)])
    common = np.ones(24, bool)
    first = match_spans(block, common)
    block["scores"]["raw"] = np.full(24, np.nan)
    second = match_spans(block, common)
    assert first == second
    pairs, unmatched, diagnostics = first
    assert unmatched == []
    assert [(row["error_start"], row["error_end"]) for row in pairs] == block["gold"]
    assert [(row["normal_start"], row["normal_end"]) for row in pairs] == [(5, 8), (14, 16)]
    assert [row["previous_error"] for row in pairs] == [False, True]
    assert diagnostics["entropy_controlled_pairs"] == 0
    assert all(row["entropy_gap"] is None for row in pairs)


def test_missing_score_in_error_span_does_not_shorten_or_move_gold():
    block = block_with_spans([(8, 11)])
    common = np.ones(24, bool)
    common[9] = False
    pairs, unmatched, _ = match_spans(block, common)
    assert pairs == []
    assert unmatched[0]["reason"] == "incomplete_common_coverage"
    assert (unmatched[0]["error_start"], unmatched[0]["error_end"]) == (8, 11)


def test_candidate_coverage_and_fixed_neighborhood_change_only_controls():
    block = block_with_spans([(8, 11)])
    common = np.ones(24, bool)
    common[7] = False
    pairs, _, _ = match_spans(block, common)
    assert pairs[0]["normal_start"] == 4
    pairs, _, diagnostics = match_spans(block, np.ones(24, bool), neighborhood=2)
    assert pairs[0]["normal_start"] == 3
    assert diagnostics["control_observation_neighborhood"] == 2


def test_surface_fallback_matches_exact_pieces_not_stripped_words():
    pieces = [f"w{index} " for index in range(24)]
    pieces[8:11] = ["repeat ", "repeat ", "other "]
    pieces[5:8] = ["normal ", "normal ", "end "]
    block = block_with_spans([(8, 11)], pieces=pieces)
    del block["token_ids"]
    pairs, unmatched, diagnostics = match_spans(block, np.ones(24, bool))
    assert not unmatched
    assert pairs[0]["normal_start"] == 5
    assert pairs[0]["error_repetition"] == 1 - 2 / 3
    assert pairs[0]["repetition_gap"] == 0
    assert diagnostics["repetition_basis"] == "surface_pieces"


def test_repeated_error_cannot_match_nonrepeated_normal_and_reports_rejection():
    block = block_with_spans([(8, 11)])
    block["token_ids"][8:11] = 100
    pairs, unmatched, _ = match_spans(block, np.ones(24, bool))
    assert pairs == []
    assert unmatched[0]["rejection_counts"]["repetition_gap_exceeds_limit"] > 0


def test_entropy_is_used_only_when_both_onsets_are_observed():
    block = block_with_spans([(8, 11)])
    block["entropy"] = np.zeros(24)
    block["entropy"][8] = 1
    pairs, unmatched, _ = match_spans(block, np.ones(24, bool))
    assert pairs == []
    assert unmatched[0]["rejection_counts"]["entropy_gap_exceeds_limit"] > 0
    block["entropy"][5] = np.nan
    pairs, _, diagnostics = match_spans(block, np.ones(24, bool))
    assert pairs[0]["normal_start"] == 5
    assert pairs[0]["entropy_gap"] is None
    assert diagnostics["entropy_available"]
    assert diagnostics["entropy_controlled_pairs"] == 0


def test_normal_intervals_are_never_reused_or_overlapped():
    block = block_with_spans([(8, 10), (13, 15), (15, 17)])
    pairs, _, _ = match_spans(block, np.ones(24, bool), position_limit=1)
    controls = [(pair["normal_start"], pair["normal_end"]) for pair in pairs]
    assert len(controls) == 3
    assert all(end <= other_start or other_end <= start
               for index, (start, end) in enumerate(controls)
               for other_start, other_end in controls[index + 1:])


def test_first_surface_type_and_prior_error_are_required():
    pieces = [f"word{index} " for index in range(24)]
    pieces[8] = "12 "
    pieces[12] = "34 "
    block = block_with_spans([(8, 11)], pieces=pieces)
    pairs, unmatched, _ = match_spans(block, np.ones(24, bool))
    assert pairs == []
    counts = unmatched[0]["rejection_counts"]
    assert counts["different_first_surface_kind"] > 0
    assert counts["different_previous_error_state"] > 0
