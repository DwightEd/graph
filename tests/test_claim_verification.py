"""Checks that affect the interpretation of span-level pilot results."""

import json

import numpy as np

from experiments.native_support.claim_verification.text import (
    flatten_source, parse_claims, project_scores, sentences,
)


def test_partition_retains_decimals_and_all_characters():
    text = "Score 3.5 stars. Open 9:00–22:30.\nNo WiFi."
    units = sentences(text)
    assert "".join(unit["text"] for unit in units) == text
    assert units[0]["text"] == "Score 3.5 stars. "


def test_exact_unique_quote_only_and_evidence_not_assumed_valid():
    raw = json.dumps(dict(claims=[
        dict(quote="private pension at 5%", claim="Private pension is taxed at 5%.", evidence="invented", alternative=""),
        dict(quote="5%", claim="Tax is 5%.", evidence="Tax", alternative=""),
    ]))
    claims, invalid = parse_claims(raw, "Salary at 5%; private pension at 5%", "Salary tax 5%")
    assert len(claims) == len(invalid) == 1
    assert not claims[0]["evidence_exact"]
    assert invalid[0]["reason"] == "nonunique_or_missing_quote"


def test_projection_preserves_gaps_and_reports_overlap():
    offsets = [(0, 4), (4, 8), (8, 12), (12, 16)]
    units = [dict(start=0, stop=8), dict(start=8, stop=16)]
    claims = [dict(start=5, stop=11), dict(start=8, stop=12)]
    direct, atomic, covered = project_scores(offsets, units, [-2, -3], claims, [4, 1])
    np.testing.assert_array_equal(direct, [-2, -2, -3, -3])
    np.testing.assert_array_equal(atomic, [-2, 4, 4, -3])
    np.testing.assert_array_equal(covered, [False, True, True, False])


def test_invalid_generation_retains_baseline():
    claims, invalid = parse_claims('{"claims": [', "No WiFi", "WiFi=no")
    assert not claims and invalid
    direct, atomic, covered = project_scores([(0, 2)], [dict(start=0, stop=2)], [3], claims, [])
    np.testing.assert_array_equal(direct, atomic)
    assert not covered.any()


def test_flatten_preserves_paths_types_and_empty_values():
    value = {"WiFi": "no", "Ambience": {"intimate": False}, "reviews": ["x", "y"], "empty": {}}
    leaves = [json.loads(line) for line in flatten_source(value)]
    assert dict(path=["Ambience", "intimate"], value=False) in leaves
    assert dict(path=["WiFi"], value="no") in leaves
    assert dict(path=["reviews", 1], value="y") in leaves
    assert dict(path=["empty"], value={}) in leaves


def test_official_labels_keep_character_overlap_and_unlabelled_tokens():
    from experiments.native_support.claim_verification.evaluate import label_record

    record = dict(text="No WiFi.", source_id="s", offsets=[(0, 2), (2, 5), (5, 7), (7, 8)])
    official = dict(response="No WiFi.", source_id="s", labels=[dict(start=3, end=7)])
    label_record(record, official)
    np.testing.assert_array_equal(record["labels"], [False, True, True, False])
    assert record["spans"][0]["text"] == "WiFi"


def test_zero_threshold_does_not_turn_a_tie_into_an_alarm():
    from experiments.native_support.claim_verification.evaluate import classification

    result = classification(np.asarray([True, False, False]), np.asarray([2., 0., -2.]))
    assert result["tp"] == 1 and result["fp"] == 0
    assert result["precision"] == result["recall"] == 1


def test_partial_group_cannot_open_official_labels():
    from types import SimpleNamespace
    import pytest
    from experiments.native_support.claim_verification.evaluate import evaluate

    with pytest.raises(ValueError, match="all regression and holdout"):
        evaluate(SimpleNamespace(group="regression"))


def test_blind_questions_map_in_sentence_not_another_occurrence():
    from experiments.native_support.claim_verification.blind import parse_questions

    text = "Salary 5%. Pension 5%."
    units = sentences(text)
    raw = json.dumps(dict(questions=[dict(sentence=1, quote="5%", question="How is pension taxed?")]))
    questions, invalid = parse_questions(raw, units)
    assert not invalid
    assert text[questions[0]["start"]:questions[0]["stop"]] == "5%"
    assert questions[0]["start"] == text.rindex("5%")


def test_blind_answer_ids_cannot_omit_or_duplicate_a_question():
    from experiments.native_support.claim_verification.blind import parse_answers

    questions = [{}, {}]
    assert parse_answers('{"answers":[{"id":0,"answer":"a"},{"id":0,"answer":"b"}]}', questions) is None
    assert parse_answers('{"answers":[{"id":1,"answer":"b"},{"id":0,"answer":"a"}]}', questions) == ["a", "b"]


def test_blind_checker_does_not_receive_answer_quote_or_original_text():
    from experiments.native_support.claim_verification.blind import answer_questions

    class CaptureReader:
        def generate(self, messages, max_new_tokens):
            self.messages = messages
            return []

    reader = CaptureReader()
    record = dict(source="Source says red.", text="CLAIMED_SECRET_VALUE")
    question = dict(quote="CLAIMED_SECRET_VALUE", question="What color is it?")
    answer_questions(reader, [record], [[question]])
    serialized = json.dumps(reader.messages)
    assert "CLAIMED_SECRET_VALUE" not in serialized
    assert "Source says red." in serialized
