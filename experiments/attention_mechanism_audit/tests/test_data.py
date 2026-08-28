import copy
import json

import pytest

from experiments.attention_mechanism_audit.data import (
    AuditBranch,
    AuditPair,
    TokenSpan,
    load_pairs,
)


def valid_record():
    return {
        "sample_id": "pair-1",
        "source_id": "source-1",
        "question_only": {
            "input_ids": [1, 40, 30, 31],
            "predictor_index": 3,
        },
        "context_a": {
            "input_ids": [1, 10, 20, 11, 21, 30, 31],
            "predictor_index": 6,
        },
        "context_b": {
            "input_ids": [1, 11, 21, 10, 20, 30, 31],
            "predictor_index": 6,
        },
        "relevant_span": [1, 3],
        "irrelevant_span": [3, 5],
        "history_span": [5, 7],
        "candidate_a_token_id": 70,
        "candidate_b_token_id": 71,
        "decision_prefix_is_neutral": True,
    }


def test_load_pairs_exposes_direct_token_contract(tmp_path):
    path = tmp_path / "audit_pairs.jsonl"
    path.write_text(json.dumps(valid_record()) + "\n", encoding="utf-8")

    pairs = load_pairs(path)

    assert len(pairs) == 1
    pair = pairs[0]
    assert isinstance(pair, AuditPair)
    assert pair.question_only == AuditBranch((1, 40, 30, 31), 3)
    assert pair.relevant_span == TokenSpan(1, 3)
    assert pair.history_span == TokenSpan(5, 7)
    assert pair.candidate_a_token_id == 70
    assert pair.candidate_b_token_id == 71
    assert pair.decision_prefix_is_neutral


def test_pair_requires_equal_position_aligned_contexts():
    record = valid_record()
    record["context_b"]["input_ids"].insert(0, 99)
    record["context_b"]["predictor_index"] += 1
    with pytest.raises(ValueError, match="equal length"):
        AuditPair.from_json(record)

    record = valid_record()
    record["context_b"]["input_ids"][0] = 99
    with pytest.raises(ValueError, match="only inside the value slots"):
        AuditPair.from_json(record)


def test_fact_swap_must_have_equal_capacity_and_exchange_exact_tokens():
    record = valid_record()
    record["irrelevant_span"] = [3, 4]
    with pytest.raises(ValueError, match="equal capacity"):
        AuditPair.from_json(record)

    record = valid_record()
    record["context_b"]["input_ids"][3:5] = [11, 21]
    with pytest.raises(ValueError, match="exactly exchange"):
        AuditPair.from_json(record)

    record = valid_record()
    # The union still has the same token multiset, but the two controls are
    # not token-for-token swaps of one another.
    record["context_b"]["input_ids"][1:5] = [11, 20, 10, 21]
    with pytest.raises(ValueError, match="exactly exchange"):
        AuditPair.from_json(record)


def test_all_branches_must_share_the_exact_answer_history():
    record = valid_record()
    record["context_b"]["input_ids"][-1] = 32
    with pytest.raises(ValueError, match="same answer history"):
        AuditPair.from_json(record)

    record = valid_record()
    record["question_only"]["input_ids"][-2] = 32
    with pytest.raises(ValueError, match="question_only must end"):
        AuditPair.from_json(record)


def test_spans_define_disjoint_prompt_facts_and_history_suffix():
    record = valid_record()
    record["irrelevant_span"] = [2, 5]
    with pytest.raises(ValueError, match="overlaps"):
        AuditPair.from_json(record)

    record = valid_record()
    record["history_span"] = [5, 6]
    with pytest.raises(ValueError, match="end at the predictor"):
        AuditPair.from_json(record)

    record = valid_record()
    record["history_span"] = [6, 7]
    with pytest.raises(ValueError, match="key before the predictor"):
        AuditPair.from_json(record)


def test_predictor_and_first_divergence_candidates_are_explicit():
    record = valid_record()
    record["question_only"]["predictor_index"] = 2
    with pytest.raises(ValueError, match="final supplied token"):
        AuditPair.from_json(record)

    record = valid_record()
    record["candidate_b_token_id"] = record["candidate_a_token_id"]
    with pytest.raises(ValueError, match="must differ"):
        AuditPair.from_json(record)


def test_shared_decision_prefix_must_be_declared_neutral():
    record = valid_record()
    record["decision_prefix_is_neutral"] = False
    with pytest.raises(ValueError, match="declared neutral"):
        AuditPair.from_json(record)


def test_a_and_b_are_a_symmetric_input_pair():
    record = valid_record()
    record["context_a"], record["context_b"] = (
        record["context_b"],
        record["context_a"],
    )
    record["candidate_a_token_id"], record["candidate_b_token_id"] = (
        record["candidate_b_token_id"],
        record["candidate_a_token_id"],
    )

    pair = AuditPair.from_json(record)

    assert pair.context_a.input_ids[1:3] == (11, 21)
    assert pair.candidate_a_token_id == 71


def test_loader_rejects_legacy_text_fields_and_duplicate_ids(tmp_path):
    record = valid_record()
    record["prompt"] = "do not reconstruct this"
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain exactly"):
        load_pairs(path)

    first = valid_record()
    duplicate = copy.deepcopy(first)
    path = tmp_path / "duplicate.jsonl"
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(duplicate) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_pairs(path)
