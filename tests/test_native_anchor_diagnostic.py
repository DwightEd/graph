"""The read-only diagnostic must retain only valid reader condition tables."""

import json

import pytest

from experiments.diagnose_native_anchors import diagnose
from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest


def _run(tmp_path, questions):
    row = {
        "id": "1",
        "source_id": "source-1",
        "task": "QA",
        "response": "alpha beta gamma",
    }
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text(json.dumps(row) + "\n")
    settings = {"input_path": str(inputs), "input_sha256": file_sha256(inputs)}
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    data = {
        "words": [
            {"start": 0, "end": 5, "coverage_state": "assertion", "abstain": True},
            {"start": 6, "end": 10, "coverage_state": "coverage_unknown", "abstain": True},
            {"start": 11, "end": 16, "coverage_state": "nonassertion", "abstain": False},
        ],
        "questions": questions,
    }
    artifact = {
        "settings_sha256": digest(settings),
        "row_sha256": digest(row),
        "content_sha256": digest(data),
        "upstream": {},
        "data": data,
    }
    (tmp_path / "A").mkdir()
    (tmp_path / "A" / "1.json").write_text(json.dumps(artifact))
    return row, artifact


def _question(role, *, evidence_valid=True):
    return {
        "id": role,
        "status": "uncertain",
        "claim_span": [0, 10],
        "atomic_target_role": role,
        "source_answer": {
            "atomic_condition_binding": {"valid": True},
            "condition_table_valid": evidence_valid,
            "condition_checks": [
                {"condition_role": "time", "status": "missing"}
            ],
        },
    }


def test_only_evidence_valid_tables_produce_multi_mask_stat_and_word_union(tmp_path):
    _run(tmp_path, [_question("subject"), _question("object")])

    result = diagnose(tmp_path)

    assert result["counts"]["multiple_masks_with_non_target_failure_spans"] == 1
    assert result["counts"]["multiple_masks_with_non_target_failure_words"] == 2
    assert result["counts"]["all_words"] == 3
    assert result["counts"]["word_coverage_unknown"] == 1
    assert result["counts"]["semantic_scored_words"] == 1
    assert result["interpretation"].startswith("Reader-reported")


def test_structurally_bound_but_evidence_invalid_tables_are_not_diagnostic_failures(tmp_path):
    _run(
        tmp_path,
        [_question("subject", evidence_valid=False), _question("object", evidence_valid=False)],
    )

    result = diagnose(tmp_path)

    assert result["counts"]["multiple_masks_with_non_target_failure_spans"] == 0
    assert result["counts"]["multiple_masks_with_non_target_failure_words"] == 0
    assert result["counts"]["all_words"] == 3
    assert result["counts"]["word_coverage_unknown"] == 1


def test_rejects_hash_mismatched_a_artifact_and_input_roster(tmp_path):
    _row, artifact = _run(tmp_path, [_question("subject")])
    original = dict(artifact)
    artifact["content_sha256"] = "invalid"
    (tmp_path / "A" / "1.json").write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="A artifact identity/hash"):
        diagnose(tmp_path)

    (tmp_path / "A" / "1.json").write_text(json.dumps(original))
    with (tmp_path / "inputs.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="input roster hash"):
        diagnose(tmp_path)
