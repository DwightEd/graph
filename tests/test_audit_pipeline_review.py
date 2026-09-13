"""CPU regression checks for orchestration semantics reviewed on 2026-09-13."""

import json
import os
from types import SimpleNamespace

import pytest

from route_graph.audit_output import merge_response
from route_graph.audit_runner import read_artifact, settings
from route_graph.audit_semantics import label_fixed_pool
from route_graph.frozen_reader import digest


class Reader:
    def ask(self, instruction, payload, **_):
        del instruction, payload
        return {
            "candidates": [
                {
                    "id": "source-a",
                    "relation": "unrelated",
                    "evidence_quotes": [],
                    "conditions_covered": False,
                    "target_answerability": "not_stated",
                },
                {
                    "id": "source-b",
                    "relation": "uncertain",
                    "evidence_quotes": [],
                    "conditions_covered": False,
                    "target_answerability": "uncertain",
                },
            ]
        }


def test_null_requires_every_frozen_source_unit_to_be_not_stated():
    frozen = {
        "views": [
            {"id": "source-a", "role": "source", "keys": [1], "segments": ["A"]},
            {"id": "source-b", "role": "source", "keys": [2], "segments": ["B"]},
        ],
        "text_node_ids": ["source-a", "source-b"],
    }
    result = label_fixed_pool(
        Reader(),
        {"prompt": "AB", "source_span": [0, 2]},
        {"id": "question", "question": "What value?"},
        frozen,
    )
    assert result["null_source_unit_check"] is False


def test_supported_correction_is_recorded_as_recovery_not_error_continuation():
    prior = {
        "id": "prior",
        "claim_span": [0, 5],
        "answer_span": [0, 1],
        "status": "unsupported",
        "localization": "slot",
        "risk": 0.9,
        "effective_scores": {"C": 0.9, "N": 0.0},
    }
    current = {
        "id": "current",
        "claim_span": [6, 11],
        "answer_span": [6, 7],
        "previous_claim_span": [0, 5],
        "status": "supported",
        "localization": "slot",
        "risk": 0.1,
        "effective_scores": {"E": 0.9, "C": 0.0, "N": 0.0},
        "audit_role": "supported_recovery_control",
    }
    validation = {
        "positions": [
            {
                "id": "history-position",
                "passed": True,
                "delta": 0.4,
                "group": {"kind": "content", "role": "history"},
            }
        ],
        "origins": {
            "history-position": {
                "passed": True,
                "delta": 0.3,
                "origin": [1],
                "origin_selection": "mapped_previous_answer_slot",
            }
        },
    }
    output = merge_response(
        {
            "id": "1",
            "source_id": "s",
            "task": "QA",
            "generator": "g",
            "response_sha256": "x",
            "response": "wrong right",
        },
        {
            "questions": [prior, current],
            "words": [
                {"start": 0, "end": 5, "coverage_state": "assertion"},
                {"start": 6, "end": 11, "coverage_state": "assertion"},
            ],
            "risk_claims_total": 1,
            "risk_claims_with_valid_contrast": 1,
        },
        {"current": {"frozen": {"finalists": []}}},
        {
            "prior": {},
            "current": {
                "relation_check": {
                    "valid": True,
                    "relation": "correction",
                    "slot_link_valid": True,
                    "previous_question_id": "prior",
                    "previous_answer_span": [0, 1],
                },
                "null_source_unit_check": True,
            },
        },
        {"prior": {}, "current": validation},
    )
    edge = output["graph"]["conditional_history_edges"][0]
    assert edge["type"] == "supported_recovery_dependency"
    assert edge["from_question_id"] == "prior"
    assert edge["from_answer_span"] == [0, 1]
    assert edge["from_token_positions"] == [1]
    assert "error_history_continuation" not in output["graph"]["claim_nodes"][1]["mechanisms"]
    assert output["coverage"]["word_states"] == {"assertion": 2}


def test_null_position_only_template_does_not_count_internal_certificate():
    question = {
        "id": "missing",
        "claim_span": [0, 5],
        "answer_span": [0, 5],
        "answer_quote": "three",
        "claim_quote": "three",
        "status": "unsupported",
        "localization": "slot",
        "risk": 0.9,
        "effective_scores": {"C": 0.0, "N": 0.9},
        "audit_role": "risk_claim",
    }
    output = merge_response(
        {
            "id": "1",
            "source_id": "s",
            "task": "QA",
            "generator": "g",
            "response_sha256": "x",
            "response": "three",
        },
        {
            "questions": [question],
            "words": [{"start": 0, "end": 5, "coverage_state": "assertion"}],
            "risk_claims_total": 1,
            "risk_claims_with_valid_contrast": 1,
        },
        {"missing": {"frozen": {"finalists": []}}},
        {"missing": {"relation_check": {}, "null_source_unit_check": True}},
        {
            "missing": {
                "status": "validated",
                "positions": [
                    {
                        "id": "position-only",
                        "passed": True,
                        "delta": 0.8,
                        "group": {"kind": "content", "role": "source"},
                        "semantic": {"relation": "nonapplicable"},
                    }
                ],
                "origins": {"position-only": {"passed": False}},
                "template": {
                    "passed": True,
                    "origin_passed": False,
                    "primary_id": "position-only",
                },
            }
        },
    )
    claim = output["graph"]["claim_nodes"][0]
    assert claim["native_certificate_ids"] == []
    assert claim["raw_position_certificate_ids"] == ["position-only"]
    assert output["coverage"]["risk_claims_with_internal_certificate"] == 0
    assert output["coverage"]["risk_claims_with_resolved_mechanism"] == 0


def test_resume_rejects_a_phase_artifact_without_its_fixed_upstream(tmp_path):
    row = {"id": "1", "response": "text"}
    data = {"status": "proposed"}
    path = tmp_path / "B" / "1.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "settings_sha256": "settings",
                "row_sha256": digest(row),
                "content_sha256": digest(data),
                "upstream": {},
                "data": data,
            }
        )
    )
    with pytest.raises(ValueError, match="exact required upstream"):
        read_artifact(tmp_path, "B", row, "settings")


def test_resume_rehashes_model_bytes_even_if_metadata_is_restored(tmp_path):
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("{}\n")
    observer, reader = tmp_path / "observer", tmp_path / "reader"
    observer.mkdir()
    reader.mkdir()
    for directory in (observer, reader):
        (directory / "config.json").write_bytes(b"a")
    args = SimpleNamespace(
        output=tmp_path / "output",
        inputs=inputs,
        observer_model=observer,
        reader_model=reader,
    )
    settings(args)
    target = observer / "config.json"
    stat = target.stat()
    target.write_bytes(b"b")
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="bytes or metadata changed"):
        settings(args)
