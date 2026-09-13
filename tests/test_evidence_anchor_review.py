"""Regression checks for the A/C semantic interface contract."""

from route_graph import audit_prompts as prompts
from route_graph.evidence_anchor import _make_edits, _source_anchor


class ScriptedReader:
    def __init__(self):
        self.calls = []

    def ask(self, instruction, payload, **kwargs):
        self.calls.append((instruction, payload))
        if instruction == prompts.SOURCE_QA:
            return {
                "answerability": "not_stated",
                "source_answer": None,
                "source_answer_quote": None,
                "evidence_quotes": [],
                "all_conditions_covered": False,
                "unresolved_candidates": [],
            }
        if instruction == prompts.VERIFY_SOURCE:
            return {
                "status": "not_stated",
                "all_premises_supported": False,
                "all_conditions_covered": False,
                "evidence_quotes": [],
                "unresolved_candidates": [],
            }
        if instruction == prompts.CHECK_EDIT:
            return {
                "grammatical": True,
                "only_target_changed": True,
                "conditions_preserved": True,
            }
        raise AssertionError("unexpected reader call")


def test_source_verification_does_not_receive_prior_answerability_label():
    reader = ScriptedReader()
    _source_anchor(reader, "The source does not give a duration.", "What duration?")
    _, verification_payload = reader.calls[1]
    # N needs an independent second source estimate, not a full first-pass
    # response whose answerability already says `not_stated`.
    assert "source_answer" not in verification_payload


def test_grounded_replacement_must_be_an_evidence_citation():
    reader = ScriptedReader()
    question = {
        "claim_quote": "The duration was 9.",
        "claim_span": [0, 19],
        "answer_span": [17, 18],
        "answer_quote": "9",
        "slot_type": "duration",
    }
    answer = {
        "source_answer": "8",
        "source_answer_quote": "7",  # present in source, but not the cited answer.
        "evidence_quotes": ["8"],
    }
    result = _make_edits(reader, "The source says 8; it also mentions 7.", question, answer, {"C": 0.9, "N": 0.0})
    assert result["status"] == "contrast_requires_rewrite"


def test_source_answer_quote_must_resolve_to_the_source():
    class BadAnswerQuoteReader(ScriptedReader):
        def ask(self, instruction, payload, **kwargs):
            self.calls.append((instruction, payload))
            if instruction == prompts.SOURCE_QA:
                return {
                    "answerability": "answerable",
                    "source_answer": "8",
                    "source_answer_quote": "invented answer",
                    "evidence_quotes": ["The duration is 8."],
                    "all_conditions_covered": True,
                    "unresolved_candidates": [],
                }
            if instruction == prompts.VERIFY_SOURCE:
                return {
                    "status": "supported",
                    "source_answer": "8",
                    "source_answer_quote": "invented answer",
                    "all_premises_supported": True,
                    "all_conditions_covered": True,
                    "evidence_quotes": ["The duration is 8."],
                    "unresolved_candidates": [],
                }
            return super().ask(instruction, payload, **kwargs)

    _, verification = _source_anchor(
        BadAnswerQuoteReader(), "The duration is 8.", "What duration?"
    )
    assert verification["status"] == "uncertain"
