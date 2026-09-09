import json

import numpy as np
import pytest

from experiments.reanchor_flow.message_dag.counterfactual_pairs import (
    CounterfactualPair,
)


class WordTokenizer:
    def __init__(self):
        self.vocabulary = {}

    def _encode(self, text):
        return [
            self.vocabulary.setdefault(token, len(self.vocabulary) + 1)
            for token in text.strip().split()
        ]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return [101] + self._encode(messages[0]["content"]) + [102]

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return self._encode(text)


def record():
    return {
        "pair_id": "entity-001",
        "constraint_kind": "entity",
        "prompt_plus": "Records Alice owns red. Question owner Alice",
        "prompt_minus": "Records Bob owns blue. Question owner Alice",
        "constraint_plus": "Alice owns red.",
        "constraint_minus": "Bob owns blue.",
        "response_prefix": "Based on the records the requested color is",
        "candidate_plus": "red",
        "candidate_minus": "blue",
        "reviewed": True,
        "source_id": "source-7",
        "dataset": "controlled",
    }


def test_pair_compilation_freezes_one_declared_edit_and_candidates():
    pair = CounterfactualPair.compile(record(), WordTokenizer())

    assert pair.pair_id == "entity-001"
    assert pair.constraint_kind == "entity"
    assert pair.response_start > max(pair.changed_positions)
    np.testing.assert_array_equal(
        pair.token_ids[0, pair.response_start:],
        pair.token_ids[1, pair.response_start:],
    )
    assert pair.candidate_plus_ids != pair.candidate_minus_ids
    assert pair.labels_used is False

    restored = CounterfactualPair.from_manifest(json.loads(json.dumps(pair.to_manifest())))
    np.testing.assert_array_equal(restored.token_ids, pair.token_ids)
    assert restored.study_identity == pair.study_identity


def test_pair_rejects_unreviewed_or_nonminimal_worlds_before_tokenization():
    invalid = record()
    invalid["reviewed"] = False
    with pytest.raises(ValueError, match="reviewed"):
        CounterfactualPair.compile(invalid, WordTokenizer())

    invalid = record()
    invalid["prompt_minus"] += " unrelated change"
    with pytest.raises(ValueError, match="outside"):
        CounterfactualPair.compile(invalid, WordTokenizer())

    invalid = record()
    invalid["prompt_plus"] += " Alice owns red."
    with pytest.raises(ValueError, match="exactly once"):
        CounterfactualPair.compile(invalid, WordTokenizer())


def test_pair_rejects_position_drift_and_tampered_frozen_tokens():
    invalid = record()
    invalid["constraint_minus"] = "Bob owns very blue."
    invalid["prompt_minus"] = "Records Bob owns very blue. Question owner Alice"
    with pytest.raises(ValueError, match="token-aligned"):
        CounterfactualPair.compile(invalid, WordTokenizer())

    pair = CounterfactualPair.compile(record(), WordTokenizer())
    frozen = pair.to_manifest()
    frozen["token_ids"][1][-1] += 1
    with pytest.raises(ValueError, match="response prefix"):
        CounterfactualPair.from_manifest(frozen)
