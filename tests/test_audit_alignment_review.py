"""Regression checks for frozen B/A token-to-span accounting."""

import pytest

from route_graph.audit_alignment import slot_masks
from route_graph.causal_contrast import ContinuationContrast


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_slot_masks_rejects_text_that_does_not_match_frozen_branch_tokens():
    contrast = ContinuationContrast.from_sequences(
        [[0, ord("a"), ord("b")], [0, ord("a"), ord("c")]], 1
    )
    metadata = {
        # Same post-prefix lengths as the frozen branches, but different tokens.
        "original": "xy",
        "alternative": "zw",
        "answer_spans": [[1, 2], [1, 2]],
    }
    with pytest.raises(ValueError, match="token"):
        slot_masks(contrast, metadata, CharacterTokenizer())
