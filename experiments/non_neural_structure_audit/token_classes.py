"""Evaluation-only token strata; all tokens remain in lineage propagation."""

from __future__ import annotations

import numpy as np


def content_token_mask(token_ids: np.ndarray, tokenizer) -> np.ndarray:
    text = [
        tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]
    return np.asarray([any(character.isalnum() for character in piece) for piece in text])
