"""Token-level hallucination onset matching used only during evaluation."""

from __future__ import annotations

import numpy as np


def positive_onsets(label) -> np.ndarray:
    label = np.asarray(label, dtype=bool)
    return np.flatnonzero(label & ~np.r_[False, label[:-1]])


def match_onsets(label, token, *, window: int = 5, caliper: int = 32):
    label = np.asarray(label, dtype=bool)
    token = np.asarray(token)
    candidates = [
        index
        for index in range(window, len(label) - window)
        if not label[index - window : index + window + 1].any()
    ]
    used: set[int] = set()
    pairs = []
    for onset in positive_onsets(label):
        if onset < window or onset + window >= len(label):
            continue
        available = [
            index
            for index in candidates
            if index not in used and abs(index - onset) <= caliper
        ]
        if not available:
            continue
        same = [index for index in available if token[index] == token[onset]]
        control = min(same or available, key=lambda index: (abs(index - onset), index))
        used.add(control)
        pairs.append((int(onset), int(control)))
    return pairs
