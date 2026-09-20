"""Offline frozen-layer key swaps: fixed query and positions, moved key content."""

from itertools import combinations

import numpy as np

from .measurements import cosine, source_masks


def rotate(states: np.ndarray, cosine_values: np.ndarray, sine_values: np.ndarray):
    half = states.shape[-1] // 2
    perpendicular = np.concatenate((-states[..., half:], states[..., :half]), axis=-1)
    return states * cosine_values + perpendicular * sine_values


def attention_from_keys(trace: dict, keys: np.ndarray) -> np.ndarray:
    query = rotate(
        trace["query"], trace["cosine"][trace["queries"]], trace["sine"][trace["queries"]]
    )
    keys = rotate(keys, trace["cosine"], trace["sine"])
    keys = np.repeat(keys, len(query) // len(keys), axis=0)
    logits = np.einsum("htd,hsd->hts", query, keys) * trace["scale"]
    logits += trace["attention_bias"]
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    return weights / weights.sum(-1, keepdims=True)


def evidence_pairs(masks: dict, source_count: int, width: int = 8, limit: int = 8):
    pairs = []
    for left, right in combinations(range(source_count), 2):
        a = np.flatnonzero(masks["ordinary"] & (masks["sources"] == left))
        b = np.flatnonzero(masks["ordinary"] & (masks["sources"] == right))
        count = min(len(a), len(b), width)
        if count:
            pairs.append((a[:count], b[:count]))
    return pairs[:limit]


def swap_probe(trace: dict, answer: dict) -> list[dict]:
    base = attention_from_keys(trace, trace["key"])
    native_error = float(np.max(abs(base - trace["attention"])))
    np.testing.assert_allclose(base, trace["attention"], atol=2e-3, rtol=2e-2, equal_nan=False)
    masks = source_masks(answer, base.shape[-1])
    pairs = evidence_pairs(masks, len(answer["evidence"]))
    rows = []
    for pair_index, (left, right) in enumerate(pairs):
        changed_keys = trace["key"].copy()
        changed_keys[:, left], changed_keys[:, right] = (
            trace["key"][:, right],
            trace["key"][:, left],
        )
        changed = attention_from_keys(trace, changed_keys)
        # Destination RoPE and native causal/sliding masks are unchanged.
        before = np.stack([base[..., left].mean(-1), base[..., right].mean(-1)], -1)
        after = np.stack([changed[..., left].mean(-1), changed[..., right].mean(-1)], -1)
        probes = probe_rows(before, after, len(left), pair_index, native_error)
        for row in probes:
            row.update(left_keys=" ".join(map(str, left)), right_keys=" ".join(map(str, right)))
        rows.extend(probes)
    return rows


def probe_rows(before, after, width: int, pair_index: int, native_error: float) -> list[dict]:
    positional, symbolic = cosine(after, before), cosine(after, before[..., ::-1])
    total = before.sum(-1)
    contrast = np.zeros_like(total)
    np.divide(abs(before[..., 0] - before[..., 1]), total, out=contrast, where=total > 0)
    rows = []
    for head, token in np.ndindex(total.shape):
        mass = float(total[head, token] * width)
        rows.append(
            dict(
                target=token,
                head=head,
                pair=pair_index,
                positional=float(positional[head, token]),
                symbolic=float(symbolic[head, token]),
                swapped_mass=mass,
                relative_contrast=float(contrast[head, token]),
                identifiable=mass >= 0.01 and contrast[head, token] >= 0.1,
                before_left=float(before[head, token, 0]),
                before_right=float(before[head, token, 1]),
                after_left=float(after[head, token, 0]),
                after_right=float(after[head, token, 1]),
                native_max_error=native_error,
            )
        )
    return rows
