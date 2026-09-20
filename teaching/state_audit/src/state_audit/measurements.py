"""Pure array calculations. No files, models, labels, or fitted detector."""

import numpy as np


def source_masks(answer: dict, length: int) -> dict:
    keys = np.arange(length)
    ordinary = ~np.isin(answer["token_ids"][:length], answer["special_token_ids"])
    prompt = answer["prompt_length"]
    sources = np.full(length, -1, dtype=int)
    sources[:prompt] = answer["key_sources"]
    return dict(
        ordinary=ordinary,
        evidence=ordinary & (sources >= 0),
        history=ordinary & (keys >= prompt),
        other=ordinary & (keys < prompt) & (sources < 0),
        sources=sources,
    )


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    result = np.full(denominator.shape, np.nan)
    return np.divide((left * right).sum(-1), denominator, out=result, where=denominator > 0)


def source_messages(trace: dict, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Native A·V·W_O for one source group; do not renormalize attention."""
    attention = trace["attention"]
    heads, _, _ = attention.shape
    values = np.repeat(trace["value"], heads // trace["value"].shape[0], axis=0)
    readout = np.einsum("hts,hsd->thd", attention * mask, values)
    projection = weights.reshape(weights.shape[0], heads, values.shape[-1])
    return np.einsum("thd,ohd->tho", readout, projection)


def measure_heads(trace: dict, weights: np.ndarray, answer: dict) -> dict:
    attention = trace["attention"]
    masks = source_masks(answer, attention.shape[-1])
    mass = (attention * masks["ordinary"]).sum(-1)
    normalized = np.full_like(attention, np.nan)
    np.divide(
        attention * masks["ordinary"], mass[..., None], out=normalized, where=mass[..., None] > 0
    )
    result = {
        f"{group}_mass": (normalized * masks[group]).sum(-1).T
        for group in ("evidence", "history", "other")
    }
    result["ordinary_mass"] = mass.T
    result["attention_entropy"] = -(normalized * np.log(np.maximum(normalized, 1e-30))).sum(-1).T
    source_mass = [
        (normalized * (masks["sources"] == i)).sum(-1).T for i in range(len(answer["evidence"]))
    ]
    result["source_mass"] = np.stack(source_mass, -1) if source_mass else None
    evidence = source_messages(trace, weights, masks["evidence"])
    history = source_messages(trace, weights, masks["history"])
    result["evidence_write_norm"] = np.linalg.norm(evidence, axis=-1)
    result["history_write_norm"] = np.linalg.norm(history, axis=-1)
    result["evidence_history_cosine"] = cosine(evidence, history)
    norms = result["evidence_write_norm"].sum(-1)
    ratio = np.full(norms.shape, np.nan)
    np.divide(np.linalg.norm(evidence.sum(1), axis=-1), norms, out=ratio, where=norms > 0)
    result["evidence_head_cancellation"] = 1 - ratio
    return result


def reanchor_candidates(
    source_mass: np.ndarray | None, window: int, minimum_mass: float, minimum_rise: float
) -> list[dict]:
    """Rise at the SAME evidence source versus preceding queries; no look-ahead."""
    if source_mass is None:
        return []
    nodes = []
    for token in range(window, len(source_mass)):
        previous = source_mass[token - window : token].mean(0)
        rise = source_mass[token] - previous
        selected = (
            (source_mass[token] >= minimum_mass) & (rise >= minimum_rise) & (source_mass[token] > 0)
        )
        for head, source in zip(*np.where(selected)):
            nodes.append(
                dict(
                    target=int(token),
                    head=int(head),
                    source=int(source),
                    mass=float(source_mass[token, head, source]),
                    baseline=float(previous[head, source]),
                    rise=float(rise[head, source]),
                )
            )
    return nodes
