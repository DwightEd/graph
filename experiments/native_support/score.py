"""A signed prompt term plus causal history reuse, in the same write budget."""

import numpy as np

NUMERIC_TERMS = (
    "attention_bias_score", "attention_reconstruction_error", "attention_add_roundoff",
    "mlp_add_roundoff", "norm_roundoff", "unembedding_roundoff", "output_bias_score",
)


def unit_vector(values):
    norm = np.linalg.norm(values)
    return values / norm if norm > 0 else np.zeros_like(values)


def write_budget(arrays):
    edges = np.asarray(arrays["edge_logit_write"], dtype=np.float64)
    ffns = np.asarray(arrays["mlp_score"], dtype=np.float64)
    numerical = sum(np.abs(arrays[name]).sum() for name in NUMERIC_TERMS)
    other = np.abs(ffns).sum() + abs(float(arrays["initial_score"])) + numerical
    return edges, ffns, float(np.abs(edges).sum() + other), float(other)


def head_fingerprint(edges, groups):
    """Physical [layer, head, source, sign] axes survive until the dot product."""
    parts = []
    for group in range(3):
        selected = edges[..., groups == group]
        parts.extend((np.maximum(selected, 0).sum(-1), np.maximum(-selected, 0).sum(-1)))
    return unit_vector(np.stack(parts, axis=-1).reshape(-1)).astype(np.float32)


def support_step(arrays, prompt_length, previous_support, previous_fingerprints):
    """s[t] = prompt[t] + sum_j weight[t,j] * s[j], with j strictly before t.

    This is a detector-defined graph, not a causal attribution of hidden provenance.
    FFN/embedding/special terms consume budget; they are not labelled hallucinations.
    """
    edges, ffns, budget, other = write_budget(arrays)
    groups = arrays["group_ids"]
    fingerprint = head_fingerprint(edges, groups)
    prompt_net = edges[..., groups == 0].sum()
    history = np.maximum(edges[..., prompt_length:], 0).sum(axis=(0, 1))
    history *= groups[prompt_length:] == 1
    similarity = np.empty(0)
    if len(previous_support):
        similarity = np.clip(np.asarray(previous_fingerprints) @ fingerprint, 0, 1)
    weights = history * similarity
    direct = 0.0
    if budget > 0:
        direct = float(prompt_net / budget)
        weights /= budget
    inherited = float(weights @ np.asarray(previous_support))
    support = direct + inherited
    diagnostics = support_diagnostics(arrays, edges, ffns, budget, other)
    diagnostics.update(
        direct_risk=-direct, inherited_risk=-inherited, risk=-support,
        support=support, history_weight=float(weights.sum()),
        history_similarity=float(similarity.mean()) if len(similarity) else 0.0,
        history_parents=len(weights),
    )
    return diagnostics, fingerprint, weights


def support_diagnostics(arrays, edges, ffns, budget, other):
    groups = arrays["group_ids"]
    prompt = edges[..., groups == 0]
    positive = np.maximum(edges, 0).sum() + np.maximum(ffns, 0).sum()
    negative = np.maximum(-edges, 0).sum() + np.maximum(-ffns, 0).sum()
    signed_total = positive + negative
    return {
        "write_budget": budget,
        "non_attention_fraction": float(other / budget) if budget else 0.0,
        "special_fraction": float(np.abs(edges[..., groups == 2]).sum() / budget) if budget else 0.0,
        "prompt_positive": float(np.maximum(prompt, 0).sum()),
        "prompt_negative": float(np.maximum(-prompt, 0).sum()),
        "ffn_positive": float(np.maximum(ffns, 0).sum()),
        "ffn_negative": float(np.maximum(-ffns, 0).sum()),
        "cancellation": float(2 * min(positive, negative) / signed_total) if signed_total else 0.0,
        "prompt_read": float(arrays["attention"][..., groups == 0].sum(-1).mean()),
        "entropy": float(arrays["entropy"]), "surprisal": float(arrays["surprisal"]),
        "logit_gap": float(arrays["logit_gap"]), "ledger_error": float(arrays["ledger_error"]),
    }


def prompt_focus(attention, prompt_length, previous_attention, width=8):
    """Same source window in current/past rows; no local-to-distant prerequisite."""
    prompt = np.asarray(attention[..., :prompt_length], dtype=np.float64)
    width = min(width, prompt_length)
    cumulative = np.pad(prompt.cumsum(-1), ((0, 0), (0, 0), (1, 0)))
    mass = cumulative[..., width:] - cumulative[..., :-width]
    starts = mass.argmax(-1)
    peak = np.take_along_axis(mass, starts[..., None], axis=-1)[..., 0]
    gain = np.full_like(peak, np.nan)
    if previous_attention:
        previous = np.mean(previous_attention, axis=0)
        cumulative = np.pad(previous.cumsum(-1), ((0, 0), (0, 0), (1, 0)))
        mass = cumulative[..., width:] - cumulative[..., :-width]
        past_same_source = np.take_along_axis(mass, starts[..., None], axis=-1)[..., 0]
        gain = peak - past_same_source
    return {"focus_start": starts, "focus_mass": peak, "focus_gain": gain}
