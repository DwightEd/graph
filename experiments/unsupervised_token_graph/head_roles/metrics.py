"""Paper Appendix A.3 scores, with explicit uninformative-swap diagnostics."""

import numpy as np


def cosine(left, right):
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide((left * right).sum(-1), denominator,
                     out=np.full(denominator.shape, np.nan), where=denominator > 0)


def swap_scores(before, after, temperature=.05):
    """Inputs [swap, head, 2] are ordinary-key block-average probabilities.

    Same-location comparison is positional; reversed reference is symbolic.
    All-zero blocks are missing observations, never perfect invariances.
    """
    positional = cosine(before, after)
    symbolic = cosine(before[..., ::-1], after)
    valid = np.isfinite(positional) & np.isfinite(symbolic)
    contrast = np.abs(before[..., 0] - before[..., 1])
    logits = contrast / temperature
    logits -= logits.max(axis=0, keepdims=True)
    weights = np.where(valid, np.exp(logits), 0.)
    total = weights.sum(0)
    weights = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)
    output = {}
    for name, values in (("positional", positional), ("symbolic", symbolic)):
        output[name] = (weights * np.nan_to_num(values)).sum(0)
        output[name][total == 0] = np.nan
    magnitude = before.sum(-1)
    relative = np.divide(contrast, magnitude, out=np.zeros_like(contrast), where=magnitude > 0)
    output["contrast"] = (weights * relative).sum(0)
    output["valid_swaps"] = valid.sum(0)
    output["gap"] = output["symbolic"] - output["positional"]
    return output


def choose_masks(layers, heads, gap, informative, fraction, seed, random_controls):
    """Remove only identifiable preferences; random controls match each layer's count."""
    count = len(layers) * len(heads)
    positional, symbolic = np.zeros(count, bool), np.zeros(count, bool)
    for layer_index in range(len(layers)):
        indices = layer_index * len(heads) + np.arange(len(heads))
        budget = int(np.floor(fraction * len(heads)))
        for selected, direction in ((positional, -1), (symbolic, 1)):
            eligible = indices[informative[indices] & (direction * gap[indices] > .05)]
            order = eligible[np.argsort(-direction * gap[eligible], kind="stable")]
            selected[order[:budget]] = True
    masks = dict(all=np.ones(count, bool), drop_positional=~positional, drop_symbolic=~symbolic)
    for control in range(random_controls):
        random = np.random.default_rng([seed, control])
        keep = np.ones(count, bool)
        for layer_index in range(len(layers)):
            indices = layer_index * len(heads) + np.arange(len(heads))
            removed = random.choice(indices, positional[indices].sum(), replace=False)
            keep[removed] = False
        masks[f"random_{control}"] = keep
    return masks
