"""Fixed-prefix candidate comparisons, independent of dataset and intervention type."""

import numpy as np

from .interventions import score_targets


def score_contrast(model, prefix_ids, candidates, operations=()):
    """Return token logp and first-minus-second margins; factual support is caller metadata."""
    if not prefix_ids or len(candidates) != 2 or any(not ids for ids in candidates):
        raise ValueError("A contrast needs a nonempty prefix and two nonempty candidates")
    if list(candidates[0]) == list(candidates[1]):
        raise ValueError("Candidate token sequences must differ")
    token_logp = []
    for candidate in candidates:
        answer = dict(
            prompt_length=len(prefix_ids),
            token_ids=[*prefix_ids, *candidate],
            response_ids=candidate,
        )
        scores = score_targets(model, answer, list(range(len(candidate))), operations)
        token_logp.append(scores["target_logp"])
    totals = [sum(values) for values in token_logp]
    means = [float(np.mean(values)) for values in token_logp]
    return dict(
        token_logp=token_logp,
        sum_logp=totals,
        mean_logp=means,
        sum_margin=totals[0] - totals[1],
        mean_margin=means[0] - means[1],
        candidate_lengths=[len(ids) for ids in candidates],
    )
