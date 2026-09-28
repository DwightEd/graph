"""Label-free scalar readout, independent of graph models and CUDA."""

import numpy as np


def scalar_scores(pack):
    context, observed = pack["context"], pack["observations"]
    return dict(local=context[:, 0], full=context[:, 1], pair=context[:, :2].mean(axis=1),
                route=observed[:, 7],
                token_source=(context[:, :2] + observed[:, :2]).mean(axis=1))


def fit_ranks(scores):
    return {name: np.quantile(values, np.linspace(0, 1, 10001)) for name, values in scores.items()}


def rank_scores(scores, references):
    # Preserve the historical midrank and fixed fusion bit for bit.
    result = {}
    for name, values in scores.items():
        reference = references[name]
        left = np.searchsorted(reference, values, side="left")
        right = np.searchsorted(reference, values, side="right")
        result[name] = (left + right) / (2 * len(reference))
    result["fixed_unsupervised"] = .75 * result["pair"] + .25 * result["route"]
    return result
