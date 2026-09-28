"""Source-first refinements: exact lexicographic ties or bounded rank residuals."""

import numpy as np

from .scalar import rank_scores, scalar_scores


ANCHORS = ("local", "full", "pair")
DETAILS = ("token_source", "source5", "route")
WEIGHTS = (.025, .1)
PRIMARY = "pair_source5_residual_0.1"
STRICT = "pair_source5_tie"


def unit_groups(pack):
    identity = np.column_stack((pack["answer_index"], pack["unit_index"]))
    return np.unique(identity, axis=0, return_inverse=True)[1]


def centered(values, groups):
    mean = np.bincount(groups, weights=values) / np.bincount(groups)
    return values - mean[groups]


def unit_window(values, pack, radius=2):
    """Offline mean within a saved unit and original target-distance radius.

    Missing special tokens are excluded, not replaced by more distant tokens.
    Pack order must be contiguous by answer/unit and increasing by target.
    """
    changes = ((np.diff(pack["answer_index"]) != 0) | (np.diff(pack["unit_index"]) != 0))
    cuts = np.r_[0, np.flatnonzero(changes) + 1, len(values)]
    result = np.empty(len(values), dtype=np.float64)
    for start, stop in zip(cuts[:-1], cuts[1:]):
        positions = pack["target"][start:stop]
        left = np.searchsorted(positions, positions - radius, side="left")
        right = np.searchsorted(positions, positions + radius, side="right")
        prefix = np.r_[0., np.cumsum(values[start:stop], dtype=np.float64)]
        result[start:stop] = (prefix[right] - prefix[left]) / (right - left)
    return result


def observations(pack):
    raw = scalar_scores(pack)
    raw["source5"] = unit_window(raw["token_source"], pack)
    return raw


def lexicographic_cutoff(anchor, detail, quantile=.95):
    order = np.lexsort((detail, anchor))
    index = order[int(np.ceil(quantile * (len(order) - 1)))]
    return dict(kind="lexicographic", anchor=float(anchor[index]), detail=float(detail[index]))


def lexicographic_scores(anchor, detail, cutoff):
    """Encode exact source order and detail ties, including a frozen cutoff.

    Integer blocks are built from query anchors AND the calibration cutoff,
    never training bins. Detail is in [-1,1], so different blocks cannot cross.
    Numeric values vary by cohort; pairwise ordering and cutoff decisions do not.
    Scores are rankings, not calibrated probabilities.
    """
    levels = np.unique(np.r_[anchor, cutoff["anchor"]])
    score = np.searchsorted(levels, anchor).astype(float) + .25 * detail
    threshold = float(np.searchsorted(levels, cutoff["anchor"]) + .25 * cutoff["detail"])
    return score, threshold


def score_candidates(pack, raw, references, calibration=None):
    ranked = rank_scores(raw, references)
    scores = dict(ranked)
    groups = unit_groups(pack)
    residuals = {name: centered(ranked[name], groups) for name in DETAILS}
    limits = {} if calibration is None else dict(calibration)
    thresholds = {}
    for anchor in ANCHORS:
        for detail in DETAILS:
            residual = residuals[detail]
            name = f"{anchor}_{detail}_tie"
            if calibration is None:
                limits[name] = lexicographic_cutoff(raw[anchor], residual)
            scores[name], thresholds[name] = lexicographic_scores(raw[anchor], residual, limits[name])
            for weight in WEIGHTS:
                scores[f"{anchor}_{detail}_residual_{weight:g}"] = ranked[anchor] + weight * residual
    for name, values in scores.items():
        if name in thresholds:
            continue
        if calibration is None:
            limits[name] = dict(kind="scalar", threshold=float(np.quantile(values, .95, method="higher")))
        thresholds[name] = limits[name]["threshold"]
    return scores, thresholds, limits
