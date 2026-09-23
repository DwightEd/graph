"""Read signed provenance once, after propagation through the native value path."""

import numpy as np
from scipy.special import softmax, xlogy

from .routes import EPS, route_scores

METHODS = ("root_route", "direct_choice_route", "root_route_mean", "raw_route",
           "route_offline_mean", "raw_attention", "entropy")


def offline_mean(values):
    positions = np.arange(len(values))
    left = np.maximum(positions - 7, 0)
    right = np.minimum(positions + 9, len(values))
    cumulative = np.r_[0., np.cumsum(values)]
    return (cumulative[right] - cumulative[left]) / (right - left)


def choice_route(positive, negative, source_count):
    """History net minus source net / all unsigned action; candidate axis last."""
    net = positive - negative
    difference = net[..., source_count, :] - net[..., :source_count, :].sum(-2)
    magnitude = (positive + negative).sum(-2)
    return difference / np.maximum(magnitude, EPS)


def read_token(arrays, response, source_count, evidence):
    probability = softmax(arrays["candidate_logits"][1:].astype(float))
    positive, negative = arrays["root_positive"], arrays["root_negative"]
    root = float(choice_route(positive, negative, source_count) @ probability)
    # Sum physical heads at one layer only. Layers are separate readout sites.
    direct = choice_route(arrays["head_positive"].sum(1), arrays["head_negative"].sum(1), source_count)
    positions = np.arange(int(arrays["query"]) + 1)
    groups = (positions >= response["prompt_length"]).astype(int)
    groups[arrays["group_ids"] == source_count + 2] = 2
    magnitude = np.sqrt(arrays["edge_value_energy"])
    routing = route_scores(arrays["attention"], magnitude, groups, response["prompt_length"], evidence)
    prefix = "prompt_" if evidence is None else ""
    mass = (positive + negative) @ probability
    share = mass / max(mass.sum(), EPS)
    sources = mass[:source_count]
    distribution = sources / max(sources.sum(), EPS)
    return {"root_route": root, "direct_choice_route": float((direct @ probability).mean()),
        "raw_route": routing[prefix + "routing_imbalance"],
        "raw_attention": routing[prefix + "attention_displacement"],
        "entropy": float(arrays["entropy"]), "root_profile": share,
        "source_positive": float(positive[:source_count].sum(0) @ probability),
        "source_negative": float(negative[:source_count].sum(0) @ probability),
        "history_positive": float(positive[source_count] @ probability),
        "history_negative": float(negative[source_count] @ probability),
        "source_share": float(share[:source_count].sum()), "root_mass": float(mass.sum()),
        "source_entropy": float(-xlogy(distribution, distribution).sum()),
        "effective_sources": float(1 / max(np.square(distribution).sum(), EPS)) if sources.sum() else np.nan,
        "candidate_tail_mass": float(arrays["candidate_tail_mass"]),
        "ledger_error": float(np.abs(arrays["ledger_error"]).max())}


def score_answer(response, arrays, source_count, evidence):
    observed = [read_token(row, response, source_count, evidence) for row in arrays]
    scores = {name: np.asarray([row[name] for row in observed]) for name in observed[0]}
    for name in ("target", "query", "token_id"):
        scores[name] = np.asarray([row[name] for row in arrays])
    scores["root_route_mean"] = offline_mean(scores["root_route"])
    scores["route_offline_mean"] = offline_mean(scores["raw_route"])
    # Compare original source identities, never target-dependent candidate ranks.
    profile = scores.pop("root_profile")
    shift = np.square(np.diff(np.sqrt(profile), axis=0)).sum(-1) / 2
    scores["source_profile_shift"] = np.r_[np.nan, shift]
    scores["risk"] = scores["root_route"]
    return scores, profile


def token_rows(response, scores):
    rows = []
    for target in range(len(scores["token_id"])):
        values = {name: float(value[target]) for name, value in scores.items()
                  if name not in ("target", "query", "token_id")}
        rows.append({"response_id": response["id"], "source_id": response["source_id"],
            "target": target, "query": int(scores["query"][target]),
            "token_id": int(scores["token_id"][target]),
            "token": response["token_text"][response["prompt_length"] + target], **values})
    return rows


def source_rows(response, arrays, source_count):
    rows = []
    for current in arrays:
        for group in range(source_count + 3):
            role = "source" if group < source_count else ("history", "other", "special")[group - source_count]
            for choice, alternative in enumerate(current["candidate_ids"][1:]):
                positive = float(current["root_positive"][group, choice])
                negative = float(current["root_negative"][group, choice])
                rows.append({"response_id": response["id"], "target": int(current["target"]),
                    "source_group": group, "role": role,
                    "observed_id": int(current["token_id"]), "alternative_id": int(alternative),
                    "positive": positive, "negative": negative, "net": positive - negative})
    return rows
