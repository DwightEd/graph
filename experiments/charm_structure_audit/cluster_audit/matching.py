"""Select normal controls WITHOUT looking at detector scores or embeddings."""

import numpy as np

from .features import CALIPERS, STRUCTURE, marginal_windows, structure_at


TIERS = ("context", "cluster", "cluster_heads")


def merged_spans(spans):
    """Merge overlap in token space, but retain merely adjacent annotations."""
    result = []
    for start, end in sorted((int(a), int(b)) for a, b in spans):
        if result and start < result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def surface_class(text):
    text = text.strip()
    if not text:
        return "empty"
    if any(char.isdigit() for char in text):
        return "number"
    if any(char.isalpha() for char in text):
        return "word"
    return "punctuation"


def start_classes(sample):
    return np.asarray([surface_class(sample["response"][a:b]) for a, b in sample["offsets"]])


def repeat_fraction(token_ids, length):
    return np.asarray([1 - len(np.unique(token_ids[i:i + length])) / length
                       for i in range(len(token_ids) - length + 1)])


def candidates(sample, data, span, position_caliper=.25, repeat_caliper=.15):
    """Same answer, length, start surface type and prior-error exposure."""
    start, end = span
    length = end - start
    labels = sample["gold"].astype(bool)
    starts = np.arange(len(labels) - length + 1)
    prefix = np.r_[0, np.cumsum(labels)]
    valid_text = sample["offsets"][:, 1] > sample["offsets"][:, 0]
    bad_text = np.r_[0, np.cumsum(~valid_text)]
    normal = (prefix[length:] - prefix[:-length] == 0)
    normal &= bad_text[length:] - bad_text[:-length] == 0
    normal &= (prefix[starts] > 0) == (prefix[start] > 0)
    normal &= abs(starts - start) / len(labels) <= position_caliper
    classes = start_classes(sample)
    normal &= classes[starts] == classes[start]
    tokens = data["graph"]["token_ids"][data["prompt"]:]
    repeated = repeat_fraction(tokens, length)
    normal &= abs(repeated - repeated[start]) <= repeat_caliper
    return starts[normal], repeated


def pair_options(sample, data, span, config, cache):
    start, end = span
    length = end - start
    options, repeated = candidates(sample, data, span, config["position_caliper"],
                                    config["repeat_caliper"])
    if length not in cache:
        cache[length] = dict(structure={}, marginal=marginal_windows(data, length))
    stored = cache[length]
    for position in np.r_[start, options]:
        if int(position) not in stored["structure"]:
            stored["structure"][int(position)] = structure_at(data, int(position), length)
    target = stored["structure"][start]
    rows = []
    for normal in options:
        value = stored["structure"][int(normal)]
        delta = abs(value - target) / CALIPERS
        marginal = abs(stored["marginal"][normal] - stored["marginal"][start])
        rms = np.sqrt(np.mean(np.square(marginal), axis=1))
        maximum = marginal.max(axis=1)
        rows.append(dict(normal_start=int(normal), error_start=start, length=length,
                         cluster_ok=bool(np.all(delta <= config["caliper_multiplier"]) and target[0] > 0 and value[0] > 0),
                         heads_ok=bool(np.all(rms <= config["head_rms_caliper"]) and
                                       np.all(maximum <= config["head_max_caliper"])),
                         structure_distance=float(np.sqrt(np.mean(delta ** 2))),
                         position_gap=abs(int(normal) - start) / len(sample["gold"]),
                         repeat_gap=float(repeated[normal] - repeated[start]),
                         error_structure=target.tolist(), normal_structure=value.tolist(),
                         head_rms=rms.tolist(), head_max=maximum.tolist()))
    return rows


def choose_disjoint(options, tier, count):
    """Deterministic rarest-first matching; never reuse/overlap a normal token."""
    eligible = {}
    for start, rows in options.items():
        accepted = [r for r in rows if (tier == "context" or r["cluster_ok"]) and
                    (tier != "cluster_heads" or r["heads_ok"])]
        eligible[start] = sorted(accepted, key=lambda r: (
            r["position_gap"] if tier == "context" else r["structure_distance"],
            r["position_gap"], r["normal_start"]))
    used = np.zeros(count, bool)
    pairs, status = [], []
    for start in sorted(eligible, key=lambda s: (len(eligible[s]), s)):
        chosen = None
        for row in eligible[start]:
            normal, length = row["normal_start"], row["length"]
            if not used[normal:normal + length].any():
                chosen = row
                used[normal:normal + length] = True
                break
        status.append(dict(error_start=start, tier=tier, candidates=len(eligible[start]),
                           matched=chosen is not None,
                           reason="matched" if chosen else "no_candidate" if not eligible[start] else "normal_overlap_conflict"))
        if chosen is not None:
            pairs.append(dict(chosen, tier=tier))
    return pairs, status


def match_answer(sample, data, config):
    """Intervals are evaluation windows, NOT automatically discovered claims."""
    options, skipped, cache = {}, [], {}
    for start, end in merged_spans(sample["spans"]):
        if end - start < 2:
            skipped.append(dict(error_start=start, length=end - start, reason="singleton_has_no_internal_cluster"))
            continue
        if np.any(sample["offsets"][start:end, 1] <= sample["offsets"][start:end, 0]):
            skipped.append(dict(error_start=start, length=end - start, reason="nontext_token_in_error_span"))
            continue
        options[start] = pair_options(sample, data, [start, end], config, cache)
    all_pairs, all_status = [], []
    for tier in TIERS:
        pairs, status = choose_disjoint(options, tier, len(sample["gold"]))
        all_pairs.extend(pairs)
        all_status.extend(status)
    return all_pairs, all_status, skipped
