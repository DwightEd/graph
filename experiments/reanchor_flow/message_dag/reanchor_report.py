"""Source-balanced incidence and downstream outcomes for reanchor morphology."""

from collections import defaultdict

import numpy as np

from .reanchor import REANCHOR_TYPES

LABELS = (("N", 0), ("H", 1))
CATEGORIES = ("any_reanchor", *REANCHOR_TYPES, "none")
HORIZONS = (("1", 1, 1), ("2-4", 2, 4), ("5-16", 5, 16))
MORPHOLOGY_METRICS = (
    "active_head_fraction",
    "all_layer_head_fraction",
    "active_layer_fraction",
    "head_focality",
    "focal_head_fraction",
    "effective_sources",
    "source_agreement",
    "gain_source_agreement",
    "mean_distance",
    "peak_prompt_fraction",
    "peak_history_fraction",
    "peak_evidence_fraction",
)


def _estimate(values, bootstrap, seed):
    source_values = np.asarray(list(values.values()), dtype=float)
    finite = source_values[np.isfinite(source_values)]
    result = {
        "mean": float(finite.mean()) if len(finite) else float("nan"),
        "ci95": [float("nan"), float("nan")],
        "sources": len(finite),
    }
    if len(finite) > 1 and bootstrap:
        rng = np.random.default_rng(seed)
        draws = finite[
            rng.integers(0, len(finite), size=(bootstrap, len(finite)))
        ].mean(1)
        result["ci95"] = np.quantile(draws, (0.025, 0.975)).tolist()
    return result


def _rate(counts, bootstrap, seed):
    rates = {
        source: numerator / denominator
        for source, (denominator, numerator) in counts.items()
        if denominator
    }
    result = _estimate(rates, bootstrap, seed)
    result.update(
        tokens=sum(value[0] for value in counts.values()),
        anchors=sum(value[1] for value in counts.values()),
    )
    return result, rates


def _difference(first, second, bootstrap, seed):
    sources = sorted(set(first) | set(second))
    paired = np.asarray(
        [[first.get(source, np.nan), second.get(source, np.nan)] for source in sources],
        dtype=float,
    ).reshape(-1, 2)
    difference = paired[:, 0] - paired[:, 1]
    values = {
        source: value
        for source, value in zip(sources, difference)
        if np.isfinite(value)
    }
    result = _estimate(values, bootstrap, seed)
    result.update(
        first_sources=int(np.isfinite(paired[:, 0]).sum()),
        second_sources=int(np.isfinite(paired[:, 1]).sum()),
    )
    return result


def _source_means(values):
    return {
        source: float(np.mean(observations))
        for source, observations in values.items()
        if observations
    }


def _outcome(values, bootstrap, seed, target_counts=None):
    by_source = _source_means(values)
    result = _estimate(by_source, bootstrap, seed)
    observations = sum(len(items) for items in values.values())
    result["observations"] = observations
    if target_counts is not None:
        result["anchors"] = observations
        result["target_tokens"] = sum(target_counts.values())
    return result


def _matches(category, kind, is_reanchor):
    if category == "any_reanchor":
        return is_reanchor
    return kind == category


def summarize_reanchor(records, *, bootstrap=200):
    """Keep incidence and morphology-conditional future outcomes separate."""

    if bootstrap < 0:
        raise ValueError("bootstrap repetitions must be nonnegative")
    incidence = defaultdict(lambda: defaultdict(lambda: np.zeros(2, dtype=int)))
    hallucination = defaultdict(lambda: defaultdict(list))
    negative_logprob = defaultdict(lambda: defaultdict(list))
    hallucination_targets = defaultdict(lambda: defaultdict(int))
    logprob_targets = defaultdict(lambda: defaultdict(int))
    morphology = defaultdict(lambda: defaultdict(list))
    coverage = defaultdict(lambda: np.zeros(3, dtype=int))

    for record in records:
        group, source = str(record["group"]), str(record["source"])
        if not group or not source:
            raise ValueError("group and source identities are required")
        coverage[group]  # Retain groups whose responses have no eligible query.
        profile = record["profile"]
        rows = np.asarray(profile["row_position"], dtype=int)
        eligible = np.asarray(profile["eligible"], dtype=bool)
        kinds = np.asarray(profile["reanchor_type"], dtype=str)
        is_reanchor = np.asarray(profile["is_reanchor"], dtype=bool)
        special = np.asarray(record["special_mask"], dtype=bool)
        labels = np.asarray(record["labels"], dtype=int)
        start = int(record["response_start"])
        logprob = np.asarray(record["predictor_logprob"], dtype=float)
        if not (
            rows.shape
            == eligible.shape
            == kinds.shape
            == is_reanchor.shape
            == logprob.shape
        ):
            raise ValueError("profile and predictor arrays must share the row axis")
        if labels.shape != (len(special) - start,):
            raise ValueError("labels must cover the generated token positions")
        row_index = {int(position): index for index, position in enumerate(rows)}
        profile_metrics = {}
        for name in MORPHOLOGY_METRICS:
            if name not in profile:
                continue
            value = np.asarray(profile[name], dtype=float)
            if value.shape != rows.shape:
                raise ValueError(f"{name} must share the profile row axis")
            profile_metrics[name] = value

        for index in np.flatnonzero(eligible):
            query = int(rows[index])
            target = query + 1
            coverage[group][0] += 1
            coverage[group][1] += int(is_reanchor[index])
            for category in CATEGORIES:
                counts = incidence[group, category, "all"][source]
                counts[0] += 1
                counts[1] += int(
                    _matches(category, kinds[index], is_reanchor[index])
                )
            if is_reanchor[index]:
                for category in ("any_reanchor", kinds[index]):
                    for name, values in profile_metrics.items():
                        if np.isfinite(values[index]):
                            morphology[group, category, name][source].append(
                                float(values[index])
                            )
            label = (
                int(labels[target - start])
                if target < len(special) and not special[target]
                else -1
            )
            coverage[group][2] += int(label in (0, 1))
            if label in (0, 1):
                label_name = "H" if label == 1 else "N"
                for category in CATEGORIES:
                    counts = incidence[group, category, label_name][source]
                    counts[0] += 1
                    counts[1] += int(
                        _matches(category, kinds[index], is_reanchor[index])
                    )

            for category in CATEGORIES:
                if not _matches(category, kinds[index], is_reanchor[index]):
                    continue
                for horizon_name, begin, end in HORIZONS:
                    anchor_hallucination = []
                    anchor_logprob = []
                    for offset in range(begin, end + 1):
                        future = query + offset
                        predictor = row_index.get(future - 1)
                        if (
                            future >= len(special)
                            or special[future]
                            or predictor is None
                        ):
                            continue
                        outcome = int(labels[future - start])
                        if outcome in (0, 1):
                            anchor_hallucination.append(float(outcome))
                        if np.isfinite(logprob[predictor]):
                            anchor_logprob.append(-float(logprob[predictor]))
                    key = (group, category, horizon_name)
                    if anchor_hallucination:
                        hallucination[key][source].append(
                            float(np.mean(anchor_hallucination))
                        )
                        hallucination_targets[key][source] += len(
                            anchor_hallucination
                        )
                    if anchor_logprob:
                        negative_logprob[key][source].append(
                            float(np.mean(anchor_logprob))
                        )
                        logprob_targets[key][source] += len(anchor_logprob)

    result = {
        "estimand": (
            "all-token reanchor incidence plus morphology-conditional fixed-horizon "
            "outcomes"
        ),
        "horizons": {name: [begin, end] for name, begin, end in HORIZONS},
        "groups": {},
        "labels_used_for_profile": False,
        "labels_used_for_outcomes": True,
    }
    for group_index, group in enumerate(sorted(coverage)):
        group_result = {
            "coverage": {
                "eligible_queries": int(coverage[group][0]),
                "reanchor_queries": int(coverage[group][1]),
                "known_next_token_labels": int(coverage[group][2]),
            },
            "incidence": {},
            "morphology": {},
            "downstream": {},
        }
        for category_index, category in enumerate(CATEGORIES):
            label_rates = {}
            category_result = {}
            for label_index, (label_name, _) in enumerate((
                ("all", None),
                *LABELS,
            )):
                summary, rates = _rate(
                    incidence[group, category, label_name],
                    bootstrap,
                    1000 + 100 * group_index + 10 * category_index + label_index,
                )
                category_result[label_name] = summary
                if label_name != "all":
                    label_rates[label_name] = rates
            category_result["H_minus_N"] = _difference(
                label_rates["H"],
                label_rates["N"],
                bootstrap,
                2000 + 100 * group_index + category_index,
            )
            group_result["incidence"][category] = category_result

            category_outcomes = {}
            for horizon_index, (horizon_name, _, _) in enumerate(HORIZONS):
                seed = 3000 + 100 * group_index + 10 * category_index + horizon_index
                hallucination_values = hallucination[group, category, horizon_name]
                logprob_values = negative_logprob[group, category, horizon_name]
                outcome = {
                    "hallucination_rate": _outcome(
                        hallucination_values,
                        bootstrap,
                        seed,
                        hallucination_targets[group, category, horizon_name],
                    ),
                    "negative_logprob": _outcome(
                        logprob_values,
                        bootstrap,
                        seed + 5000,
                        logprob_targets[group, category, horizon_name],
                    ),
                }
                if category != "none":
                    outcome["hallucination_rate_minus_none"] = _difference(
                        _source_means(hallucination_values),
                        _source_means(hallucination[group, "none", horizon_name]),
                        bootstrap,
                        seed + 10000,
                    )
                    outcome["negative_logprob_minus_none"] = _difference(
                        _source_means(logprob_values),
                        _source_means(
                            negative_logprob[group, "none", horizon_name]
                        ),
                        bootstrap,
                        seed + 15000,
                    )
                category_outcomes[horizon_name] = outcome
            group_result["downstream"][category] = category_outcomes
        for category_index, category in enumerate(("any_reanchor", *REANCHOR_TYPES)):
            group_result["morphology"][category] = {
                name: _outcome(
                    morphology[group, category, name],
                    bootstrap,
                    9000 + 100 * group_index + 10 * category_index + metric_index,
                )
                for metric_index, name in enumerate(MORPHOLOGY_METRICS)
            }
        result["groups"][group] = group_result
    return result
