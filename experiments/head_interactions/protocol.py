"""Explicit condition schedule; the teaching package owns scoring, messages, and operations."""

from dataclasses import replace
from itertools import combinations, permutations

import numpy as np
from state_audit.analysis.interactions import adaptation_effects, factorial_effects
from state_audit.experiments.messages import restore_messages

READOUTS = ("sum_margin", "mean_margin", "candidate_sum_margin")


def deletions(sites, strength=1.0):
    return [site.deletion(strength) for site in sites]


def compare_readout(baseline, changed, atol):
    differences = [
        np.max(np.abs(np.asarray(left) - right))
        for left, right in zip(baseline[0]["token_logp"], changed[0]["token_logp"])
    ]
    error = float(max(differences))
    margin_error = abs(baseline[0]["sum_margin"] - changed[0]["sum_margin"])
    return dict(
        max_token_logp_error=error,
        margin_error=margin_error,
        passed=bool(np.isfinite(error) and max(error, margin_error) <= atol),
    )


def controls(trials, baseline, atol):
    sham = trials.evaluate("zero_strength", deletions(trials.sites, 0.0))
    restored = restore_messages(
        trials.restorer("self_restore"), trials.sites, baseline[1], current=baseline
    )
    operations = deletions(trials.sites)
    deleted = trials.evaluate("round_trip/delete", operations)
    round_trip = restore_messages(
        trials.restorer("round_trip/restore"), trials.sites, baseline[1], operations, deleted
    )
    return dict(
        zero_strength=compare_readout(baseline, sham, atol),
        self_restore=compare_readout(baseline, restored, atol),
        delete_restore=compare_readout(baseline, round_trip, atol),
    )


def factorial_rows(trials, groups, baseline, singles, reference="zero", donor=None):
    rows = []
    for left, right in combinations(groups, 2):
        sites = [*groups[left], *groups[right]]
        name = f"{reference}/joint/{left}/{right}"
        if donor is None:
            joint = trials.evaluate(name, deletions(sites))
        else:
            joint = restore_messages(trials.restorer(name), sites, donor, current=baseline)
        results = {"11": baseline, "01": singles[left], "10": singles[right], "00": joint}
        for readout in READOUTS:
            values = {key: result[0][readout] for key, result in results.items()}
            rows.append(
                dict(
                    a=left,
                    b=right,
                    reference=reference,
                    readout=readout,
                    **factorial_effects(values),
                )
            )
    return rows


def single_conditions(trials, groups, baseline, gain):
    removed, rows = {}, []
    for name, sites in groups.items():
        removed[name] = trials.evaluate(f"zero/single/{name}", deletions(sites))
        boosted = trials.evaluate(f"gain/single/{name}", deletions(sites, 1.0 - gain))
        for readout in READOUTS:
            base = baseline[0][readout]
            cut, boost = removed[name][0][readout], boosted[0][readout]
            rows.append(
                dict(
                    group=name,
                    readout=readout,
                    baseline=base,
                    removed=cut,
                    boosted=boost,
                    native_effect=base - cut,
                    boost_effect=boost - base,
                )
            )
    return removed, rows


def compensation_rows(trials, groups, baseline, removed):
    rows, skipped = [], []
    for upstream, downstream in permutations(groups, 2):
        first, second = groups[upstream], groups[downstream]
        if max(site.layer for site in first) >= min(site.layer for site in second):
            skipped.append(dict(a=upstream, b=downstream, reason="not_strict_layer_order"))
            continue
        if min(position for site in first for position in site.positions) > max(
            position for site in second for position in site.positions
        ):
            skipped.append(dict(a=upstream, b=downstream, reason="receiver_precedes_upstream"))
            continue
        name = f"restore/{upstream}/{downstream}"
        restored = restore_messages(
            trials.restorer(name), second, baseline[1], deletions(first), removed[upstream]
        )
        for readout in READOUTS:
            values = [result[0][readout] for result in (baseline, removed[upstream], restored)]
            rows.append(
                dict(a=upstream, b=downstream, readout=readout, **adaptation_effects(*values))
            )
    return rows, skipped


def donor_factorial(trials, groups, baseline, donor):
    singles = {}
    for name, sites in groups.items():
        singles[name] = restore_messages(
            trials.restorer(f"donor/single/{name}"), sites, donor, current=baseline
        )
    return factorial_rows(trials, groups, baseline, singles, "donor", donor)


def run_protocol(trials, groups, atol, gain, donor=None, compensation=True):
    baseline = trials.evaluate("baseline")
    checks = controls(trials, baseline, atol)
    result = dict(
        controls=checks,
        valid=all(row["passed"] for row in checks.values()),
        singles=[],
        interactions=[],
        adaptation=[],
        adaptation_skipped=[],
    )
    if not result["valid"]:
        return result
    removed, result["singles"] = single_conditions(trials, groups, baseline, gain)
    result["interactions"] = factorial_rows(trials, groups, baseline, removed)
    if compensation:
        result["adaptation"], result["adaptation_skipped"] = compensation_rows(
            trials, groups, baseline, removed
        )
    if donor is not None:
        result["interactions"].extend(donor_factorial(trials, groups, baseline, donor))
    return result


def random_groups(groups, model, seed):
    """One fixed head permutation per layer; same counts, source spans, and queries."""
    random = np.random.default_rng(seed)
    occupied = {}
    for sites in groups.values():
        for site in sites:
            occupied.setdefault(site.layer, set()).update(site.heads)
    mapping = {}
    for layer, used in sorted(occupied.items()):
        count, _, _ = model.head_layout(layer)
        available = sorted(set(range(count)) - used)
        if len(available) < len(used):
            raise ValueError(f"Layer {layer}: not enough disjoint heads for the random control")
        mapping[layer] = dict(zip(sorted(used), random.choice(available, len(used), replace=False)))
    return {
        name: [
            replace(site, heads=tuple(int(mapping[site.layer][head]) for head in site.heads))
            for site in sites
        ]
        for name, sites in groups.items()
    }
