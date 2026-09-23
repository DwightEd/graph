"""Tied-score ranking changes and exact AP accounting on frozen scores."""

import numpy as np

from .comparison_evaluation import group_metrics, phase_masks
from .dynamics_audit_rank import budget_rows, half_pair_rows, rank_ledger

CANDIDATE = "transport_route"
CONTROLS = ("raw_route", "route_offline_mean")
RANK_SCOPES = ("all_error", "front_half", "back_half")
IDENTITY = ("response_id", "source_id", "target", "label", "phase", "half")


def audit_scopes(tokens):
    labels = np.asarray([row["label"] for row in tokens])
    onsets = np.asarray([row["is_span_onset"] for row in tokens], dtype=bool)
    first = np.asarray([row["is_answer_first_error"] for row in tokens], dtype=bool)
    front = np.asarray([row["half"] == "front" for row in tokens])
    later = onsets & ~first
    return {**phase_masks(labels, onsets, first),
            "later_onset_vs_normal": ((labels == 0) | later, later),
            "front_half": (front, labels), "back_half": (~front, labels)}


def metric_rows(tokens, methods):
    answers = np.asarray([row["response_id"] for row in tokens])
    sources = np.asarray([row["source_id"] for row in tokens])
    result = []
    groups = [("ALL", np.ones(len(tokens), dtype=bool))]
    groups.extend((str(identity), answers == identity) for identity in np.unique(answers))
    for identity, selected in groups:
        for scope, (mask, labels) in audit_scopes(tokens).items():
            mask = mask & selected
            for method in methods:
                scores = np.asarray([row[method] for row in tokens])
                metrics = group_metrics(labels[mask], scores[mask], answers[mask], sources[mask])
                result.append({"response_id": identity, "scope": scope, "method": method,
                    **{key: metrics[key] for key in ("tokens", "positives", "negatives", "auroc", "ap")},
                    "source_auroc": metrics["source_balanced"]["auroc"],
                    "within_answer_auroc": metrics["within_answer"]["pair_weighted_auroc"]})
    return result


def within_ranks(tokens, scores):
    answers = np.asarray([row["response_id"] for row in tokens])
    ranks = np.empty(len(tokens))
    for identity in np.unique(answers):
        selected = answers == identity
        ledger = rank_ledger(np.zeros(selected.sum(), dtype=int), scores[selected])
        ranks[selected] = (ledger["rank_first"] + ledger["rank_last"]) / 2
    return ranks


def selection_weight(scores, budget):
    """Inclusion probability at top-k under random ordering within a tie."""
    threshold = np.sort(scores)[-budget]
    above, tied = scores > threshold, scores == threshold
    result = above.astype(float)
    result[tied] = (budget - above.sum()) / tied.sum()
    return result


def ranking_changes(tokens, scores, ledgers, scope):
    ranks = {method: within_ranks(tokens, value) for method, value in scores.items()}
    candidate = ledgers[CANDIDATE]
    result = []
    for control in CONTROLS:
        baseline = ledgers[control]
        for index, token in enumerate(tokens):
            row = {name: token[name] for name in IDENTITY}
            row.update(scope=scope, control=control, candidate=CANDIDATE,
                       delta_score=float(scores[CANDIDATE][index] - scores[control][index]),
                       within_rank_gain=float(ranks[control][index] - ranks[CANDIDATE][index]))
            for prefix, ledger in (("candidate", candidate), ("control", baseline)):
                row.update({f"{prefix}_{name}": ledger[name][index].item()
                            for name in ("rank_first", "rank_last", "normals_above", "ap_credit")})
            row["rank_gain"] = ((row["control_rank_first"] + row["control_rank_last"])
                                - (row["candidate_rank_first"] + row["candidate_rank_last"])) / 2
            row["delta_ap_credit"] = row["candidate_ap_credit"] - row["control_ap_credit"]
            result.append(row)
    return result


def ap_attribution(tokens, ledgers, scope):
    """Credits partition THIS scope's AP, not independent answer AP or causality."""
    result = []
    for partition in ("response_id", "phase", "half"):
        groups = np.asarray([row[partition] for row in tokens])
        for group in np.unique(groups):
            selected = groups == group
            candidate = float(ledgers[CANDIDATE]["ap_credit"][selected].sum())
            for control in CONTROLS:
                baseline = float(ledgers[control]["ap_credit"][selected].sum())
                result.append({"scope": scope, "partition": partition, "group": str(group),
                    "tokens": int(selected.sum()), "candidate": CANDIDATE, "control": control,
                    "candidate_ap_credit": candidate, "control_ap_credit": baseline,
                    "delta_ap_credit": candidate - baseline})
    return result


def budget_changes(tokens, scores, budgets, scope):
    result = []
    for budget in budgets:
        candidate = selection_weight(scores[CANDIDATE], budget)
        for control in CONTROLS:
            baseline = selection_weight(scores[control], budget)
            delta = candidate - baseline
            for index in np.flatnonzero(delta):
                result.append({**{name: tokens[index][name] for name in IDENTITY},
                    "scope": scope, "budget": budget, "candidate": CANDIDATE, "control": control,
                    "candidate_inclusion": float(candidate[index]),
                    "control_inclusion": float(baseline[index]),
                    "selection_change": float(delta[index])})
    return result


def ranking_tables(tokens, methods):
    tables = {name: [] for name in ("ranking_changes", "ap_attribution", "top_budget", "budget_changes")}
    for scope, (mask, labels) in audit_scopes(tokens).items():
        if scope not in RANK_SCOPES or not mask.any():
            continue
        selected = [tokens[index] for index in np.flatnonzero(mask)]
        scores = {name: np.asarray([row[name] for row in selected]) for name in methods}
        ledgers = {name: rank_ledger(labels[mask], value) for name, value in scores.items()}
        tables["ranking_changes"].extend(ranking_changes(selected, scores, ledgers, scope))
        tables["ap_attribution"].extend(ap_attribution(selected, ledgers, scope))
        for method in methods:
            budgets = budget_rows(labels[mask], scores[method], method)
            tables["top_budget"].extend({"scope": scope, **row} for row in budgets)
        tables["budget_changes"].extend(budget_changes(selected, scores, [r["budget"] for r in budgets], scope))
    tables["metrics"] = metric_rows(tokens, methods)
    tables["auc_half_pairs"] = [row for method in methods for row in
                               half_pair_rows(tokens, np.asarray([t[method] for t in tokens]), method)]
    return tables
