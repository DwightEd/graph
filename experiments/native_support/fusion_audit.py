"""Frozen-reference alarms and paired AUC contributions, evaluated after scoring."""

import numpy as np
from state_audit.storage import write_csv, write_json

from .comparison_evaluation import phase_masks
from .evaluate import evaluation_records
from .risk_envelope import fit_distribution, percentile


def alarm_counts(labels, alarms):
    positive = labels.astype(bool)
    tp = int((positive & alarms).sum())
    fp = int((~positive & alarms).sum())
    positives = int(positive.sum())
    negatives = int((~positive).sum())
    return {"tp": tp, "fp": fp, "fn": positives - tp, "tn": negatives - fp,
            "recall": tp / positives if positives else None,
            "fpr": fp / negatives if negatives else None,
            "precision": tp / (tp + fp) if tp + fp else None}


def paired_audit(joined, columns, control, thresholds):
    candidate = joined["scores"][:, columns.index("risk_envelope")]
    baseline = joined["scores"][:, columns.index(control)]
    candidate_alarm = candidate > thresholds["risk_envelope"]["threshold"]
    baseline_alarm = baseline > thresholds[control]["threshold"]
    normal = joined["labels"] == 0
    # Each error's fraction of outranked normal tokens is its exact pooled AUC contribution.
    ranks = []
    for scores in (candidate, baseline):
        if normal.any():
            distribution = fit_distribution(scores[normal], np.ones(normal.sum()))
            ranks.append(percentile(scores, distribution))
        else:
            ranks.append(np.full(len(scores), np.nan))
    delta = ranks[0] - ranks[1]
    phases = phase_masks(joined["labels"], joined["onsets"], joined["firsts"])
    comparisons = {}
    for phase, (mask, labels) in phases.items():
        error = mask & labels.astype(bool)
        comparisons[phase] = {
            "candidate": alarm_counts(labels[mask], candidate_alarm[mask]),
            "control": alarm_counts(labels[mask], baseline_alarm[mask]),
            "recovered_error_tokens": int((error & candidate_alarm & ~baseline_alarm).sum()),
            "lost_error_tokens": int((error & ~candidate_alarm & baseline_alarm).sum()),
            "added_normal_alarms": int((normal & candidate_alarm & ~baseline_alarm).sum()),
            "removed_normal_alarms": int((normal & ~candidate_alarm & baseline_alarm).sum()),
            "delta_auroc_from_error_ranks": float(delta[error].mean()) if error.any() and normal.any() else None,
        }
    return comparisons, candidate_alarm, baseline_alarm, delta


def audit_token_rows(records, rows, control, candidate_alarm, baseline_alarm, delta):
    lookup = {(row["response_id"], int(row["target"])): row for row in rows}
    result = []
    offset = 0
    for record in records:
        for index, target in enumerate(record["target"]):
            position = offset + index
            changed = candidate_alarm[position] != baseline_alarm[position]
            if not (changed or record["onsets"][index] or record["firsts"][index]):
                continue
            source = lookup[record["id"], int(target)]
            context = [lookup[record["id"], t]["token"]
                       for t in range(max(0, target - 5), min(record["response_length"], target + 6))]
            result.append({**source, "control": control, "label": int(record["labels"][index]),
                           "is_span_onset": bool(record["onsets"][index]),
                           "is_answer_first_error": bool(record["firsts"][index]),
                           "candidate_alarm": bool(candidate_alarm[position]),
                           "control_alarm": bool(baseline_alarm[position]),
                           "error_rank_delta": float(delta[position]) if record["labels"][index] and np.isfinite(delta[position]) else None,
                           "context_for_review_only": ''.join(context)})
        offset += len(record["labels"])
    return result


def write_fusion_audit(output, destination, annotations, methods, rows, thresholds, baseline):
    columns = tuple(methods.values())
    records = evaluation_records(output, annotations, destination, columns)
    joined = {name: np.concatenate([r[name] for r in records]) for name in ("labels", "onsets", "firsts", "scores")}
    comparisons, cases = {}, []
    for control in (baseline, "route_state"):
        result, candidate_alarm, baseline_alarm, delta = paired_audit(joined, columns, control, thresholds)
        comparisons[control] = result
        cases.extend(audit_token_rows(records, rows, control, candidate_alarm, baseline_alarm, delta))
    audit = {"status": "evaluated", "threshold_source": "unlabeled_external_reference",
             "normal_fpr_guarantee": False, "thresholds": thresholds, "comparisons": comparisons,
             "labels_used_for_scoring_or_thresholds": False,
             "context_contains_future_text_for_review_only": True}
    write_json(destination / "complementarity.json", audit)
    fields = list(cases[0]) if cases else ["response_id", "target", "control", "label"]
    write_csv(destination / "alarm_changes.csv", cases, fields)
    return audit
