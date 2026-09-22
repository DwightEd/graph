"""Detection deadlines, normal-answer alarms and exit behavior on frozen scores."""

import numpy as np

DEADLINES = (0, 1, 3, 7)
RECOVERY_STEPS = 15


def deadline_measurements(block, method, start, end):
    """Deadline d permits offsets 0..d, always stopping at the annotated end."""
    result = {}
    for deadline in DEADLINES:
        stop = min(end, start + deadline + 1)
        finite = np.isfinite(block["scores"][method][start:stop])
        alarms = block["alarms"][method][start:stop] & finite
        result[f"detected_by_{deadline}"] = bool(alarms.any())
        result[f"unresolved_by_{deadline}"] = bool(not alarms.any() and not finite.all())
    return result


def recovery_measurement(block, method, end, next_start):
    """Observe normal tokens after a span, censoring at missing data or a new error.

    A tail is the consecutive alarm run starting at end, not the time of the
    last later alarm. These are gold-aligned diagnostics, not detector inputs.
    """
    stop = min(end + RECOVERY_STEPS, next_start, block["tokens"])
    finite = np.isfinite(block["scores"][method][end:stop])
    alarms = block["alarms"][method][end:stop] & finite
    if stop == block["tokens"]:
        reason = "answer_end"
    elif stop == next_start:
        reason = "next_error"
    else:
        reason = "horizon"
    tail = 0
    for observed, alarm in zip(finite, alarms):
        if not observed or not alarm:
            reason = "missing" if not observed else "clear"
            break
        tail += 1
    end_observed = np.isfinite(block["scores"][method][end - 1])
    return {
        "end_alarm": bool(block["alarms"][method][end - 1]) if end_observed else None,
        "post_end_tokens": len(finite),
        "post_end_scored_tokens": int(finite.sum()),
        "post_end_alarms": int(alarms.sum()),
        "alarm_tail_lower_bound": tail,
        "alarm_tail_exact": reason == "clear",
        "alarm_tail_stop": reason,
    }


def lifecycle_summary(rows):
    count = len(rows)
    result = {}
    for deadline in DEADLINES:
        detected = sum(row[f"detected_by_{deadline}"] for row in rows)
        unresolved = sum(row[f"unresolved_by_{deadline}"] for row in rows)
        result[f"detected_by_{deadline}"] = detected
        result[f"recall_by_{deadline}_lower"] = detected / count if count else None
        result[f"recall_by_{deadline}_upper"] = (detected + unresolved) / count if count else None
    recovery = [row for row in rows if row["post_end_tokens"] > 0]
    continuing = [row for row in recovery if row["end_alarm"] is True]
    observed = sum(row["post_end_scored_tokens"] for row in recovery)
    alarms = sum(row["post_end_alarms"] for row in recovery)
    result.update(
        spans_with_recovery=len(recovery),
        post_end_scored_tokens=observed,
        post_end_alarms=alarms,
        post_end_alarm_rate=alarms / observed if observed else None,
        late_only_alarm_spans=sum(
            row["complete"] and not row["observed_any_alarm"] and row["post_end_alarms"] > 0
            for row in recovery
        ),
        spans_ending_in_alarm=len(continuing),
        continuing_alarm_tail_mean_lower_bound=float(
            np.mean([row["alarm_tail_lower_bound"] for row in continuing])
        )
        if continuing
        else None,
        continuing_alarm_tail_censored=sum(not row["alarm_tail_exact"] for row in continuing),
    )
    return result


def answer_measurements(blocks, method, metadata):
    rows = []
    for block in blocks:
        labels, eligible = block["views"]["all_error"]
        finite = block["common_finite"] & eligible
        alarms = block["alarms"][method] & finite
        normal, error = finite & ~labels, finite & labels
        rows.append(
            dict(
                metadata,
                id=block["record"]["id"],
                source_id=block["record"]["source_id"],
                eligible_tokens=int(eligible.sum()),
                scored_tokens=int(finite.sum()),
                missing_tokens=int((eligible & ~finite).sum()),
                all_normal=bool(eligible.any() and not labels[eligible].any()),
                tp=int((alarms & error).sum()),
                fn=int((~alarms & error).sum()),
                fp=int((alarms & normal).sum()),
                tn=int((~alarms & normal).sum()),
            )
        )
    return rows


def answer_summary(rows):
    normal = [row for row in rows if row["all_normal"]]
    detected = sum(row["fp"] > 0 for row in normal)
    unresolved = sum(row["fp"] == 0 and row["missing_tokens"] > 0 for row in normal)
    return {
        "answers": len(rows),
        "all_normal_answers": len(normal),
        "all_normal_answers_with_alarm": detected,
        "all_normal_answers_unresolved": unresolved,
        "normal_answer_alarm_rate_lower": detected / len(normal) if normal else None,
        "normal_answer_alarm_rate_upper": (detected + unresolved) / len(normal) if normal else None,
        **{name: sum(row[name] for row in rows) for name in ("tp", "fn", "fp", "tn")},
    }
