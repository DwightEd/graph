"""Retrospective span attribution and earliest score availability are different quantities."""

import numpy as np

from ..dual_state.report import intervals
from .aggregation import TOKEN_METHODS, select_whole_units


def span_availability(record, units, selected, method, offline=False):
    rows = []
    for start, stop in intervals(record, True):
        hits = [unit for unit, alarm in zip(units, selected)
                if alarm and unit["start"] < stop and unit["stop"] > start]
        onset = next(unit for unit in units if unit["start"] <= start < unit["stop"])
        first_end = min((unit["stop"] - 1 for unit in hits), default=None)
        rows.append(dict(method=method, response_id=record["id"], start=start, stop=stop,
            retrospectively_flagged=bool(hits),
            onset_unit_flagged=any(unit["start"] == onset["start"] for unit in hits),
            onset_unit_future_tokens=onset["stop"] - 1 - start,
            first_flagged_unit_end=first_end,
            earliest_score_delay=first_end - start if first_end is not None and not offline else None,
            score_available_before_span_end=not offline and first_end is not None and first_end < stop,
            score_scope="completed_answer_and_reference" if offline else "unit_end"))
    return rows


def unit_budgets(records, rows, token_methods=TOKEN_METHODS, offline_methods=()):
    """Top-decile unit ranking; every selected unit flags its entire original interval."""
    summary, delays, alarms = {}, [], []
    # Ranking uses every measured unit. Missing truth affects evaluation, not selection.
    eligible = rows
    for method in token_methods:
        selected = select_whole_units([row[method] for row in eligible])
        lookup = {(row["response_id"], row["unit_id"]): bool(hit)
                  for row, hit in zip(eligible, selected)}
        normal_tokens = false_tokens = flagged_tokens = error_tokens = caught_tokens = 0
        clean_answers = clean_alarms = 0
        method_delays = []
        for record in records:
            flags = [lookup.get((record["id"], i), False) for i in range(len(record["units"]))]
            alarm = np.zeros(record["response_length"], dtype=bool)
            for unit, hit in zip(record["units"], flags):
                alarm[unit["start"]:unit["stop"]] = hit
            valid, labels = record["valid"], record["labels"]
            normal_tokens += int((valid & (labels == 0)).sum())
            false_tokens += int((alarm & valid & (labels == 0)).sum())
            error_tokens += int((valid & (labels == 1)).sum())
            caught_tokens += int((alarm & valid & (labels == 1)).sum())
            flagged_tokens += int((alarm & valid).sum())
            if valid.all() and not labels.any():
                clean_answers += 1
                clean_alarms += int(alarm.any())
            method_delays.extend(span_availability(record, record["units"], flags, method, method in offline_methods))
        chosen = [row for row, hit in zip(eligible, selected) if hit]
        summary[method] = dict(eligible_units=len(eligible), selected_units=len(chosen),
            selected_unit_fraction=len(chosen) / len(eligible) if eligible else None,
            false_positive_units=sum(row["fully_annotated"] and row["error_tokens"] == 0 for row in chosen),
            selected_incomplete_units=sum(not row["fully_annotated"] for row in chosen),
            flagged_tokens=flagged_tokens, flagged_error_tokens=caught_tokens,
            token_precision=caught_tokens / flagged_tokens if flagged_tokens else None,
            token_recall=caught_tokens / error_tokens if error_tokens else None,
            normal_token_fpr=false_tokens / normal_tokens if normal_tokens else None,
            all_normal_answers=clean_answers, all_normal_answer_alarms=clean_alarms)
        delays.extend(method_delays)
        alarms.extend(dict(method=method, response_id=row["response_id"], unit_id=row["unit_id"],
                           selected=bool(hit), score=row[method]) for row, hit in zip(eligible, selected))
    return dict(unit_budget_fraction=.1, tie_policy="include_all_boundary_ties",
                deployment_threshold=False, availability="earliest_after_reading_unit_end; excludes_boundary_lookahead",
                offline_methods=list(offline_methods), offline_delay="not_an_online_detection_delay",
                methods=summary), delays, alarms
