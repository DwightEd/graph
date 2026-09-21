"""Label-assisted same-answer controls, chosen without reading detector scores."""

from collections import Counter
from types import SimpleNamespace

import numpy as np

from ..span_audit.matching import has_text, overlaps, repetition_rate, token_kind
from ..span_audit.units import Span


def matching_answer(block, common_finite):
    """Use saved token IDs, or report exact offset pieces as the lexical proxy."""
    offsets = np.asarray(block["offsets"])
    tokens = block.get("token_ids")
    basis = "token_ids"
    if tokens is None:
        tokens = [block["text"][start:end] for start, end in offsets]
        basis = "surface_pieces"
    error = np.asarray(block["views"]["all_error"][0], dtype=bool)
    entropy = block.get("entropy")
    return SimpleNamespace(
        text=block["text"], offsets=offsets, response_ids=np.asarray(tokens),
        error_mask=error, common_finite=np.asarray(common_finite, dtype=bool),
        previous_error=np.r_[False, np.logical_or.accumulate(error)[:-1]],
        entropy=None if entropy is None else np.asarray(entropy),
        repetition_basis=basis,
    )


def control_rejection(answer, error, control, used, neighborhood):
    """Exclude contaminated, reused or unobserved normal intervals first."""
    begin = max(0, control.start - neighborhood)
    end = min(len(answer.error_mask), control.end + neighborhood)
    if answer.error_mask[begin:end].any():
        return "error_in_control_neighborhood"
    if any(overlaps(control, previous) for previous in used):
        return "normal_interval_already_used"
    if not answer.common_finite[control.start:control.end].all():
        return "incomplete_common_coverage"
    if not has_text(answer, control):
        return "missing_surface_offsets"
    if answer.previous_error[control.start] != answer.previous_error[error.start]:
        return "different_previous_error_state"
    if token_kind(answer, control.start) != token_kind(answer, error.start):
        return "different_first_surface_kind"
    return None


def candidate_features(answer, error, control):
    error_repetition = repetition_rate(answer, error)
    normal_repetition = repetition_rate(answer, control)
    entropy_gap = None
    if answer.entropy is not None:
        values = answer.entropy[[error.start, control.start]]
        if np.isfinite(values).all():
            entropy_gap = float(abs(values[0] - values[1]))
    return {
        "normal_start": control.start, "normal_end": control.end,
        "position_gap": abs(error.start - control.start) / len(answer.error_mask),
        "repetition_gap": abs(error_repetition - normal_repetition),
        "error_repetition": error_repetition, "normal_repetition": normal_repetition,
        "first_surface_kind": token_kind(answer, error.start),
        "previous_error": bool(answer.previous_error[error.start]),
        "entropy_gap": entropy_gap, "entropy_controlled": entropy_gap is not None,
        "repetition_basis": answer.repetition_basis,
    }


def candidate_controls(answer, error, used, limits):
    candidates, rejected = [], Counter()
    for start in range(len(answer.error_mask) - error.length + 1):
        control = Span(start, start + error.length)
        reason = control_rejection(answer, error, control, used, limits["neighborhood"])
        if reason is None:
            candidate = candidate_features(answer, error, control)
            for feature in ("position", "repetition", "entropy"):
                gap = candidate[feature + "_gap"]
                if gap is not None and gap > limits[feature + "_limit"]:
                    reason = feature + "_gap_exceeds_limit"
                    break
        if reason is None:
            candidates.append(candidate)
        else:
            rejected[reason] += 1
    return candidates, rejected


def interval_text(answer, start, end):
    return answer.text[answer.offsets[start, 0]:answer.offsets[end - 1, 1]]


def choose_pairs(answer, gold, response_id, limits):
    pairs, unmatched, used = [], [], []
    for span_index, (start, end) in enumerate(gold):
        error = Span(int(start), int(end))
        row = {
            "pair_id": f"{response_id}:{span_index}", "span_index": span_index,
            "error_start": error.start, "error_end": error.end, "length": error.length,
        }
        if not answer.common_finite[start:end].all() or not has_text(answer, error):
            reason = "incomplete_common_coverage"
            if not has_text(answer, error):
                reason = "missing_surface_offsets"
            unmatched.append(dict(row, reason=reason, rejection_counts={}))
            continue
        candidates, rejected = candidate_controls(answer, error, used, limits)
        if not candidates:
            unmatched.append(dict(row, reason="no_eligible_normal_interval",
                                  rejection_counts=dict(rejected)))
            continue
        best = min(candidates, key=lambda item: (
            item["position_gap"], item["repetition_gap"], item["normal_start"]))
        row.update(best)
        row["error_text"] = interval_text(answer, error.start, error.end)
        row["normal_text"] = interval_text(answer, best["normal_start"], best["normal_end"])
        pairs.append(row)
        used.append(Span(best["normal_start"], best["normal_end"]))
    return pairs, unmatched


def match_spans(block, common_finite, *, position_limit=.25, repetition_limit=.15,
                entropy_limit=.5, neighborhood=0):
    """Return pairs, unmatched spans and diagnostics; never use score magnitudes.

    Inputs are response-aligned arrays. Gold and returned intervals are half-open.
    The caller supplies one common finite mask for every compared method. A zero
    neighborhood compares span interiors; a positive radius excludes nearby gold
    errors too. Missing entropy remains unmeasured, never a synthetic zero.
    """
    answer = matching_answer(block, common_finite)
    limits = {
        "position_limit": position_limit, "repetition_limit": repetition_limit,
        "entropy_limit": entropy_limit, "neighborhood": neighborhood,
    }
    pairs, unmatched = choose_pairs(answer, block["gold"], block["record"]["id"], limits)
    diagnostics = dict(
        total_gold_spans=len(block["gold"]), matched=len(pairs), unmatched=len(unmatched),
        repetition_basis=answer.repetition_basis,
        entropy_available=answer.entropy is not None,
        entropy_controlled_pairs=sum(pair["entropy_controlled"] for pair in pairs),
        unmatched_reasons=dict(Counter(row["reason"] for row in unmatched)),
        selection="same_answer_greedy_gold_order_without_scores",
        control_observation_neighborhood=neighborhood, **limits,
    )
    return pairs, unmatched, diagnostics
