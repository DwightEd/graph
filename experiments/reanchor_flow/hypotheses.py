"""Hypothesis tests and source-clustered summaries for re-anchor evaluation."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .claims import FORCED_CHUNK
from .events import boundary_events, onset_pairs, positive_onsets
from .metrics import (
    cluster_curve,
    cluster_group_contrast,
    cluster_summary,
    metric_with_cluster_ci,
)
from .signals import finite_mean, same_model


def third_change(series) -> float:
    series = np.asarray(series, dtype=np.float64)
    width = len(series) // 3
    if width < 2:
        return float("nan")
    return finite_mean(series[-width:]) - finite_mean(series[:width])


def effect(records: list[dict], first: str, second: str, field: str):
    value, source = [], []
    for record in records:
        if record.get(second) is None:
            continue
        left, right = record[first].get(field), record[second].get(field)
        if left is None or right is None:
            continue
        difference = float(left) - float(right)
        if np.isfinite(difference):
            value.append(difference)
            source.append(record["source_id"])
    value = np.asarray(value, dtype=np.float64)
    source = np.asarray(source)
    return value, source


def summarize_effect(
    records: list[dict],
    first: str,
    second: str,
    field: str,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    value, source = effect(records, first, second, field)
    return cluster_summary(value, source, repeats=bootstrap, seed=seed)


def method_difference(
    functional: list[dict],
    attention: list[dict],
    first: str,
    second: str,
    field: str,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    """Functional event contrast minus its attention-only counterpart."""

    def keyed(records):
        return {
            (record["sample_id"], record[first]["center"]): record
            for record in records
            if record.get(second) is not None
        }

    first_map, second_map = keyed(functional), keyed(attention)
    keys = sorted(first_map.keys() & second_map.keys())
    value, source = [], []
    for key in keys:
        scalars = (
            first_map[key][first].get(field),
            first_map[key][second].get(field),
            second_map[key][first].get(field),
            second_map[key][second].get(field),
        )
        if any(item is None for item in scalars) or not np.isfinite(scalars).all():
            continue
        functional_effect = float(scalars[0]) - float(scalars[1])
        attention_effect = float(scalars[2]) - float(scalars[3])
        value.append(functional_effect - attention_effect)
        source.append(first_map[key]["source_id"])
    return cluster_summary(value, source, repeats=bootstrap, seed=seed)


def paired_association(
    records: list[dict],
    positive: str,
    negative: str,
    field: str,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    label = np.tile([True, False], len(records))
    score = np.asarray(
        [-record[side][field] for record in records for side in (positive, negative)]
    )
    source = np.repeat([record["source_id"] for record in records], 2)
    result = metric_with_cluster_ci(
        label, score, source, repeats=bootstrap, seed=seed
    )
    result["warning"] = (
        "This is 1:1 matched discrimination. Its prevalence and AP are not "
        "comparable to a natural-prevalence hallucination detector."
    )
    return result


def boundary_group_effect(
    correct: list[dict],
    hallucinated: list[dict],
    field: str,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    records = [*correct, *hallucinated]
    value = [record["boundary"][field] for record in records]
    label = [False] * len(correct) + [True] * len(hallucinated)
    source = [record["source_id"] for record in records]
    return cluster_group_contrast(
        value, label, source, repeats=bootstrap, seed=seed
    )


def matched_boundary_effect(
    clean: list[dict],
    hallucinated: list[dict],
    field: str,
    *,
    bootstrap: int,
    seed: int,
    position_caliper: int = 64,
) -> dict[str, object]:
    """Pair each hallucinated boundary to a nearby clean boundary in its response."""

    available = list(clean)
    differences, sources, distances, punctuation_matches = [], [], [], []
    for positive in sorted(
        hallucinated,
        key=lambda item: (item["sample_id"], item["boundary"]["center"]),
    ):
        candidate_indices = [
            index
            for index, candidate in enumerate(available)
            if candidate["sample_id"] == positive["sample_id"]
            and abs(
                candidate["boundary"]["center"] - positive["boundary"]["center"]
            )
            <= position_caliper
        ]
        if not candidate_indices:
            continue
        control_index = min(
            candidate_indices,
            key=lambda index: (
                available[index]["preceding_token_id"]
                != positive["preceding_token_id"],
                abs(
                    available[index]["boundary"]["center"]
                    - positive["boundary"]["center"]
                ),
                available[index]["boundary"]["center"],
            ),
        )
        control = available.pop(control_index)
        differences.append(
            positive["boundary"][field] - control["boundary"][field]
        )
        sources.append(positive["source_id"])
        distances.append(
            abs(positive["boundary"]["center"] - control["boundary"]["center"])
        )
        punctuation_matches.append(
            positive["preceding_token_id"] == control["preceding_token_id"]
        )
    result = cluster_summary(
        differences, sources, repeats=bootstrap, seed=seed
    )
    result.update(
        definition="hallucinated minus clean boundary, paired within response and position caliper",
        candidate_hallucinations=len(hallucinated),
        matched_pairs=len(differences),
        position_caliper=position_caliper,
        median_position_distance=(float(np.median(distances)) if distances else None),
        punctuation_token_match_rate=(
            float(np.mean(punctuation_matches)) if punctuation_matches else None
        ),
    )
    return result


def event_curve_report(
    records: list[dict],
    side: str,
    field: str,
    *,
    bootstrap: int,
    seed: int,
    width: int,
) -> dict[str, object]:
    kept = [record for record in records if record[side].get(field) is not None]
    if not kept:
        return cluster_curve(
            np.empty((0, width)), np.empty(0), repeats=bootstrap, seed=seed
        )
    values = np.stack([record[side][field] for record in kept])
    source = np.asarray([record["source_id"] for record in kept])
    return cluster_curve(values, source, repeats=bootstrap, seed=seed)


def drift_report(rows: list[dict], bootstrap: int, seed: int) -> dict:
    fields = (
        "evidence_enrichment",
        "other_prompt_enrichment",
        "history_enrichment",
        "evidence_specificity",
    )
    source = []
    change = {field: [] for field in fields}
    raw_evidence, raw_history = [], []
    for row in rows:
        if len(row["evidence_enrichment"]) < 6:
            continue
        source.append(row["source_id"])
        for field in fields:
            change[field].append(third_change(row[field]))
        raw_evidence.append(third_change(row["raw_evidence_share"]))
        raw_history.append(third_change(row["raw_history_share"]))
    return {
        "definition": (
            "last response third minus first response third, within sample; "
            "primary fields are log(observed share / availability null)"
        ),
        "evidence_enrichment_change": cluster_summary(
            change["evidence_enrichment"], source, repeats=bootstrap, seed=seed
        ),
        "other_prompt_enrichment_change": cluster_summary(
            change["other_prompt_enrichment"], source, repeats=bootstrap, seed=seed + 1
        ),
        "history_enrichment_change": cluster_summary(
            change["history_enrichment"], source, repeats=bootstrap, seed=seed + 2
        ),
        "evidence_specificity_change": cluster_summary(
            change["evidence_specificity"], source, repeats=bootstrap, seed=seed + 3
        ),
        "descriptive_raw_share_change": {
            "warning": "Raw shares contain deterministic growth of the history source pool.",
            "evidence": cluster_summary(
                raw_evidence, source, repeats=bootstrap, seed=seed + 4
            ),
            "history": cluster_summary(
                raw_history, source, repeats=bootstrap, seed=seed + 5
            ),
        },
    }


def direction(summary: dict, expected: int, minimum_sources: int = 5) -> str:
    low, high = summary["ci95"]
    if summary.get("sources", 0) < minimum_sources or low is None:
        return "inconclusive"
    if expected > 0:
        return "supported" if low > 0 else ("contradicted" if high < 0 else "inconclusive")
    return "supported" if high < 0 else ("contradicted" if low > 0 else "inconclusive")


def joint_status(*statuses: str) -> str:
    if statuses and all(status == "supported" for status in statuses):
        return "supported"
    if any(status == "contradicted" for status in statuses):
        return "contradicted"
    return "inconclusive"


def next_step(h1: str, h2: str, h3: str, generation_scope: bool) -> str:
    if not generation_scope:
        return "use the same model for generation and observation before making generation-mechanism claims"
    if h1 == h2 == h3 == "supported":
        return "add claim-specific support alignment, then run localized layer/message interventions"
    if h1 == "supported" and h2 == "supported" and h3 == "contradicted":
        return "a re-anchor rhythm exists, but missed entry is not supported as its hallucination mechanism"
    if h2 == "supported" and h3 == "inconclusive":
        return "increase exact-boundary hallucination power; do not train a detector yet"
    return "do not optimize a detector; improve atomic-claim/support alignment or stop this hypothesis"


def optional_dependence(events: list[dict], field: str, bootstrap: int, seed: int):
    values, source = [], []
    for event in events:
        value = event["boundary"].get(field)
        if value is not None and np.isfinite(value):
            values.append(value)
            source.append(event["source_id"])
    if not values:
        return {"status": "not_run"}
    result = cluster_summary(values, source, repeats=bootstrap, seed=seed)
    result["status"] = "observed"
    return result


def task_report(
    task: str,
    rows: list[dict],
    *,
    bootstrap: int,
    seed: int,
    pre: int,
    post: int,
    curve_low: int,
    curve_high: int,
) -> dict:
    boundaries = [
        event
        for row in rows
        for event in boundary_events(
            row, pre=pre, post=post, curve_low=curve_low, curve_high=curve_high
        )
    ]
    correct = [event for event in boundaries if event["correct"]]
    correct_with_control = [event for event in correct if event["control"] is not None]
    at_boundary = [
        event
        for event in boundaries
        if event["prefix_clean"] and event["onset_at_boundary"]
    ]
    near_boundary = [
        event
        for event in boundaries
        if event["prefix_clean"] and event["onset_near_boundary"]
    ]
    late = [
        event
        for event in boundaries
        if event["prefix_clean"] and event["late_onset"]
    ]
    onset = [
        pair
        for row in rows
        for pair in onset_pairs(
            row, pre=pre, post=post, curve_low=curve_low, curve_high=curve_high
        )
    ]

    attention_rows = [
        {
            **row,
            "evidence_specificity": row["attention_evidence_specificity"],
            "history_enrichment": row["attention_history_enrichment"],
            "functional_log_lift_trace": None,
        }
        for row in rows
    ]
    attention_boundaries = [
        event
        for row in attention_rows
        for event in boundary_events(
            row, pre=pre, post=post, curve_low=curve_low, curve_high=curve_high
        )
    ]
    attention_correct = [
        event for event in attention_boundaries if event["correct"] and event["control"] is not None
    ]
    attention_at_boundary = [
        event
        for event in attention_boundaries
        if event["prefix_clean"] and event["onset_at_boundary"]
    ]
    attention_onset = [
        pair
        for row in attention_rows
        for pair in onset_pairs(
            row, pre=pre, post=post, curve_low=curve_low, curve_high=curve_high
        )
    ]

    clean_rows = [row for row in rows if not row["label"].any()]
    drift = drift_report(clean_rows, bootstrap, seed)
    drift["clean_responses"] = len(clean_rows)
    drift["all_response_sensitivity"] = drift_report(rows, bootstrap, seed + 6)
    boundary_test = {
        "definition": (
            "at natural sentence-boundary proxies only: boundary entry change minus "
            "the mean of up to three distributed local controls in the same clean span"
        ),
        "primary_measure": (
            "functional evidence log-lift minus other-prompt log-lift, each relative "
            "to a visible-source capacity null"
        ),
        "correct_boundary_pairs": len(correct_with_control),
        "evidence_specificity": summarize_effect(
            correct_with_control,
            "boundary",
            "control",
            "evidence_entry",
            bootstrap=bootstrap,
            seed=seed + 10,
        ),
        "history_release_secondary": summarize_effect(
            correct_with_control,
            "boundary",
            "control",
            "history_entry_release",
            bootstrap=bootstrap,
            seed=seed + 11,
        ),
        "post_window_descriptive": summarize_effect(
            correct_with_control,
            "boundary",
            "control",
            "evidence_post_pulse",
            bootstrap=bootstrap,
            seed=seed + 12,
        ),
    }
    for number, stage in enumerate(("early", "middle", "late")):
        field = f"{stage}_evidence_entry"
        if correct_with_control and field in correct_with_control[0]["boundary"]:
            boundary_test[f"{stage}_layer_evidence"] = summarize_effect(
                correct_with_control,
                "boundary",
                "control",
                field,
                bootstrap=bootstrap,
                seed=seed + 13 + number,
            )

    exact_difference = matched_boundary_effect(
        correct,
        at_boundary,
        "evidence_entry",
        bootstrap=bootstrap,
        seed=seed + 20,
    )
    missed_test = {
        "definition": (
            "difference-in-differences: each exact-onset boundary entry change is "
            "paired to the nearest clean natural boundary in the same response"
        ),
        "expected_sign": "negative",
        "clean_boundaries": len(correct),
        "exact_boundary_hallucinations": len(at_boundary),
        "near_boundary_hallucinations": len(near_boundary),
        "late_onset_claims": len(late),
        "exact_boundary_primary": exact_difference,
        "near_boundary_sensitivity": boundary_group_effect(
            correct,
            near_boundary,
            "evidence_entry",
            bootstrap=bootstrap,
            seed=seed + 21,
        ),
        "late_onset_boundary_antecedent": boundary_group_effect(
            correct,
            late,
            "evidence_entry",
            bootstrap=bootstrap,
            seed=seed + 22,
        ),
        "warning": (
            "Near and late onsets are not pooled into the primary status. Curves "
            "after offset 0 describe persistence/recovery after generated tokens enter history."
        ),
    }

    onset_test = {
        "definition": (
            "secondary local association: hallucination-onset entry change minus a "
            "clean token in the same response; token identity is used only within a position caliper"
        ),
        "pairs": len(onset),
        "pair_classes": dict(Counter(pair["onset_class"] for pair in onset)),
        "token_matched_pairs": sum(pair["token_matched"] for pair in onset),
        "median_position_distance": (
            float(np.median([pair["position_distance"] for pair in onset])) if onset else None
        ),
        "evidence_entry": summarize_effect(
            onset,
            "positive",
            "control",
            "evidence_entry",
            bootstrap=bootstrap,
            seed=seed + 30,
        ),
        "post_window_descriptive": summarize_effect(
            onset,
            "positive",
            "control",
            "evidence_post_pulse",
            bootstrap=bootstrap,
            seed=seed + 31,
        ),
        "matched_discrimination_secondary": paired_association(
            onset,
            "positive",
            "control",
            "evidence_entry",
            bootstrap=bootstrap,
            seed=seed + 32,
        ),
    }

    attention_control = {
        "boundary_evidence_specificity": summarize_effect(
            attention_correct,
            "boundary",
            "control",
            "evidence_entry",
            bootstrap=bootstrap,
            seed=seed + 40,
        ),
        "exact_boundary_missed_entry": matched_boundary_effect(
            [event for event in attention_boundaries if event["correct"]],
            attention_at_boundary,
            "evidence_entry",
            bootstrap=bootstrap,
            seed=seed + 41,
        ),
        "onset_entry": summarize_effect(
            attention_onset,
            "positive",
            "control",
            "evidence_entry",
            bootstrap=bootstrap,
            seed=seed + 42,
        ),
        "functional_minus_attention_boundary": method_difference(
            correct_with_control,
            attention_correct,
            "boundary",
            "control",
            "evidence_entry",
            bootstrap=bootstrap,
            seed=seed + 43,
        ),
        "interpretation": (
            "If attention changes without functional AVW_O specificity, the effect "
            "is selection-only rather than transported-content evidence."
        ),
    }

    dependence = {
        "status": "run" if any(row.get("causal_cuts", False) for row in rows) else "not_run",
        "scope_warning": (
            "These are whole-response evidence-source cuts with MLPs active. They "
            "measure attention-mediated evidence dependence, not a localized re-anchor circuit."
        ),
        "direct_response_evidence": {
            "clean_boundary": optional_dependence(
                correct, "direct_evidence_cut_dependence", bootstrap, seed + 50
            ),
            "exact_hallucination_boundary": optional_dependence(
                at_boundary, "direct_evidence_cut_dependence", bootstrap, seed + 51
            ),
            "near_hallucination_boundary": optional_dependence(
                near_boundary, "direct_evidence_cut_dependence", bootstrap, seed + 52
            ),
            "late_onset_claim_boundary": optional_dependence(
                late, "direct_evidence_cut_dependence", bootstrap, seed + 53
            ),
        },
        "all_attention_paths_from_evidence": {
            "clean_boundary": optional_dependence(
                correct, "global_evidence_cut_dependence", bootstrap, seed + 54
            ),
            "exact_hallucination_boundary": optional_dependence(
                at_boundary, "global_evidence_cut_dependence", bootstrap, seed + 55
            ),
            "near_hallucination_boundary": optional_dependence(
                near_boundary, "global_evidence_cut_dependence", bootstrap, seed + 56
            ),
            "late_onset_claim_boundary": optional_dependence(
                late, "global_evidence_cut_dependence", bootstrap, seed + 57
            ),
        },
    }

    width = curve_high - curve_low + 1
    curves = {
        "offset": np.arange(curve_low, curve_high + 1).tolist(),
        "correct_boundary_evidence": event_curve_report(
            correct, "boundary", "evidence_curve", bootstrap=bootstrap, seed=seed + 60, width=width
        ),
        "within_claim_control_evidence": event_curve_report(
            correct_with_control, "control", "evidence_curve", bootstrap=bootstrap, seed=seed + 61, width=width
        ),
        "hallucination_onset_evidence": event_curve_report(
            onset, "positive", "evidence_curve", bootstrap=bootstrap, seed=seed + 62, width=width
        ),
        "matched_token_evidence": event_curve_report(
            onset, "control", "evidence_curve", bootstrap=bootstrap, seed=seed + 63, width=width
        ),
        "correct_boundary_history_release": event_curve_report(
            correct, "boundary", "history_curve", bootstrap=bootstrap, seed=seed + 64, width=width
        ),
        "hallucination_onset_history_release": event_curve_report(
            onset, "positive", "history_curve", bootstrap=bootstrap, seed=seed + 65, width=width
        ),
    }

    drift_status = joint_status(
        direction(drift["evidence_enrichment_change"], -1),
        direction(drift["history_enrichment_change"], 1),
    )
    boundary_status = direction(boundary_test["evidence_specificity"], 1)
    missed_association_status = direction(exact_difference, -1)
    missed_status = (
        missed_association_status
        if boundary_status == "supported"
        else "not_tested_without_H2"
    )
    missed_test["association_status_before_H2_gate"] = missed_association_status
    missed_test["phenomenon_status_after_H2_gate"] = missed_status
    observer_status = {
        "H1_exposure_adjusted_preference_drift": drift_status,
        "H2_natural_boundary_evidence_specificity": boundary_status,
        "H3_exact_boundary_missed_entry_association": missed_status,
    }

    same_model_count = sum(
        same_model(row.get("generator_model", ""), row.get("observer_model", ""))
        for row in rows
    )
    generation_scope = same_model_count == len(rows)
    generation_status = (
        observer_status
        if generation_scope
        else {name: "not_tested_for_generation" for name in observer_status}
    )
    total_spans = sum(len(row["claim_start"]) for row in rows)
    forced = sum(
        int(np.count_nonzero(row["claim_boundary_kind"] == FORCED_CHUNK)) for row in rows
    )
    return {
        "task": task,
        "samples": len(rows),
        "sources": len({row["source_id"] for row in rows}),
        "hallucination_runs": int(sum(len(positive_onsets(row["label"])) for row in rows)),
        "window": {
            "pre": pre,
            "post_descriptive": post,
            "curve_low": curve_low,
            "curve_high": curve_high,
            "primary_pre_outcome_entry_offset": 0,
        },
        "evaluation_config": {"bootstrap": bootstrap, "task_seed": seed},
        "event_selection": {
            "all_sentence_like_spans": total_spans,
            "forced_length_boundaries_excluded": forced,
            "natural_boundaries_with_complete_scalar_window": len(boundaries),
            "correct_boundaries_with_within_span_control": len(correct_with_control),
            "note": "Curve completeness never controls scalar-test inclusion.",
        },
        "normal_autoregressive_drift": drift,
        "correct_boundary_vs_within_claim": boundary_test,
        "missed_reanchor_at_claim_boundary": missed_test,
        "hallucination_onset_vs_matched_token_secondary": onset_test,
        "attention_only_control": attention_control,
        "whole_evidence_dependence_control": dependence,
        "event_curves": curves,
        "observer_hypothesis_status": observer_status,
        "hypothesis_status": generation_status,
        "recommended_next_step": next_step(
            drift_status, boundary_status, missed_status, generation_scope
        ),
        "model_scope": {
            "observer_models": dict(Counter(row.get("observer_model") or "unknown" for row in rows)),
            "generator_models": dict(Counter(row.get("generator_model") or "unknown" for row in rows)),
            "same_generator_observer_samples": same_model_count,
            "generation_claims_allowed": generation_scope,
            "interpretation": (
                "same-generator observer trace under teacher forcing"
                if generation_scope
                else "teacher-forced observer processing; generation-mechanism status is withheld"
            ),
        },
        "measurement_scope": {
            "boundary": "natural punctuation proxy, not a gold atomic fact boundary",
            "evidence": (
                "complete RAG evidence span; evidence-vs-other-prompt specificity is tested, "
                "but claim-specific support/validator alignment is still required"
            ),
            "route": (
                "layer-resolved A and A*||W_OV|| relative to availability nulls; "
                "this is observed routing magnitude, not signed causal attribution"
            ),
        },
        "decision_rule": (
            "A hypothesis is supported only when its source-bootstrap 95% interval "
            "has the preregistered sign. Crossing zero means inconclusive, never absence."
        ),
    }
