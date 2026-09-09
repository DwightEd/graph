"""Post-hoc label join for a completed, label-free mechanism subset."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from .artifact_schema import AUDIT_SCHEMA
from .artifacts import save_json
from .dataset import LabelSource, RagTruthLabelSource
from .route_model import EVIDENCE, RESPONSE
from .subset import MANIFEST_NAME, MANIFEST_SCHEMA

REPORT_NAME = "mechanism_evaluation.json"

# These are fixed, untrained axes. Direction only converts each raw value to
# hallucination risk for evaluation; it does not alter the stored mechanism.
AXIS_DIRECTION = {
    "route_origin_competition": 1.0,
    "evidence_adoption": -1.0,
}
NEUTRAL_AXES = ("temporal_switch_score",)
RAW_AXIS_NAMES = (*AXIS_DIRECTION, *NEUTRAL_AXES)
REANCHOR_SOURCE_KINDS = ("prompt_evidence", "other_prompt", "remote_response")
EVALUATION_FIELDS = (
    "selected_root_evaluated",
    "corridor_evaluated",
)
CONFIRMATION_FIELDS = (
    "selected_root_confirmed",
    "corridor_confirmed",
    "corridor_restoration_valid",
    "carrier_any_confirmed",
    "full_chain_confirmed",
)
EFFECT_FIELDS = (
    "root_value_effect",
    "selected_root_value_necessity",
    "selected_root_causal_score",
    "corridor_necessity",
    "corridor_conditional_rescue",
    "corridor_mediated_rescue",
)
ORIGIN_COMPONENT_FIELDS = (
    "route_evidence_origin_action",
    "route_evidence_origin_signed_sum",
    "route_evidence_origin_absolute_budget",
    "route_evidence_origin_head_agreement",
    "route_response_origin_action",
    "route_response_origin_signed_sum",
    "route_response_origin_absolute_budget",
    "route_response_origin_head_agreement",
)
QUERY_COVERAGE_FIELDS = (
    "route_query_retained_fraction",
    "route_query_unobserved_fraction",
)
CONFIRMATION_DENOMINATOR = {
    "selected_root_confirmed": "selected_root_evaluated",
    "corridor_confirmed": "corridor_evaluated",
    "corridor_restoration_valid": "corridor_evaluated",
    "carrier_any_confirmed": "carrier_evaluated",
    "full_chain_confirmed": "full_chain_evaluated",
}


def _item(artifact: Mapping[str, object], name: str):
    value = np.asarray(artifact[name])
    if value.shape != ():
        raise ValueError(f"mechanism artifact {name} must be scalar")
    return value.item()


def _slot(position: object, target: int, *, name: str) -> int:
    matches = np.flatnonzero(np.asarray(position, dtype=np.int64) == target)
    if len(matches) != 1:
        raise ValueError(f"target position is absent from {name}")
    return int(matches[0])


def _origin_action(action: np.ndarray) -> tuple[float, float, float, float]:
    """Aggregate heads without averaging or erasing within-layer conflict."""

    layer_signed = action.sum(axis=1)
    layer_budget = np.abs(action).sum(axis=1)
    layer_agreement = np.divide(
        np.abs(layer_signed),
        layer_budget,
        out=np.zeros_like(layer_signed),
        where=layer_budget > 0,
    )
    origin_action = float(np.sum(layer_signed * layer_agreement))
    signed_sum = float(layer_signed.sum())
    absolute_budget = float(layer_budget.sum())
    head_agreement = (
        float(np.sum(np.abs(layer_signed)) / absolute_budget)
        if absolute_budget > 0
        else 0.0
    )
    return origin_action, signed_sum, absolute_budget, head_agreement


def _target_reanchor_axes(
    artifact: Mapping[str, object],
    query: int,
    row_slot: int,
    route_shape: tuple[int, int, int],
) -> dict[str, float | bool | str]:
    """Return selection-aware axes for this artifact's own query.

    Candidate rows earlier than ``query`` can describe geometry on a route to
    the audited target, but their downstream action is not an immediate
    next-token effect.  They therefore cannot be joined to this target's
    label.  Likewise, context rows around a selected event and no-event
    fallbacks are not silently relabeled as event centers.
    """

    recorded = bool(_item(artifact, "target_reanchor_selection_recorded"))
    has_event = bool(_item(artifact, "target_reanchor_has_event"))
    is_center = bool(_item(artifact, "target_reanchor_is_center"))
    fallback = bool(_item(artifact, "target_reanchor_fallback"))
    center = int(_item(artifact, "target_reanchor_center_position"))
    offset = int(_item(artifact, "target_reanchor_window_offset"))
    layer = int(_item(artifact, "target_reanchor_layer"))
    head = int(_item(artifact, "target_reanchor_head"))
    source_kind = str(_item(artifact, "target_reanchor_source_kind"))
    selection_score = float(_item(artifact, "target_reanchor_score"))
    if has_event:
        if (
            not recorded
            or fallback
            or query - center != offset
            or source_kind not in {"prompt_evidence", "other_prompt", "remote_response"}
            or not 0 <= layer < route_shape[0]
            or not 0 <= head < route_shape[1]
            or not math.isfinite(selection_score)
            or selection_score <= 0
        ):
            raise ValueError("target re-anchor event selection is inconsistent")
        if is_center != (offset == 0):
            raise ValueError("target re-anchor center flag disagrees with its offset")
    elif is_center or fallback and not recorded:
        raise ValueError("target re-anchor non-event selection is inconsistent")

    temporal_score = float("nan")
    recomputed_score = float("nan")
    temporal_evaluated = False
    evidence_adoption = float("nan")
    evidence_adoption_evaluated = False
    if is_center:
        if center != query:
            raise ValueError("target re-anchor center names another query")
        dense_score = np.asarray(artifact["reanchor_score"], dtype=np.float64)
        if dense_score.shape != route_shape:
            raise ValueError(
                "re-anchor score must share the route [layer,head,row] axes"
            )
        current_score = float(dense_score[layer, head, row_slot])
        if not math.isfinite(current_score) or current_score < 0:
            raise ValueError("current-target structural switch score is invalid")
        # Preserve the full-response discovery score.  A shorter bf16 prefix
        # can change the continuous score or even a near-tied source winner;
        # expose that difference without reranking the frozen event.
        recomputed_score = current_score
        temporal_score = selection_score
        temporal_evaluated = True

        if source_kind == "prompt_evidence":
            bucket_name = np.asarray(artifact["reanchor_bucket_name"]).astype(str)
            bucket = np.flatnonzero(bucket_name == "prompt_evidence")
            if len(bucket) != 1:
                raise ValueError("re-anchor bucket table lacks prompt evidence")
            bucket_action = np.asarray(
                artifact["reanchor_bucket_downstream_action"], dtype=np.float64
            )
            expected = (*route_shape, len(bucket_name))
            if bucket_action.shape != expected:
                raise ValueError("re-anchor bucket action has the wrong shape")
            evidence_adoption = float(
                bucket_action[layer, head, row_slot, int(bucket[0])]
            )
            if not math.isfinite(evidence_adoption):
                raise ValueError("prompt-evidence adoption action is not finite")
            evidence_adoption_evaluated = True

    # A bounded candidate table may contain earlier downstream-to-query events.
    # Validate its marker, but never consume its downstream action here.
    position = np.asarray(artifact["reanchor_candidate_position"], dtype=np.int64)
    current = np.asarray(
        artifact["reanchor_candidate_current_target_match"], dtype=bool
    )
    if position.ndim != 1 or current.shape != position.shape:
        raise ValueError("re-anchor candidate fields must be aligned vectors")
    if bool(np.any(current & (position != query))):
        raise ValueError("current-target re-anchor candidate has another position")
    return {
        "temporal_switch_score": temporal_score,
        "temporal_switch_recomputed_score": recomputed_score,
        "temporal_switch_score_delta": recomputed_score - temporal_score,
        "temporal_switch_evaluated": temporal_evaluated,
        "target_reanchor_selection_recorded": recorded,
        "target_reanchor_has_event": has_event,
        "target_reanchor_is_center": is_center,
        "target_reanchor_fallback": fallback,
        "target_reanchor_window_offset": offset,
        "temporal_switch_source_kind": source_kind,
        "evidence_adoption": evidence_adoption,
        "evidence_adoption_evaluated": evidence_adoption_evaluated,
    }


def mechanism_axes(artifact: Mapping[str, object]) -> dict[str, float | bool | str]:
    """Compute fixed raw query axes and one optional exact diagnostic.

    Route-origin competition remains the static baseline.  Temporal switch is
    transport-only, while evidence adoption is signed target action at this
    artifact's query.  They remain separate so evaluation cannot manufacture a
    favorable composite after seeing labels.  Exact intervention values never
    enter any raw axis and remain missing when not evaluated.
    """

    query = int(_item(artifact, "query_position"))
    row_slot = _slot(artifact["route_row_position"], query, name="route rows")

    action = np.asarray(artifact["route_head_action"], dtype=np.float64)
    if action.ndim != 4:
        raise ValueError("head action tensor must use [layer,head,row,origin]")

    row_total = np.asarray(artifact["route_row_total"], dtype=np.float64)
    row_retained = np.asarray(artifact["route_row_retained"], dtype=np.float64)
    if row_total.shape != action.shape[:3] or row_retained.shape != row_total.shape:
        raise ValueError("row coverage tensors disagree with route rows")
    query_total = float(row_total[:, :, row_slot].sum())
    query_retained = float(row_retained[:, :, row_slot].sum())
    query_unobserved = max(query_total - query_retained, 0.0)
    query_retained_fraction = (
        query_retained / query_total if query_total > 0 else float("nan")
    )
    query_unobserved_fraction = (
        query_unobserved / query_total if query_total > 0 else float("nan")
    )

    evidence_action = action[:, :, row_slot, EVIDENCE]
    (
        evidence_origin_action,
        evidence_signed,
        evidence_budget,
        evidence_agreement,
    ) = _origin_action(evidence_action)
    reanchor_axes = _target_reanchor_axes(artifact, query, row_slot, action.shape[:3])
    response_action = action[:, :, row_slot, RESPONSE]
    (
        response_origin_action,
        response_signed,
        response_budget,
        response_agreement,
    ) = _origin_action(response_action)
    combined_action_budget = response_budget + evidence_budget
    route_origin_competition_evaluated = combined_action_budget > 0
    route_origin_competition = (
        (response_origin_action - evidence_origin_action)
        / (combined_action_budget + np.finfo(np.float64).eps)
        if route_origin_competition_evaluated
        else float("nan")
    )

    exact_evaluated = bool(
        _item(artifact, "selected_root_evaluated")
        and _item(artifact, "corridor_evaluated")
    )
    exact_bottleneck = float("nan")
    if exact_evaluated:
        exact_values = np.asarray(
            [
                float(_item(artifact, "selected_root_value_necessity")),
                float(_item(artifact, "corridor_necessity")),
                float(_item(artifact, "corridor_mediated_rescue")),
            ],
            dtype=np.float64,
        )
        if np.all(np.isfinite(exact_values)):
            root_effect = float(_item(artifact, "root_value_effect"))
            direction = 1.0 if root_effect >= 0 else -1.0
            exact_bottleneck = float((direction * exact_values).min())
    return {
        "route_origin_competition": float(route_origin_competition),
        "route_origin_competition_evaluated": route_origin_competition_evaluated,
        **reanchor_axes,
        "route_query_total_mass": query_total,
        "route_query_retained_mass": query_retained,
        "route_query_unobserved_mass": query_unobserved,
        "route_query_retained_fraction": query_retained_fraction,
        "route_query_unobserved_fraction": query_unobserved_fraction,
        "route_evidence_origin_action": evidence_origin_action,
        "route_evidence_origin_signed_sum": evidence_signed,
        "route_evidence_origin_absolute_budget": evidence_budget,
        "route_evidence_origin_head_agreement": evidence_agreement,
        "route_response_origin_action": response_origin_action,
        "route_response_origin_signed_sum": response_signed,
        "route_response_origin_absolute_budget": response_budget,
        "route_response_origin_head_agreement": response_agreement,
        "selected_root_exact_bottleneck": exact_bottleneck,
        "selected_root_exact_evaluated": exact_evaluated,
    }


def _mean(rows: list[dict], name: str) -> float | None:
    values = [
        float(row[name])
        for row in rows
        if row.get(name) is not None and math.isfinite(float(row[name]))
    ]
    return float(np.mean(values)) if values else None


def _rate(rows: list[dict], name: str) -> float | None:
    return float(np.mean([bool(row[name]) for row in rows])) if rows else None


def _confirmation_rates(
    rows: list[dict],
) -> tuple[dict[str, float | None], dict[str, int]]:
    rates = {}
    evaluated_targets = {}
    for name, denominator in CONFIRMATION_DENOMINATOR.items():
        evaluated = [row for row in rows if bool(row[denominator])]
        evaluated_targets[name] = len(evaluated)
        rates[name] = (
            float(np.mean([bool(row[name]) for row in evaluated]))
            if evaluated
            else None
        )
    return rates, evaluated_targets


def summarize(rows: list[dict]) -> dict:
    confirmation_rate, confirmation_evaluated_targets = _confirmation_rates(rows)
    return {
        "targets": len(rows),
        "samples": len({row["sample_id"] for row in rows}),
        "evaluation_rate": {
            **{name: _rate(rows, name) for name in EVALUATION_FIELDS},
            "route_origin_competition_evaluated": _rate(
                rows, "route_origin_competition_evaluated"
            ),
            "temporal_switch_evaluated": _rate(rows, "temporal_switch_evaluated"),
            "evidence_adoption_evaluated": _rate(rows, "evidence_adoption_evaluated"),
            "target_reanchor_selection_recorded": _rate(
                rows, "target_reanchor_selection_recorded"
            ),
            "target_reanchor_has_event": _rate(rows, "target_reanchor_has_event"),
            "target_reanchor_is_center": _rate(rows, "target_reanchor_is_center"),
            "target_reanchor_fallback": _rate(rows, "target_reanchor_fallback"),
            "carrier_evaluated": _rate(rows, "carrier_evaluated"),
            "full_chain_evaluated": _rate(rows, "full_chain_evaluated"),
        },
        "confirmation_rate": confirmation_rate,
        "confirmation_evaluated_targets": confirmation_evaluated_targets,
        "mean_exact_effect": {name: _mean(rows, name) for name in EFFECT_FIELDS},
        "mean_raw_axis": {name: _mean(rows, name) for name in RAW_AXIS_NAMES},
        "mean_origin_component": {
            name: _mean(rows, name) for name in ORIGIN_COMPONENT_FIELDS
        },
        "mean_query_coverage": {
            name: _mean(rows, name) for name in QUERY_COVERAGE_FIELDS
        },
    }


def raw_axis_evaluation(rows: list[dict]) -> dict[str, dict]:
    """Evaluate separate raw axes without fitting a classifier.

    A long-range structural switch has no universal hallucination direction:
    prompt evidence, instructions, and remote response hubs can have different
    roles.  Its main AUROC/AUPRC therefore use a transparent raw-higher
    reporting convention and also report the negated orientation.  Directional
    axes retain their independently registered risk direction.
    """

    result = {}
    for name in RAW_AXIS_NAMES:
        direction = AXIS_DIRECTION.get(name)
        finite = [
            row
            for row in rows
            if row.get(name) is not None and math.isfinite(float(row[name]))
        ]
        label = np.asarray(
            [row["hallucination_label"] for row in finite], dtype=np.int8
        )
        raw = np.asarray([row[name] for row in finite], dtype=np.float64)
        risk = raw if direction is None else direction * raw
        metric = {
            "total_targets": len(rows),
            "evaluated_targets": len(label),
            "targets": len(label),
            "positives": int(label.sum()),
            "prevalence": float(label.mean()) if len(label) else None,
            "hallucination_direction": (
                "neutral_raw_higher_reporting_convention"
                if direction is None
                else "lower"
                if direction < 0
                else "higher"
            ),
            "auroc": None,
            "auprc": None,
        }
        if direction is None:
            metric["negated_auroc"] = None
            metric["negated_auprc"] = None
        if len(np.unique(label)) == 2:
            metric["auroc"] = float(roc_auc_score(label, risk))
            metric["auprc"] = float(average_precision_score(label, risk))
            if direction is None:
                metric["negated_auroc"] = float(roc_auc_score(label, -raw))
                metric["negated_auprc"] = float(average_precision_score(label, -raw))
        result[name] = metric
    return result


def temporal_switch_by_source_kind(rows: list[dict]) -> dict[str, dict]:
    """Stratify event-center discrimination without assigning a shared meaning."""

    result = {}
    for source_kind in REANCHOR_SOURCE_KINDS:
        source_rows = [
            row
            for row in rows
            if bool(row["temporal_switch_evaluated"])
            and row["temporal_switch_source_kind"] == source_kind
        ]
        result[source_kind] = raw_axis_evaluation(source_rows)["temporal_switch_score"]
    return result


def _capture_rows(output: Path, manifest: dict) -> list[dict]:
    """Load all label-free artifacts before the label store is opened."""

    rows = []
    entries = tuple(manifest["audits"].values())
    for entry in tqdm(
        entries,
        desc=f"{output.name} evaluation artifacts",
        unit="artifact",
        dynamic_ncols=True,
    ):
        with np.load(output / entry["result"], allow_pickle=False) as artifact:
            if int(_item(artifact, "subset_audit_schema")) != AUDIT_SCHEMA:
                raise ValueError("unsupported subset audit schema")
            row = {
                "sample_id": str(_item(artifact, "dataset_sample_id")),
                "task_type": str(_item(artifact, "task_type")),
                "query_position": int(_item(artifact, "query_position")),
                "prediction_position": int(_item(artifact, "prediction_position")),
                "response_start": int(_item(artifact, "response_start")),
            }
            row.update(
                {name: bool(_item(artifact, name)) for name in EVALUATION_FIELDS}
            )
            row["carrier_evaluated_count"] = int(
                _item(artifact, "carrier_evaluated_count")
            )
            row["carrier_evaluated"] = row["carrier_evaluated_count"] > 0
            row["full_chain_evaluated"] = bool(_item(artifact, "full_chain_evaluated"))
            row.update(
                {name: bool(_item(artifact, name)) for name in CONFIRMATION_FIELDS}
            )
            row.update({name: float(_item(artifact, name)) for name in EFFECT_FIELDS})
            row.update(mechanism_axes(artifact))
            rows.append(row)
    return rows


def _join_labels(
    label_source: LabelSource,
    rows: list[dict],
    *,
    sample_ids: list[str] | None = None,
) -> dict[str, np.ndarray]:
    sample_ids = tuple(
        dict.fromkeys(
            sample_ids if sample_ids is not None else [row["sample_id"] for row in rows]
        )
    )
    label_by_sample = label_source.load(sample_ids)

    for row in rows:
        relative = row.pop("prediction_position") - row.pop("response_start")
        sample_label = label_by_sample[row["sample_id"]]
        if not 0 <= relative < len(sample_label):
            raise ValueError(
                f"audit target lies outside labels: {row['sample_id']} "
                f"q={row['query_position']}"
            )
        row["hallucination_label"] = int(sample_label[relative])
    return label_by_sample


def _json_rows(rows: list[dict]) -> list[dict]:
    """Represent unavailable numeric diagnostics as JSON null, never zero."""

    return [
        {
            name: (
                None
                if isinstance(value, (float, np.floating))
                and not math.isfinite(float(value))
                else value
            )
            for name, value in row.items()
        }
        for row in rows
    ]


def evaluate_subset_split(
    dataset_location: str | Path,
    output_root: str | Path,
    *,
    label_source: LabelSource | None = None,
    plot: bool = False,
) -> dict:
    """Join labels after capture, then summarize mechanisms and fixed axes."""

    dataset_location = Path(dataset_location)
    output = Path(output_root)
    manifest_path = output / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("subset_manifest_schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported subset manifest schema")
    if not manifest.get("analysis_complete"):
        raise ValueError("subset capture is incomplete")
    if manifest.get("labels_used_for_capture") is not False:
        raise ValueError("capture manifest violates the label firewall")
    config = manifest["config"]
    location_key = (
        "dataset_manifest" if "dataset_manifest" in config else "dataset_root"
    )
    captured_location = Path(config[location_key]).resolve()
    if captured_location != dataset_location.resolve():
        raise ValueError(
            f"evaluation {location_key} differs from capture manifest"
        )
    if label_source is None:
        if location_key != "dataset_root":
            raise ValueError("manifest evaluation requires an explicit label source")
        label_source = RagTruthLabelSource(dataset_location)

    rows = _capture_rows(output, manifest)
    label_by_sample = _join_labels(
        label_source, rows, sample_ids=list(manifest["samples"])
    )

    groups = {}
    task_names = [
        "ALL",
        *sorted(
            {row["task_type"] for row in rows}
            | {entry["task_type"] for entry in manifest["selection"]}
        ),
    ]
    for task in task_names:
        task_rows = (
            rows if task == "ALL" else [row for row in rows if row["task_type"] == task]
        )
        groups[task] = {
            "all": summarize(task_rows),
            "clean": summarize(
                [row for row in task_rows if row["hallucination_label"] == 0]
            ),
            "hallucinated": summarize(
                [row for row in task_rows if row["hallucination_label"] == 1]
            ),
            "raw_axis_evaluation": raw_axis_evaluation(task_rows),
            "temporal_switch_by_source_kind": temporal_switch_by_source_kind(task_rows),
        }

    report = {
        "subset_evaluation_schema": 5,
        "labels_accessed_after_capture": True,
        "analysis_scope": manifest.get("analysis_scope", "selected_target_function"),
        "selection_is_not_population_evaluation": True,
        "hypothesis_status": (
            "unvalidated raw axes: route-origin competition retains a fixed "
            "higher-risk direction; prompt-evidence adoption retains a fixed "
            "lower-risk direction; mixed-source temporal switches are "
            "direction-neutral and reported in raw and negated orientations"
        ),
        "claim_scope": (
            "label-free event-center measurements for the teacher-forced "
            "observed-token contrast; event-window context rows, no-event "
            "fallbacks, and earlier route events are excluded from temporal and "
            "adoption axes; no axis establishes factual accuracy or free-running "
            "generation causality"
        ),
        "axis_definition": {
            "route_origin_competition": (
                "(response-origin layer-consistent signed action minus "
                "evidence-origin layer-consistent signed action) divided by "
                "their combined absolute head-action budget; higher is the "
                "fixed hallucination-risk direction"
            ),
            "temporal_switch_score": (
                "frozen transport-only local-to-long-range dominance-flip score "
                "at the structurally selected layer and head, only when this "
                "artifact's query is the selected event center; source kind is "
                "preserved per target, context/fallback rows and earlier-event "
                "actions are excluded, and both raw and negated AUROC/AUPRC are "
                "reported overall and by source kind because mixed-source switches "
                "have no universal direction"
            ),
            "evidence_adoption": (
                "full-row prompt-evidence bucket signed grad-message action at "
                "the structurally selected event layer, head, and current query; "
                "defined only for prompt-evidence event centers, with higher "
                "registered as lower hallucination risk; this is observed-token "
                "support, not source-specific causal mediation or factual accuracy"
            ),
        },
        "axis_role": {
            "route_origin_competition": "static baseline",
            "temporal_switch_score": (
                "direction-neutral event-center structural hypothesis"
            ),
            "evidence_adoption": ("prompt-evidence event-center functional hypothesis"),
        },
        "evaluation_limitations": {
            "label_firewall": (
                "labels are joined only after label-free capture and cannot enter "
                "event discovery, target selection, or raw-axis construction"
            ),
            "selection": (
                "event-conditioned targets do not estimate population performance; "
                "context and fallback rows are reported descriptively but do not "
                "enter event-center AUROC/AUPRC"
            ),
            "teacher_forcing": (
                "signed action explains the recorded next-token contrast under the "
                "observed prefix; it does not measure counterfactual free-running "
                "sequence behavior"
            ),
        },
        "secondary_diagnostic_definition": {
            "temporal_switch_recomputed_score": (
                "prefix recomputation at the frozen event layer/head/query; "
                "zero can mean the switch did not reproduce; never used to "
                "replace or rerank the discovery event"
            ),
            "temporal_switch_score_delta": (
                "prefix-recomputed score minus frozen full-response discovery "
                "score; a numerical/reproduction diagnostic, not a risk axis"
            ),
            "selected_root_exact_bottleneck": (
                "minimum selected-root/corridor exact effect aligned to the "
                "measured root-effect direction; null unless both intervention "
                "stages were evaluated and never combined with the all-evidence "
                "discovery score"
            ),
        },
        "groups": groups,
        "targets": _json_rows(rows),
    }
    from .cohort_plot import COHORT_REPORT_NAME, summarize_cohort

    cohort = summarize_cohort(
        output, manifest, label_by_sample, target_rows=rows, plot=plot
    )
    report["cohort_report"] = COHORT_REPORT_NAME
    report["cohort_plots"] = cohort["plots"]
    report["full_scan_coverage"] = {
        task: group["coverage"] for task, group in cohort["groups"].items()
    }
    save_json(output / REPORT_NAME, report)
    return report
