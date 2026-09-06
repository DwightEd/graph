"""Post-hoc label join for a completed, label-free mechanism subset."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from research_dataset import open_research_dataset

from .artifacts import save_json
from .route_model import EVIDENCE, RESPONSE
from .subset import MANIFEST_NAME, MANIFEST_SCHEMA
from .subset_artifacts import AUDIT_SCHEMA

REPORT_NAME = "mechanism_evaluation.json"

# These are fixed, untrained axes. Direction only converts each raw value to
# hallucination risk for evaluation; it does not alter the stored mechanism.
AXIS_DIRECTION = {
    "native_source_mediated_observed_margin": -1.0,
    "response_origin_supporting_action_candidate": 1.0,
}
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


def _agreement(action: np.ndarray) -> tuple[float, float, float]:
    """Return signed sum, absolute budget, and cancellation agreement."""

    signed = float(action.sum())
    budget = float(np.abs(action).sum())
    agreement = abs(signed) / budget if budget > 0 else 0.0
    return signed, budget, agreement


def _weighted(values: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(weight.sum())
    return float(np.dot(values, weight) / denominator) if denominator > 0 else 0.0


def mechanism_axes(artifact: Mapping[str, object]) -> dict[str, float | bool]:
    """Compute two registered raw axes while retaining head-resolved inputs.

    Native-source support is an operator-specific logit-margin bottleneck. The
    exact value is admitted only when the root-lineage route, source-cut
    integration ledger, and exact intervention ladder agree on positive
    support for the observed-token contrast. Response-origin supporting action
    is a separate first-order candidate. It is not called lock-in because this
    audit does not intervene on response history.
    """

    query = int(_item(artifact, "query_position"))
    row_slot = _slot(artifact["route_row_position"], query, name="route rows")
    stage_slot = _slot(artifact["route_stage_position"], query, name="stage trace")

    transport = np.asarray(artifact["route_head_transport"], dtype=np.float64)
    action = np.asarray(artifact["route_head_action"], dtype=np.float64)
    integration = np.asarray(artifact["route_head_integration"], dtype=np.float64)
    if transport.shape != action.shape or transport.ndim != 4:
        raise ValueError("head transport/action tensors disagree")
    if integration.shape[:3] != action.shape[:3] or integration.shape[-1] != 4:
        raise ValueError("head integration tensor disagrees with route rows")

    evidence_transport = float(transport[:, :, row_slot, EVIDENCE].sum())
    evidence_action = action[:, :, row_slot, EVIDENCE]
    evidence_signed, evidence_budget, evidence_agreement = _agreement(evidence_action)
    integration_action = integration[:, :, row_slot, 3]
    integration_signed, integration_budget, integration_agreement = _agreement(
        integration_action
    )

    exact_values = np.asarray(
        [
            float(_item(artifact, "selected_root_value_necessity")),
            float(_item(artifact, "corridor_necessity")),
            float(_item(artifact, "corridor_mediated_rescue")),
        ],
        dtype=np.float64,
    )
    exact_bottleneck = float(exact_values.min())
    exact_confirmed = bool(
        _item(artifact, "selected_root_confirmed")
        and _item(artifact, "corridor_confirmed")
        and _item(artifact, "corridor_restoration_valid")
    )
    lineage_present = evidence_transport > 0
    evidence_supporting = evidence_signed > 0
    integration_supporting = integration_signed > 0
    source_support_gate = bool(
        exact_confirmed
        and lineage_present
        and evidence_supporting
        and integration_supporting
        and exact_bottleneck > 0
    )
    source_support = exact_bottleneck if source_support_gate else 0.0

    response_action = action[:, :, row_slot, RESPONSE]
    response_layer_signed = response_action.sum(axis=1)
    response_layer_budget = np.abs(response_action).sum(axis=1)
    response_layer_agreement = np.divide(
        np.abs(response_layer_signed),
        response_layer_budget,
        out=np.zeros_like(response_layer_signed),
        where=response_layer_budget > 0,
    )
    continuity = np.asarray(artifact["route_state_continuity"], dtype=np.float64)[
        :, stage_slot
    ]
    positive_response = np.clip(response_layer_signed, 0.0, None)
    response_support = float(np.sum(positive_response * response_layer_agreement))
    response_signed, response_budget, response_agreement = _agreement(response_action)

    module_agreement = np.asarray(
        artifact["route_module_functional_agreement"], dtype=np.float64
    )[:, stage_slot]
    module_cosine = np.asarray(
        artifact["route_module_vector_cosine"], dtype=np.float64
    )[:, stage_slot]
    cross_head_vector = np.asarray(
        artifact["route_cross_head_vector_coherence"], dtype=np.float64
    )[:, row_slot]
    cross_head_function = np.asarray(
        artifact["route_cross_head_functional_agreement"], dtype=np.float64
    )[:, row_slot]
    integration_layer_budget = np.abs(integration_action).sum(axis=1)
    return {
        "native_source_mediated_observed_margin": source_support,
        "native_source_exact_bottleneck_ungated": exact_bottleneck,
        "native_source_support_gate": source_support_gate,
        "selected_source_lineage_present": lineage_present,
        "selected_source_action_supporting": evidence_supporting,
        "selected_source_integration_supporting": integration_supporting,
        "selected_source_transport_sum": evidence_transport,
        "selected_source_action_signed_sum": evidence_signed,
        "selected_source_action_absolute_budget": evidence_budget,
        "selected_source_action_functional_agreement": evidence_agreement,
        "selected_source_integration_action_signed_sum": integration_signed,
        "selected_source_integration_action_absolute_budget": integration_budget,
        "selected_source_integration_functional_agreement": integration_agreement,
        "response_origin_supporting_action_candidate": response_support,
        "response_origin_action_signed_sum": response_signed,
        "response_origin_action_absolute_budget": response_budget,
        "response_origin_functional_agreement": response_agreement,
        "source_conditioned_state_continuity": _weighted(
            continuity, integration_layer_budget
        ),
        # Module diagnostics remain separate because they are conditioned on
        # the selected-root cut, not a response-origin intervention.
        "source_conditioned_module_functional_agreement": _weighted(
            module_agreement, integration_layer_budget
        ),
        "source_conditioned_module_vector_cosine": _weighted(
            module_cosine, integration_layer_budget
        ),
        "source_conditioned_cross_head_vector_coherence": _weighted(
            cross_head_vector, integration_layer_budget
        ),
        "source_conditioned_cross_head_functional_agreement": _weighted(
            cross_head_function, integration_layer_budget
        ),
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


def summarize(rows: list[dict]) -> dict:
    return {
        "targets": len(rows),
        "samples": len({row["sample_id"] for row in rows}),
        "confirmation_rate": {name: _rate(rows, name) for name in CONFIRMATION_FIELDS},
        "mean_exact_effect": {name: _mean(rows, name) for name in EFFECT_FIELDS},
        "mean_raw_axis": {name: _mean(rows, name) for name in AXIS_DIRECTION},
    }


def raw_axis_evaluation(rows: list[dict]) -> dict[str, dict]:
    """Evaluate the two fixed axes without fitting a classifier or threshold."""

    result = {}
    for name, direction in AXIS_DIRECTION.items():
        finite = [
            row
            for row in rows
            if row.get(name) is not None and math.isfinite(float(row[name]))
        ]
        label = np.asarray(
            [row["hallucination_label"] for row in finite], dtype=np.int8
        )
        risk = direction * np.asarray([row[name] for row in finite], dtype=np.float64)
        metric = {
            "targets": len(label),
            "positives": int(label.sum()),
            "prevalence": float(label.mean()) if len(label) else None,
            "hallucination_direction": "lower" if direction < 0 else "higher",
            "auroc": None,
            "auprc": None,
        }
        if len(np.unique(label)) == 2:
            metric["auroc"] = float(roc_auc_score(label, risk))
            metric["auprc"] = float(average_precision_score(label, risk))
        result[name] = metric
    return result


def _capture_rows(output: Path, manifest: dict) -> list[dict]:
    """Load all label-free artifacts before the label store is opened."""

    rows = []
    for entry in manifest["audits"].values():
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
                {name: bool(_item(artifact, name)) for name in CONFIRMATION_FIELDS}
            )
            row.update({name: float(_item(artifact, name)) for name in EFFECT_FIELDS})
            row.update(mechanism_axes(artifact))
            rows.append(row)
    return rows


def _join_labels(dataset_root: Path, rows: list[dict]) -> None:
    sample_ids = list(dict.fromkeys(row["sample_id"] for row in rows))
    dataset = open_research_dataset(
        dataset_root,
        device="cpu",
        retain_embedded_labels=True,
    )
    labels = dataset.prepare_evaluation_labels(sample_ids)
    label_by_sample = {}
    for sample_id in sample_ids:
        sample = dataset[sample_id]
        try:
            label_by_sample[sample_id] = (
                labels.response_labels(sample).detach().cpu().numpy()
            )
        finally:
            sample.release_attention()

    for row in rows:
        relative = row.pop("prediction_position") - row.pop("response_start")
        sample_label = label_by_sample[row["sample_id"]]
        if not 0 <= relative < len(sample_label):
            raise ValueError(
                f"audit target lies outside labels: {row['sample_id']} "
                f"q={row['query_position']}"
            )
        row["hallucination_label"] = int(sample_label[relative])


def evaluate_subset_split(
    dataset_root: str | Path,
    output_root: str | Path,
) -> dict:
    """Join labels after capture, then summarize mechanisms and fixed axes."""

    dataset_root = Path(dataset_root)
    output = Path(output_root)
    manifest_path = output / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("subset_manifest_schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported subset manifest schema")
    if not manifest.get("analysis_complete"):
        raise ValueError("subset capture is incomplete")
    if manifest.get("labels_used_for_capture") is not False:
        raise ValueError("capture manifest violates the label firewall")
    captured_dataset_root = Path(manifest["config"]["dataset_root"]).resolve()
    if captured_dataset_root != dataset_root.resolve():
        raise ValueError("evaluation dataset_root differs from capture manifest")

    rows = _capture_rows(output, manifest)
    _join_labels(dataset_root, rows)

    groups = {}
    task_names = ["ALL", *sorted({row["task_type"] for row in rows})]
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
        }

    report = {
        "subset_evaluation_schema": 3,
        "labels_accessed_after_capture": True,
        "selection_is_not_population_evaluation": True,
        "claim_scope": (
            "native source-operator support for an observed-token contrast "
            "and response-origin supporting-action candidates; neither axis "
            "identifies factual correctness"
        ),
        "axis_definition": {
            "native_source_mediated_observed_margin": (
                "exact root/corridor/mediated-rescue bottleneck gated by "
                "root-lineage transport, signed source action, and signed "
                "source-cut integration for the observed-token margin"
            ),
            "response_origin_supporting_action_candidate": (
                "positive response-origin signed head action weighted by "
                "head functional agreement; no response-history intervention"
            ),
        },
        "groups": groups,
        "targets": rows,
    }
    save_json(output / REPORT_NAME, report)
    return report
