"""Translate audit evidence into conservative model-authorization decisions."""

from __future__ import annotations

import numpy as np

from .config import EvaluationConfig

QUESTIONS = (
    ("A0", "Are scores, tokens, and labels strictly aligned?", "subsequent_audits"),
    ("A1", "Does direct prompt/response role carry signal?", "linear_lookback"),
    (
        "A2",
        "Does exact response endpoint identity add signal beyond coarse lag?",
        "exact_response_history_graph",
    ),
    ("A3", "Does head fracture add conditional signal?", "head_resolved_module"),
    ("A4", "Does layer order matter?", "ordered_recurrent_update"),
    (
        "A5",
        "Does multi-hop lineage outperform direct or one-hop lineage?",
        "multi_hop_message_passing",
    ),
    ("A6", "Which aggregation form is necessary?", "typed_aggregator"),
    (
        "A7",
        "Is there response-history lock-in after an error?",
        "gated_temporal_state",
    ),
    ("A8", "Does structure change before the first error?", "online_change_point"),
    (
        "A9",
        "Do nonlinear interactions outperform an additive form?",
        "neural_joint_model",
    ),
    ("A10", "Does the proxy have a causal effect in the base model?", "causal_claim"),
)


def _gate(
    audit: str,
    status: str,
    question: str,
    authorized: str,
    evidence: str,
) -> dict[str, str]:
    return {
        "audit": audit,
        "status": status,
        "question": question,
        "authorized_model_component": authorized,
        "evidence": evidence,
    }


def _finite(row: dict[str, object], *names: str) -> bool:
    return all(
        name in row and row[name] is not None and np.isfinite(float(row[name]))
        for name in names
    )


def gate_decisions(
    relation_rows: list[dict[str, object]],
    *,
    scope: str,
    samples: int,
    source_groups: int,
    positive_responses: int,
    artifact_binding_verified: bool,
    full_a0_verified: bool,
    config: EvaluationConfig,
) -> list[dict[str, str]]:
    if not artifact_binding_verified:
        return [
            _gate(
                audit,
                "INCONCLUSIVE_ARTIFACT_BINDING_FAILED"
                if audit == "A0"
                else "BLOCKED_BY_A0",
                question,
                "none",
                "dataset, score, source, or endpoint-null invariant binding failed",
            )
            for audit, question, _ in QUESTIONS
        ]

    decisions = [
        _gate(
            "A0",
            "PASS" if full_a0_verified else "INCONCLUSIVE_A0_CONTROLS_MISSING",
            QUESTIONS[0][1],
            QUESTIONS[0][2] if full_a0_verified else "none",
            (
                "artifact binding, gold alignment, and full-pipeline label "
                "permutation were verified"
                if full_a0_verified
                else "artifact binding passed; raw/tokenizer gold alignment and "
                "full-pipeline label permutation are not implemented"
            ),
        )
    ]
    if scope == "smoke":
        decisions.extend(
            _gate(
                audit,
                "NOT_EVALUATED_SMOKE",
                question,
                "none",
                "confirmation was not opened",
            )
            for audit, question, _ in QUESTIONS[1:]
        )
        return decisions
    if not full_a0_verified:
        decisions.extend(
            _gate(
                audit,
                "BLOCKED_BY_A0",
                question,
                "none",
                "formal structure gates require complete A0",
            )
            for audit, question, _ in QUESTIONS[1:]
        )
        return decisions

    relation = {row["relation"]: row for row in relation_rows}
    direct = relation["direct_role"]
    lineage = relation["lineage_margin"]
    if scope == "discovery":
        a2_status = (
            "PILOT_COARSE_NULL_ONLY"
            if bool(lineage.get("endpoint_null_valid", False))
            else "INCONCLUSIVE_NULL_INVALID"
        )
        decisions.extend(
            (
                _gate(
                    "A1",
                    "DISCOVERY_ASSOCIATION_AVAILABLE",
                    QUESTIONS[1][1],
                    "none",
                    "association is exploratory until nuisance controls are added",
                ),
                _gate(
                    "A2",
                    a2_status,
                    QUESTIONS[2][1],
                    "none",
                    "pilot null preserves coarse lag and count degree, not weighted "
                    "strength or proven mixing",
                ),
                _gate(
                    "A3",
                    "INCONCLUSIVE_CONTROL_MISSING",
                    QUESTIONS[3][1],
                    "none",
                    "head-mean conditional control is not implemented",
                ),
                _gate(
                    "A4",
                    "DISCOVERY_CONTROL_AVAILABLE",
                    QUESTIONS[4][1],
                    "none",
                    "layer permutation is exploratory until unordered baselines exist",
                ),
                _gate(
                    "A5",
                    "INCONCLUSIVE_DEPTH_CONTROL_MISSING",
                    QUESTIONS[5][1],
                    "none",
                    "direct/one-hop/two-hop/full-depth ablation is required",
                ),
                _gate(
                    "A6",
                    "NOT_IMPLEMENTED_WEIGHT_NULL",
                    QUESTIONS[6][1],
                    "none",
                    "fixed-endpoint weight shuffle is required",
                ),
                _gate(
                    "A7",
                    "EXPLORATORY_ONLY",
                    QUESTIONS[7][1],
                    "none",
                    "temporal summaries are discovery-only",
                ),
                _gate(
                    "A8",
                    "NOT_IMPLEMENTED_CHANGE_POINT",
                    QUESTIONS[8][1],
                    "none",
                    "a discovery-frozen threshold and FPR are required",
                ),
                _gate(
                    "A9",
                    "EXPLORATORY_DISCOVERY_ONLY",
                    QUESTIONS[9][1],
                    "none",
                    "grouped CV may screen one frozen joint form",
                ),
                _gate(
                    "A10",
                    "NOT_AVAILABLE_FROM_ATTENTION_CACHE",
                    QUESTIONS[10][1],
                    "none",
                    "matched interventions in the base LLM are required",
                ),
            )
        )
        return decisions

    token_classes_present = int(direct.get("positive_tokens", 0)) > 0 and int(
        direct.get("tokens", 0)
    ) > int(direct.get("positive_tokens", 0))
    enough_responses = (
        samples >= config.minimum_confirmation_samples
        and source_groups >= config.grouped_cv_folds
        and positive_responses >= config.minimum_positive_responses
    )

    if (
        not enough_responses
        or not token_classes_present
        or not _finite(
            direct,
            "association_shift_q",
            "auprc",
            "prevalence",
            "auprc_ci_low",
        )
    ):
        a1_status = "INCONCLUSIVE_LOW_POWER"
    else:
        a1_status = "INCONCLUSIVE_REQUIRED_CONTROLS_MISSING"
    decisions.append(
        _gate(
            "A1",
            a1_status,
            QUESTIONS[1][1],
            "none",
            "association is measured; matched nuisance controls and d_z are missing",
        )
    )

    endpoint_valid = bool(lineage.get("endpoint_null_valid", False))
    if not endpoint_valid:
        a2_status = "INCONCLUSIVE_NULL_INVALID"
    else:
        a2_status = "INCONCLUSIVE_REQUIRED_NULL_MISSING"
    decisions.append(
        _gate(
            "A2",
            a2_status,
            QUESTIONS[2][1],
            "none",
            "pilot null does not preserve weighted source strength or establish mixing",
        )
    )
    decisions.append(
        _gate(
            "A3",
            "INCONCLUSIVE_CONTROL_MISSING",
            QUESTIONS[3][1],
            "none",
            "head-mean conditional control is not implemented",
        )
    )

    if (
        not enough_responses
        or not token_classes_present
        or not _finite(
            lineage,
            "layer_auprc_delta",
            "layer_auprc_delta_ci_low",
            "layer_shuffle_q",
        )
    ):
        a4_status = "INCONCLUSIVE_LOW_POWER"
    else:
        a4_status = "INCONCLUSIVE_REQUIRED_CONTROLS_MISSING"
    decisions.append(
        _gate(
            "A4",
            a4_status,
            QUESTIONS[4][1],
            "none",
            "layer permutation is measured; unordered baselines are missing",
        )
    )
    decisions.extend(
        (
            _gate(
                "A5",
                "INCONCLUSIVE_DEPTH_CONTROL_MISSING",
                QUESTIONS[5][1],
                "none",
                "direct/one-hop/two-hop/full-depth ablation is required",
            ),
            _gate(
                "A6",
                "NOT_IMPLEMENTED_WEIGHT_NULL",
                QUESTIONS[6][1],
                "none",
                "fixed-endpoint weight shuffle is required",
            ),
            _gate(
                "A7",
                "EXPLORATORY_ONLY",
                QUESTIONS[7][1],
                "none",
                "temporal effect is reported but persistence is not gated",
            ),
            _gate(
                "A8",
                "NOT_IMPLEMENTED_CHANGE_POINT",
                QUESTIONS[8][1],
                "none",
                "a discovery-frozen threshold and confirmation FPR are required",
            ),
            _gate(
                "A9",
                "EXPLORATORY_DISCOVERY_ONLY",
                QUESTIONS[9][1],
                "none",
                "current grouped CV refits all relations and is not a frozen model",
            ),
            _gate(
                "A10",
                "NOT_AVAILABLE_FROM_ATTENTION_CACHE",
                QUESTIONS[10][1],
                "none",
                "matched interventions in the base LLM are required",
            ),
        )
    )
    return decisions
