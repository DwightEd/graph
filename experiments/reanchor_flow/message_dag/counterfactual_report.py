"""Offline coverage and local-to-finite agreement for CLCMS artifacts."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .artifacts import atomic_path


def _write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def evaluate(output):
    output = Path(output)
    screen = json.loads((output / "screen_plan.json").read_text(encoding="utf-8"))
    witness_path = output / "witness_plan.json"
    witness = (
        json.loads(witness_path.read_text(encoding="utf-8"))
        if witness_path.exists()
        else {"witnesses": [], "planned_witnesses": 0, "completed_witnesses": 0}
    )
    finite = []
    message_capture_calls = 0
    intervention_calls = 0
    for item in witness["witnesses"]:
        path = output / item["path"]
        with np.load(path, allow_pickle=False) as stored:
            local = float(stored["local_selective"])
            response = float(stored["intervention_selective"][-1])
            evidence_effect = stored["evidence_log_odds"][:, -1] - stored[
                "evidence_log_odds"
            ][:, 0]
            message_capture_calls += int(stored["message_capture_forward_calls"])
            intervention_calls += int(stored["forward_calls"])
            finite.append(
                {
                    "pair_id": item["pair_id"],
                    "carrier": item["carrier"],
                    "local_selective": local,
                    "finite_selective": response,
                    "sign_agreement": bool(local * response > 0),
                    "plus_evidence_effect": float(evidence_effect[0]),
                    "minus_evidence_effect": float(evidence_effect[1]),
                    "bidirectional_constraint_weakening": bool(
                        np.all(evidence_effect < 0)
                    ),
                }
            )
    statuses = Counter(pair["status"] for pair in screen["pairs"])
    summary = {
        "method": "counterfactual_last_crossing_mediation_screen",
        "claim_scope": "screen plus bounded bidirectional post-WO value-channel witness",
        "pairs": len(screen["pairs"]),
        "pair_status": dict(sorted(statuses.items())),
        "selected_pairs": statuses["selected"],
        "planned_witnesses": witness["planned_witnesses"],
        "completed_witnesses": witness["completed_witnesses"],
        "finite_witness_forward_calls": {
            "message_capture": message_capture_calls,
            "baseline_and_intervention": intervention_calls,
            "total": message_capture_calls + intervention_calls,
        },
        "local_to_finite_sign_accuracy": (
            float(np.mean([row["sign_agreement"] for row in finite]))
            if finite
            else float("nan")
        ),
        "bidirectional_constraint_weakening_rate": (
            float(np.mean([row["bidirectional_constraint_weakening"] for row in finite]))
            if finite
            else float("nan")
        ),
        "finite_results": finite,
        "labels_used": False,
        "limitations": [
            "screened carriers are selected candidates, not an unbiased population effect",
            "post-WO exchange is a bounded witness, not a natural indirect effect",
            "teacher-forced sequence odds do not establish free-generation behavior",
            "K-routing and V-content screen branches are not separate finite interventions",
        ],
    }
    _write_json(output / "counterfactual_summary.json", summary)
    lines = [
        "# Counterfactual last-crossing mediation screen",
        "",
        (
            f"Pairs: {summary['pairs']}; selected: {summary['selected_pairs']}; "
            f"finite witnesses: {summary['completed_witnesses']}/"
            f"{summary['planned_witnesses']}."
        ),
        "",
        (
            "No-event, no-edge and no-response pairs remain in the pair denominator. "
            "The finite result is a bounded post-WO witness, not full mediation."
        ),
        "",
        "Exact per-pair curves and both exchange directions remain in witness NPZ files.",
    ]
    with atomic_path(output / "counterfactual_summary.md") as temporary:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
