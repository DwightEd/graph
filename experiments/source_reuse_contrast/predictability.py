"""Label-free comparison of current, birth, and dynamic route predictors."""

from __future__ import annotations

from pathlib import Path

from .artifacts import read_json, write_json


def write_predictability_gate(
    *,
    training_paths: dict[str, str | Path],
    manifest_paths: dict[str, str | Path],
    output_path: str | Path,
) -> dict:
    """Write the pre-label evidence required for a source-reuse claim."""

    modes = sorted(set(training_paths) & set(manifest_paths))
    if not modes:
        raise ValueError("matching training and score manifests are required")

    rows = {}
    for mode in modes:
        training = read_json(training_paths[mode])
        manifest = read_json(manifest_paths[mode])
        best = min(
            (
                row
                for row in training["history"]
                if row.get("validation_nll") is not None
            ),
            key=lambda row: row["validation_nll"],
        )
        rows[mode] = {
            "validation_nll": float(best["validation_nll"]),
            "validation_shuffled_nll": float(best["validation_shuffled_nll"]),
            "validation_shuffle_gap": float(best["validation_shuffle_gap"]),
            "validation_accuracy": float(best["validation_accuracy"]),
            "validation_margin": float(best["validation_margin"]),
            "validation_coverage": float(best["validation_coverage"]),
            **manifest["diagnostics"],
        }

    deltas = {}
    if "dynamic" in rows:
        for baseline in ("current", "birth"):
            if baseline in rows:
                deltas[f"dynamic_minus_{baseline}"] = {
                    "validation_nll_reduction": rows[baseline]["validation_nll"]
                    - rows["dynamic"]["validation_nll"],
                    "validation_accuracy_gain": rows["dynamic"]["validation_accuracy"]
                    - rows[baseline]["validation_accuracy"],
                }

    gates = {
        "dynamic_beats_current_validation_nll": bool(
            "dynamic" in rows
            and "current" in rows
            and rows["dynamic"]["validation_nll"] < rows["current"]["validation_nll"]
        ),
        "dynamic_beats_birth_validation_nll": bool(
            "dynamic" in rows
            and "birth" in rows
            and rows["dynamic"]["validation_nll"] < rows["birth"]["validation_nll"]
        ),
        "dynamic_uses_exact_memory_mapping": bool(
            "dynamic" in rows and rows["dynamic"]["validation_shuffle_gap"] > 0.0
        ),
        "dynamic_score_not_collapsed": bool(
            "dynamic" in rows
            and rows["dynamic"]["unique_endpoint_nll_1e6"] >= 32
            and rows["dynamic"]["endpoint_nll_std"] > 1e-4
        ),
    }
    report = {
        "schema": "source-reuse-predictability-gate-v2",
        "labels_read": False,
        "modes": rows,
        "deltas": deltas,
        "gates": gates,
        "source_reuse_claim_admitted": bool(all(gates.values())),
    }
    write_json(output_path, report)
    return report
