"""Effect-size and wording-robustness audit for condition/value binding results."""

import numpy as np
import pandas as pd


THRESHOLDS = (0.01, 0.02, 0.05)
KEYS = ["case_id", "panel", "layer", "head"]


def effect_state(condition, value, threshold):
    if not np.isfinite(condition) or not np.isfinite(value):
        return "not_tested"
    if value > threshold and condition > threshold:
        return "value_with_condition"
    if value > threshold and condition <= threshold:
        return "value_without_condition"
    if value < -threshold and condition < -threshold:
        return "both_wrong"
    if value < -threshold and condition >= -threshold:
        return "wrong_value_without_condition"
    return "weak"


def sensitivity_table(binding):
    tested = binding[
        np.isfinite(binding.final_condition) & np.isfinite(binding.final_value)
    ].copy()
    rows = []
    for threshold in THRESHOLDS:
        frame = tested[
            ["case_id", "panel", "side", "layer", "head"]
        ].copy()
        frame["threshold"] = threshold
        frame["state"] = [
            effect_state(condition, value, threshold)
            for condition, value in zip(
                tested.final_condition, tested.final_value
            )
        ]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def paired_side_effects(binding):
    tested = binding[
        np.isfinite(binding.final_condition) & np.isfinite(binding.final_value)
    ].copy()
    fields = [
        "final_condition",
        "final_value",
        "final_evidence",
        "final_wrong_source",
        "joint_nonadditivity",
    ]
    supported = tested[tested.side == "supported"][KEYS + fields]
    unsupported = tested[tested.side == "unsupported"][KEYS + fields]
    paired = unsupported.merge(
        supported, on=KEYS, suffixes=("_unsupported", "_supported")
    )
    for field in fields:
        paired["delta_" + field] = (
            paired[field + "_unsupported"] - paired[field + "_supported"]
        )
    for side in ("supported", "unsupported"):
        paired["condition_value_interaction_" + side] = (
            paired["final_condition_" + side]
            + paired["final_value_" + side]
            - paired["final_evidence_" + side]
        )
    for threshold in THRESHOLDS:
        for side in ("supported", "unsupported"):
            paired[f"state_{side}_{threshold:g}"] = [
                effect_state(condition, value, threshold)
                for condition, value in zip(
                    paired["final_condition_" + side],
                    paired["final_value_" + side],
                )
            ]
    return paired


def panel_consistency(binding):
    tested = binding[
        np.isfinite(binding.final_condition) & np.isfinite(binding.final_value)
    ].copy()
    natural = tested[tested.panel == "natural"]
    parallel = tested[tested.panel == "parallel_singular"]
    keys = ["case_id", "side", "layer", "head"]
    fields = ["final_condition", "final_value", "final_evidence"]
    paired = natural[keys + fields].merge(
        parallel[keys + fields],
        on=keys,
        suffixes=("_natural", "_parallel"),
    )
    for threshold in THRESHOLDS:
        for panel in ("natural", "parallel"):
            paired[f"state_{panel}_{threshold:g}"] = [
                effect_state(condition, value, threshold)
                for condition, value in zip(
                    paired[f"final_condition_{panel}"],
                    paired[f"final_value_{panel}"],
                )
            ]
        paired[f"same_state_{threshold:g}"] = (
            paired[f"state_natural_{threshold:g}"]
            == paired[f"state_parallel_{threshold:g}"]
        )
        paired[f"partial_both_{threshold:g}"] = (
            (paired[f"state_natural_{threshold:g}"] == "value_without_condition")
            & (paired[f"state_parallel_{threshold:g}"] == "value_without_condition")
        )
    return paired


def write_binding_robustness(binding, output):
    sensitivity = sensitivity_table(binding)
    paired = paired_side_effects(binding)
    panels = panel_consistency(binding)
    sensitivity.to_csv(output / "binding_sensitivity.csv", index=False)
    paired.to_csv(output / "binding_supported_vs_unsupported.csv", index=False)
    panels.to_csv(output / "binding_panel_consistency.csv", index=False)
    counts = (
        sensitivity.groupby(
            ["case_id", "panel", "side", "threshold", "state"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "heads"})
    )
    counts.to_csv(output / "binding_sensitivity_counts.csv", index=False)
    return sensitivity, paired, panels
