"""Compare one reference embedding with aligned alternatives."""

import numpy as np

from .metrics import paired_delta


def control_deltas(
    label: np.ndarray,
    source_id: np.ndarray,
    unsupervised_scores: dict[str, dict[str, np.ndarray]],
    probe_scores: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    report = {}
    for control in unsupervised_scores:
        if control == "real":
            continue

        unsupervised = {}
        for detector, real_score in unsupervised_scores["real"].items():
            unsupervised[detector] = paired_delta(
                label,
                real_score,
                unsupervised_scores[control][detector],
                source_id,
                replicates,
                seed,
            )

        supervised = {}
        for reader in ("linear_node", "node_mlp"):
            supervised[reader] = paired_delta(
                label,
                probe_scores[f"{reader}__real"],
                probe_scores[f"{reader}__{control}"],
                source_id,
                replicates,
                seed,
            )
        report[control] = {
            "unsupervised": unsupervised,
            "supervised_readability": supervised,
        }
    return report
