"""Compare real and controlled constructions with the same readers."""

import numpy as np

from .metrics import paired_delta


CONTROL_NAMES = ("no_message", "endpoint_rewire", "weight_shuffle")


def control_deltas(
    label: np.ndarray,
    source_id: np.ndarray,
    unsupervised_scores: dict[str, dict[str, np.ndarray]],
    probe_scores: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    report = {}
    for control in CONTROL_NAMES:
        if control not in unsupervised_scores:
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
            real_name = f"{reader}__real"
            control_name = f"{reader}__{control}"
            supervised[reader] = paired_delta(
                label,
                probe_scores[real_name],
                probe_scores[control_name],
                source_id,
                replicates,
                seed,
            )
        report[control] = {
            "unsupervised": unsupervised,
            "supervised_readability": supervised,
        }
    return report
