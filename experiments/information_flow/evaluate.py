"""Evaluate information-flow embeddings with the shared node-only readers."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from experiments.grounded_route.evaluation.data import (
    EmbeddingTable,
    align_table,
    load_labels,
)
from experiments.grounded_route.evaluation.detectors import (
    DetectorConfig,
    score_detectors,
)
from experiments.grounded_route.evaluation.metrics import (
    binary_metrics,
    paired_delta,
    source_bootstrap,
)
from experiments.grounded_route.evaluation.probes import (
    ProbeConfig,
    readability_scores,
)

from .config import VIEW_NAMES


PRIMARY_DETECTORS = ("pca_knn", "isolation_forest")


COMPARISONS = {
    "ordered_trace_minus_reverse": ("full_trace", "reverse_trace"),
    "ordered_final_minus_reverse": ("full_final", "reverse_final"),
    "all_layers_minus_last_layer": ("full_final", "last_layer"),
    "progressive_minus_layer_mean": ("full_final", "layer_mean"),
    "trajectory_minus_final": ("full_trace", "full_final"),
    "flow_minus_identity": ("full_trace", "identity"),
}


def load_tables(root) -> dict[str, EmbeddingTable]:
    root = Path(root)
    tables = {
        name: EmbeddingTable.load(root / f"index_{name}.npz")
        for name in VIEW_NAMES
    }
    reference = tables["full_trace"]
    return {
        name: reference if name == "full_trace" else align_table(reference, table)
        for name, table in tables.items()
    }


def metric_report(label, score):
    return binary_metrics(label, score)


def compare_views(
    label,
    source_id,
    unsupervised_scores,
    probe_scores,
    bootstrap,
    seed,
):
    report = {}
    for name, (left, right) in COMPARISONS.items():
        unsupervised = {
            detector: paired_delta(
                label,
                unsupervised_scores[left][detector],
                unsupervised_scores[right][detector],
                source_id,
                bootstrap,
                seed,
            )
            for detector in PRIMARY_DETECTORS
        }
        supervised = {
            reader: paired_delta(
                label,
                probe_scores[f"{reader}__{left}"],
                probe_scores[f"{reader}__{right}"],
                source_id,
                bootstrap,
                seed,
            )
            for reader in ("linear_node", "node_mlp")
        }
        report[name] = {
            "unsupervised": unsupervised,
            "supervised_readability": supervised,
        }
    return report


def evaluate(
    calibration_dir,
    test_dir,
    test_root,
    output_dir,
    *,
    device: str = "cpu",
    folds: int = 5,
    epochs: int = 20,
    bootstrap: int = 1_000,
    seeds: tuple[int, ...] = (20260827,),
) -> dict[str, object]:
    calibration = load_tables(calibration_dir)
    test = load_tables(test_dir)
    reference = test["full_trace"]

    detector_config = DetectorConfig(epochs=epochs, seeds=seeds)
    unsupervised_scores = {
        name: score_detectors(
            calibration[name].embedding,
            test[name].embedding,
            detector_config,
            device,
        )
        for name in VIEW_NAMES
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_scores = {
        "sample_id": reference.sample_id,
        "source_id": reference.source_id,
        "token_index": reference.token_index,
        "response_length": reference.response_length,
        "response_token_id": reference.response_token_id,
    }
    for view, scores in unsupervised_scores.items():
        for detector, score in scores.items():
            frozen_scores[f"{view}__{detector}"] = score
    frozen_scores["absolute_position"] = reference.token_index.astype(np.float32)
    frozen_scores["relative_position"] = (
        reference.token_index / np.maximum(reference.response_length - 1, 1)
    ).astype(np.float32)
    np.savez_compressed(output_dir / "unsupervised_scores.npz", **frozen_scores)

    label = load_labels(reference, test_root)
    probe_config = ProbeConfig(folds=folds, epochs=epochs, seeds=seeds)
    probe_scores = readability_scores(
        {name: table.embedding for name, table in test.items()},
        label,
        reference.source_id,
        reference.token_index,
        reference.response_length,
        probe_config,
        device,
    )
    np.savez_compressed(output_dir / "probe_scores.npz", **probe_scores)

    unsupervised = {
        view: {
            detector: metric_report(label, score)
            for detector, score in scores.items()
        }
        for view, scores in unsupervised_scores.items()
    }
    position = {
        name: metric_report(label, frozen_scores[name])
        for name in ("absolute_position", "relative_position")
    }
    readability = {
        name: metric_report(label, score)
        for name, score in probe_scores.items()
    }

    primary_bootstrap = {
        "full_trace__pca_knn": source_bootstrap(
            label,
            unsupervised_scores["full_trace"]["pca_knn"],
            reference.source_id,
            bootstrap,
            seeds[0],
        ),
        "linear_node__full_trace": source_bootstrap(
            label,
            probe_scores["linear_node__full_trace"],
            reference.source_id,
            bootstrap,
            seeds[0],
        ),
    }

    report = {
        "experiment": "attention_only_information_flow",
        "samples": int(len(np.unique(reference.sample_id))),
        "tokens": int(len(label)),
        "positive_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        "views": list(VIEW_NAMES),
        "detector_config": asdict(detector_config),
        "probe_config": asdict(probe_config),
        "unsupervised": unsupervised,
        "position_baselines": position,
        "supervised_readability": readability,
        "primary_bootstrap": primary_bootstrap,
        "comparisons": compare_views(
            label,
            reference.source_id,
            unsupervised_scores,
            probe_scores,
            bootstrap,
            seeds[0],
        ),
        "artifacts": {
            "unsupervised_scores": str(
                (output_dir / "unsupervised_scores.npz").resolve()
            ),
            "probe_scores": str((output_dir / "probe_scores.npz").resolve()),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**report, "report": str(report_path.resolve())}
