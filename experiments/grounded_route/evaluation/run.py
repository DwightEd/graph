"""Evaluate saved node embeddings without reconstructing the graph."""

import argparse
import json
from pathlib import Path

import numpy as np

from .controls import control_deltas
from .data import load_labels, load_variants
from .detectors import DetectorConfig, score_detectors
from .metrics import binary_metrics, source_bootstrap
from .probes import ProbeConfig, readability_scores


def command_line() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GroundedRoute node embeddings")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--control",
        nargs=3,
        action="append",
        default=[],
        metavar=("NAME", "CALIBRATION_INDEX", "TEST_INDEX"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260825, 20260826, 20260827])
    return parser


def metric_report(
    label: np.ndarray,
    score: np.ndarray,
    source_id: np.ndarray,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    return {
        **binary_metrics(label, score),
        "bootstrap": source_bootstrap(label, score, source_id, bootstrap, seed),
    }


def main() -> None:
    arguments = command_line().parse_args()
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    calibration_paths = {"real": arguments.calibration}
    test_paths = {"real": arguments.test_index}
    for name, calibration, test in arguments.control:
        calibration_paths[name] = calibration
        test_paths[name] = test

    calibration = load_variants(calibration_paths)
    test = load_variants(test_paths)
    detector_config = DetectorConfig(
        epochs=arguments.epochs,
        seeds=tuple(arguments.seeds),
    )

    unsupervised_scores = {}
    score_artifact = {}
    for name in test:
        current = score_detectors(
            calibration[name].embedding,
            test[name].embedding,
            detector_config,
            arguments.device,
        )
        unsupervised_scores[name] = current
        for detector, score in current.items():
            score_artifact[f"{name}__{detector}"] = score

    reference = test["real"]
    score_artifact["absolute_position"] = reference.token_index.astype(np.float32)
    score_artifact["relative_position"] = (
        reference.token_index / np.maximum(reference.response_length - 1, 1)
    ).astype(np.float32)
    np.savez_compressed(output / "unsupervised_scores.npz", **score_artifact)

    label = load_labels(reference, arguments.test_root)
    probe_config = ProbeConfig(
        folds=arguments.folds,
        epochs=arguments.epochs,
        seeds=tuple(arguments.seeds),
    )
    probe_scores = readability_scores(
        {name: table.embedding for name, table in test.items()},
        label,
        reference.source_id,
        reference.token_index,
        reference.response_length,
        probe_config,
        arguments.device,
    )
    np.savez_compressed(output / "probe_scores.npz", **probe_scores)

    unsupervised_metrics = {
        name: {
            detector: metric_report(
                label,
                score,
                reference.source_id,
                arguments.bootstrap,
                arguments.seed,
            )
            for detector, score in scores.items()
        }
        for name, scores in unsupervised_scores.items()
    }
    position_metrics = {
        name: metric_report(
            label,
            score_artifact[name],
            reference.source_id,
            arguments.bootstrap,
            arguments.seed,
        )
        for name in ("absolute_position", "relative_position")
    }
    probe_metrics = {
        name: metric_report(
            label,
            score,
            reference.source_id,
            arguments.bootstrap,
            arguments.seed,
        )
        for name, score in probe_scores.items()
    }

    report = {
        "experiment": "grounded_route_node_embedding_evaluation",
        "samples": int(len(np.unique(reference.sample_id))),
        "tokens": int(len(label)),
        "positive_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        "embedding_dimension": int(reference.embedding.shape[1]),
        "variants": list(test),
        "unsupervised": unsupervised_metrics,
        "position_baselines": position_metrics,
        "supervised_readability": probe_metrics,
        "construction_controls": control_deltas(
            label,
            reference.source_id,
            unsupervised_scores,
            probe_scores,
            arguments.bootstrap,
            arguments.seed,
        ),
        "artifacts": {
            "unsupervised_scores": str((output / "unsupervised_scores.npz").resolve()),
            "probe_scores": str((output / "probe_scores.npz").resolve()),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
