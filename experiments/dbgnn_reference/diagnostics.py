"""Compare GCN and DBGNN using the same frozen node-only readers."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from experiment_protocol import scalar_text
from experiments.grounded_route.evaluation.data import (
    EmbeddingTable,
    align_table,
    load_labels,
)
from experiments.grounded_route.evaluation.detectors import (
    DetectorConfig,
    score_detectors,
    score_pca_knn,
)
from experiments.grounded_route.evaluation.metrics import binary_metrics, paired_delta
from experiments.grounded_route.evaluation.probes import ProbeConfig, readability_scores

from .data import load_bundle


ENCODERS = ("gcn", "dbgnn_no_transition", "dbgnn")


def embedding_table(bundle) -> EmbeddingTable:
    index = bundle.index
    return EmbeddingTable(
        sample_id=index.sample_id.astype(str),
        source_id=index.source_id.astype(str),
        token_index=index.token_index.astype(np.int32),
        response_length=index.response_length.astype(np.int32),
        response_token_id=index.response_token_id.astype(np.int64),
        embedding=index.embedding.astype(np.float32),
    )


def aligned_tables(bundles) -> dict[str, EmbeddingTable]:
    reference = embedding_table(bundles["gcn"])
    return {
        name: reference if name == "gcn" else align_table(reference, embedding_table(bundle))
        for name, bundle in bundles.items()
    }


def position_features(table: EmbeddingTable) -> np.ndarray:
    return np.column_stack(
        (
            table.token_index / np.maximum(table.response_length - 1, 1),
            np.log1p(table.response_length),
        )
    ).astype(np.float32)


def compare(
    gcn_calibration_path,
    gcn_test_path,
    no_transition_calibration_path,
    no_transition_test_path,
    dbgnn_calibration_path,
    dbgnn_test_path,
    test_root,
    output_dir,
    *,
    detector_config: DetectorConfig | None = None,
    probe_config: ProbeConfig | None = None,
    device: str = "cpu",
    bootstrap_replicates: int = 1_000,
    seed: int = 20260826,
) -> dict[str, object]:
    detector_config = DetectorConfig() if detector_config is None else detector_config
    probe_config = ProbeConfig() if probe_config is None else probe_config

    calibration_bundles = {
        "gcn": load_bundle(gcn_calibration_path),
        "dbgnn_no_transition": load_bundle(no_transition_calibration_path),
        "dbgnn": load_bundle(dbgnn_calibration_path),
    }
    test_bundles = {
        "gcn": load_bundle(gcn_test_path),
        "dbgnn_no_transition": load_bundle(no_transition_test_path),
        "dbgnn": load_bundle(dbgnn_test_path),
    }
    calibration = aligned_tables(calibration_bundles)
    test = aligned_tables(test_bundles)

    scores_by_encoder = {
        name: score_detectors(
            calibration[name].embedding,
            test[name].embedding,
            detector_config,
            device,
        )
        for name in ENCODERS
    }
    flat_scores = {
        f"{detector}__{name}": score
        for name, scores in scores_by_encoder.items()
        for detector, score in scores.items()
    }
    flat_scores["position_pca_knn"] = score_pca_knn(
        position_features(calibration["gcn"]),
        position_features(test["gcn"]),
        detector_config,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = test["gcn"]
    score_path = output_dir / "unsupervised_scores.npz"
    np.savez_compressed(
        score_path,
        sample_id=reference.sample_id,
        source_id=reference.source_id,
        token_index=reference.token_index,
        response_length=reference.response_length,
        response_token_id=reference.response_token_id,
        **flat_scores,
    )

    label = load_labels(reference, test_root)
    probes = readability_scores(
        {name: table.embedding for name, table in test.items()},
        label,
        reference.source_id,
        reference.token_index,
        reference.response_length,
        probe_config,
        device,
    )
    probe_path = output_dir / "probe_scores.npz"
    np.savez_compressed(probe_path, **probes)

    unsupervised = {
        name: binary_metrics(label, score)
        for name, score in flat_scores.items()
    }
    readability = {
        name: binary_metrics(label, score)
        for name, score in probes.items()
    }

    comparisons = {}
    for comparison, baseline in (
        ("causal_minus_no_transition", "dbgnn_no_transition"),
        ("causal_minus_gcn", "gcn"),
    ):
        comparisons[comparison] = {
            "unsupervised": {
                detector: paired_delta(
                    label,
                    scores_by_encoder["dbgnn"][detector],
                    scores_by_encoder[baseline][detector],
                    reference.source_id,
                    bootstrap_replicates,
                    seed,
                )
                for detector in scores_by_encoder["dbgnn"]
            },
            "supervised_readability": {
                reader: paired_delta(
                    label,
                    probes[f"{reader}__dbgnn"],
                    probes[f"{reader}__{baseline}"],
                    reference.source_id,
                    bootstrap_replicates,
                    seed,
                )
                for reader in ("linear_node", "node_mlp")
            },
        }

    report = {
        "experiment": "dbgnn_gcn_node_embedding_comparison",
        "samples": int(len(np.unique(reference.sample_id))),
        "tokens": int(len(label)),
        "positive_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        "detector_config": asdict(detector_config),
        "probe_config": asdict(probe_config),
        "encoders": {
            name: {
                "family": scalar_text(test_bundles[name].metadata, "encoder_family"),
                "higher_order_mode": scalar_text(
                    test_bundles[name].metadata, "higher_order_mode"
                ),
                "debruijn_order": int(
                    np.asarray(test_bundles[name].metadata["debruijn_order"]).item()
                ),
                "parameter_count": int(
                    np.asarray(test_bundles[name].metadata["parameter_count"]).item()
                ),
            }
            for name in ENCODERS
        },
        "unsupervised": unsupervised,
        "supervised_readability": readability,
        "comparisons": comparisons,
        "artifacts": {
            "unsupervised_scores": str(score_path.resolve()),
            "probe_scores": str(probe_path.resolve()),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**report, "report": str(report_path.resolve())}


def command_line() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare GCN and DBGNN node embeddings")
    parser.add_argument("--gcn-calibration", required=True)
    parser.add_argument("--gcn-test", required=True)
    parser.add_argument("--no-transition-calibration", required=True)
    parser.add_argument("--no-transition-test", required=True)
    parser.add_argument("--dbgnn-calibration", required=True)
    parser.add_argument("--dbgnn-test", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260826])
    parser.add_argument("--bootstrap", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser


def main() -> None:
    arguments = command_line().parse_args()
    report = compare(
        arguments.gcn_calibration,
        arguments.gcn_test,
        arguments.no_transition_calibration,
        arguments.no_transition_test,
        arguments.dbgnn_calibration,
        arguments.dbgnn_test,
        arguments.test,
        arguments.output,
        detector_config=DetectorConfig(
            epochs=arguments.epochs,
            seeds=tuple(arguments.seeds),
        ),
        probe_config=ProbeConfig(
            folds=arguments.folds,
            epochs=arguments.epochs,
            seeds=tuple(arguments.seeds),
        ),
        device=arguments.device,
        bootstrap_replicates=arguments.bootstrap,
        seed=arguments.seed,
    )
    print(f"report: {report['report']}")


if __name__ == "__main__":
    main()
