"""Compare frozen first-order GCN and order-2 DBGNN node representations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from experiment_protocol import FrozenEvaluation, scalar_text, validate_source_audit
from research_dataset import open_research_dataset
from experiments.grounded_route.artifacts import load_npz, save_npz, sha256
from experiments.grounded_route.detection import PCAKNNConfig, PCAWhitenedKNN
from experiments.grounded_route.graph_effectiveness.data import load_bundle
from experiments.grounded_route.graph_effectiveness.detectors import (
    DETECTOR_NAMES,
    DetectorConfig,
    score_detectors,
)
from experiments.grounded_route.graph_effectiveness.metrics import (
    binary_metrics,
    paired_source_delta,
)
from experiments.grounded_route.graph_effectiveness.upper_bound import (
    ProbeConfig,
    fit_readability_probes,
)
from experiments.grounded_route.pipeline import validate_calibration_provenance


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
    bootstrap_replicates: int = 2_000,
    seed: int = 20260826,
) -> dict[str, object]:
    """Freeze every node-only score before labels are opened for diagnostics."""

    detector_config = DetectorConfig() if detector_config is None else detector_config
    probe_config = ProbeConfig() if probe_config is None else probe_config
    calibration = {
        "gcn": load_bundle(gcn_calibration_path),
        "dbgnn_no_transition": load_bundle(no_transition_calibration_path),
        "dbgnn": load_bundle(dbgnn_calibration_path),
    }
    test = {
        "gcn": load_bundle(gcn_test_path),
        "dbgnn_no_transition": load_bundle(no_transition_test_path),
        "dbgnn": load_bundle(dbgnn_test_path),
    }
    _verify_protocol(calibration, test)
    for name in ("dbgnn_no_transition", "dbgnn"):
        _verify_pair(calibration["gcn"], calibration[name])
        _verify_pair(test["gcn"], test[name])
        _verify_topology(calibration["gcn"], calibration[name])
        _verify_topology(test["gcn"], test[name])

    score = {}
    for name in ("gcn", "dbgnn_no_transition", "dbgnn"):
        values = score_detectors(
            calibration[name].index.embedding,
            test[name].index.embedding,
            config=detector_config,
            device=device,
        )
        score.update({f"{detector}__{name}": value for detector, value in values.items()})
    score["position_pca_knn"] = _position_score(
        calibration["gcn"].index,
        test["gcn"].index,
        detector_config,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "frozen_unsupervised_scores.npz"
    reference = test["gcn"]
    rows = reference.index.arrays()
    rows.pop("embedding")
    save_npz(
        score_path,
        schema=np.asarray("dbgnn-gcn-node-only-score-comparison"),
        version=np.asarray(1, dtype=np.int32),
        labels_included=np.asarray(False),
        **rows,
        **score,
        dataset_manifest_sha256=np.asarray(
            scalar_text(reference.metadata, "dataset_manifest_sha256")
        ),
        audit_scope=np.asarray(scalar_text(reference.metadata, "audit_scope")),
        encoder_names=np.asarray(tuple(test)),
        calibration_index_sha256=np.asarray(
            [calibration[name].index_sha256 for name in test]
        ),
        test_index_sha256=np.asarray([test[name].index_sha256 for name in test]),
        checkpoint_sha256=np.asarray(
            [scalar_text(test[name].metadata, "checkpoint_sha256") for name in test]
        ),
        calibration_checkpoint_sha256=np.asarray(
            [
                scalar_text(calibration[name].metadata, "checkpoint_sha256")
                for name in test
            ]
        ),
        encoder_family=np.asarray(
            [scalar_text(test[name].metadata, "encoder_family") for name in test]
        ),
        higher_order_mode=np.asarray(
            [scalar_text(test[name].metadata, "higher_order_mode") for name in test]
        ),
        training_protocol_sha256=np.asarray(
            scalar_text(reference.metadata, "training_protocol_sha256")
        ),
        gcn_index_sha256=np.asarray(reference.index_sha256),
        no_transition_index_sha256=np.asarray(
            test["dbgnn_no_transition"].index_sha256
        ),
        dbgnn_index_sha256=np.asarray(test["dbgnn"].index_sha256),
    )

    for bundle in (*calibration.values(), *test.values()):
        bundle.reverify()
    frozen = FrozenEvaluation.capture(score_path, expected_split="test")
    score_rows = load_npz(score_path)
    dataset = open_research_dataset(
        test_root,
        device="cpu",
        retain_embedded_labels=True,
    )
    labels = frozen.align_loaded(dataset, score_rows)
    label = labels.token_label.astype(np.int8)
    source_id = labels.source_id.astype(str)

    probes = fit_readability_probes(
        {name: bundle.index.embedding for name, bundle in test.items()},
        label,
        source_id,
        reference.index.token_index,
        reference.index.response_length,
        config=probe_config,
        device=device,
    )
    oof_path = output_dir / "supervised_readability_oof.npz"
    save_npz(
        oof_path,
        schema=np.asarray("dbgnn-gcn-readability-ceiling"),
        version=np.asarray(1, dtype=np.int32),
        labels_included=np.asarray(False),
        labels_read=np.asarray(True),
        labels_used_to_fit=np.asarray(True),
        sample_id=reference.index.sample_id,
        source_id=reference.index.source_id,
        token_index=reference.index.token_index,
        response_length=reference.index.response_length,
        dataset_manifest_sha256=np.asarray(
            scalar_text(reference.metadata, "dataset_manifest_sha256")
        ),
        input_names=np.asarray(tuple(test)),
        input_index_sha256=np.asarray(
            [test[name].index_sha256 for name in test]
        ),
        fold_id=probes.fold_id,
        **probes.score,
        **probes.seed_score,
    )

    unsupervised = {
        name: binary_metrics(label, values)
        for name, values in score.items()
    }
    readability = {
        name: binary_metrics(label, values)
        for name, values in probes.score.items()
    }
    detector_deltas = {
        comparison: {
            detector: paired_source_delta(
                label,
                score[f"{detector}__dbgnn"],
                score[f"{detector}__{baseline}"],
                source_id,
                replicates=bootstrap_replicates,
                seed=seed,
            )
            for detector in DETECTOR_NAMES
        }
        for comparison, baseline in (
            ("causal_minus_no_transition", "dbgnn_no_transition"),
            ("causal_minus_gcn", "gcn"),
        )
    }
    probe_deltas = {
        comparison: {
            family: paired_source_delta(
                label,
                probes.score[f"{family}__dbgnn"],
                probes.score[f"{family}__{baseline}"],
                source_id,
                replicates=bootstrap_replicates,
                seed=seed,
            )
            for family in ("linear_node", "node_mlp")
        }
        for comparison, baseline in (
            ("causal_minus_no_transition", "dbgnn_no_transition"),
            ("causal_minus_gcn", "gcn"),
        )
    }
    report = {
        "schema": "dbgnn-gcn-representation-diagnostic",
        "version": 1,
        "labels_read": True,
        "labels_used_during_unsupervised_fit": False,
        "labels_used_during": "posthoc_metrics_and_source_grouped_oof_probe_only",
        "primary_comparison": "causal_dbgnn_minus_same_architecture_no_transition",
        "auxiliary_comparison": "causal_dbgnn_minus_first_order_gcn",
        "downstream_edges_used": False,
        "topology_identity_verified": True,
        "encoders": {
            name: {
                "family": scalar_text(bundle.metadata, "encoder_family"),
                "order": int(np.asarray(bundle.metadata["debruijn_order"]).item()),
                "parameter_count": int(
                    np.asarray(bundle.metadata["parameter_count"]).item()
                ),
            }
            for name, bundle in test.items()
        },
        "samples": len(reference.records),
        "tokens": len(label),
        "positive_tokens": int(label.sum()),
        "prevalence": float(label.mean()),
        "detector_config": asdict(detector_config),
        "probe_config": asdict(probe_config),
        "unsupervised": unsupervised,
        "supervised_readability_ceiling": readability,
        "comparisons": {
            "unsupervised": detector_deltas,
            "readability": probe_deltas,
        },
        "artifacts": {
            "frozen_scores": str(score_path.resolve()),
            "frozen_scores_sha256": frozen.artifact.sha256,
            "readability_oof": str(oof_path.resolve()),
            "readability_oof_sha256": sha256(oof_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {**report, "report": str(report_path.resolve())}


def _verify_pair(left, right) -> None:
    for field in (
        "sample_id",
        "source_id",
        "task_type",
        "token_index",
        "response_length",
        "response_token_id",
    ):
        if not np.array_equal(getattr(left.index, field), getattr(right.index, field)):
            raise ValueError(f"GCN and DBGNN rows differ in {field}")


def _verify_protocol(calibration, test) -> None:
    expected = {
        "gcn": ("official_lisiq_gcn", "no_transition", 1),
        "dbgnn_no_transition": (
            "official_lisiq_dbgnn",
            "no_transition",
            2,
        ),
        "dbgnn": ("official_lisiq_dbgnn", "causal", 2),
    }
    reference_protocol = scalar_text(
        calibration["gcn"].metadata,
        "training_protocol_sha256",
    )
    train_source_index = scalar_text(
        calibration["gcn"].metadata,
        "source_index_sha256",
    )
    test_source_index = scalar_text(test["gcn"].metadata, "source_index_sha256")
    implementation = scalar_text(
        calibration["gcn"].metadata,
        "implementation_sha256",
    )
    for name, protocol in expected.items():
        train_meta = calibration[name].metadata
        test_meta = test[name].metadata
        validate_calibration_provenance(calibration[name].index, train_meta)
        validate_source_audit(
            reserved_source_ids=test_meta["reserved_source_ids"],
            test_source_ids=test_meta["test_source_ids"],
            test_sample_ids=test_meta["test_sample_ids"],
            row_sample_ids=test[name].index.sample_id,
            row_source_ids=test[name].index.source_id,
            audit_scope=scalar_text(test_meta, "audit_scope"),
        )
        observed = (
            scalar_text(test_meta, "encoder_family"),
            scalar_text(test_meta, "higher_order_mode"),
            int(np.asarray(test_meta["debruijn_order"]).item()),
        )
        if observed != protocol:
            raise ValueError(f"{name} has the wrong encoder/order control protocol")
        for field in (
            "checkpoint_sha256",
            "implementation_sha256",
            "encoder_family",
            "higher_order_mode",
            "upstream_commit",
            "training_protocol_sha256",
        ):
            if scalar_text(train_meta, field) != scalar_text(test_meta, field):
                raise ValueError(f"{name} calibration/test differ in {field}")
        if scalar_text(train_meta, "implementation_sha256") != implementation:
            raise ValueError("encoder controls use different implementations")
        if scalar_text(train_meta, "training_protocol_sha256") != reference_protocol:
            raise ValueError("GCN and DBGNN training hyperparameters differ")
        if scalar_text(train_meta, "source_index_sha256") != train_source_index:
            raise ValueError("encoder controls use different train graph inputs")
        if scalar_text(test_meta, "source_index_sha256") != test_source_index:
            raise ValueError("encoder controls use different test graph inputs")


def _verify_topology(left, right) -> None:
    for left_record, right_record in zip(left.records, right.records, strict=True):
        left_graph = left_record.load()
        right_graph = right_record.load()
        for field in (
            "token_ids",
            "edge_index",
            "edge_layer",
            "edge_head",
            "edge_weight",
            "diagonal",
            "unresolved",
        ):
            if not torch.equal(getattr(left_graph, field), getattr(right_graph, field)):
                raise ValueError(f"GCN and DBGNN topology differs in {field}")


def _position_score(calibration, test, config: DetectorConfig) -> np.ndarray:
    def features(index):
        return np.column_stack(
            (
                index.token_index / np.maximum(index.response_length - 1, 1),
                np.log1p(index.response_length),
            )
        ).astype(np.float32)

    reference = PCAWhitenedKNN.fit(
        features(calibration),
        PCAKNNConfig(
            components=2,
            neighbors=config.neighbors,
            max_reference=config.max_reference,
            seed=config.seed,
        ),
    )
    return reference.score(features(test))


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
    parser.add_argument("--bootstrap", type=int, default=2_000)
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
            neural_epochs=arguments.epochs,
            seed=arguments.seed,
            neural_seeds=tuple(arguments.seeds),
        ),
        probe_config=ProbeConfig(
            folds=arguments.folds,
            epochs=arguments.epochs,
            split_seed=arguments.seed,
            seeds=tuple(arguments.seeds),
        ),
        device=arguments.device,
        bootstrap_replicates=arguments.bootstrap,
        seed=arguments.seed,
    )
    print(f"report: {report['report']}")


if __name__ == "__main__":
    main()
