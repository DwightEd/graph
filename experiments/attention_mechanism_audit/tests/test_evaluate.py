import json

import numpy as np
import torch

import experiments.attention_mechanism_audit.evaluate as evaluate_module
from experiments.attention_mechanism_audit.capture import ROLE_NAMES, SELF
from experiments.attention_mechanism_audit.data import EVIDENCE
from experiments.attention_mechanism_audit.evaluate import (
    _mean_or_none,
    _position_matched_difference,
    _summarize_arrays,
    token_metrics,
)


def test_missing_single_sample_class_is_serialized_as_null():
    assert _mean_or_none(np.asarray([], dtype=np.float32)) is None
    assert _mean_or_none(np.asarray([1.0, 3.0])) == 2.0


def test_token_metrics_keep_layer_drift_and_separate_causal_effects():
    layers, responses, heads, roles = 3, 3, 2, len(ROLE_NAMES)
    edge = torch.zeros(layers, responses, heads, roles)
    edge[:, 0, :, EVIDENCE] = 1.0
    edge[0, 1, :, EVIDENCE] = 2.0
    edge[1:, 1, :, SELF] = 2.0
    edge[:, 2, :, len(ROLE_NAMES) - 2] = 1.0
    route = torch.zeros_like(edge)
    route[:, 0, :, EVIDENCE] = 1.0
    route[0, 1, :, EVIDENCE] = 1.0
    route[1, 1, :, SELF] = 1.0
    route[2, 1, 0, EVIDENCE] = 1.0
    route[2, 1, 1, SELF] = 1.0
    route[:, 2, :, len(ROLE_NAMES) - 2] = 1.0
    artifact = {
        "trace": {
            "role_edge_magnitude": edge,
            "role_attention": route,
            "source_message_entropy": torch.zeros(layers, responses),
            "message_coherence": torch.ones(layers, responses),
            "source_role": torch.tensor(
                [
                    [EVIDENCE, SELF, -1, -1],
                    [EVIDENCE, 1, SELF, -1],
                    [EVIDENCE, 1, len(ROLE_NAMES) - 2, SELF],
                ],
                dtype=torch.int8,
            ),
        },
        "mechanism": {
            "evidence_message_effect": torch.tensor([0.25, -0.5, 0.1]),
            "response_message_effect": torch.tensor([0.1, 0.8, 0.4]),
            "evidence_response_removed_margin": torch.tensor([-0.1, 0.2, 0.2]),
            "full_margin": torch.tensor([-0.2, 0.3, -0.1]),
        },
    }

    metrics = token_metrics(artifact)

    np.testing.assert_allclose(
        metrics["message_evidence_share_mean"], [1.0, 1 / 3, 0.0]
    )
    np.testing.assert_allclose(
        metrics["message_response_share_mean"], [0.0, 2 / 3, 1.0]
    )
    np.testing.assert_allclose(
        metrics["message_routing_drift_mean"], [-1.0, 1 / 3, 1.0]
    )
    np.testing.assert_allclose(
        metrics["message_routing_drift_layer_shift"], [0.0, 2.0, 0.0]
    )
    np.testing.assert_allclose(
        metrics["attention_routing_drift_mean"], [-1.0, 0.0, 1.0]
    )
    np.testing.assert_allclose(
        metrics["attention_routing_drift_layer_shift"], [0.0, 1.0, 0.0]
    )
    np.testing.assert_allclose(metrics["message_source_dispersion_mean"], 0.0)
    np.testing.assert_allclose(
        metrics["head_role_disagreement_mean"], [0.0, np.log(2) / 3, 0.0]
    )
    np.testing.assert_allclose(
        metrics["head_role_disagreement_layer_shift"], [0.0, np.log(2), 0.0]
    )
    np.testing.assert_allclose(metrics["message_coherence_mean"], 1.0)
    np.testing.assert_allclose(
        metrics["evidence_message_effect"], [0.25, -0.5, 0.1]
    )
    np.testing.assert_allclose(
        metrics["message_independent_capture_signature"], [0.0, 1.0, 0.0]
    )


def test_primary_contrast_is_within_response_not_cross_response():
    result = _position_matched_difference(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([True, True, False, False]),
        np.asarray(["hall", "hall", "correct", "correct"]),
        np.asarray(["source"] * 4),
        np.asarray(["model"] * 4),
        np.asarray([0, 1, 0, 1]),
        np.asarray([20] * 4),
        position_bin=16,
        bootstrap=100,
        seed=3,
    )

    assert result["sources"] == 0
    assert result["matched_samples"] == 0


def test_primary_contrast_reports_hallucinated_token_coverage():
    result = _position_matched_difference(
        np.asarray([1.0, 3.0, 5.0, 9.0]),
        np.asarray([False, True, False, True]),
        np.asarray(["mixed"] * 4),
        np.asarray(["source"] * 4),
        np.asarray(["model"] * 4),
        np.asarray([0, 1, 10, 11]),
        np.asarray([100] * 4),
        position_bin=16,
        bootstrap=100,
        seed=4,
    )

    assert result["sources"] == 1
    assert result["matched_samples"] == 1
    assert result["matched_cells"] == 2
    assert result["covered_hallucinated_tokens"] == 2
    assert result["hallucinated_token_coverage"] == 1.0


def test_single_class_smoke_report_is_strict_json():
    metrics = {
        "message_routing_drift_mean": np.asarray([0.1, 0.2]),
        "message_source_dispersion_mean": np.asarray([0.2, 0.3]),
        "head_role_disagreement_mean": np.asarray([0.1, 0.1]),
        "evidence_message_effect": np.asarray([0.4, 0.5]),
        "message_independent_capture_signature": np.asarray([0.0, 0.0]),
        "full_margin": np.asarray([1.0, -1.0]),
    }

    report = _summarize_arrays(
        label=np.asarray([False, False]),
        sample_id=np.asarray(["sample", "sample"]),
        source_id=np.asarray(["source", "source"]),
        generator=np.asarray(["model", "model"]),
        token_index=np.asarray([0, 1]),
        response_length=np.asarray([10, 10]),
        metrics=metrics,
        samples=1,
        position_bin=16,
        bootstrap=20,
        seed=5,
    )

    assert report["observer_readout"]["target_preferred_hallucinated"] is None
    json.dumps(report, allow_nan=False)


def test_combine_saved_recomputes_all_split_statistics_and_preserves_samples(
    tmp_path, monkeypatch
):
    def write_split(name, routing, evidence):
        root = tmp_path / name
        root.mkdir()
        sample_id = f"{name}:sample"
        arrays_path = root / "token_metrics.npz"
        np.savez_compressed(
            arrays_path,
            label=np.asarray([False, True]),
            sample_id=np.asarray([sample_id, sample_id]),
            source_id=np.asarray(["shared-source", "shared-source"]),
            generator_model=np.asarray([f"{name}-generator"] * 2),
            split=np.asarray([name, name]),
            token_index=np.asarray([0, 1], dtype=np.int32),
            response_length=np.asarray([100, 100], dtype=np.int32),
            message_routing_drift_mean=np.asarray(routing, dtype=np.float32),
            message_source_dispersion_mean=np.asarray([0.2, 0.3], dtype=np.float32),
            evidence_message_effect=np.asarray(evidence, dtype=np.float32),
            full_margin=np.asarray([1.0, 1.0], dtype=np.float32),
            message_independent_capture_signature=np.asarray(
                [0.0, 1.0], dtype=np.float32
            ),
        )
        audit_path = root / "sample.json"
        audit_path.write_text(
            json.dumps({"sample_id": sample_id, "label": [0, 1]}),
            encoding="utf-8",
        )
        index_path = root / "sample_audits.jsonl"
        index_path.write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "split": name,
                    "audit": str(audit_path),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report_path = root / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "samples": 1,
                    "token_metrics": str(arrays_path),
                    "sample_audits": {"index": str(index_path), "count": 1},
                    "coverage": {
                        "cached_candidates": 1,
                        "eligible_qa": 1,
                        "captured": 1,
                        "evaluated": 1,
                        "complete": True,
                        "mixed_label_samples": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        return report_path

    train_report = write_split("train", [0.0, 2.0], [1.0, 0.0])
    test_report = write_split("test", [1.0, 5.0], [2.0, 0.0])
    plotted = {}

    def record_plot(report, records, output):
        plotted["samples"] = [record["sample_id"] for record in records]
        plotted["output"] = output
        plotted["report"] = report

    monkeypatch.setattr(evaluate_module, "plot_population", record_plot)
    output = tmp_path / "all" / "report.json"
    report = evaluate_module.combine_saved(
        inputs=[("train", train_report), ("test", test_report)],
        output=output,
        bootstrap=100,
        seed=7,
    )

    assert report["scope"] == "all_available_qa_splits"
    assert report["samples"] == 2
    assert report["tokens"] == 4
    assert report["hallucinated_tokens"] == 2
    np.testing.assert_allclose(
        report["summaries"]["message_routing_drift_mean"][
            "position_matched_source_equal_difference"
        ],
        3.0,
    )
    assert set(report["by_split"]) == {"train", "test"}
    assert report["coverage"] == {
        "cached_candidates": 2,
        "eligible_qa": 2,
        "captured": 2,
        "evaluated": 2,
        "complete": True,
        "mixed_label_samples": 2,
        "matched_samples": 2,
        "matched_cells": 2,
        "covered_hallucinated_tokens": 2,
        "hallucinated_token_coverage": 1.0,
    }
    assert report["source_overlap_between_splits"] == ["shared-source"]
    assert report["sample_audits"]["count"] == 2
    assert plotted["samples"] == ["train:sample", "test:sample"]
    assert plotted["output"] == tmp_path / "all" / "figures"

    with np.load(tmp_path / "all" / "token_metrics.npz") as merged:
        assert merged["sample_id"].tolist() == [
            "train:sample",
            "train:sample",
            "test:sample",
            "test:sample",
        ]
        assert merged["split"].tolist() == ["train", "train", "test", "test"]
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["coverage"]["evaluated"] == 2
